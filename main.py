from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.provider import Provider, STTProvider
from astrbot.core.star.star_tools import StarTools

from .core.audio_processor import download_media, extract_audio
from .core.bili_resolver import (
    BilibiliResult,
    download_bili_stream,
    is_bilibili_url,
    resolve_bilibili,
)
from .core.bili_subtitle import fetch_bili_subtitle
from .core.context_formatter import (
    format_media_metadata,
    format_media_work,
    select_hot_comments,
)
from .core.douyin_resolver import DouyinResult, is_douyin_url, resolve_douyin
from .core.media_input import (
    VideoReference,
    cleanup_owned_video_path,
    cleanup_resolved_video,
    direct_video_attachment_path,
    extract_direct_video_references,
    localize_direct_videos,
    remove_direct_video_attachment_parts,
    resolve_direct_video,
)
from .core.models import HotComment, MediaWork
from .core.url_extractor import extract_video_url
from .core.video_captioner import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_COMMENT_MEDIA_PROMPT,
    DEFAULT_VISUAL_REFUSAL_KEYWORDS,
    caption_comment_media,
    caption_from_frames,
    caption_from_media_urls,
    caption_from_url,
)
from .core.video_context import (
    VideoContextEntry,
    delete_video_context,
    list_video_contexts,
    prune_video_contexts,
    wrap_video_context,
)
from .core.video_resolver import VideoSource, resolve_video_url

T = TypeVar("T")
DEFAULT_DIRECT_VIDEO_QUESTION = "请概括这个视频的主要内容。"
ATTACHMENT_ONLY_PROMPTS = {"<attachment>", "[视频]"}
ATTACHMENT_ONLY_VIDEO_PATTERN = re.compile(r"\[(?:视频|Video)\d*\]", re.IGNORECASE)
ANALYSIS_FAILURE_PREFIXES = ("视频链接解析失败", "视频附件解析失败")


def _is_analysis_failure(result: str) -> bool:
    return result.startswith(ANALYSIS_FAILURE_PREFIXES)


@dataclass(frozen=True)
class AnalysisOptions:
    comment_count: int
    comment_chars: int
    comment_reply_limit: int
    first_seconds: int
    ffmpeg_path: str
    download_dir: Path
    max_bytes: int


@register(
    "灵犀 · 视频理解",
    "灵犀",
    "自动理解直发视频与抖音/B站链接，并限制历史中的完整视频解析数量",
    "2.7.0",
    "https://github.com/gongzhudeng/astrbot_plugin_video_chat",
)
class VideoChatPlugin(Star):
    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config or {}

    @filter.event_message_type(filter.EventMessageType.ALL, priority=20_000)
    async def normalize_direct_video(self, event: AstrMessageEvent) -> None:
        if not extract_direct_video_references(event):
            return
        try:
            localized = await localize_direct_videos(
                event,
                cache_dir=self._direct_video_cache_dir(),
                max_bytes=self._max_video_bytes(),
                cache_ttl_seconds=max(
                    60,
                    int(
                        self.config.get("direct_video_cache_ttl_seconds", 3600) or 3600
                    ),
                ),
                cache_max_bytes=max(
                    1,
                    int(self.config.get("direct_video_cache_max_mb", 2048) or 2048),
                )
                * 1024
                * 1024,
            )
            if localized:
                logger.info("[video-chat] 已提前本地化 %d 个直发视频", len(localized))
        except Exception as exc:
            logger.exception("[video-chat] 直发视频提前本地化失败：%s", exc)

    @filter.on_llm_request(priority=10_000)
    async def inject_video_context(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        if not self._llm_tool_enabled() and req.func_tool:
            req.func_tool.remove_tool("analyze_video")

        if not bool(self.config.get("auto_parse_video_messages", True)):
            pruned = self._prune_request_video_contexts(req, incoming_details=0)
            await self._persist_request_history(event, req, pruned=pruned)
            return

        direct_videos = extract_direct_video_references(event)
        direct_video = direct_videos[0] if direct_videos else None
        clean_url = self._extract_supported_message_url(event)
        if direct_video is None and not clean_url:
            pruned = self._prune_request_video_contexts(req, incoming_details=0)
            await self._persist_request_history(event, req, pruned=pruned)
            return

        event.set_extra("video_chat_user_context", self._video_user_context(req.prompt))
        if direct_video is not None:
            attachment_path = direct_video_attachment_path(
                list(req.extra_user_content_parts or [])
            )
            if attachment_path:
                direct_video = replace(direct_video, path=attachment_path)
            result = await self._analyze_direct_video(event, direct_video)
            source_key = "direct-video"
        else:
            result = await self._do_analyze(event, clean_url)
            source_key = clean_url

        if not result.strip() or _is_analysis_failure(result):
            pruned = self._prune_request_video_contexts(req, incoming_details=0)
            await self._persist_request_history(event, req, pruned=pruned)
            if result.strip():
                req.extra_user_content_parts.append(TextPart(text=result.strip()))
            return

        limit = self._video_context_limit()
        pruned = self._prune_request_video_contexts(req, incoming_details=1)
        if direct_video is not None:
            req.extra_user_content_parts = remove_direct_video_attachment_parts(
                list(req.extra_user_content_parts or [])
            )
        prompt = str(req.prompt or "").strip()
        if (
            not prompt
            or prompt in ATTACHMENT_ONLY_PROMPTS
            or ATTACHMENT_ONLY_VIDEO_PATTERN.fullmatch(prompt)
        ):
            req.prompt = DEFAULT_DIRECT_VIDEO_QUESTION

        video_context = wrap_video_context(result)
        context_part = TextPart(text=video_context)
        if limit == 0:
            context_part.mark_as_temp()
        req.extra_user_content_parts.append(context_part)
        event.set_extra("video_chat_processed_source", source_key)
        details_hash = hashlib.sha256(result.encode("utf-8")).hexdigest()[:12]
        logger.info(
            "[video-chat] 视频解析已组装：source=%s mode=%s details_chars=%d "
            "details_hash=%s request_has_video_context=true",
            source_key,
            "temporary" if limit == 0 else "persistent",
            len(result),
            details_hash,
        )
        await self._persist_request_history(
            event,
            req,
            pruned=pruned,
            include_current_user=True,
            current_video_context=video_context if limit > 0 else None,
        )

    @filter.command("清理视频上下文")
    async def cmd_clear_video_context(self, event: AstrMessageEvent) -> None:
        """清理当前会话历史中的全部视频解析详情。"""
        conversation, history, error = await self._load_current_video_history(event)
        if error:
            yield event.plain_result(error)
            return
        assert conversation is not None and history is not None

        pruned = prune_video_contexts(history, max_details=0)
        if not pruned:
            yield event.plain_result("当前会话没有完整视频解析详情。")
            return

        error = await self._save_video_history(event, conversation, history)
        if error:
            yield event.plain_result(error)
            return
        yield event.plain_result(f"已清理 {pruned} 个完整视频解析详情。")

    @filter.command("视频上下文")
    async def cmd_list_video_context(self, event: AstrMessageEvent) -> None:
        """查看当前会话中保留的完整视频解析详情。"""
        _, history, error = await self._load_current_video_history(event)
        if error:
            yield event.plain_result(error)
            return
        assert history is not None

        entries = list_video_contexts(history)
        if not entries:
            yield event.plain_result("当前会话没有完整视频解析详情。")
            return
        lines = [f"当前会话共有 {len(entries)} 个完整视频解析详情："]
        lines.extend(
            f"{index}. {self._format_video_context_entry(entry)}"
            for index, entry in enumerate(entries, 1)
        )
        lines.append("使用 /删视频上下文 <序号> 可删除指定详情。")
        yield event.plain_result("\n".join(lines))

    @filter.command("删视频上下文")
    async def cmd_delete_video_context(
        self,
        event: AstrMessageEvent,
        index: str = "",
    ) -> None:
        """按序号删除当前会话中的视频解析详情。"""
        try:
            selected_index = int(str(index or "").strip())
        except ValueError:
            yield event.plain_result("用法：/删视频上下文 <序号>")
            return
        if selected_index < 1:
            yield event.plain_result("序号必须是大于 0 的整数。")
            return

        conversation, history, error = await self._load_current_video_history(event)
        if error:
            yield event.plain_result(error)
            return
        assert conversation is not None and history is not None

        entries = list_video_contexts(history)
        if not entries:
            yield event.plain_result("当前会话没有完整视频解析详情。")
            return
        if selected_index > len(entries):
            yield event.plain_result(
                f"序号超出范围，当前共有 {len(entries)} 个视频解析详情。"
            )
            return

        deleted = delete_video_context(history, selected_index)
        assert deleted is not None
        error = await self._save_video_history(event, conversation, history)
        if error:
            yield event.plain_result(error)
            return
        yield event.plain_result(
            "已删除视频上下文：" + self._format_video_context_entry(deleted)
        )

    async def _load_current_video_history(
        self,
        event: AstrMessageEvent,
    ) -> tuple[Any | None, list[Any] | None, str | None]:
        manager = getattr(getattr(self, "context", None), "conversation_manager", None)
        if manager is None:
            return None, None, "当前会话管理器不可用，未能读取视频上下文。"
        try:
            conversation_id = await manager.get_curr_conversation_id(
                event.unified_msg_origin
            )
            conversation = (
                await manager.get_conversation(
                    event.unified_msg_origin,
                    conversation_id,
                )
                if conversation_id
                else None
            )
        except Exception as exc:
            logger.warning("[video-chat] 获取当前会话失败：%s", exc)
            return None, None, "获取当前会话失败，未能读取视频上下文。"
        if conversation is None:
            return None, None, "当前没有可读取的会话历史。"
        try:
            history = json.loads(getattr(conversation, "history", "[]") or "[]")
        except (TypeError, ValueError):
            history = []
        return conversation, history if isinstance(history, list) else [], None

    async def _save_video_history(
        self,
        event: AstrMessageEvent,
        conversation: Any,
        history: list[Any],
    ) -> str | None:
        manager = getattr(getattr(self, "context", None), "conversation_manager", None)
        if manager is None:
            return "当前会话管理器不可用，未能更新视频上下文。"
        try:
            await manager.update_conversation(
                event.unified_msg_origin,
                conversation.cid,
                history=history,
            )
            conversation.history = json.dumps(history, ensure_ascii=False)
        except Exception as exc:
            logger.warning("[video-chat] 更新视频上下文失败：%s", exc)
            return "更新视频上下文失败，请稍后重试。"
        return None

    @staticmethod
    def _format_video_context_entry(entry: VideoContextEntry) -> str:
        identity = entry.title or entry.summary or entry.url or "未命名视频"
        identity = " ".join(identity.split())
        if len(identity) > 80:
            identity = identity[:77] + "..."
        platform = entry.platform or "未知平台"
        return f"[{entry.kind}] {platform} · {identity}"

    @filter.command("视频")
    async def cmd_video(self, event: AstrMessageEvent) -> None:
        """直接解析视频链接，用法：/视频 <链接>"""
        raw = str(event.message_str or "").strip()
        for prefix in ("/视频", "视频"):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :].strip()
                break
        url = extract_video_url(raw) if raw else None
        if not url:
            yield event.plain_result(
                "用法：/视频 <视频链接>\n例：/视频 https://v.douyin.com/xxxx/"
            )
            return
        yield event.plain_result("正在解析视频，请稍候…")
        yield event.plain_result(await self._do_analyze(event, url))

    @filter.llm_tool(name="analyze_video")
    async def analyze_video(self, event: AstrMessageEvent, url: str = "") -> str:
        """分析视频链接。结合当前人设自然参考高赞评论，不要强制逐条点评评论。

        Args:
            url(string): 抖音视频/图文链接，或 B 站 BV/av/短链。
        """
        if not self._llm_tool_enabled():
            return "视频分析工具已关闭。请让用户直接发送视频或明确的视频链接。"
        clean_url = extract_video_url(url.strip()) if url.strip() else None
        if not clean_url:
            return "未能识别有效的视频链接，请检查 URL 格式是否正确。"
        if event.get_extra("video_chat_processed_source"):
            return (
                "当前请求已解析一个视频，请直接根据已有解析结果回答用户，"
                "不要继续分析其他媒体。"
            )
        event.set_extra("video_chat_processed_source", clean_url)
        return await self._do_analyze(event, clean_url)

    def _llm_tool_enabled(self) -> bool:
        return bool(self.config.get("analyze_video_tool_enabled", False))

    def _analysis_options(self) -> AnalysisOptions:
        comment_enabled = bool(self.config.get("hot_comments_enabled", True))
        comment_count = (
            max(1, int(self.config.get("hot_comment_max_count", 10) or 10))
            if comment_enabled
            else 0
        )
        return AnalysisOptions(
            comment_count=comment_count,
            comment_chars=max(
                50, int(self.config.get("hot_comment_max_chars", 500) or 500)
            ),
            comment_reply_limit=(
                max(0, int(self.config.get("hot_comment_reply_count", 2) or 0))
                if comment_enabled
                else 0
            ),
            first_seconds=max(
                0, int(self.config.get("analyze_first_seconds", 120) or 0)
            ),
            ffmpeg_path=str(self.config.get("ffmpeg_path", "") or "").strip(),
            download_dir=self._resolve_download_dir(),
            max_bytes=max(1, int(self.config.get("max_video_size_mb", 200) or 200))
            * 1024
            * 1024,
        )

    async def _do_analyze(self, event: AstrMessageEvent, clean_url: str) -> str:
        options = self._analysis_options()

        logger.info("[video-chat] 开始解析链接：%s", clean_url)
        with tempfile.TemporaryDirectory(
            prefix="video_chat_work_",
            dir=str(options.download_dir),
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            try:
                if is_bilibili_url(clean_url):
                    work = await self._analyze_bilibili(
                        event,
                        clean_url,
                        temp_dir,
                        options.comment_count,
                        options.comment_reply_limit,
                        options.first_seconds,
                        options.ffmpeg_path,
                        options.max_bytes,
                    )
                elif is_douyin_url(clean_url):
                    work = await self._analyze_douyin(
                        event,
                        clean_url,
                        temp_dir,
                        options.comment_count,
                        options.comment_reply_limit,
                        options.first_seconds,
                        options.ffmpeg_path,
                        options.max_bytes,
                    )
                else:
                    work = await self._analyze_generic(
                        event,
                        clean_url,
                        temp_dir,
                        options.first_seconds,
                        options.ffmpeg_path,
                        options.max_bytes,
                    )
            except Exception as exc:
                logger.exception("[video-chat] 视频解析失败：%s", exc)
                return "视频链接解析失败，请稍后重试或检查链接是否有效。"

        return await self._finalize_work(event, work, options)

    async def _analyze_direct_video(
        self, event: AstrMessageEvent, reference: VideoReference
    ) -> str:
        options = self._analysis_options()
        resolved = None
        completed = False
        try:
            resolved = await resolve_direct_video(
                reference, max_bytes=options.max_bytes
            )
            with tempfile.TemporaryDirectory(
                prefix="video_chat_work_",
                dir=str(options.download_dir),
            ) as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                work = MediaWork(
                    platform="本地视频",
                    source_url="",
                    work_type="视频",
                    local_video_path=resolved.path,
                )
                work.visual_summary = await self._caption_frames_with_fallback(
                    event,
                    resolved.path,
                    options.first_seconds,
                    options.ffmpeg_path,
                    work,
                )
                if self._stt_enabled():
                    work.transcript = await self._transcribe_with_fallback(
                        event,
                        resolved.path,
                        temp_dir,
                        options.first_seconds,
                        options.ffmpeg_path,
                    )
                result = await self._finalize_work(event, work, options)
                completed = True
                return result
        except Exception as exc:
            logger.exception("[video-chat] 直发视频解析失败：%s", exc)
            return "视频附件解析失败，请重新发送视频或检查文件是否有效。"
        finally:
            if completed and resolved is not None:
                cleanup_owned_video_path(
                    resolved.path,
                    self._direct_video_cache_dir(),
                )
            cleanup_resolved_video(resolved)

    async def _finalize_work(
        self,
        event: AstrMessageEvent,
        work: MediaWork,
        options: AnalysisOptions,
    ) -> str:
        await self._caption_comment_media_if_enabled(
            event,
            work,
            options.comment_count,
            options.comment_chars,
            options.comment_reply_limit,
            options.ffmpeg_path,
        )
        return format_media_work(
            work,
            comment_max_count=options.comment_count,
            comment_max_chars=options.comment_chars,
            comment_reply_limit=options.comment_reply_limit,
        )

    def _extract_supported_message_url(self, event: AstrMessageEvent) -> str | None:
        raw = str(event.message_str or "").strip()
        clean_url = extract_video_url(raw) if raw else None
        if clean_url and (is_bilibili_url(clean_url) or is_douyin_url(clean_url)):
            return clean_url
        return None

    def _video_context_limit(self) -> int:
        return max(0, int(self.config.get("max_video_context_details", 3) or 0))

    def _prune_request_video_contexts(
        self,
        req: ProviderRequest,
        *,
        incoming_details: int,
    ) -> int:
        pruned = prune_video_contexts(
            req.contexts,
            max_details=self._video_context_limit(),
            incoming_details=incoming_details,
        )
        if pruned:
            logger.info("[video-chat] 已清理 %d 个旧视频解析上下文", pruned)
        return pruned

    async def _persist_request_history(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
        *,
        pruned: int,
        include_current_user: bool = False,
        current_video_context: str | None = None,
    ) -> None:
        if not pruned and not include_current_user:
            return

        conversation = getattr(req, "conversation", None)
        manager = getattr(getattr(self, "context", None), "conversation_manager", None)
        conversation_id = getattr(conversation, "cid", None)
        if manager is None or not conversation_id:
            return

        history = copy.deepcopy(list(req.contexts or []))
        if include_current_user:
            content: list[dict[str, str]] = []
            prompt = str(req.prompt or "").strip()
            if prompt:
                content.append({"type": "text", "text": prompt})
            if current_video_context:
                content.append({"type": "text", "text": current_video_context})
            if content:
                history.append({"role": "user", "content": content})

        video_context_count = len(list_video_contexts(history))
        try:
            await manager.update_conversation(
                event.unified_msg_origin,
                conversation_id,
                history=history,
            )
        except Exception as exc:
            logger.warning("[video-chat] 视频上下文历史写入失败：%s", exc)
            return

        try:
            conversation.history = json.dumps(history, ensure_ascii=False)
        except Exception:
            pass
        logger.info(
            "[video-chat] 视频上下文历史已更新：cid=%s pruned=%d "
            "history_video_contexts=%d current_user_saved=%s",
            conversation_id,
            pruned,
            video_context_count,
            include_current_user,
        )

    async def _analyze_bilibili(
        self,
        event: AstrMessageEvent,
        url: str,
        temp_dir: Path,
        comment_count: int,
        comment_reply_limit: int,
        first_seconds: int,
        ffmpeg_path: str,
        max_bytes: int,
    ) -> MediaWork:
        metadata = await resolve_bilibili(
            url,
            include_media=False,
            comment_count=comment_count,
            comment_reply_limit=comment_reply_limit,
        )
        if metadata is None:
            raise RuntimeError("B站作品信息解析失败")
        work = self._work_from_bilibili(metadata, url)

        sessdata = str(self.config.get("bilibili_sessdata", "") or "").strip()
        if sessdata:
            try:
                work.subtitle = (
                    await fetch_bili_subtitle(metadata.canonical_url, sessdata) or ""
                )
            except Exception as exc:
                logger.warning("[video-chat] B站字幕获取失败：%s", exc)

        plus_frames = bool(self.config.get("bili_subtitle_plus_frames", False))
        if work.subtitle and not plus_frames:
            return work

        media = await resolve_bilibili(url, include_media=True, comment_count=0)
        if media is None or not media.video_url:
            return work
        video_path = temp_dir / "bili_video.m4s"
        if await download_bili_stream(media.video_url, video_path):
            work.local_video_path = video_path
            work.visual_summary = await self._caption_frames_with_fallback(
                event, video_path, first_seconds, ffmpeg_path, work
            )
        if not work.subtitle and self._stt_enabled():
            audio_source = work.local_video_path
            if media.audio_url:
                audio_source = temp_dir / "bili_audio.m4s"
                try:
                    await download_media(
                        media.audio_url,
                        audio_source,
                        headers={"Referer": "https://www.bilibili.com/"},
                        max_bytes=max_bytes,
                    )
                except Exception as exc:
                    logger.warning("[video-chat] B站音轨下载失败：%s", exc)
            if audio_source and audio_source.exists():
                work.transcript = await self._transcribe_with_fallback(
                    event, audio_source, temp_dir, first_seconds, ffmpeg_path
                )
        return work

    async def _analyze_douyin(
        self,
        event: AstrMessageEvent,
        url: str,
        temp_dir: Path,
        comment_count: int,
        comment_reply_limit: int,
        first_seconds: int,
        ffmpeg_path: str,
        max_bytes: int,
    ) -> MediaWork:
        cookies_file = self._cookies_file()
        result = await resolve_douyin(
            url,
            cookies_file=cookies_file,
            comment_count=comment_count,
            comment_reply_limit=comment_reply_limit,
            comment_cdp_fallback_enabled=bool(
                self.config.get("douyin_comment_browser_fallback_enabled", False)
            ),
            comment_cdp_url=str(
                self.config.get("douyin_comment_cdp_url", "http://127.0.0.1:9222")
                or "http://127.0.0.1:9222"
            ).strip(),
        )
        if result is None:
            return await self._analyze_generic(
                event, url, temp_dir, first_seconds, ffmpeg_path, max_bytes
            )
        work = self._work_from_douyin(result, url)
        if result.image_urls:
            work.visual_summary = await self._caption_media_with_fallback(
                event, result.image_urls, first_seconds, ffmpeg_path, work
            )
            return work
        if not result.play_url:
            return work

        # Do not send Douyin's remote play_url as a native ``video_url`` part to
        # the chat provider.  Some OpenAI-compatible providers retain or reuse
        # native video media state, which can make subsequent main-chat requests
        # carry tens of thousands of extra prompt tokens.  Materializing the
        # video locally and sending sampled frames keeps the main conversation
        # text-only after this hook finishes.
        local_video = temp_dir / "douyin_video.mp4"
        try:
            await download_media(result.play_url, local_video, max_bytes=max_bytes)
            work.local_video_path = local_video
            work.visual_summary = await self._caption_frames_with_fallback(
                event, local_video, first_seconds, ffmpeg_path, work
            )
            if self._stt_enabled():
                work.transcript = await self._transcribe_with_fallback(
                    event, local_video, temp_dir, first_seconds, ffmpeg_path
                )
        except Exception as exc:
            logger.warning("[video-chat] 抖音媒体下载或抽帧失败：%s", exc)
        return work

    async def _analyze_generic(
        self,
        event: AstrMessageEvent,
        url: str,
        temp_dir: Path,
        first_seconds: int,
        ffmpeg_path: str,
        max_bytes: int,
    ) -> MediaWork:
        source: VideoSource | None = None
        try:
            source = await resolve_video_url(
                url,
                proxy=str(self.config.get("ytdlp_proxy", "") or "").strip() or None,
                allow_local_download=bool(
                    self.config.get("allow_local_download", False)
                ),
                download_dir=self._resolve_download_dir(),
                max_size_bytes=max_bytes,
                cookies_file=self._cookies_file(),
            )
            work = MediaWork(platform="其他", source_url=url, title=source.title or "")
            if source.has_stream_url:
                work.visual_summary = await self._caption_url_with_fallback(
                    event, source.stream_url, work
                )
            if source.has_local_file:
                work.local_video_path = source.local_path
                if not work.visual_summary:
                    work.visual_summary = await self._caption_frames_with_fallback(
                        event, source.local_path, first_seconds, ffmpeg_path, work
                    )
                if self._stt_enabled():
                    work.transcript = await self._transcribe_with_fallback(
                        event, source.local_path, temp_dir, first_seconds, ffmpeg_path
                    )
            return work
        finally:
            if source is not None:
                source.cleanup()

    def _image_preprocess_options(self) -> tuple[bool, int, int]:
        enabled = bool(self.config.get("image_preprocess_enabled", False))
        max_size = max(
            256, int(self.config.get("image_preprocess_max_size", 1280) or 1280)
        )
        quality = min(
            100,
            max(50, int(self.config.get("image_preprocess_quality", 85) or 85)),
        )
        return enabled, max_size, quality

    async def _caption_comment_media_if_enabled(
        self,
        event: AstrMessageEvent,
        work: MediaWork,
        comment_count: int,
        comment_chars: int,
        reply_limit: int,
        ffmpeg_path: str,
    ) -> None:
        if not bool(self.config.get("comment_media_caption_enabled", False)):
            return
        max_media = max(1, int(self.config.get("comment_media_max_count", 6) or 6))
        selected = select_hot_comments(
            work.comments,
            max_count=comment_count,
            max_chars=comment_chars,
            reply_limit=reply_limit,
        )
        work.comments = selected
        media_items: list[tuple[str, str]] = []
        owners: dict[str, HotComment] = {}

        def collect(comment: HotComment, prefix: str) -> None:
            for index, url in enumerate(comment.media_urls):
                if len(media_items) >= max_media:
                    return
                media_id = f"{prefix}-{index + 1}"
                media_items.append((media_id, url))
                owners[media_id] = comment
            for index, reply in enumerate(comment.replies):
                if len(media_items) >= max_media:
                    return
                collect(reply, f"{prefix}R{index + 1}")

        for index, comment in enumerate(selected, 1):
            collect(comment, f"C{index}")
            if len(media_items) >= max_media:
                break
        if not media_items:
            return

        preprocess_enabled, preprocess_max_size, preprocess_quality = (
            self._image_preprocess_options()
        )
        descriptions: dict[str, str] = {}
        for provider in self._visual_providers(event):
            try:
                descriptions = await caption_comment_media(
                    media_items,
                    provider=provider,
                    prompt=self._comment_media_caption_prompt(),
                    max_media=max_media,
                    ffmpeg_path=ffmpeg_path,
                    preprocess_enabled=preprocess_enabled,
                    preprocess_max_size=preprocess_max_size,
                    preprocess_quality=preprocess_quality,
                    refusal_keywords=self._visual_refusal_keywords(),
                )
                if descriptions:
                    break
            except Exception as exc:
                logger.warning(
                    "[video-chat] 评论图片模型 %s 调用失败，尝试下一个：%s",
                    self._provider_name(provider),
                    exc,
                )
        for media_id, description in descriptions.items():
            owner = owners.get(media_id)
            if owner is not None:
                owner.media_descriptions.append(description)

    async def _caption_url_with_fallback(
        self,
        event: AstrMessageEvent,
        url: str,
        work: MediaWork | None = None,
    ) -> str:
        return await self._try_visual_providers(
            event,
            lambda provider: caption_from_url(
                url,
                provider=provider,
                prompt=self._caption_prompt(event),
                user_context=self._caption_user_context(event),
                video_info=self._video_info_for_caption(work),
                refusal_keywords=self._visual_refusal_keywords(),
            ),
        )

    async def _caption_frames_with_fallback(
        self,
        event: AstrMessageEvent,
        path: Path,
        first_seconds: int,
        ffmpeg_path: str,
        work: MediaWork | None = None,
    ) -> str:
        preprocess_enabled, preprocess_max_size, preprocess_quality = (
            self._image_preprocess_options()
        )
        return await self._try_visual_providers(
            event,
            lambda provider: caption_from_frames(
                path,
                provider=provider,
                prompt=self._caption_prompt(event),
                user_context=self._caption_user_context(event),
                video_info=self._video_info_for_caption(work),
                frames_per_second=float(
                    self.config.get("frames_per_second", 1.0) or 1.0
                ),
                max_frames=max(1, int(self.config.get("max_frames", 30) or 30)),
                analyze_first_seconds=first_seconds,
                ffmpeg_path=ffmpeg_path,
                preprocess_max_size=(preprocess_max_size if preprocess_enabled else 0),
                preprocess_quality=preprocess_quality,
                refusal_keywords=self._visual_refusal_keywords(),
            ),
        )

    async def _caption_media_with_fallback(
        self,
        event: AstrMessageEvent,
        urls: list[str],
        first_seconds: int,
        ffmpeg_path: str,
        work: MediaWork | None = None,
    ) -> str:
        preprocess_enabled, preprocess_max_size, preprocess_quality = (
            self._image_preprocess_options()
        )
        return await self._try_visual_providers(
            event,
            lambda provider: caption_from_media_urls(
                urls,
                provider=provider,
                prompt=self._caption_prompt(event),
                user_context=self._caption_user_context(event),
                video_info=self._video_info_for_caption(work),
                max_media=max(1, int(self.config.get("max_images", 9) or 9)),
                frames_per_second=float(
                    self.config.get("frames_per_second", 1.0) or 1.0
                ),
                max_frames=max(1, int(self.config.get("max_frames", 30) or 30)),
                analyze_first_seconds=first_seconds,
                ffmpeg_path=ffmpeg_path,
                preprocess_enabled=preprocess_enabled,
                preprocess_max_size=preprocess_max_size,
                preprocess_quality=preprocess_quality,
                refusal_keywords=self._visual_refusal_keywords(),
            ),
        )

    async def _try_visual_providers(
        self,
        event: AstrMessageEvent,
        operation: Callable[[Provider], Awaitable[str]],
    ) -> str:
        for provider in self._visual_providers(event):
            try:
                result = await operation(provider)
                if result.strip():
                    return result.strip()
            except Exception as exc:
                logger.warning(
                    "[video-chat] 视觉模型 %s 调用失败，尝试下一个：%s",
                    self._provider_name(provider),
                    exc,
                )
        return ""

    async def _transcribe_with_fallback(
        self,
        event: AstrMessageEvent,
        source: Path,
        temp_dir: Path,
        first_seconds: int,
        ffmpeg_path: str,
    ) -> str:
        follow_visual = bool(self.config.get("stt_follow_visual_duration", True))
        max_seconds = first_seconds if follow_visual else 0
        audio_path = temp_dir / "stt_audio.wav"
        try:
            await extract_audio(
                source,
                audio_path,
                ffmpeg_path=ffmpeg_path,
                max_seconds=max_seconds,
            )
        except Exception as exc:
            logger.warning("[video-chat] STT 音频准备失败：%s", exc)
            return ""

        for provider in self._stt_providers(event):
            try:
                text = str(
                    await provider.get_text(audio_url=str(audio_path)) or ""
                ).strip()
                if text:
                    return text
            except Exception as exc:
                logger.warning(
                    "[video-chat] STT 模型 %s 调用失败，尝试下一个：%s",
                    self._provider_name(provider),
                    exc,
                )
        return ""

    def _configured_fallback_ids(
        self, list_key: str, legacy_keys: tuple[str, ...]
    ) -> list[str]:
        if list_key in self.config:
            configured = self.config.get(list_key)
            if isinstance(configured, list):
                return [str(item).strip() for item in configured if str(item).strip()]
            logger.warning("[video-chat] 回退 Provider 配置不是列表：%s", list_key)
            return []
        return [
            value
            for key in legacy_keys
            if (value := str(self.config.get(key, "") or "").strip())
        ]

    def _visual_providers(self, event: AstrMessageEvent) -> list[Provider]:
        providers: list[Provider] = []
        primary_id = str(self.config.get("caption_provider_id", "") or "").strip()
        if primary_id:
            provider = self.context.get_provider_by_id(primary_id)
            if isinstance(provider, Provider):
                providers.append(provider)
            else:
                logger.warning(
                    "[video-chat] 视频转述 Provider 不可用或类型不正确：%s",
                    primary_id,
                )
        else:
            session = str(getattr(event, "unified_msg_origin", "") or "")
            provider = self.context.get_using_provider(session)
            if isinstance(provider, Provider):
                providers.append(provider)

        for fallback_id in self._configured_fallback_ids(
            "caption_fallback_provider_ids", ("caption_fallback_provider_id",)
        ):
            provider = self.context.get_provider_by_id(fallback_id)
            if isinstance(provider, Provider):
                providers.append(provider)
            else:
                logger.warning(
                    "[video-chat] 视频转述回退 Provider 不可用或类型不正确：%s",
                    fallback_id,
                )
        return self._deduplicate_providers(providers)

    def _stt_providers(self, event: AstrMessageEvent) -> list[STTProvider]:
        providers: list[STTProvider] = []
        primary_id = str(self.config.get("stt_provider_id", "") or "").strip()
        if primary_id:
            provider = self.context.get_provider_by_id(primary_id)
            if isinstance(provider, STTProvider):
                providers.append(provider)
            else:
                logger.warning(
                    "[video-chat] STT Provider 不可用或类型不正确：%s",
                    primary_id,
                )
        else:
            session = str(getattr(event, "unified_msg_origin", "") or "")
            provider = self.context.get_using_stt_provider(session)
            if isinstance(provider, STTProvider):
                providers.append(provider)

        for fallback_id in self._configured_fallback_ids(
            "stt_fallback_provider_ids", ("stt_fallback_provider_id",)
        ):
            provider = self.context.get_provider_by_id(fallback_id)
            if isinstance(provider, STTProvider):
                providers.append(provider)
            else:
                logger.warning(
                    "[video-chat] STT 回退 Provider 不可用或类型不正确：%s",
                    fallback_id,
                )
        return self._deduplicate_providers(providers)

    @staticmethod
    def _deduplicate_providers(providers: list[T]) -> list[T]:
        result: list[T] = []
        seen: set[int] = set()
        for provider in providers:
            identity = id(provider)
            if identity not in seen:
                seen.add(identity)
                result.append(provider)
        return result

    @staticmethod
    def _provider_name(provider: Any) -> str:
        try:
            return str(provider.meta().id)
        except Exception:
            return type(provider).__name__

    @staticmethod
    def _work_from_bilibili(result: BilibiliResult, source_url: str) -> MediaWork:
        return MediaWork(
            platform="哔哩哔哩",
            source_url=source_url,
            work_id=result.bvid or str(result.aid),
            title=result.title,
            description=result.description,
            topics=result.topics,
            author=result.author,
            author_id=result.author_id,
            published_at=result.published_at,
            comments=result.comments,
        )

    @staticmethod
    def _work_from_douyin(result: DouyinResult, source_url: str) -> MediaWork:
        return MediaWork(
            platform="抖音",
            source_url=source_url,
            work_type="图文/动图" if result.image_urls else "视频",
            work_id=result.aweme_id,
            title=result.title,
            description=result.description,
            topics=result.topics,
            author=result.author,
            author_id=result.author_id,
            published_at=result.published_at,
            video_url=result.play_url,
            image_urls=result.image_urls,
            comments=result.comments,
        )

    def _resolve_download_dir(self) -> Path:
        custom = str(self.config.get("download_dir", "") or "").strip()
        if custom:
            path = Path(custom)
            path.mkdir(parents=True, exist_ok=True)
            return path
        path = StarTools.get_data_dir("astrbot_plugin_video_chat") / "temp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _direct_video_cache_dir(self) -> Path:
        path = (
            StarTools.get_data_dir("astrbot_plugin_video_chat") / "direct_video_cache"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _max_video_bytes(self) -> int:
        return (
            max(1, int(self.config.get("max_video_size_mb", 200) or 200)) * 1024 * 1024
        )

    def _video_user_context(self, prompt: str | None) -> str:
        value = str(prompt or "")
        value = re.sub(
            r"<image_context\b[^>]*>.*?</image_context>", "", value, flags=re.DOTALL
        )
        value = re.sub(
            r"<quoted_message\b[^>]*>.*?</quoted_message>", "", value, flags=re.DOTALL
        )
        value = re.sub(r"\[(?:视频|Video)(?:\d+)?\]", "", value, flags=re.IGNORECASE)
        value = " ".join(value.split()).strip()
        if not value or value in ATTACHMENT_ONLY_PROMPTS:
            return ""
        max_chars = max(
            0, int(self.config.get("video_user_context_max_chars", 500) or 0)
        )
        return value[:max_chars] if max_chars else ""

    def _caption_user_context(self, event: AstrMessageEvent | None) -> str:
        if event is None:
            return ""
        context = str(event.get_extra("video_chat_user_context", "") or "").strip()
        if not context:
            return ""
        return f"<user_context>\n用户同轮聊天记录：\n{context}\n</user_context>"

    @staticmethod
    def _video_info_for_caption(work: MediaWork | None) -> str:
        if work is None:
            return ""
        metadata = format_media_metadata(work)
        if work.source_url:
            metadata += f"\n链接：{work.source_url}"
        return f"<video_info>\n视频信息：\n{metadata}\n</video_info>"

    def _caption_prompt(self, event: AstrMessageEvent | None = None) -> str:
        base = (
            str(self.config.get("caption_prompt", "") or "").strip()
            or DEFAULT_CAPTION_PROMPT
        )
        if "视频信息" in base or "视频元信息" in base:
            return base
        return (
            f"{base}\n\n"
            "## 五、额外补充信息\n"
            "后方可能提供两类辅助信息：用户聊天记录（与该视频同轮发送），"
            "以及平台提供的视频信息（例如作者、标题、描述、话题标签和发布时间）。"
            "请参考它们理解用户的关注重点和视频背景，但不要把辅助信息当作画面中已经出现的事实。"
        )

    def _comment_media_caption_prompt(self) -> str:
        return (
            str(self.config.get("comment_media_caption_prompt", "") or "").strip()
            or DEFAULT_COMMENT_MEDIA_PROMPT
        )

    def _visual_refusal_keywords(self) -> list[str]:
        configured = self.config.get(
            "caption_refusal_keywords", list(DEFAULT_VISUAL_REFUSAL_KEYWORDS)
        )
        if not isinstance(configured, list):
            logger.warning("[video-chat] 视频模型拒绝判定关键词配置不是列表")
            return []
        return [str(keyword).strip() for keyword in configured if str(keyword).strip()]

    def _cookies_file(self) -> str | None:
        return (
            str(self.config.get("ytdlp_cookies_file", "") or "").strip().strip("\"'")
            or None
        )

    def _stt_enabled(self) -> bool:
        return bool(self.config.get("stt_enabled", False))
