from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
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
from .core.context_formatter import format_media_work, select_hot_comments
from .core.douyin_resolver import DouyinResult, is_douyin_url, resolve_douyin
from .core.models import HotComment, MediaWork
from .core.url_extractor import extract_video_url
from .core.video_captioner import (
    DEFAULT_CAPTION_PROMPT,
    DEFAULT_COMMENT_MEDIA_PROMPT,
    caption_comment_media,
    caption_from_frames,
    caption_from_media_urls,
    caption_from_url,
)
from .core.video_resolver import VideoSource, resolve_video_url

T = TypeVar("T")


@register(
    "灵犀 · 视频链接理解",
    "灵犀",
    "发送视频链接，AI 自动理解视频内容，支持抖音（视频/图文/动图）、B站（含字幕提取）",
    "1.3.1",
    "https://github.com/gongzhudeng/astrbot_plugin_video_chat",
)
class VideoChatPlugin(Star):
    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config or {}

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
        clean_url = extract_video_url(url.strip()) if url.strip() else None
        if not clean_url:
            return "未能识别有效的视频链接，请检查 URL 格式是否正确。"
        return await self._do_analyze(event, clean_url)

    async def _do_analyze(self, event: AstrMessageEvent, clean_url: str) -> str:
        comment_enabled = bool(self.config.get("hot_comments_enabled", True))
        comment_count = (
            max(1, int(self.config.get("hot_comment_max_count", 10) or 10))
            if comment_enabled
            else 0
        )
        comment_chars = max(
            50, int(self.config.get("hot_comment_max_chars", 500) or 500)
        )
        comment_reply_limit = (
            max(0, int(self.config.get("hot_comment_reply_count", 2) or 0))
            if comment_enabled
            else 0
        )
        first_seconds = max(0, int(self.config.get("analyze_first_seconds", 120) or 0))
        ffmpeg_path = str(self.config.get("ffmpeg_path", "") or "").strip()
        download_dir = self._resolve_download_dir()
        max_bytes = (
            max(1, int(self.config.get("max_video_size_mb", 200) or 200)) * 1024 * 1024
        )

        logger.info("[video-chat] 开始解析链接：%s", clean_url)
        with tempfile.TemporaryDirectory(
            prefix="video_chat_work_",
            dir=str(download_dir),
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            try:
                if is_bilibili_url(clean_url):
                    work = await self._analyze_bilibili(
                        event,
                        clean_url,
                        temp_dir,
                        comment_count,
                        comment_reply_limit,
                        first_seconds,
                        ffmpeg_path,
                        max_bytes,
                    )
                elif is_douyin_url(clean_url):
                    work = await self._analyze_douyin(
                        event,
                        clean_url,
                        temp_dir,
                        comment_count,
                        comment_reply_limit,
                        first_seconds,
                        ffmpeg_path,
                        max_bytes,
                    )
                else:
                    work = await self._analyze_generic(
                        event,
                        clean_url,
                        temp_dir,
                        first_seconds,
                        ffmpeg_path,
                        max_bytes,
                    )
            except Exception as exc:
                logger.exception("[video-chat] 视频解析失败：%s", exc)
                return "视频链接解析失败，请稍后重试或检查链接是否有效。"

        await self._caption_comment_media_if_enabled(
            event,
            work,
            comment_count,
            comment_chars,
            comment_reply_limit,
            ffmpeg_path,
        )
        return format_media_work(
            work,
            comment_max_count=comment_count,
            comment_max_chars=comment_chars,
            comment_reply_limit=comment_reply_limit,
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
                event, video_path, first_seconds, ffmpeg_path
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
                event, result.image_urls, first_seconds, ffmpeg_path
            )
            return work
        if not result.play_url:
            return work

        work.visual_summary = await self._caption_url_with_fallback(
            event, result.play_url
        )
        if self._stt_enabled():
            local_video = temp_dir / "douyin_video.mp4"
            try:
                await download_media(result.play_url, local_video, max_bytes=max_bytes)
                work.local_video_path = local_video
                work.transcript = await self._transcribe_with_fallback(
                    event, local_video, temp_dir, first_seconds, ffmpeg_path
                )
                if not work.visual_summary:
                    work.visual_summary = await self._caption_frames_with_fallback(
                        event, local_video, first_seconds, ffmpeg_path
                    )
            except Exception as exc:
                logger.warning("[video-chat] 抖音媒体下载失败：%s", exc)
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
                    event, source.stream_url
                )
            if source.has_local_file:
                work.local_video_path = source.local_path
                if not work.visual_summary:
                    work.visual_summary = await self._caption_frames_with_fallback(
                        event, source.local_path, first_seconds, ffmpeg_path
                    )
                if self._stt_enabled():
                    work.transcript = await self._transcribe_with_fallback(
                        event, source.local_path, temp_dir, first_seconds, ffmpeg_path
                    )
            return work
        finally:
            if source is not None:
                source.cleanup()

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

        descriptions: dict[str, str] = {}
        for provider in self._visual_providers(event):
            try:
                descriptions = await caption_comment_media(
                    media_items,
                    provider=provider,
                    prompt=self._comment_media_caption_prompt(),
                    max_media=max_media,
                    ffmpeg_path=ffmpeg_path,
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
        self, event: AstrMessageEvent, url: str
    ) -> str:
        return await self._try_visual_providers(
            event,
            lambda provider: caption_from_url(
                url,
                provider=provider,
                prompt=self._caption_prompt(),
            ),
        )

    async def _caption_frames_with_fallback(
        self,
        event: AstrMessageEvent,
        path: Path,
        first_seconds: int,
        ffmpeg_path: str,
    ) -> str:
        return await self._try_visual_providers(
            event,
            lambda provider: caption_from_frames(
                path,
                provider=provider,
                prompt=self._caption_prompt(),
                frames_per_second=float(
                    self.config.get("frames_per_second", 1.0) or 1.0
                ),
                max_frames=max(1, int(self.config.get("max_frames", 30) or 30)),
                analyze_first_seconds=first_seconds,
                ffmpeg_path=ffmpeg_path,
            ),
        )

    async def _caption_media_with_fallback(
        self,
        event: AstrMessageEvent,
        urls: list[str],
        first_seconds: int,
        ffmpeg_path: str,
    ) -> str:
        return await self._try_visual_providers(
            event,
            lambda provider: caption_from_media_urls(
                urls,
                provider=provider,
                prompt=self._caption_prompt(),
                max_media=max(1, int(self.config.get("max_images", 9) or 9)),
                frames_per_second=float(
                    self.config.get("frames_per_second", 1.0) or 1.0
                ),
                max_frames=max(1, int(self.config.get("max_frames", 30) or 30)),
                analyze_first_seconds=first_seconds,
                ffmpeg_path=ffmpeg_path,
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

    def _visual_providers(self, event: AstrMessageEvent) -> list[Provider]:
        providers: list[Provider] = []
        primary_id = str(self.config.get("caption_provider_id", "") or "").strip()
        if primary_id:
            provider = self.context.get_provider_by_id(primary_id)
            if isinstance(provider, Provider):
                providers.append(provider)
        else:
            session = str(getattr(event, "unified_msg_origin", "") or "")
            provider = self.context.get_using_provider(session)
            if provider:
                providers.append(provider)
        fallback_id = str(
            self.config.get("caption_fallback_provider_id", "") or ""
        ).strip()
        if fallback_id:
            provider = self.context.get_provider_by_id(fallback_id)
            if isinstance(provider, Provider):
                providers.append(provider)
        return self._deduplicate_providers(providers)

    def _stt_providers(self, event: AstrMessageEvent) -> list[STTProvider]:
        providers: list[STTProvider] = []
        primary_id = str(self.config.get("stt_provider_id", "") or "").strip()
        if primary_id:
            provider = self.context.get_provider_by_id(primary_id)
            if isinstance(provider, STTProvider):
                providers.append(provider)
        else:
            session = str(getattr(event, "unified_msg_origin", "") or "")
            provider = self.context.get_using_stt_provider(session)
            if provider:
                providers.append(provider)
        fallback_id = str(self.config.get("stt_fallback_provider_id", "") or "").strip()
        if fallback_id:
            provider = self.context.get_provider_by_id(fallback_id)
            if isinstance(provider, STTProvider):
                providers.append(provider)
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

    def _caption_prompt(self) -> str:
        return (
            str(self.config.get("caption_prompt", "") or "").strip()
            or DEFAULT_CAPTION_PROMPT
        )

    def _comment_media_caption_prompt(self) -> str:
        return (
            str(self.config.get("comment_media_caption_prompt", "") or "").strip()
            or DEFAULT_COMMENT_MEDIA_PROMPT
        )

    def _cookies_file(self) -> str | None:
        return (
            str(self.config.get("ytdlp_cookies_file", "") or "").strip().strip("\"'")
            or None
        )

    def _stt_enabled(self) -> bool:
        return bool(self.config.get("stt_enabled", False))
