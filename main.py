from __future__ import annotations

import tempfile
from pathlib import Path

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.star.star_tools import StarTools

from .core.url_extractor import extract_video_url
from .core.video_resolver import resolve_video_url, VideoSource
from .core.video_captioner import (
    DEFAULT_CAPTION_PROMPT,
    caption_from_url,
    caption_from_frames,
    caption_from_image_urls,
)
from .core.bili_subtitle import extract_bvid, fetch_bili_subtitle
from .core.bili_resolver import is_bilibili_url, resolve_bilibili, download_bili_stream
from .core.douyin_resolver import is_douyin_url, resolve_douyin


@register(
    "灵犀 · 视频链接理解",
    "灵犀",
    "发送视频链接，AI 自动理解视频内容，支持抖音（视频/图文）、B站（含字幕提取）",
    "1.1.0",
    "https://github.com/gongzhudeng/astrbot_plugin_video_chat",
)
class VideoChatPlugin(Star):
    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config or {}

    # ------------------------------------------------------------------
    # Command: /视频 <url>  （直接触发，不经过 LLM 中转）
    # ------------------------------------------------------------------

    @filter.command("视频")
    async def cmd_video(self, event: AstrMessageEvent) -> None:
        """直接解析视频链接，用法：/视频 <链接>"""
        raw = str(event.message_str or "").strip()
        # message_str includes the command prefix; strip leading /视频
        for prefix in ("/视频", "视频"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
                break

        url = extract_video_url(raw) if raw else None
        if not url:
            yield event.plain_result("用法：/视频 <视频链接>\n例：/视频 https://v.douyin.com/xxxx/")
            return

        yield event.plain_result(f"正在解析视频，请稍候…")
        result = await self._do_analyze(event, url)
        yield event.plain_result(result)

    # ------------------------------------------------------------------
    # LLM Tool: analyze_video
    # ------------------------------------------------------------------

    @filter.llm_tool(name="analyze_video")
    async def analyze_video(self, event: AstrMessageEvent, url: str = "") -> str:
        """分析视频链接内容。当用户发送或提到视频链接，或明确要求你看某个视频时调用此工具。

        Args:
            url(string): 视频链接。支持抖音视频/图文（v.douyin.com 短链或完整链接）、
                         B站（BV号/链接/短链 b23.tv）。
        """
        clean_url = extract_video_url(url.strip()) if url.strip() else None
        if not clean_url:
            return "未能识别有效的视频链接，请检查 URL 格式是否正确。"
        return await self._do_analyze(event, clean_url)

    # ------------------------------------------------------------------
    # Core analysis logic (shared by command and llm_tool)
    # ------------------------------------------------------------------

    async def _do_analyze(self, event: AstrMessageEvent, clean_url: str) -> str:
        allow_dl = bool(self.config.get("allow_local_download", False))
        proxy = str(self.config.get("ytdlp_proxy", "") or "").strip() or None
        max_mb = int(self.config.get("max_video_size_mb", 200) or 200)
        max_bytes = max_mb * 1024 * 1024
        cookies_file = str(self.config.get("ytdlp_cookies_file", "") or "").strip().strip("\"'") or None
        download_dir = self._resolve_download_dir()

        logger.info("[video-chat] 开始解析链接：%s", clean_url)

        # --- Douyin: native HTTP first, yt-dlp as fallback ---
        source: VideoSource | None = None
        _douyin_image_urls: list[str] = []
        if is_douyin_url(clean_url):
            logger.info("[video-chat] 抖音链接，尝试原生 HTTP 解析...")
            try:
                douyin_result = await resolve_douyin(clean_url, cookies_file=cookies_file)
            except Exception as exc:
                logger.warning("[video-chat] 原生抖音解析异常：%s", exc)
                douyin_result = None

            if douyin_result and douyin_result.is_image_post:
                # Image/note post — caption images directly, skip yt-dlp
                _douyin_image_urls = douyin_result.image_urls
            elif douyin_result and douyin_result.play_url:
                logger.info("[video-chat] 原生抖音解析成功（视频）")
                source = VideoSource(stream_url=douyin_result.play_url, title=douyin_result.title)
            else:
                logger.warning("[video-chat] 原生抖音解析失败，回落 yt-dlp")

        # --- Douyin image/note post: caption images directly, no video needed ---
        if _douyin_image_urls:
            provider = self._resolve_caption_provider(event)
            if provider is None:
                return "未找到可用的视觉模型，请在插件配置中填写 caption_provider_id，或确认当前会话已绑定模型。"
            prompt = str(self.config.get("caption_prompt", "") or "").strip() or DEFAULT_CAPTION_PROMPT
            max_images = max(1, int(self.config.get("max_images", 9) or 9))
            try:
                caption = await caption_from_image_urls(
                    _douyin_image_urls,
                    provider=provider,
                    prompt=prompt,
                    max_images=max_images,
                )
            except RuntimeError as exc:
                logger.warning("[video-chat] 图文转述失败：%s", exc)
                return f"图文内容理解失败：{exc}"
            return f"[图文内容转述]\n{caption}"

        # --- Bilibili: bilibili-api-python (curl_cffi) first, yt-dlp as fallback ---
        if source is None and is_bilibili_url(clean_url):
            logger.info("[video-chat] B站链接，尝试原生 bilibili-api 解析...")
            try:
                bili_result = await resolve_bilibili(clean_url)
            except Exception as exc:
                logger.warning("[video-chat] 原生 B站解析异常：%s", exc)
                bili_result = None

            if bili_result:
                v_url, _a_url, bili_title = bili_result
                logger.info("[video-chat] 原生 B站解析成功，开始下载视频流...")
                # B站 CDN URL 有防盗链保护，外部模型无法直接访问，必须先下载
                _bili_tmpdir = tempfile.TemporaryDirectory(
                    prefix="video_chat_bili_",
                    dir=str(download_dir) if download_dir else None,
                )
                _bili_dest = Path(_bili_tmpdir.name) / "bili_video.m4s"
                ok = await download_bili_stream(v_url, _bili_dest)
                if ok and _bili_dest.exists():
                    logger.info("[video-chat] B站视频流下载完成: %s", _bili_dest)
                    source = VideoSource(
                        local_path=_bili_dest,
                        title=bili_title,
                        _tmpdir=_bili_tmpdir,
                    )
                else:
                    _bili_tmpdir.cleanup()
                    logger.warning("[video-chat] B站视频流下载失败，回落 yt-dlp")
            else:
                logger.warning("[video-chat] 原生 B站解析失败，回落 yt-dlp")

        if source is None:
            try:
                source = await resolve_video_url(
                    clean_url,
                    proxy=proxy,
                    allow_local_download=allow_dl,
                    download_dir=download_dir,
                    max_size_bytes=max_bytes,
                    cookies_file=cookies_file,
                )
            except RuntimeError as exc:
                logger.warning("[video-chat] 链接解析失败：%s", exc)
                return f"视频链接解析失败：{exc}"

        provider = self._resolve_caption_provider(event)
        if provider is None:
            source.cleanup()
            return "未找到可用的视觉模型，请在插件配置中填写 caption_provider_id，或确认当前会话已绑定模型。"

        prompt = str(self.config.get("caption_prompt", "") or "").strip() or DEFAULT_CAPTION_PROMPT
        fps = float(self.config.get("frames_per_second", 1.0) or 1.0)
        max_frames = max(1, int(self.config.get("max_frames", 30) or 30))
        first_secs = max(0, int(self.config.get("analyze_first_seconds", 120) or 120))
        ffmpeg_path = str(self.config.get("ffmpeg_path", "") or "").strip()

        # --- Bilibili subtitle priority ---
        sessdata = str(self.config.get("bilibili_sessdata", "") or "").strip()
        subtitle_text: str | None = None
        if sessdata and extract_bvid(clean_url):
            logger.info("[video-chat] 检测到 B 站链接，尝试获取字幕...")
            try:
                subtitle_text = await fetch_bili_subtitle(clean_url, sessdata)
            except Exception as exc:
                logger.warning("[video-chat] 字幕获取异常，降级处理：%s", exc)
                subtitle_text = None

        plus_frames = bool(self.config.get("bili_subtitle_plus_frames", False))

        if subtitle_text and not plus_frames:
            source.cleanup()
            logger.info("[video-chat] 使用字幕文本路径（共 %d 行）", subtitle_text.count("\n") + 1)
            return f"[B站视频字幕内容]\n{subtitle_text}"

        try:
            if subtitle_text and plus_frames:
                if not source.has_local_file:
                    source.cleanup()
                    return f"[B站视频字幕内容]\n{subtitle_text}"
                logger.info("[video-chat] 字幕+抽帧模式")
                frame_caption = await caption_from_frames(
                    source.local_path,
                    provider=provider,
                    prompt=prompt,
                    frames_per_second=fps,
                    max_frames=max_frames,
                    analyze_first_seconds=first_secs,
                    ffmpeg_path=ffmpeg_path,
                )
                caption = f"[字幕]\n{subtitle_text}\n\n[画面转述]\n{frame_caption}"

            elif source.has_stream_url:
                logger.info("[video-chat] 使用 video_url 直传路径：%s", source.stream_url)
                try:
                    caption = await caption_from_url(
                        source.stream_url,
                        provider=provider,
                        prompt=prompt,
                    )
                except RuntimeError as exc:
                    logger.warning("[video-chat] video_url 路径失败：%s", exc)
                    if allow_dl and source.has_local_file:
                        logger.info("[video-chat] 回退到抽帧路径（本地文件）")
                        caption = await caption_from_frames(
                            source.local_path,
                            provider=provider,
                            prompt=prompt,
                            frames_per_second=fps,
                            max_frames=max_frames,
                            analyze_first_seconds=first_secs,
                            ffmpeg_path=ffmpeg_path,
                        )
                    else:
                        return (
                            f"视觉模型拒绝了 video_url 输入：{exc}\n"
                            "如需使用抽帧路径，请在插件配置中开启 allow_local_download。"
                        )

            elif source.has_local_file:
                logger.info("[video-chat] 使用抽帧路径：%s", source.local_path)
                caption = await caption_from_frames(
                    source.local_path,
                    provider=provider,
                    prompt=prompt,
                    frames_per_second=fps,
                    max_frames=max_frames,
                    analyze_first_seconds=first_secs,
                    ffmpeg_path=ffmpeg_path,
                )
            else:
                return "视频解析结果无效，既没有直链也没有本地文件。"
        except RuntimeError as exc:
            logger.warning("[video-chat] 视频转述失败：%s", exc)
            return f"视频内容理解失败：{exc}"
        finally:
            source.cleanup()

        title_prefix = f"【{source.title}】\n" if source.title else ""
        return f"{title_prefix}[视频内容转述]\n{caption}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_download_dir(self) -> Path | None:
        custom = str(self.config.get("download_dir", "") or "").strip()
        if custom:
            p = Path(custom)
            p.mkdir(parents=True, exist_ok=True)
            return p
        data_dir = StarTools.get_data_dir("astrbot_plugin_video_chat")
        temp_dir = data_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _resolve_caption_provider(self, event: AstrMessageEvent) -> object | None:
        provider_id = str(self.config.get("caption_provider_id", "") or "").strip()
        if provider_id:
            p = self.context.get_provider_by_id(provider_id)
            if p is None:
                logger.warning("[video-chat] 未找到配置的转述模型 ID：%s", provider_id)
            return p
        session = str(getattr(event, "unified_msg_origin", "") or "")
        return self.context.get_using_provider(session)