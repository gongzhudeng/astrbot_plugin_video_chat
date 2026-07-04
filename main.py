from __future__ import annotations

from pathlib import Path

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.star.star_tools import StarTools

from .core.url_extractor import extract_video_url
from .core.video_resolver import resolve_video_url
from .core.video_captioner import (
    DEFAULT_CAPTION_PROMPT,
    caption_from_url,
    caption_from_frames,
)
from .core.bili_subtitle import extract_bvid, fetch_bili_subtitle


@register(
    "灵犀 · 视频链接理解",
    "灵犀",
    "发送视频链接，AI 自动理解视频内容，支持抖音、B站（含字幕提取）、YouTube 等主流平台",
    "1.1.0",
    "https://github.com/gongzhudeng/astrbot_plugin_video_chat",
)
class VideoChatPlugin(Star):
    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config or {}

    # ------------------------------------------------------------------
    # LLM Tool: analyze_video
    # ------------------------------------------------------------------

    @filter.llm_tool(name="analyze_video")
    async def analyze_video(self, event: AstrMessageEvent, url: str = "") -> str:
        """分析视频链接内容。当用户发送或提到视频链接，或明确要求你看某个视频时调用此工具。

        Args:
            url(string): 视频链接。支持抖音、B站（BV号/链接/短链）、YouTube（含短链）、
                         微博、快手、TikTok、Twitter/X、Instagram 等平台。
        """
        # --- 1. Clean and validate the URL ---
        clean_url = extract_video_url(url.strip()) if url.strip() else None
        if not clean_url:
            return "未能识别有效的视频链接，请检查 URL 格式是否正确。"

        # --- 2. Resolve to stream URL or local file ---
        allow_dl = bool(self.config.get("allow_local_download", False))
        proxy = str(self.config.get("ytdlp_proxy", "") or "").strip() or None
        max_mb = int(self.config.get("max_video_size_mb", 200) or 200)
        max_bytes = max_mb * 1024 * 1024

        download_dir = self._resolve_download_dir()

        logger.info("[video-chat] 开始解析链接：%s", clean_url)
        try:
            source = await resolve_video_url(
                clean_url,
                proxy=proxy,
                allow_local_download=allow_dl,
                download_dir=download_dir,
                max_size_bytes=max_bytes,
            )
        except RuntimeError as exc:
            logger.warning("[video-chat] 链接解析失败：%s", exc)
            return f"视频链接解析失败：{exc}"

        # --- 3. Get the vision provider ---
        provider = self._resolve_caption_provider(event)
        if provider is None:
            source.cleanup()
            return "未找到可用的视觉模型，请在插件配置中填写 caption_provider_id，或确认当前会话已绑定模型。"

        # --- 4. Read frame extraction config ---
        prompt = str(self.config.get("caption_prompt", "") or "").strip() or DEFAULT_CAPTION_PROMPT
        fps = float(self.config.get("frames_per_second", 1.0) or 1.0)
        max_frames = max(1, int(self.config.get("max_frames", 30) or 30))
        first_secs = max(0, int(self.config.get("analyze_first_seconds", 120) or 120))
        ffmpeg_path = str(self.config.get("ffmpeg_path", "") or "").strip()

        # --- 5. B站字幕优先分支 ---
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
            # Subtitle-only path: pass text directly to LLM, skip video download
            source.cleanup()
            logger.info("[video-chat] 使用字幕文本路径（共 %d 行）", subtitle_text.count("\n") + 1)
            return (
                f"[B站视频字幕内容]\n{subtitle_text}"
            )

        # --- 6. Caption the video (frames or URL) ---
        try:
            if subtitle_text and plus_frames:
                # Subtitle + frames: need local file for frame extraction
                if not source.has_local_file:
                    # subtitle alone is good enough; no local file available
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
        """Return the directory to use for temporary video downloads."""
        custom = str(self.config.get("download_dir", "") or "").strip()
        if custom:
            p = Path(custom)
            p.mkdir(parents=True, exist_ok=True)
            return p
        # Default: <plugin data dir>/temp
        data_dir = StarTools.get_data_dir("astrbot_plugin_video_chat")
        temp_dir = data_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _resolve_caption_provider(self, event: AstrMessageEvent) -> object | None:
        """Return the vision provider to use for captioning."""
        provider_id = str(self.config.get("caption_provider_id", "") or "").strip()
        if provider_id:
            p = self.context.get_provider_by_id(provider_id)
            if p is None:
                logger.warning("[video-chat] 未找到配置的转述模型 ID：%s", provider_id)
            return p
        # Fall back to the current session provider
        session = str(getattr(event, "unified_msg_origin", "") or "")
        return self.context.get_using_provider(session)