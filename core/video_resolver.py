from __future__ import annotations

import asyncio
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoSource:
    """Result returned by resolve_video_url."""

    # Direct streamable URL — present when the platform provides one.
    stream_url: str | None = None
    # Local file path — present only when a file was downloaded.
    local_path: Path | None = None
    # Human-readable title from the platform, if available.
    title: str | None = None
    # Temporary directory that owns local_path; caller must call cleanup().
    _tmpdir: object = None  # tempfile.TemporaryDirectory instance

    @property
    def has_stream_url(self) -> bool:
        return bool(self.stream_url)

    @property
    def has_local_file(self) -> bool:
        return self.local_path is not None and self.local_path.exists()

    def cleanup(self) -> None:
        """Delete the temporary directory if one was created."""
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None
            self.local_path = None


def _build_ydl_opts(
    *,
    proxy: str | None = None,
    download: bool = False,
    output_template: str | None = None,
    max_filesize: int | None = None,
) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": not download,
    }
    if proxy:
        opts["proxy"] = proxy
    if output_template:
        opts["outtmpl"] = output_template
    if max_filesize is not None:
        opts["max_filesize"] = max_filesize
    return opts


def _extract_stream_url(info: dict) -> str | None:
    """Pick the best direct stream URL from yt-dlp info dict."""
    # Prefer a single consolidated url over split audio/video formats
    url = info.get("url")
    if url and url.startswith("http"):
        return url

    # Walk formats, prefer combined streams (both video and audio)
    formats: list[dict] = info.get("formats") or []
    for fmt in reversed(formats):
        vcodec = fmt.get("vcodec") or ""
        acodec = fmt.get("acodec") or ""
        fmt_url = fmt.get("url") or ""
        if vcodec not in ("none", "") and acodec not in ("none", "") and fmt_url.startswith("http"):
            return fmt_url

    # Fallback: any video stream
    for fmt in reversed(formats):
        vcodec = fmt.get("vcodec") or ""
        fmt_url = fmt.get("url") or ""
        if vcodec not in ("none", "") and fmt_url.startswith("http"):
            return fmt_url

    return None


async def resolve_video_url(
    url: str,
    *,
    proxy: str | None = None,
    allow_local_download: bool = False,
    download_dir: Path | None = None,
    max_size_bytes: int | None = None,
) -> VideoSource:
    """Resolve a platform video URL to a streamable URL or local file.

    Steps:
    1. Extract info without downloading (yt-dlp extract_info).
    2. Return stream URL if one exists.
    3. If allow_local_download is True and no usable stream URL, download the
       video to a temporary directory and return the local path.
    4. If allow_local_download is False and no stream URL, raise RuntimeError.
    """
    loop = asyncio.get_event_loop()

    # --- Step 1: extract info only (no download) ---
    info = await loop.run_in_executor(None, _extract_info_sync, url, proxy)
    title: str | None = info.get("title") if info else None

    # --- Step 2: try to get a direct stream URL ---
    stream_url: str | None = _extract_stream_url(info) if info else None
    if stream_url:
        return VideoSource(stream_url=stream_url, title=title)

    # --- Step 3: optional local download ---
    if not allow_local_download:
        raise RuntimeError(
            "无法获取视频直链，且 allow_local_download 已关闭。"
            "请在插件配置中开启 allow_local_download，或使用支持 video_url 的视觉模型。"
        )

    tmpdir = await loop.run_in_executor(
        None,
        _download_to_temp_sync,
        url,
        proxy,
        download_dir,
        max_size_bytes,
    )
    video_file = _find_downloaded_video(Path(tmpdir.name))
    if video_file is None:
        tmpdir.cleanup()
        raise RuntimeError("视频下载完成但未找到输出文件，请检查 yt-dlp 日志。")

    return VideoSource(title=title, local_path=video_file, _tmpdir=tmpdir)


# ---------------------------------------------------------------------------
# Synchronous helpers (run in executor to avoid blocking the event loop)
# ---------------------------------------------------------------------------

def _extract_info_sync(url: str, proxy: str | None) -> dict | None:
    try:
        import yt_dlp  # type: ignore[import]
    except ImportError:
        raise RuntimeError("yt-dlp 未安装，请检查 requirements.txt 并重启 AstrBot。")

    opts = _build_ydl_opts(proxy=proxy, download=False)
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            return ydl.extract_info(url, download=False)
        except Exception as exc:
            raise RuntimeError(f"yt-dlp 提取视频信息失败：{exc}") from exc


def _download_to_temp_sync(
    url: str,
    proxy: str | None,
    download_dir: Path | None,
    max_size_bytes: int | None,
) -> tempfile.TemporaryDirectory:
    try:
        import yt_dlp  # type: ignore[import]
    except ImportError:
        raise RuntimeError("yt-dlp 未安装，请检查 requirements.txt 并重启 AstrBot。")

    tmpdir = tempfile.TemporaryDirectory(
        prefix="video_chat_",
        dir=str(download_dir) if download_dir else None,
    )
    output_template = str(Path(tmpdir.name) / "%(id)s.%(ext)s")
    opts = _build_ydl_opts(
        proxy=proxy,
        download=True,
        output_template=output_template,
        max_filesize=max_size_bytes,
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([url])
        except Exception as exc:
            tmpdir.cleanup()
            raise RuntimeError(f"yt-dlp 下载失败：{exc}") from exc

    return tmpdir


_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".flv", ".avi", ".mov", ".m4v"}


def _find_downloaded_video(directory: Path) -> Path | None:
    """Return the first video file found in directory."""
    for f in directory.iterdir():
        if f.suffix.lower() in _VIDEO_SUFFIXES:
            return f
    return None