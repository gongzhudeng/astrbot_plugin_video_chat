from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from astrbot.api.message_components import Video
from astrbot.core.message.components import ComponentType

from .audio_processor import download_media

DIRECT_VIDEO_ATTACHMENT_PREFIX = "[Video Attachment:"
DIRECT_VIDEO_ATTACHMENT_PATH_MARKER = ", path "
SUPPORTED_VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
}


@dataclass(frozen=True)
class VideoReference:
    file: str = ""
    path: str = ""
    url: str = ""


@dataclass(frozen=True)
class ResolvedVideoInput:
    path: Path
    cleanup: bool = False
    source: str = "direct"


def _is_video_component(item: Any) -> bool:
    if isinstance(item, Video):
        return True
    value = getattr(item, "type", None)
    if value == ComponentType.Video:
        return True
    return str(getattr(value, "value", value) or "").lower() == "video"


def _is_http_url(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("http://", "https://"))


def _raw_message_segments(event: Any) -> list[dict[str, Any]]:
    raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
    if raw is None:
        return []
    if isinstance(raw, dict):
        message = raw.get("message")
    else:
        try:
            message = raw.get("message")
        except (AttributeError, TypeError):
            message = getattr(raw, "message", None)
    return [segment for segment in message or [] if isinstance(segment, dict)]


def _raw_video_urls(event: Any) -> list[str]:
    urls: list[str] = []
    for segment in _raw_message_segments(event):
        if segment.get("type") != "video":
            continue
        data = segment.get("data")
        if not isinstance(data, dict):
            continue
        url = str(data.get("url", "") or "").strip()
        if _is_http_url(url):
            urls.append(url)
    return urls


def _video_suffix(reference: VideoReference) -> str:
    for candidate in (reference.path, reference.file, reference.url):
        parsed = urlparse(str(candidate or ""))
        suffix = Path(parsed.path).suffix.lower()
        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            return suffix
    return ".mp4"


def _is_path_owned_by(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
    return True


def cleanup_direct_video_cache(
    cache_dir: Path,
    *,
    ttl_seconds: int,
    max_bytes: int,
    now: float | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    current_time = time.time() if now is None else now
    files: list[tuple[Path, float, int]] = []
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        expired = ttl_seconds > 0 and current_time - stat.st_mtime > ttl_seconds
        if expired:
            path.unlink(missing_ok=True)
            continue
        if path.name.endswith(".part"):
            continue
        files.append((path, stat.st_mtime, stat.st_size))

    total = sum(size for _, _, size in files)
    if max_bytes <= 0 or total <= max_bytes:
        return
    for path, _, size in sorted(files, key=lambda item: item[1]):
        path.unlink(missing_ok=True)
        total -= size
        if total <= max_bytes:
            break


async def localize_direct_videos(
    event: Any,
    *,
    cache_dir: Path,
    max_bytes: int,
    cache_ttl_seconds: int,
    cache_max_bytes: int,
) -> list[Path]:
    cleanup_direct_video_cache(
        cache_dir,
        ttl_seconds=cache_ttl_seconds,
        max_bytes=cache_max_bytes,
    )
    message = list(getattr(getattr(event, "message_obj", None), "message", None) or [])
    raw_urls = iter(_raw_video_urls(event))
    localized: list[Path] = []
    for item in message:
        if not _is_video_component(item):
            continue
        raw_url = next(raw_urls, "")
        reference = VideoReference(
            file=str(getattr(item, "file", "") or ""),
            path=str(getattr(item, "path", "") or ""),
            url=str(getattr(item, "url", "") or "") or raw_url,
        )
        local_path = next(
            (
                candidate
                for value in (reference.path, reference.file)
                if (candidate := _normalize_local_path(value)) is not None
            ),
            None,
        )
        if local_path is None:
            remote = reference.url or (
                reference.file if _is_http_url(reference.file) else ""
            )
            if not remote:
                continue
            suffix = _video_suffix(reference)
            destination = cache_dir / f"direct_{uuid.uuid4().hex}{suffix}"
            partial = destination.with_name(destination.name + ".part")
            try:
                await download_media(remote, partial, max_bytes=max_bytes)
                _validate_video_path(partial, max_bytes, allowed_suffix=suffix)
                partial.replace(destination)
                local_path = destination.resolve()
            except Exception:
                partial.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                raise

        item.file = str(local_path)
        item.path = str(local_path)
        localized.append(local_path)
    return localized


def cleanup_owned_video_path(path: Path, cache_dir: Path) -> None:
    if _is_path_owned_by(path, cache_dir):
        path.unlink(missing_ok=True)


def _normalize_local_path(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.startswith("file:///"):
        raw = unquote(raw[8:])
    path = Path(raw)
    return path.resolve() if path.is_file() else None


def direct_video_attachment_path(parts: list[Any]) -> str:
    for part in parts:
        text = str(getattr(part, "text", "") or "").strip()
        if not text.startswith(DIRECT_VIDEO_ATTACHMENT_PREFIX):
            continue
        _, marker, path = text.rpartition(DIRECT_VIDEO_ATTACHMENT_PATH_MARKER)
        if marker and path.endswith("]"):
            return path[:-1].strip()
    return ""


def remove_direct_video_attachment_parts(parts: list[Any]) -> list[Any]:
    return [
        part
        for part in parts
        if not str(getattr(part, "text", "") or "").startswith(
            DIRECT_VIDEO_ATTACHMENT_PREFIX
        )
    ]


def extract_direct_video_references(event: Any) -> list[VideoReference]:
    message = list(getattr(getattr(event, "message_obj", None), "message", None) or [])
    return [
        VideoReference(
            file=str(getattr(item, "file", "") or ""),
            path=str(getattr(item, "path", "") or ""),
            url=str(getattr(item, "url", "") or ""),
        )
        for item in message
        if _is_video_component(item)
    ]


async def resolve_direct_video(
    reference: VideoReference,
    *,
    max_bytes: int,
) -> ResolvedVideoInput:
    for candidate in (reference.path, reference.file, reference.url):
        local_path = _normalize_local_path(candidate)
        if local_path is not None:
            _validate_video_path(local_path, max_bytes)
            return ResolvedVideoInput(path=local_path)

    remote = reference.url or reference.file
    if not remote.startswith(("http://", "https://")):
        raise FileNotFoundError("视频组件没有可用的本地路径或下载地址")
    video = Video(file=remote)
    path = Path(await video.convert_to_file_path()).resolve()
    _validate_video_path(path, max_bytes)
    return ResolvedVideoInput(path=path, cleanup=True)


def cleanup_resolved_video(video_input: ResolvedVideoInput | None) -> None:
    if video_input is None or not video_input.cleanup:
        return
    try:
        os.remove(video_input.path)
    except FileNotFoundError:
        return


def _validate_video_path(
    path: Path,
    max_bytes: int,
    *,
    allowed_suffix: str | None = None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"视频文件不存在：{path}")
    suffix = allowed_suffix or path.suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"不支持的视频文件格式：{suffix or '无扩展名'}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("视频文件为空")
    if size > max(1, max_bytes):
        raise ValueError(f"视频文件超过大小限制：{size / 1024 / 1024:.1f} MB")
