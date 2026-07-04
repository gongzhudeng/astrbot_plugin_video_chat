from __future__ import annotations

import asyncio
import base64
import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from astrbot import logger

DEFAULT_CAPTION_PROMPT = (
    "请用不超过 300 字简洁概括这个视频的核心内容，"
    "包括主要人物、事件和关键信息。"
    "如果画面中有字幕，请以字幕内容为准。"
    "不要编造没有出现的信息。"
)


async def caption_from_url(
    stream_url: str,
    *,
    provider: object,
    prompt: str = DEFAULT_CAPTION_PROMPT,
) -> str:
    """Ask the vision model to caption a video via direct URL.

    Returns the caption text, or raises RuntimeError on failure.
    """
    contexts = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video_url",
                    "video_url": {"url": stream_url},
                },
            ],
        }
    ]
    try:
        resp = await provider.text_chat(contexts=contexts)
    except Exception as exc:
        raise RuntimeError(f"视觉模型调用失败（video_url 路径）：{exc}") from exc

    text = str(getattr(resp, "completion_text", "") or "").strip()
    if not text:
        raise RuntimeError("视觉模型返回了空文本（video_url 路径）")
    return text


def _to_jpeg_bytes(data: bytes) -> bytes | None:
    """Convert raw image bytes to JPEG using Pillow.

    Returns JPEG bytes, or None if conversion fails (unsupported format).
    Falls back to returning the original data if it already looks like JPEG/PNG/GIF/WEBP/BMP.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        pass

    # Pillow not available or unsupported — accept only known-safe formats by magic bytes
    if data[:3] == b"\xff\xd8\xff":
        return data  # JPEG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return data  # PNG
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return data  # GIF
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data  # WEBP
    if data[:2] == b"BM":
        return data  # BMP
    return None  # Skip AVIF/HEIC/unknown


async def caption_from_image_urls(
    image_urls: list[str],
    *,
    provider: object,
    prompt: str = DEFAULT_CAPTION_PROMPT,
    max_images: int = 9,
) -> str:
    """Download image URLs, encode to base64, and ask the vision model to caption them.

    max_images caps how many images are sent (first N).
    Returns caption text, or raises RuntimeError on failure.
    """
    import aiohttp

    selected = image_urls[:max(1, max_images)]
    image_blocks = []

    # Prefer JPEG/PNG/WEBP — avoids AVIF responses from Douyin/XHS CDNs
    _img_headers = {
        "Accept": "image/jpeg,image/png,image/webp,image/gif,image/*;q=0.8",
        "Referer": "https://www.douyin.com/",
    }

    async with aiohttp.ClientSession(headers=_img_headers) as session:
        for img_url in selected:
            try:
                async with session.get(img_url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    raw = await resp.read()
                # Convert to JPEG (handles AVIF/HEIC/unknown formats via Pillow)
                converted = _to_jpeg_bytes(raw)
                if converted is None:
                    logger.warning("[video-chat] 跳过不支持的图片格式 %s", img_url[:80])
                    continue
                # Determine MIME type from converted bytes
                if converted[:3] == b"\xff\xd8\xff":
                    mime = "image/jpeg"
                elif converted[:8] == b"\x89PNG\r\n\x1a\n":
                    mime = "image/png"
                elif converted[:6] in (b"GIF87a", b"GIF89a"):
                    mime = "image/gif"
                elif converted[:4] == b"RIFF" and converted[8:12] == b"WEBP":
                    mime = "image/webp"
                else:
                    mime = "image/jpeg"
                payload = base64.b64encode(converted).decode("utf-8")
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{payload}"},
                })
            except Exception as exc:
                logger.warning("[video-chat] 图片下载失败 %s：%s", img_url[:80], exc)

    if not image_blocks:
        raise RuntimeError("所有图片下载均失败，无法进行图片转述")

    contexts = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_blocks,
            ],
        }
    ]

    logger.info("[video-chat] 图文转述：发送 %d/%d 张图片给视觉模型", len(image_blocks), len(image_urls))
    try:
        resp = await provider.text_chat(contexts=contexts)
    except Exception as exc:
        raise RuntimeError(f"视觉模型调用失败（图片路径）：{exc}") from exc

    text = str(getattr(resp, "completion_text", "") or "").strip()
    if not text:
        raise RuntimeError("视觉模型返回了空文本（图片路径）")
    return text


async def caption_from_frames(
    local_path: Path,
    *,
    provider: object,
    prompt: str = DEFAULT_CAPTION_PROMPT,
    frames_per_second: float = 1.0,
    max_frames: int = 30,
    analyze_first_seconds: int = 120,
    ffmpeg_path: str = "",
) -> str:
    """Extract frames with ffmpeg and ask the vision model to caption them.

    Returns the caption text, or raises RuntimeError on failure.
    """
    frame_data_urls = await asyncio.get_event_loop().run_in_executor(
        None,
        _extract_frames_sync,
        local_path,
        frames_per_second,
        max_frames,
        analyze_first_seconds,
        ffmpeg_path,
    )

    image_blocks = [
        {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        for data_url in frame_data_urls
    ]

    contexts = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_blocks,
            ],
        }
    ]

    try:
        resp = await provider.text_chat(contexts=contexts)
    except Exception as exc:
        raise RuntimeError(f"视觉模型调用失败（抽帧路径）：{exc}") from exc

    text = str(getattr(resp, "completion_text", "") or "").strip()
    if not text:
        raise RuntimeError("视觉模型返回了空文本（抽帧路径）")
    return text


# ---------------------------------------------------------------------------
# Synchronous ffmpeg helper
# ---------------------------------------------------------------------------

def _extract_frames_sync(
    local_path: Path,
    frames_per_second: float,
    max_frames: int,
    analyze_first_seconds: int,
    ffmpeg_path: str,
) -> list[str]:
    """Extract frames using fps-rate + window + cap strategy, return as data URLs.

    Strategy:
      analyze_duration = min(video_duration, analyze_first_seconds)  # 0 = full
      raw_count        = ceil(analyze_duration * frames_per_second)
      actual_count     = min(raw_count, max_frames)
    ffmpeg only processes the first analyze_duration seconds.
    """
    import math

    ffmpeg_cmd = ffmpeg_path.strip() or "ffmpeg"
    if shutil.which(ffmpeg_cmd) is None and not os.path.exists(ffmpeg_cmd):
        raise RuntimeError(
            f"未找到 ffmpeg（{ffmpeg_cmd}）。"
            "请安装 ffmpeg 并加入 PATH，或在插件配置中填写 ffmpeg_path。"
        )

    duration = _probe_duration(local_path, ffmpeg_path)

    # Determine the window to analyze
    if analyze_first_seconds > 0 and duration and duration > analyze_first_seconds:
        analyze_duration = float(analyze_first_seconds)
    else:
        analyze_duration = duration  # None means unknown; handle below

    # Compute target frame count
    if analyze_duration and analyze_duration > 0:
        raw_count = math.ceil(analyze_duration * max(frames_per_second, 0.001))
    else:
        # Duration unknown: fall back to max_frames spread over whatever is there
        raw_count = max_frames

    actual_count = min(raw_count, max(max_frames, 1))

    # Build fps expression so ffmpeg emits exactly actual_count frames
    if analyze_duration and analyze_duration > 0:
        fps_expr = f"{max(actual_count / analyze_duration, 0.001):.6f}"
    else:
        fps_expr = f"{max(frames_per_second, 0.001):.6f}"

    with tempfile.TemporaryDirectory(prefix="vchat_frames_") as tmpdir:
        out_pattern = os.path.join(tmpdir, "frame_%03d.jpg")
        cmd = [ffmpeg_cmd, "-hide_banner", "-loglevel", "error", "-y"]
        # Limit input duration when a window is configured
        if analyze_first_seconds > 0 and analyze_duration:
            cmd += ["-t", str(analyze_duration)]
        cmd += [
            "-i", str(local_path),
            "-vf", f"fps={fps_expr}",
            "-frames:v", str(actual_count),
            "-q:v", "5",
            out_pattern,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg 抽帧失败：{exc.stderr}") from exc

        frame_files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
        if not frame_files:
            raise RuntimeError("ffmpeg 运行成功但未生成任何帧文件，请检查视频文件是否有效。")

        data_urls = []
        for path in frame_files:
            payload = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            data_urls.append(f"data:image/jpeg;base64,{payload}")

        logger.info(
            "[video-chat] 抽帧完成：共 %d 帧（分析前 %ss，上限 %d），视频=%s",
            len(data_urls),
            str(analyze_first_seconds) if analyze_first_seconds > 0 else "全程",
            max_frames,
            local_path.name,
        )
        return data_urls


def _probe_duration(local_path: Path, ffmpeg_path: str) -> float | None:
    """Use ffprobe to get video duration in seconds."""
    ffprobe_cmd = (
        os.path.join(os.path.dirname(ffmpeg_path.strip()), "ffprobe")
        if ffmpeg_path.strip()
        else "ffprobe"
    )
    if shutil.which(ffprobe_cmd) is None and not os.path.exists(ffprobe_cmd):
        return None
    try:
        result = subprocess.run(
            [
                ffprobe_cmd,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(local_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None