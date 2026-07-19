from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import aiohttp


async def download_media(
    url: str,
    destination: Path,
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: int = 300,
    max_bytes: int | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(headers=headers or {}) as session:
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()
            written = 0
            with destination.open("wb") as file:
                async for chunk in response.content.iter_chunked(65536):
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise RuntimeError("媒体文件超过配置的大小限制")
                    file.write(chunk)
    return destination


async def extract_audio(
    source: Path,
    destination: Path,
    *,
    ffmpeg_path: str = "",
    max_seconds: int = 0,
) -> Path:
    await asyncio.get_running_loop().run_in_executor(
        None,
        _extract_audio_sync,
        source,
        destination,
        ffmpeg_path,
        max_seconds,
    )
    return destination


def _extract_audio_sync(
    source: Path,
    destination: Path,
    ffmpeg_path: str,
    max_seconds: int,
) -> None:
    ffmpeg = ffmpeg_path.strip() or "ffmpeg"
    if shutil.which(ffmpeg) is None and not os.path.exists(ffmpeg):
        raise RuntimeError(f"未找到 ffmpeg（{ffmpeg}）")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if max_seconds > 0:
        command.extend(["-t", str(max_seconds)])
    command.extend(
        ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"音频提取失败：{exc.stderr}") from exc
