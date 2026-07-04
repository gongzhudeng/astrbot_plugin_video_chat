"""Bilibili resolver using bilibili-api-python with curl_cffi browser impersonation.

Bypasses the HTTP 412 anti-scraping that blocks yt-dlp on Bilibili.

Flow:
  b23.tv short link → HTTP redirect → canonical URL
  canonical URL → extract BV/av ID
  bilibili-api-python (curl_cffi impersonate) → raw download URL dict
  parse dash/durl directly → return (video_url, audio_url, title)
"""
from __future__ import annotations

import re
from typing import Optional

import aiohttp

from astrbot import logger

_BVID_RE = re.compile(r"(BV[A-Za-z0-9]{10})")
_AVID_RE = re.compile(r"[Aa][Vv](\d+)")

_BILI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}


def is_bilibili_url(url: str) -> bool:
    return any(d in url for d in ("bilibili.com", "b23.tv", "bili2233.cn"))


def extract_bvid_from_url(url: str) -> Optional[str]:
    m = _BVID_RE.search(url)
    return m.group(1) if m else None


def extract_avid_from_url(url: str) -> Optional[str]:
    m = _AVID_RE.search(url)
    return m.group(1) if m else None


async def _follow_redirect(url: str) -> str:
    """Follow a short link redirect and return the final URL."""
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=12),
                headers=_BILI_HEADERS,
            ) as resp:
                final = str(resp.url)
                logger.info("[bilibili] 短链解析 → %s", final)
                return final
    except Exception as exc:
        logger.warning("[bilibili] 短链解析失败: %s", exc)
        return url


def _pick_best_video_url(download_data: dict) -> Optional[str]:
    """Extract the best video stream URL from bilibili's download URL response dict."""
    # DASH format: data.dash.video[0].baseUrl
    dash = download_data.get("dash")
    if isinstance(dash, dict):
        videos = dash.get("video") or []
        if videos and isinstance(videos, list):
            # videos are sorted by quality descending; pick highest ≤ 720p
            # id: 16=360p 32=480p 64=720p 74=720p60 80=1080p ...
            best_url: Optional[str] = None
            best_quality = 0
            for v in videos:
                if not isinstance(v, dict):
                    continue
                q = v.get("id", 0)
                url = v.get("baseUrl") or v.get("base_url") or ""
                if url and q <= 64 and q > best_quality:
                    best_quality = q
                    best_url = url
            # If nothing ≤720p found, just take first entry
            if not best_url and videos:
                v0 = videos[0]
                if isinstance(v0, dict):
                    best_url = v0.get("baseUrl") or v0.get("base_url")
            if best_url:
                return best_url

    # FLV/MP4 format: data.durl[0].url
    durl = download_data.get("durl")
    if isinstance(durl, list) and durl:
        first = durl[0]
        if isinstance(first, dict):
            return first.get("url")

    return None


def _pick_best_audio_url(download_data: dict) -> Optional[str]:
    """Extract the best audio stream URL from bilibili's DASH response."""
    dash = download_data.get("dash")
    if not isinstance(dash, dict):
        return None
    audios = dash.get("audio") or []
    if not audios or not isinstance(audios, list):
        return None
    # audio list is also sorted by quality descending; just take first
    first = audios[0]
    if isinstance(first, dict):
        return first.get("baseUrl") or first.get("base_url")
    return None


async def download_bili_stream(v_url: str, dest: "Path") -> bool:
    """Download a B站 CDN video stream to dest using the required Referer header.

    B站 CDN URLs are hot-link protected; external vision models cannot fetch
    them directly.  We download the stream locally so frame extraction works.
    """
    from pathlib import Path as _Path  # noqa: F401 (type hint only above)

    headers = {**_BILI_HEADERS, "Range": "bytes=0-"}
    try:
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                v_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status not in (200, 206):
                    logger.warning("[bilibili] 视频流下载失败，HTTP %d", resp.status)
                    return False
                with open(dest, "wb") as f:
                    async for chunk in resp.content.iter_chunked(65536):
                        f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("[bilibili] 视频流下载异常: %s", exc)
        return False


async def resolve_bilibili(
    url: str,
) -> Optional[tuple[str, Optional[str], Optional[str]]]:
    """Resolve a Bilibili URL to (video_stream_url, audio_stream_url, title).

    Uses bilibili-api-python with curl_cffi browser impersonation to bypass
    the 412 Precondition Failed that yt-dlp gets on Bilibili.

    Returns None if resolution fails or bilibili-api-python is not installed.
    """
    try:
        from bilibili_api import request_settings, select_client
        from bilibili_api.video import Video
    except ImportError:
        logger.warning(
            "[bilibili] bilibili-api-python 未安装，无法使用原生 B 站解析。"
            "请重启 AstrBot 等待依赖自动安装。"
        )
        return None

    # Use curl_cffi to impersonate Chrome 131 — same as the reference plugin
    select_client("curl_cffi")
    request_settings.set("impersonate", "chrome131")

    # Resolve short links (b23.tv / bili2233.cn) to get the canonical URL
    if "b23.tv" in url or "bili2233.cn" in url:
        url = await _follow_redirect(url)

    bvid = extract_bvid_from_url(url)
    avid_str = extract_avid_from_url(url) if not bvid else None

    if not bvid and not avid_str:
        logger.warning("[bilibili] 无法从 URL 提取 BV/av 号: %s", url)
        return None

    try:
        video = Video(bvid=bvid) if bvid else Video(aid=int(avid_str))

        info = await video.get_info()
        title: Optional[str] = info.get("title") if isinstance(info, dict) else None

        # get_download_url returns the raw API dict — parse it directly
        # to avoid breaking enum issues inside VideoDownloadURLDataDetecter
        raw = await video.get_download_url(page_index=0)

        # bilibili-api wraps the response; actual payload may be under 'data' or at top level
        download_data: dict = raw if isinstance(raw, dict) else {}
        if "data" in download_data and isinstance(download_data["data"], dict):
            download_data = download_data["data"]

        v_url = _pick_best_video_url(download_data)
        if not v_url:
            logger.warning("[bilibili] 未能从下载数据中提取视频 URL")
            return None

        a_url = _pick_best_audio_url(download_data)

        logger.info("[bilibili] 原生解析成功: 标题=%s", title)
        return v_url, a_url, title

    except Exception as exc:
        logger.warning("[bilibili] bilibili-api 解析失败: %s", exc)
        return None