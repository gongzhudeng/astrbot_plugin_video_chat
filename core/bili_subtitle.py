"""Bilibili subtitle fetcher.

Flow:
  BV/av number
    → GET /x/web-interface/view   → aid, cid
    → GET /x/player/v2            → subtitle list URLs
    → download subtitle JSON      → plain text
"""

from __future__ import annotations

import re

import aiohttp

from astrbot import logger

_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
_PLAYER_V2_API = "https://api.bilibili.com/x/player/v2"

_BVID_RE = re.compile(r"BV[A-Za-z0-9]+")
_AVID_RE = re.compile(r"[Aa][Vv](\d+)")

_COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}


def extract_bvid(url: str) -> str | None:
    """Extract BV number from a Bilibili URL or plain text."""
    m = _BVID_RE.search(url)
    return m.group(0) if m else None


def extract_avid(url: str) -> str | None:
    """Extract av number (digits only) from a Bilibili URL."""
    m = _AVID_RE.search(url)
    return m.group(1) if m else None


def _build_cookies(sessdata: str) -> dict[str, str]:
    return {"SESSDATA": sessdata} if sessdata.strip() else {}


async def _get_video_info(
    session: aiohttp.ClientSession,
    *,
    bvid: str | None = None,
    avid: str | None = None,
) -> tuple[int | None, int | None]:
    """Return (aid, cid) for the first part of the video, or (None, None) on failure."""
    params: dict[str, str] = {}
    if bvid:
        params["bvid"] = bvid
    elif avid:
        params["aid"] = avid
    else:
        return None, None

    try:
        async with session.get(
            _VIEW_API, params=params, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            data = await resp.json(content_type=None)
    except Exception as exc:
        logger.warning("[bili-subtitle] 获取视频信息失败：%s", exc)
        return None, None

    if data.get("code") != 0:
        logger.warning(
            "[bili-subtitle] /x/web-interface/view 返回错误：%s", data.get("message")
        )
        return None, None

    video_data = data.get("data") or {}
    aid = video_data.get("aid")
    # cid of the first page
    pages = video_data.get("pages") or []
    cid = pages[0].get("cid") if pages else video_data.get("cid")
    return aid, cid


async def _get_subtitle_urls(
    session: aiohttp.ClientSession,
    aid: int,
    cid: int,
) -> list[tuple[str, str]]:
    """Return list of (language, subtitle_url) pairs."""
    params = {"aid": str(aid), "cid": str(cid)}
    try:
        async with session.get(
            _PLAYER_V2_API, params=params, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            data = await resp.json(content_type=None)
    except Exception as exc:
        logger.warning("[bili-subtitle] 获取字幕列表失败：%s", exc)
        return []

    if data.get("code") != 0:
        logger.warning("[bili-subtitle] /x/player/v2 返回错误：%s", data.get("message"))
        return []

    subtitles = (data.get("data") or {}).get("subtitle", {}).get("subtitles") or []
    results: list[tuple[str, str]] = []
    for item in subtitles:
        url = str(item.get("subtitle_url") or "").strip()
        lang = str(item.get("lan_doc") or item.get("lan") or "").strip()
        if url:
            if url.startswith("//"):
                url = "https:" + url
            results.append((lang, url))
    return results


async def _download_subtitle_text(
    session: aiohttp.ClientSession,
    url: str,
) -> str | None:
    """Download subtitle JSON and convert body entries to plain text."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            payload = await resp.json(content_type=None)
    except Exception as exc:
        logger.warning("[bili-subtitle] 下载字幕 JSON 失败：%s", exc)
        return None

    body = payload.get("body") or []
    lines: list[str] = []
    for entry in body:
        content = str(entry.get("content") or "").strip()
        if content:
            lines.append(content)

    return "\n".join(lines) if lines else None


async def fetch_bili_subtitle(url: str, sessdata: str) -> str | None:
    """Fetch subtitle text for a Bilibili video URL.

    Returns plain-text subtitle string, or None if unavailable.
    Prefers Chinese subtitles; falls back to the first available language.
    """
    bvid = extract_bvid(url)
    avid = extract_avid(url) if not bvid else None

    if not bvid and not avid:
        logger.debug("[bili-subtitle] 无法从 URL 中提取 BV/av 号：%s", url)
        return None

    cookies = _build_cookies(sessdata)
    headers = dict(_COMMON_HEADERS)

    async with aiohttp.ClientSession(headers=headers, cookies=cookies) as session:
        aid, cid = await _get_video_info(session, bvid=bvid, avid=avid)
        if not aid or not cid:
            return None

        subtitle_list = await _get_subtitle_urls(session, aid, cid)
        if not subtitle_list:
            logger.info("[bili-subtitle] 视频无字幕（aid=%s cid=%s）", aid, cid)
            return None

        # Prefer Chinese (zh-CN / ai-zh), then first available
        preferred_url: str | None = None
        fallback_url: str | None = None
        for lang, sub_url in subtitle_list:
            lang_lower = lang.lower()
            if "zh" in lang_lower or "中" in lang_lower:
                preferred_url = sub_url
                break
            if fallback_url is None:
                fallback_url = sub_url

        target_url = preferred_url or fallback_url
        if not target_url:
            return None

        text = await _download_subtitle_text(session, target_url)
        if text:
            logger.info("[bili-subtitle] 字幕获取成功，共 %d 行", text.count("\n") + 1)
        return text
