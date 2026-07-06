"""Native Douyin resolver — bypasses yt-dlp's fresh-cookie requirement.

Strategy:
  1. Follow the short URL redirect to get aweme_id.
  2. Fetch share page from iesdouyin.com or m.douyin.com using iOS UA.
  3. Extract window._ROUTER_DATA JSON.
  4. Walk loaderData → video_(id)/page or note_(id)/page → videoInfoRes → item_list[0]
     - Video post: video.play_addr.url_list[0]  (strip "playwm" watermark)
     - Image/note post: images[].url_list[0]    (return list, no ffmpeg needed)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

from astrbot import logger


@dataclass
class DouyinResult:
    """Result of a Douyin resolution attempt.

    Exactly one of play_url or image_urls will be non-empty.
    """
    play_url: Optional[str] = None
    image_urls: list[str] = field(default_factory=list)
    title: str = ""

    @property
    def is_image_post(self) -> bool:
        return bool(self.image_urls) and not self.play_url


# Matches /video/1234..., /note/1234..., or /slides/1234... in the redirected canonical URL
_AWEME_ID_RE = re.compile(r"/(?:video|note|slides)/(\d{15,20})")

# iOS UA — iesdouyin.com returns _ROUTER_DATA to mobile clients
_IOS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/16.6 Mobile/15E148 Safari/604.1 Edg/132.0.0.0"
    ),
    "Referer": "https://www.douyin.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def is_douyin_url(url: str) -> bool:
    return "douyin.com" in url


def _parse_douyin_cookies(cookies_file: str) -> dict[str, str]:
    """Extract douyin.com cookies from a Netscape cookie file."""
    cookies: dict[str, str] = {}
    try:
        lines = Path(cookies_file).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        logger.warning("[douyin] 读取 cookies 文件失败：%s", exc)
        return cookies

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) == 8 and parts[7] == "":
            parts = parts[:7]
        if len(parts) < 7:
            continue
        domain = parts[0].lstrip(".")
        name = parts[5].strip()
        value = parts[6] if len(parts) > 6 else ""
        if not name:
            continue
        if "douyin.com" in domain:
            cookies[name] = value

    logger.debug("[douyin] 从 cookies 文件中提取了 %d 个 douyin.com cookie", len(cookies))
    return cookies


async def _resolve_short_url(url: str, session: aiohttp.ClientSession) -> str:
    """Single-hop redirect — return the Location header URL."""
    try:
        async with session.get(
            url,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=12),
        ) as resp:
            location = resp.headers.get("Location", url)
            logger.info("[douyin] 短链解析 → %s", location)
            return location
    except Exception as exc:
        logger.warning("[douyin] 短链重定向失败：%s", exc)
        return url


def _extract_aweme_id(url: str) -> Optional[str]:
    m = _AWEME_ID_RE.search(url)
    return m.group(1) if m else None


def _extract_from_router_data(html: str, aweme_id: str) -> Optional[DouyinResult]:
    """Parse window._ROUTER_DATA and return a DouyinResult (video or image post)."""
    m = re.search(
        r"window\._ROUTER_DATA\s*=\s*(.*?)</script>",
        html,
        re.DOTALL,
    )
    if not m:
        logger.debug("[douyin] 未找到 window._ROUTER_DATA")
        return None

    raw = m.group(1).strip().rstrip(";")
    try:
        data = json.loads(raw)
    except Exception as exc:
        logger.debug("[douyin] _ROUTER_DATA JSON 解析失败：%s", exc)
        return None

    loader_data: dict = data.get("loaderData", {})

    for key_tmpl in ("video_(id)/page", "note_(id)/page", "slides_(id)/page"):
        page = loader_data.get(key_tmpl)
        if not isinstance(page, dict):
            continue
        item_list: list = page.get("videoInfoRes", {}).get("item_list", [])
        if not item_list or not isinstance(item_list[0], dict):
            continue
        item = item_list[0]
        title: str = item.get("desc", "") or ""

        # --- Video post ---
        video_obj = item.get("video")
        if isinstance(video_obj, dict):
            url_list: list = video_obj.get("play_addr", {}).get("url_list", [])
            if url_list and isinstance(url_list[0], str):
                play_url = url_list[0].replace("playwm", "play")
                _lower = play_url.lower()
                is_audio = any(
                    tok in _lower
                    for tok in (".mp3", ".m4a", ".aac", "ies-music", "mime_type=audio", "/music/")
                )
                if not is_audio:
                    logger.info("[douyin] 从 _ROUTER_DATA[%s] 提取到 play_url", key_tmpl)
                    return DouyinResult(play_url=play_url, title=title)
                logger.info("[douyin] play_url 为音频，尝试图片字段")

        # --- Image/slides post ---
        images: list = item.get("images") or item.get("image_list") or []
        image_urls: list[str] = []
        for img in images:
            if not isinstance(img, dict):
                continue
            candidates = [
                img.get("url_list"),
                img.get("download_url_list"),
                img.get("animated_cover", {}).get("url_list") if isinstance(img.get("animated_cover"), dict) else None,
                img.get("video", {}).get("play_addr", {}).get("url_list") if isinstance(img.get("video"), dict) else None,
            ]
            for urls in candidates:
                if urls and isinstance(urls, list) and isinstance(urls[0], str):
                    image_urls.append(urls[0])
                    break

        if image_urls:
            logger.info(
                "[douyin] 图文/动图帖子，提取到 %d 个素材 (key=%s)",
                len(image_urls),
                key_tmpl,
            )
            return DouyinResult(image_urls=image_urls, title=title)

    logger.debug(
        "[douyin] _ROUTER_DATA 中未找到 play_url 或图片，loaderData 键：%s",
        list(loader_data.keys())[:10],
    )
    return None


async def _fetch_and_extract(
    aweme_id: str, session: aiohttp.ClientSession
) -> Optional[DouyinResult]:
    """Try iesdouyin.com then m.douyin.com, return DouyinResult or None."""
    candidates = [
        f"https://www.iesdouyin.com/share/video/{aweme_id}",
        f"https://www.iesdouyin.com/share/slides/{aweme_id}",
        f"https://m.douyin.com/share/video/{aweme_id}",
        f"https://m.douyin.com/share/slides/{aweme_id}",
    ]

    for candidate_url in candidates:
        try:
            async with session.get(
                candidate_url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                html = await resp.text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("[douyin] 页面请求失败 %s：%s", candidate_url, exc)
            continue

        logger.debug("[douyin] 获取页面 %s (%d 字节)", candidate_url, len(html))
        result = _extract_from_router_data(html, aweme_id)
        if result:
            return result

        logger.debug("[douyin] %s 未找到有效内容，尝试下一候选", candidate_url)

    logger.warning("[douyin] 所有候选页面均未找到有效内容")
    return None


async def resolve_douyin(
    url: str, cookies_file: Optional[str] = None
) -> Optional[DouyinResult]:
    """Resolve a Douyin URL.

    Returns DouyinResult:
      - .play_url set → video post, use stream_url path
      - .image_urls set → image/note post, use image captioning path
    Returns None if resolution fails (caller may fall back to yt-dlp).
    """
    cookies = _parse_douyin_cookies(cookies_file) if cookies_file else {}

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(
        headers=_IOS_HEADERS,
        cookies=cookies,
        connector=connector,
    ) as session:
        resolved = await _resolve_short_url(url, session)
        aweme_id = _extract_aweme_id(resolved)

        if not aweme_id:
            logger.warning("[douyin] 无法从重定向 URL 提取 aweme_id：%s", resolved)
            return None

        logger.info("[douyin] aweme_id = %s，开始提取内容", aweme_id)
        result = await _fetch_and_extract(aweme_id, session)

        if result:
            if result.is_image_post:
                logger.info("[douyin] 原生解析成功（图文帖，%d 张图片）", len(result.image_urls))
            else:
                logger.info("[douyin] 原生解析成功（视频）：%s…", (result.play_url or "")[:80])
        else:
            logger.warning("[douyin] 原生解析失败，将回落 yt-dlp")

        return result