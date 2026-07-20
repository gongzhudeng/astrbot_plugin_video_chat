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

import asyncio
import base64
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import aiohttp

from astrbot import logger

from .douyin_signer import generate_a_bogus
from .models import HotComment


@dataclass
class DouyinResult:
    """Result of a Douyin resolution attempt.

    Exactly one of play_url or image_urls will be non-empty.
    """

    play_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    title: str = ""
    description: str = ""
    topics: list[str] = field(default_factory=list)
    author: str = ""
    author_id: str = ""
    published_at: str = ""
    aweme_id: str = ""
    comments: list[HotComment] = field(default_factory=list)

    @property
    def is_image_post(self) -> bool:
        return bool(self.image_urls) and not self.play_url


# Matches /video/1234..., /note/1234..., or /slides/1234... in the redirected canonical URL
_AWEME_REF_RE = re.compile(r"/(video|note|slides)/(\d{15,20})")

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
        lines = (
            Path(cookies_file)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
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

    logger.debug(
        "[douyin] 从 cookies 文件中提取了 %d 个 douyin.com cookie", len(cookies)
    )
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
            parsed = urlsplit(location)
            safe_location = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            logger.info("[douyin] 短链解析 → %s", safe_location)
            return location
    except Exception as exc:
        logger.warning("[douyin] 短链重定向失败：%s", exc)
        return url


def _extract_aweme_ref(url: str) -> tuple[str | None, str | None]:
    match = _AWEME_REF_RE.search(url)
    if not match:
        return None, None
    return match.group(2), match.group(1)


def _extract_topics(item: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    for extra in item.get("text_extra") or []:
        if not isinstance(extra, dict):
            continue
        name = str(extra.get("hashtag_name", "") or "").strip()
        if name:
            topics.append(f"#{name}")
    return list(dict.fromkeys(topics))


def _build_result(item: dict[str, Any], aweme_id: str) -> DouyinResult:
    title = str(item.get("desc", "") or "").strip()
    author = item.get("author") or {}
    published_at = ""
    if item.get("create_time"):
        published_at = datetime.fromtimestamp(int(item["create_time"])).strftime(
            "%Y-%m-%d %H:%M"
        )
    return DouyinResult(
        title=title,
        description=title,
        topics=_extract_topics(item),
        author=str(author.get("nickname", "") or "").strip(),
        author_id=(
            f"抖音号 {author['unique_id']}"
            if author.get("unique_id")
            else f"UID {author['uid']}"
            if author.get("uid")
            else ""
        ),
        published_at=published_at,
        aweme_id=aweme_id,
    )


def _extract_image_urls(item: dict[str, Any]) -> list[str]:
    images: list = item.get("images") or item.get("image_list") or []
    image_urls: list[str] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        candidates = [
            image.get("url_list"),
            image.get("download_url_list"),
            image.get("animated_cover", {}).get("url_list")
            if isinstance(image.get("animated_cover"), dict)
            else None,
            image.get("video", {}).get("play_addr", {}).get("url_list")
            if isinstance(image.get("video"), dict)
            else None,
        ]
        for urls in candidates:
            if urls and isinstance(urls, list) and isinstance(urls[0], str):
                image_urls.append(urls[0])
                break
    return image_urls


def _extract_play_url(item: dict[str, Any]) -> str | None:
    video_obj = item.get("video")
    if not isinstance(video_obj, dict):
        return None
    url_list: list = video_obj.get("play_addr", {}).get("url_list", [])
    if not url_list or not isinstance(url_list[0], str):
        return None
    play_url = url_list[0].replace("playwm", "play")
    lower_url = play_url.lower()
    if any(
        token in lower_url
        for token in (
            ".mp3",
            ".m4a",
            ".aac",
            "ies-music",
            "mime_type=audio",
            "/music/",
        )
    ):
        return None
    return play_url


def _extract_from_router_data(
    html: str,
    aweme_id: str,
    preferred_type: str | None = None,
) -> DouyinResult | None:
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
    page_types = [preferred_type] if preferred_type else []
    page_types.extend(
        page_type
        for page_type in ("video", "note", "slides")
        if page_type not in page_types
    )

    for page_type in page_types:
        key_tmpl = f"{page_type}_(id)/page"
        page = loader_data.get(key_tmpl)
        if not isinstance(page, dict):
            continue
        item_list: list = page.get("videoInfoRes", {}).get("item_list", [])
        if not item_list or not isinstance(item_list[0], dict):
            continue
        item = item_list[0]
        result = _build_result(item, aweme_id)
        image_urls = _extract_image_urls(item)
        play_url = _extract_play_url(item)

        if preferred_type in {"note", "slides"} and image_urls:
            logger.info(
                "[douyin] 图文/动图帖子，提取到 %d 个素材 (key=%s)",
                len(image_urls),
                key_tmpl,
            )
            result.image_urls = image_urls
            return result

        if play_url:
            logger.info("[douyin] 从 _ROUTER_DATA[%s] 提取到 play_url", key_tmpl)
            result.play_url = play_url
            return result

        if image_urls:
            logger.info(
                "[douyin] 图文/动图帖子，提取到 %d 个素材 (key=%s)",
                len(image_urls),
                key_tmpl,
            )
            result.image_urls = image_urls
            return result

    logger.debug(
        "[douyin] _ROUTER_DATA 中未找到 play_url 或图片，loaderData 键：%s",
        list(loader_data.keys())[:10],
    )
    return None


async def _fetch_and_extract(
    aweme_id: str,
    session: aiohttp.ClientSession,
    preferred_type: str | None = None,
) -> DouyinResult | None:
    """Try canonical share type first, then compatible fallback pages."""
    page_types = [preferred_type] if preferred_type else []
    page_types.extend(
        page_type
        for page_type in ("video", "note", "slides")
        if page_type not in page_types
    )
    candidates = [
        f"https://{host}/share/{page_type}/{aweme_id}"
        for host in ("www.iesdouyin.com", "m.douyin.com")
        for page_type in page_types
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
        result = _extract_from_router_data(
            html,
            aweme_id,
            preferred_type=preferred_type,
        )
        if result:
            return result

        logger.debug("[douyin] %s 未找到有效内容，尝试下一候选", candidate_url)

    logger.warning("[douyin] 所有候选页面均未找到有效内容")
    return None


_WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
)
_WEB_HEADERS = {
    "User-Agent": _WEB_USER_AGENT,
    "Referer": "https://www.douyin.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def _generate_ms_token() -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return "".join(secrets.choice(alphabet) for _ in range(182)) + "=="


def _web_comment_params(**values: object) -> dict[str, object]:
    params: dict[str, object] = {
        "device_platform": "webapp",
        "aid": "6383",
        "channel": "channel_pc_web",
        "item_type": "0",
        "pc_client_type": "1",
        "publish_video_strategy_type": "2",
        "pc_libra_divert": "Windows",
        "version_code": "290100",
        "version_name": "29.1.0",
        "cookie_enabled": "true",
        "screen_width": "1920",
        "screen_height": "1080",
        "browser_language": "zh-CN",
        "browser_platform": "Win32",
        "browser_name": "Edge",
        "browser_version": "131.0.0.0",
        "browser_online": "true",
        "engine_name": "Blink",
        "engine_version": "131.0.0.0",
        "os_name": "Windows",
        "os_version": "10",
        "cpu_core_num": "12",
        "device_memory": "8",
        "platform": "PC",
        "downlink": "10",
        "effective_type": "4g",
        "round_trip_time": "100",
    }
    params.update(values)
    return params


def _extract_comment_media(item: dict[str, Any]) -> list[str]:
    raw_media = item.get("image_list") or item.get("images") or []
    media_urls: list[str] = []
    for media in raw_media:
        if not isinstance(media, dict):
            continue
        candidates = [
            media.get("origin_url", {}).get("url_list")
            if isinstance(media.get("origin_url"), dict)
            else None,
            media.get("medium_url", {}).get("url_list")
            if isinstance(media.get("medium_url"), dict)
            else None,
            media.get("url_list"),
        ]
        for urls in candidates:
            if isinstance(urls, list) and urls and isinstance(urls[0], str):
                media_urls.append(urls[0])
                break
    return list(dict.fromkeys(media_urls))


def _normalize_douyin_comment(item: dict[str, Any]) -> HotComment | None:
    message = str(item.get("text", "") or "").replace("\n", " ").strip()
    media_urls = _extract_comment_media(item)
    if not message and not media_urls:
        return None
    user = item.get("user") or {}
    return HotComment(
        message=message,
        likes=int(item.get("digg_count", 0) or 0),
        username=str(user.get("nickname", "") or "").strip(),
        comment_id=str(item.get("cid", "") or ""),
        media_urls=media_urls,
        reply_count=int(item.get("reply_comment_total", 0) or 0),
    )


async def _request_signed_comments(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: dict[str, object],
) -> dict[str, Any]:
    request_params = dict(params)
    if not request_params.get("msToken"):
        cookie_token = next(
            (
                morsel.value
                for morsel in session.cookie_jar.filter_cookies(
                    "https://www.douyin.com/"
                ).values()
                if morsel.key == "msToken" and morsel.value
            ),
            "",
        )
        request_params["msToken"] = cookie_token or _generate_ms_token()
    query = urlencode(request_params)
    signed = {
        **request_params,
        "a_bogus": generate_a_bogus(query, _WEB_USER_AGENT),
    }
    async with session.get(
        endpoint,
        params=signed,
        headers=_WEB_HEADERS,
        timeout=aiohttp.ClientTimeout(total=15),
    ) as response:
        content_type = response.headers.get("Content-Type", "unknown")
        text = await response.text(encoding="utf-8", errors="replace")
        if response.status != 200:
            raise RuntimeError(
                f"评论接口 HTTP {response.status}，响应类型 {content_type}"
            )
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        summary = " ".join(text[:120].split())
        raise RuntimeError(
            f"评论接口返回非 JSON，响应类型 {content_type}，摘要 {summary!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("评论接口返回了空数据或非对象数据")
    status = int(payload.get("status_code", 0) or 0)
    if status != 0:
        raise RuntimeError(f"评论接口业务状态异常：{status}")
    return payload


async def _fetch_douyin_replies(
    session: aiohttp.ClientSession,
    aweme_id: str,
    comment_id: str,
    limit: int,
) -> list[HotComment]:
    if not comment_id or limit <= 0:
        return []
    payload = await _request_signed_comments(
        session,
        "https://www.douyin.com/aweme/v1/web/comment/list/reply/",
        _web_comment_params(
            item_id=aweme_id,
            comment_id=comment_id,
            cursor="0",
            count=str(min(20, limit)),
            cut_version="1",
        ),
    )
    return [
        comment
        for item in (payload.get("comments") or [])[:limit]
        if isinstance(item, dict)
        and (comment := _normalize_douyin_comment(item)) is not None
    ]


async def _fetch_hot_comments_signed(
    session: aiohttp.ClientSession,
    aweme_id: str,
    count: int,
    reply_limit: int,
) -> list[HotComment]:
    payload = await _request_signed_comments(
        session,
        "https://www.douyin.com/aweme/v1/web/comment/list/",
        _web_comment_params(
            aweme_id=aweme_id,
            cursor="0",
            count=str(min(20, max(count, 1))),
            insert_ids="",
            whale_cut_token="",
            cut_version="1",
            rcFT="",
        ),
    )
    comments = [
        comment
        for item in payload.get("comments") or []
        if isinstance(item, dict)
        and (comment := _normalize_douyin_comment(item)) is not None
    ]
    comments = sorted(comments, key=lambda item: item.likes, reverse=True)[:count]
    if reply_limit > 0:
        for comment in comments:
            if comment.reply_count <= 0:
                continue
            try:
                comment.replies = await _fetch_douyin_replies(
                    session, aweme_id, comment.comment_id, reply_limit
                )
            except Exception as exc:
                logger.debug(
                    "[douyin] 评论 %s 回复获取失败：%s", comment.comment_id, exc
                )
    return comments


def _normalize_cdp_comment_payload(
    payload: dict[str, Any], count: int, reply_limit: int
) -> list[HotComment]:
    comments: list[HotComment] = []
    for item in payload.get("comments") or []:
        if not isinstance(item, dict):
            continue
        comment = _normalize_douyin_comment(item)
        if comment is None:
            continue
        embedded_replies = item.get("reply_comment") or []
        comment.replies = [
            reply
            for reply_item in embedded_replies[: max(0, reply_limit)]
            if isinstance(reply_item, dict)
            and (reply := _normalize_douyin_comment(reply_item)) is not None
        ]
        comments.append(comment)
    return sorted(comments, key=lambda item: item.likes, reverse=True)[:count]


async def _fetch_replies_via_cdp_page(
    command: Any,
    source_url: str,
    aweme_id: str,
    comment_id: str,
    limit: int,
) -> list[HotComment]:
    if not comment_id or limit <= 0:
        return []
    source_params = dict(parse_qsl(urlsplit(source_url).query, keep_blank_values=True))
    source_params.pop("a_bogus", None)
    source_params.pop("aweme_id", None)
    source_params.update(
        {
            "item_id": aweme_id,
            "comment_id": comment_id,
            "cursor": "0",
            "count": str(min(20, limit)),
            "cut_version": "1",
        }
    )
    query = urlencode(source_params)
    endpoint = (
        "https://www.douyin.com/aweme/v1/web/comment/list/reply/"
        f"?{query}&a_bogus={generate_a_bogus(query, _WEB_USER_AGENT)}"
    )
    expression = (
        "(async () => {"
        "const controller = new AbortController();"
        "const timeout = setTimeout(() => controller.abort(), 8000);"
        "try {"
        f"const response = await fetch({json.dumps(endpoint)}, "
        "{credentials: 'include', signal: controller.signal});"
        "return {status: response.status, body: await response.text()};"
        "} finally { clearTimeout(timeout); }"
        "})()"
    )
    result = await command(
        "Runtime.evaluate",
        {
            "expression": expression,
            "awaitPromise": True,
            "returnByValue": True,
        },
    )
    value = (result.get("result") or {}).get("value") or {}
    if not isinstance(value, dict) or int(value.get("status", 0) or 0) != 200:
        raise RuntimeError(f"CDP 回复接口 HTTP {value.get('status', 'unknown')}")
    try:
        payload = json.loads(str(value.get("body", "") or ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("CDP 回复接口返回非 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CDP 回复接口返回空数据或非对象")
    status = int(payload.get("status_code", 0) or 0)
    if status != 0:
        raise RuntimeError(f"CDP 回复接口业务状态异常：{status}")
    return [
        reply
        for item in (payload.get("comments") or [])[:limit]
        if isinstance(item, dict)
        and (reply := _normalize_douyin_comment(item)) is not None
    ]


async def _fill_cdp_comment_replies(
    command: Any,
    comments: list[HotComment],
    source_url: str,
    aweme_id: str,
    reply_limit: int,
) -> None:
    if reply_limit <= 0:
        return
    for comment in comments:
        missing = min(reply_limit, comment.reply_count) - len(comment.replies)
        if missing <= 0:
            continue
        try:
            fetched = await _fetch_replies_via_cdp_page(
                command,
                source_url,
                aweme_id,
                comment.comment_id,
                missing,
            )
        except Exception as exc:
            logger.warning(
                "[douyin] CDP 评论 %s 回复补取失败：%s",
                comment.comment_id,
                exc,
            )
            continue
        seen = {reply.comment_id for reply in comment.replies if reply.comment_id}
        for reply in fetched:
            if reply.comment_id and reply.comment_id in seen:
                continue
            comment.replies.append(reply)
            if reply.comment_id:
                seen.add(reply.comment_id)
            if len(comment.replies) >= reply_limit:
                break


async def _fetch_comments_via_cdp(
    cdp_url: str,
    aweme_id: str,
    count: int,
    reply_limit: int,
) -> list[HotComment]:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("浏览器评论模式需要安装 websockets") from exc

    base_url = cdp_url.rstrip("/")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/json/version", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"CDP 调试端口返回 HTTP {response.status}")
                version = await response.json(content_type=None)
            async with session.get(
                f"{base_url}/json/list", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"CDP 页面列表返回 HTTP {response.status}")
                targets = await response.json(content_type=None)
    except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
        raise RuntimeError(
            f"未连接到 CDP 调试端口 {base_url}，请先启动已登录抖音的 Chrome/Edge"
        ) from exc
    except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("CDP 调试端口返回了无效 JSON") from exc

    if not isinstance(version, dict) or not version.get("Browser"):
        raise RuntimeError("CDP /json/version 未返回有效浏览器信息")
    if not isinstance(targets, list):
        raise RuntimeError("CDP /json/list 未返回页面列表")
    pages = [
        item
        for item in targets
        if isinstance(item, dict)
        and item.get("type") == "page"
        and item.get("webSocketDebuggerUrl")
    ]
    page = next(
        (item for item in pages if "douyin.com" in str(item.get("url", ""))),
        pages[0] if pages else None,
    )
    if page is None:
        raise RuntimeError("CDP 浏览器没有可用页面，请先打开一个浏览器标签页")

    request_id = 0
    deferred_events: list[dict[str, Any]] = []

    async with websockets.connect(
        page["webSocketDebuggerUrl"], open_timeout=8, close_timeout=3
    ) as socket:

        async def command(method: str, params: dict[str, Any] | None = None) -> dict:
            nonlocal request_id
            request_id += 1
            current_id = request_id
            await socket.send(
                json.dumps(
                    {
                        "id": current_id,
                        "method": method,
                        "params": params or {},
                    }
                )
            )
            while True:
                message = json.loads(await socket.recv())
                if message.get("id") != current_id:
                    if message.get("method"):
                        deferred_events.append(message)
                    continue
                if message.get("error"):
                    detail = message["error"].get("message", "CDP 命令执行失败")
                    raise RuntimeError(f"CDP {method} 失败：{detail}")
                return message.get("result") or {}

        await command("Network.enable")
        await command("Page.enable")
        await command(
            "Page.navigate", {"url": f"https://www.douyin.com/video/{aweme_id}"}
        )
        await asyncio.sleep(2)
        await command(
            "Runtime.evaluate",
            {"expression": "window.scrollTo(0, document.body.scrollHeight); true"},
        )

        async def next_event(deadline: float) -> dict[str, Any]:
            if deferred_events:
                return deferred_events.pop(0)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            return json.loads(await asyncio.wait_for(socket.recv(), timeout=remaining))

        deadline = asyncio.get_running_loop().time() + 15
        response_request_id = ""
        response_url = ""
        try:
            while True:
                event = await next_event(deadline)
                method = event.get("method")
                params = event.get("params") or {}
                if (
                    response_request_id
                    and method == "Network.loadingFinished"
                    and str(params.get("requestId", "")) == response_request_id
                ):
                    break
                if method != "Network.responseReceived":
                    continue
                response = params.get("response") or {}
                url = str(response.get("url", ""))
                if (
                    "/aweme/v1/web/comment/list/" not in url
                    or "/comment/list/reply/" in url
                ):
                    continue
                response_request_id = str(params.get("requestId", ""))
                response_url = url
                if not response_request_id:
                    continue
                if any(
                    queued.get("method") == "Network.loadingFinished"
                    and str((queued.get("params") or {}).get("requestId", ""))
                    == response_request_id
                    for queued in deferred_events
                ):
                    break
        except asyncio.TimeoutError as exc:
            page_state = await command(
                "Runtime.evaluate",
                {
                    "expression": "({url: location.href, title: document.title})",
                    "returnByValue": True,
                },
            )
            value = (page_state.get("result") or {}).get("value") or {}
            title = str(value.get("title", ""))[:60] if isinstance(value, dict) else ""
            raise RuntimeError(
                "浏览器已连接，但 15 秒内未捕获评论请求；"
                f"请确认页面已登录且能显示评论（页面标题：{title or '未知'}）"
            ) from exc

        body_result = await command(
            "Network.getResponseBody", {"requestId": response_request_id}
        )
        body = str(body_result.get("body", "") or "")
        if body_result.get("base64Encoded"):
            try:
                body = base64.b64decode(body).decode("utf-8", errors="replace")
            except Exception as exc:
                raise RuntimeError("CDP 评论响应 Base64 解码失败") from exc
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("CDP 捕获的评论响应不是有效 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("CDP 捕获的评论响应为空或不是对象")
        status = int(payload.get("status_code", 0) or 0)
        if status != 0:
            raise RuntimeError(f"CDP 评论接口业务状态异常：{status}")

        comments = _normalize_cdp_comment_payload(payload, count, reply_limit)
        await _fill_cdp_comment_replies(
            command,
            comments,
            response_url,
            aweme_id,
            reply_limit,
        )
        logger.info(
            "[douyin] 浏览器捕获评论成功：%d 条（来源 %s）",
            len(comments),
            urlsplit(response_url).path,
        )
        return comments


async def _fetch_hot_comments(
    session: aiohttp.ClientSession,
    aweme_id: str,
    count: int,
    reply_limit: int = 0,
    cdp_fallback_enabled: bool = False,
    cdp_url: str = "http://127.0.0.1:9222",
) -> list[HotComment]:
    if not aweme_id or count <= 0:
        return []
    if cdp_fallback_enabled:
        try:
            return await _fetch_comments_via_cdp(cdp_url, aweme_id, count, reply_limit)
        except Exception as exc:
            logger.warning("[douyin] 浏览器评论模式失败：%s", exc)
    try:
        return await _fetch_hot_comments_signed(session, aweme_id, count, reply_limit)
    except Exception as exc:
        logger.warning("[douyin] 轻量签名评论请求失败：%s", exc)
    return []


async def resolve_douyin(
    url: str,
    cookies_file: str | None = None,
    comment_count: int = 0,
    comment_reply_limit: int = 0,
    comment_cdp_fallback_enabled: bool = False,
    comment_cdp_url: str = "http://127.0.0.1:9222",
) -> DouyinResult | None:
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
        aweme_id, aweme_type = _extract_aweme_ref(resolved)

        if not aweme_id:
            logger.warning("[douyin] 无法从重定向 URL 提取 aweme_id：%s", resolved)
            return None

        logger.info("[douyin] aweme_id = %s，开始提取内容", aweme_id)
        result = await _fetch_and_extract(
            aweme_id,
            session,
            preferred_type=aweme_type,
        )
        if result and comment_count > 0:
            result.comments = await _fetch_hot_comments(
                session,
                aweme_id,
                comment_count,
                comment_reply_limit,
                comment_cdp_fallback_enabled,
                comment_cdp_url,
            )

        if result:
            if result.is_image_post:
                logger.info(
                    "[douyin] 原生解析成功（图文帖，%d 张图片）", len(result.image_urls)
                )
            else:
                logger.info(
                    "[douyin] 原生解析成功（视频）：%s…", (result.play_url or "")[:80]
                )
        else:
            logger.warning("[douyin] 原生解析失败，将回落 yt-dlp")

        return result
