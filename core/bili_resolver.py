"""Bilibili metadata and media resolver."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from astrbot import logger

from .models import HotComment

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


@dataclass
class BilibiliResult:
    canonical_url: str
    bvid: str = ""
    aid: int = 0
    cid: int = 0
    title: str = ""
    description: str = ""
    topics: list[str] = field(default_factory=list)
    author: str = ""
    author_id: str = ""
    published_at: str = ""
    video_url: str | None = None
    audio_url: str | None = None
    comments: list[HotComment] = field(default_factory=list)


def is_bilibili_url(url: str) -> bool:
    return any(domain in url for domain in ("bilibili.com", "b23.tv", "bili2233.cn"))


async def normalize_bilibili_url(url: str) -> str:
    if "b23.tv" not in url and "bili2233.cn" not in url:
        return url
    try:
        async with aiohttp.ClientSession(headers=_BILI_HEADERS) as session:
            async with session.get(
                url,
                allow_redirects=True,
                timeout=aiohttp.ClientTimeout(total=12),
            ) as response:
                return str(response.url)
    except Exception as exc:
        logger.warning("[bilibili] 短链展开失败：%s", exc)
        return url


def _pick_best_video_url(download_data: dict[str, Any]) -> str | None:
    dash = download_data.get("dash") or {}
    videos = dash.get("video") or []
    candidates = []
    for video in videos:
        if not isinstance(video, dict):
            continue
        url = video.get("baseUrl") or video.get("base_url")
        if url:
            candidates.append((int(video.get("id", 0) or 0), str(url)))
    limited = [item for item in candidates if item[0] <= 64]
    if limited:
        return max(limited, key=lambda item: item[0])[1]
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    durl = download_data.get("durl") or []
    if durl and isinstance(durl[0], dict):
        return durl[0].get("url")
    return None


def _pick_best_audio_url(download_data: dict[str, Any]) -> str | None:
    audios = (download_data.get("dash") or {}).get("audio") or []
    candidates = []
    for audio in audios:
        if not isinstance(audio, dict):
            continue
        url = audio.get("baseUrl") or audio.get("base_url")
        if url:
            candidates.append((int(audio.get("bandwidth", 0) or 0), str(url)))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _extract_topics(info: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    for item in info.get("honor_reply", {}).get("honor", []) or []:
        text = str(item.get("desc", "") or "").strip()
        if text:
            topics.append(f"#{text}")
    return topics


def _extract_comment_media(content: dict[str, Any]) -> list[str]:
    media_urls: list[str] = []
    for item in content.get("pictures") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("img_src") or item.get("url") or "").strip()
        if url:
            media_urls.append(url if url.startswith("http") else f"https:{url}")
    return list(dict.fromkeys(media_urls))


def _normalize_comment(item: dict[str, Any], reply_limit: int = 0) -> HotComment | None:
    content = item.get("content") or {}
    message = str(content.get("message", "") or "").replace("\n", " ").strip()
    media_urls = _extract_comment_media(content)
    if not message and not media_urls:
        return None
    member = item.get("member") or {}
    try:
        likes = int(item.get("like", 0) or 0)
    except (TypeError, ValueError):
        likes = 0
    replies = [
        reply
        for raw in (item.get("replies") or [])[: max(0, reply_limit)]
        if isinstance(raw, dict) and (reply := _normalize_comment(raw, 0)) is not None
    ]
    return HotComment(
        message=message,
        likes=likes,
        username=str(member.get("uname", "") or "").strip(),
        comment_id=str(item.get("rpid_str") or item.get("rpid") or ""),
        media_urls=media_urls,
        replies=replies,
        reply_count=int(item.get("rcount") or item.get("count") or len(replies) or 0),
    )


async def _fetch_tags(aid: int, bvid: str) -> list[str]:
    params = {"bvid": bvid} if bvid else {"aid": aid}
    try:
        async with aiohttp.ClientSession(headers=_BILI_HEADERS) as session:
            async with session.get(
                "https://api.bilibili.com/x/tag/archive/tags",
                params=params,
                timeout=aiohttp.ClientTimeout(total=12),
            ) as response:
                payload = await response.json(content_type=None)
        return [
            f"#{item['tag_name']}"
            for item in payload.get("data") or []
            if isinstance(item, dict) and item.get("tag_name")
        ]
    except Exception as exc:
        logger.warning("[bilibili] 获取标签失败：%s", exc)
        return []


async def _fetch_bili_replies(
    session: aiohttp.ClientSession,
    aid: int,
    root_id: str,
    limit: int,
) -> list[HotComment]:
    if not root_id or limit <= 0:
        return []
    async with session.get(
        "https://api.bilibili.com/x/v2/reply/reply",
        params={
            "type": 1,
            "oid": aid,
            "root": root_id,
            "pn": 1,
            "ps": min(20, limit),
        },
        timeout=aiohttp.ClientTimeout(total=15),
    ) as response:
        payload = await response.json(content_type=None)
    if not isinstance(payload, dict) or payload.get("code") != 0:
        return []
    data = payload.get("data") or {}
    return [
        reply
        for item in (data.get("replies") or [])[:limit]
        if isinstance(item, dict) and (reply := _normalize_comment(item, 0)) is not None
    ]


async def _fetch_hot_comments(
    aid: int,
    count: int,
    reply_limit: int = 0,
) -> list[HotComment]:
    if aid <= 0 or count <= 0:
        return []
    params = {"type": 1, "oid": aid, "mode": 3, "next": 0, "ps": min(20, count)}
    try:
        async with aiohttp.ClientSession(headers=_BILI_HEADERS) as session:
            async with session.get(
                "https://api.bilibili.com/x/v2/reply/main",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                payload = await response.json(content_type=None)
            if not isinstance(payload, dict) or payload.get("code") != 0:
                raise RuntimeError(
                    str(payload.get("message") or payload.get("code"))
                    if isinstance(payload, dict)
                    else "评论接口返回了非对象数据"
                )
            data = payload.get("data") or {}
            raw = [*(data.get("top_replies") or []), *(data.get("replies") or [])]
            comments = [
                comment
                for item in raw
                if isinstance(item, dict)
                and (comment := _normalize_comment(item, reply_limit)) is not None
            ]
            comments = sorted(comments, key=lambda item: item.likes, reverse=True)[
                :count
            ]
            if reply_limit > 0:
                for comment in comments:
                    if len(comment.replies) >= reply_limit:
                        continue
                    try:
                        fetched = await _fetch_bili_replies(
                            session,
                            aid,
                            comment.comment_id,
                            reply_limit,
                        )
                    except Exception as exc:
                        logger.debug(
                            "[bilibili] 评论 %s 回复获取失败：%s",
                            comment.comment_id,
                            exc,
                        )
                        continue
                    existing = {
                        reply.comment_id or f"{reply.username}:{reply.message}"
                        for reply in comment.replies
                    }
                    for reply in fetched:
                        key = reply.comment_id or f"{reply.username}:{reply.message}"
                        if key not in existing:
                            comment.replies.append(reply)
                            existing.add(key)
                        if len(comment.replies) >= reply_limit:
                            break
            return comments
    except Exception as exc:
        logger.warning("[bilibili] 获取高赞评论失败：%s", exc)
        return []


async def resolve_bilibili(
    url: str,
    *,
    include_media: bool = True,
    comment_count: int = 0,
    comment_reply_limit: int = 0,
) -> BilibiliResult | None:
    try:
        from bilibili_api import request_settings, select_client
        from bilibili_api.video import Video
    except ImportError:
        logger.warning("[bilibili] bilibili-api-python 未安装")
        return None

    select_client("curl_cffi")
    request_settings.set("impersonate", "chrome131")
    canonical_url = await normalize_bilibili_url(url)
    bvid_match = _BVID_RE.search(canonical_url)
    avid_match = _AVID_RE.search(canonical_url) if not bvid_match else None
    if not bvid_match and not avid_match:
        return None

    try:
        video = (
            Video(bvid=bvid_match.group(1))
            if bvid_match
            else Video(aid=int(avid_match.group(1)))
        )
        info = await video.get_info()
        aid = int(info.get("aid", 0) or 0)
        bvid = str(info.get("bvid", "") or "")
        pages = info.get("pages") or []
        cid = int((pages[0].get("cid") if pages else info.get("cid")) or 0)
        owner = info.get("owner") or {}
        published_at = ""
        if info.get("pubdate"):
            published_at = datetime.fromtimestamp(int(info["pubdate"])).strftime(
                "%Y-%m-%d %H:%M"
            )
        topics = await _fetch_tags(aid, bvid)
        if not topics:
            topics = _extract_topics(info)
        comments = await _fetch_hot_comments(aid, comment_count, comment_reply_limit)

        result = BilibiliResult(
            canonical_url=canonical_url,
            bvid=bvid,
            aid=aid,
            cid=cid,
            title=str(info.get("title", "") or ""),
            description=str(info.get("desc", "") or ""),
            topics=topics,
            author=str(owner.get("name", "") or ""),
            author_id=f"UID {owner['mid']}" if owner.get("mid") else "",
            published_at=published_at,
            comments=comments,
        )
        if not include_media:
            return result

        raw = await video.get_download_url(page_index=0)
        download_data = raw if isinstance(raw, dict) else {}
        if isinstance(download_data.get("data"), dict):
            download_data = download_data["data"]
        result.video_url = _pick_best_video_url(download_data)
        result.audio_url = _pick_best_audio_url(download_data)
        return result if result.video_url else None
    except Exception as exc:
        logger.warning("[bilibili] 解析失败：%s", exc)
        return None


async def download_bili_stream(url: str, destination: Path) -> bool:
    try:
        async with aiohttp.ClientSession(
            headers={**_BILI_HEADERS, "Range": "bytes=0-"}
        ) as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status not in (200, 206):
                    return False
                with destination.open("wb") as file:
                    async for chunk in response.content.iter_chunked(65536):
                        file.write(chunk)
        return True
    except Exception as exc:
        logger.warning("[bilibili] 媒体流下载失败：%s", exc)
        return False
