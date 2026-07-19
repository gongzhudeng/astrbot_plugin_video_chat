from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from .models import HotComment, MediaWork


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_comment(comment: HotComment, reply_limit: int) -> HotComment | None:
    message = _clean_text(comment.message)
    media_urls = list(dict.fromkeys(url for url in comment.media_urls if url))
    media_descriptions = [
        description
        for value in comment.media_descriptions
        if (description := _clean_text(value))
    ]
    if not message and not media_urls and not media_descriptions:
        return None

    replies: list[HotComment] = []
    for reply in comment.replies[: max(0, reply_limit)]:
        normalized = _normalize_comment(reply, 0)
        if normalized is not None:
            replies.append(normalized)

    return HotComment(
        message=message,
        likes=max(0, int(comment.likes or 0)),
        username=_clean_text(comment.username),
        comment_id=_clean_text(comment.comment_id),
        media_urls=media_urls,
        media_descriptions=media_descriptions,
        replies=replies,
        reply_count=max(len(replies), int(comment.reply_count or 0)),
    )


def _comment_lines(comment: HotComment, index: int) -> list[str]:
    message = comment.message or "图片评论"
    lines = [f"{index}. [{comment.likes}赞] {message}"]
    lines.extend(
        f"   图片：{description}" for description in comment.media_descriptions
    )
    for reply in comment.replies:
        lines.append(
            "   - 回复"
            + (f"（{reply.username}）" if reply.username else "")
            + f"：{reply.message or '图片回复'}"
        )
        lines.extend(
            f"     图片：{description}" for description in reply.media_descriptions
        )
    return lines


def _rendered_length(comment: HotComment, index: int) -> int:
    return len("\n".join(_comment_lines(comment, index)))


def select_hot_comments(
    comments: Iterable[HotComment],
    *,
    max_count: int,
    max_chars: int,
    reply_limit: int = 0,
) -> list[HotComment]:
    unique: dict[tuple[str, str, tuple[str, ...]], HotComment] = {}
    for comment in comments:
        normalized = _normalize_comment(comment, reply_limit)
        if normalized is None:
            continue
        key = (
            normalized.username,
            normalized.message,
            tuple(normalized.media_urls),
        )
        previous = unique.get(key)
        if previous is None or normalized.likes > previous.likes:
            unique[key] = normalized

    ordered = sorted(unique.values(), key=lambda item: item.likes, reverse=True)
    selected: list[HotComment] = []
    used = 0
    for comment in ordered:
        if len(selected) >= max(0, max_count):
            break

        candidate = replace(comment, replies=list(comment.replies))
        index = len(selected) + 1
        separator = 1 if selected else 0
        while candidate.replies and (
            used + separator + _rendered_length(candidate, index) > max(0, max_chars)
        ):
            candidate.replies.pop()

        line_length = _rendered_length(candidate, index)
        if used + separator + line_length > max(0, max_chars):
            break
        selected.append(candidate)
        used += separator + line_length
    return selected


def format_media_work(
    work: MediaWork,
    *,
    comment_max_count: int = 10,
    comment_max_chars: int = 500,
    comment_reply_limit: int = 0,
) -> str:
    metadata = [f"平台：{work.platform}"]
    if work.work_type:
        metadata.append(f"类型：{work.work_type}")
    if work.title:
        metadata.append(f"标题：{work.title}")
    if work.description and work.description != work.title:
        metadata.append(f"描述：{work.description}")
    if work.topics:
        metadata.append("话题：" + " ".join(work.topics))
    if work.author:
        author = work.author
        if work.author_id:
            author += f"（{work.author_id}）"
        metadata.append(f"作者：{author}")
    elif work.author_id:
        metadata.append(f"作者账号：{work.author_id}")
    if work.published_at:
        metadata.append(f"发布时间：{work.published_at}")

    sections = ["[视频解析结果]", "【作品】\n" + "\n".join(metadata)]
    if work.subtitle:
        sections.append("【字幕】\n" + work.subtitle.strip())
    elif work.transcript:
        sections.append("【语音原文】\n" + work.transcript.strip())
    if work.visual_summary:
        sections.append("【画面】\n" + work.visual_summary.strip())

    comments = select_hot_comments(
        work.comments,
        max_count=comment_max_count,
        max_chars=comment_max_chars,
        reply_limit=comment_reply_limit,
    )
    if comments:
        lines: list[str] = []
        for index, comment in enumerate(comments, 1):
            lines.extend(_comment_lines(comment, index))
        sections.append("【高赞评论】\n" + "\n".join(lines))

    return "\n\n".join(sections)
