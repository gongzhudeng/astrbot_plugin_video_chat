from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

VIDEO_CONTEXT_START = "<!-- astrbot-video-chat:context:v1:start -->"
VIDEO_CONTEXT_END = "<!-- astrbot-video-chat:context:v1:end -->"
VIDEO_CONTEXT_PRUNED = "[历史视频：详细解析内容已按上下文保留限制清理]"
VIDEO_CONTEXT_PATTERN = re.compile(
    re.escape(VIDEO_CONTEXT_START) + r"(.*?)" + re.escape(VIDEO_CONTEXT_END),
    flags=re.DOTALL,
)
VIDEO_RESULT_MARKER = "[视频解析结果]"
VIDEO_TOOL_NAME = "analyze_video"


@dataclass(frozen=True)
class VideoContextEntry:
    kind: str
    details: str
    platform: str = ""
    title: str = ""
    summary: str = ""
    url: str = ""
    message_index: int = -1
    part_index: int | None = None
    block_index: int | None = None
    tool_call_id: str = ""


def wrap_video_context(details: str) -> str:
    return "\n".join(
        (
            VIDEO_CONTEXT_START,
            details.strip(),
            VIDEO_CONTEXT_END,
        )
    )


def is_video_context_block(text: Any) -> bool:
    return bool(VIDEO_CONTEXT_PATTERN.search(str(text or "")))


def list_video_contexts(contexts: list[Any]) -> list[VideoContextEntry]:
    entries: list[VideoContextEntry] = []
    tool_results = _tool_results_by_id(contexts)

    for message_index, message in enumerate(contexts):
        role, content = _message_role_and_content(message)
        if role == "user":
            parts = content if isinstance(content, list) else [content]
            for part_index, part in enumerate(parts):
                text = (
                    _part_text(part) if isinstance(content, list) else str(part or "")
                )
                for block_index, match in enumerate(
                    VIDEO_CONTEXT_PATTERN.finditer(text)
                ):
                    entries.append(
                        _build_entry(
                            kind="自动",
                            details=match.group(1).strip(),
                            message_index=message_index,
                            part_index=part_index
                            if isinstance(content, list)
                            else None,
                            block_index=block_index,
                        )
                    )
        elif role == "assistant":
            for tool_call in _message_tool_calls(message):
                if _tool_call_name(tool_call) != VIDEO_TOOL_NAME:
                    continue
                call_id = _tool_call_id(tool_call)
                details = tool_results.get(call_id, "")
                if not call_id or not _is_complete_video_details(details):
                    continue
                entries.append(
                    _build_entry(
                        kind="工具",
                        details=details,
                        url=_tool_call_url(tool_call),
                        message_index=message_index,
                        tool_call_id=call_id,
                    )
                )
    return entries


def delete_video_context(
    contexts: list[Any],
    index: int,
) -> VideoContextEntry | None:
    entries = list_video_contexts(contexts)
    if index < 1 or index > len(entries):
        return None
    entry = entries[index - 1]
    if entry.kind == "工具":
        _delete_tool_context(contexts, entry.tool_call_id)
    else:
        _delete_marker_context(contexts, entry)
    return entry


def prune_video_contexts(
    contexts: list[Any],
    *,
    max_details: int,
    incoming_details: int = 0,
) -> int:
    keep_history = max(0, max_details - max(0, incoming_details))
    prune_count = max(0, len(list_video_contexts(contexts)) - keep_history)
    pruned = 0
    for _ in range(prune_count):
        if delete_video_context(contexts, 1) is None:
            break
        pruned += 1
    return pruned


def _build_entry(
    *,
    kind: str,
    details: str,
    url: str = "",
    message_index: int,
    part_index: int | None = None,
    block_index: int | None = None,
    tool_call_id: str = "",
) -> VideoContextEntry:
    return VideoContextEntry(
        kind=kind,
        details=details,
        platform=_extract_field(details, "平台"),
        title=_extract_field(details, "标题"),
        summary=_extract_summary(details),
        url=url,
        message_index=message_index,
        part_index=part_index,
        block_index=block_index,
        tool_call_id=tool_call_id,
    )


def _extract_field(details: str, field: str) -> str:
    match = re.search(rf"(?m)^{re.escape(field)}：\s*(.+)$", details)
    return " ".join(match.group(1).split()) if match else ""


def _extract_summary(details: str) -> str:
    description = _extract_field(details, "描述")
    if description:
        return description
    for section in ("画面", "字幕", "语音原文"):
        match = re.search(
            rf"【{section}】\s*\n(.+?)(?=\n\n【|\Z)",
            details,
            flags=re.DOTALL,
        )
        if match:
            return " ".join(match.group(1).split())
    return ""


def _is_complete_video_details(details: str) -> bool:
    return VIDEO_RESULT_MARKER in str(details or "")


def _tool_results_by_id(contexts: list[Any]) -> dict[str, str]:
    results: dict[str, str] = {}
    for message in contexts:
        role, _ = _message_role_and_content(message)
        if role != "tool":
            continue
        call_id = _message_tool_call_id(message)
        if call_id:
            results[call_id] = _message_content_text(message)
    return results


def _delete_marker_context(contexts: list[Any], entry: VideoContextEntry) -> None:
    if entry.message_index < 0 or entry.message_index >= len(contexts):
        return
    message = contexts[entry.message_index]
    _, content = _message_role_and_content(message)
    target = (
        content[entry.part_index]
        if isinstance(content, list)
        and entry.part_index is not None
        and entry.part_index < len(content)
        else message
    )
    text = _part_text(target) if target is not message else _message_content(message)
    matches = list(VIDEO_CONTEXT_PATTERN.finditer(text))
    block_index = entry.block_index or 0
    if block_index >= len(matches):
        return
    match = matches[block_index]
    text = text[: match.start()] + VIDEO_CONTEXT_PRUNED + text[match.end() :]
    if target is message:
        _set_message_content(message, text)
    else:
        _set_part_text(target, text)


def _delete_tool_context(contexts: list[Any], call_id: str) -> None:
    remove_message_indexes: list[int] = []
    for message_index, message in enumerate(contexts):
        role, _ = _message_role_and_content(message)
        if role == "tool" and _message_tool_call_id(message) == call_id:
            remove_message_indexes.append(message_index)
            continue
        if role != "assistant":
            continue
        tool_calls = _message_tool_calls(message)
        filtered = [call for call in tool_calls if _tool_call_id(call) != call_id]
        if len(filtered) == len(tool_calls):
            continue
        _set_message_tool_calls(message, filtered)
        if not filtered and not _message_content_text(message).strip():
            remove_message_indexes.append(message_index)

    for message_index in sorted(set(remove_message_indexes), reverse=True):
        del contexts[message_index]


def _message_role_and_content(message: Any) -> tuple[str, Any]:
    if isinstance(message, dict):
        return str(message.get("role", "")), message.get("content")
    return str(getattr(message, "role", "")), getattr(message, "content", None)


def _part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text", "") or "")
    return str(getattr(part, "text", "") or "")


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _message_content_text(message: Any) -> str:
    _, content = _message_role_and_content(message)
    if isinstance(content, list):
        return "\n".join(_part_text(part) for part in content)
    return str(content or "")


def _message_tool_calls(message: Any) -> list[Any]:
    calls = (
        message.get("tool_calls", [])
        if isinstance(message, dict)
        else getattr(message, "tool_calls", [])
    )
    return list(calls or [])


def _message_tool_call_id(message: Any) -> str:
    value = (
        message.get("tool_call_id", "")
        if isinstance(message, dict)
        else getattr(message, "tool_call_id", "")
    )
    return str(value or "")


def _tool_call_id(tool_call: Any) -> str:
    value = (
        tool_call.get("id", "")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "id", "")
    )
    return str(value or "")


def _tool_call_function(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get("function", {})
    return getattr(tool_call, "function", None)


def _tool_call_name(tool_call: Any) -> str:
    function = _tool_call_function(tool_call)
    value = (
        function.get("name", "")
        if isinstance(function, dict)
        else getattr(function, "name", "")
    )
    return str(value or "")


def _tool_call_url(tool_call: Any) -> str:
    function = _tool_call_function(tool_call)
    arguments = (
        function.get("arguments", "")
        if isinstance(function, dict)
        else getattr(function, "arguments", "")
    )
    if isinstance(arguments, dict):
        return str(arguments.get("url", "") or "")
    try:
        decoded = json.loads(str(arguments or "{}"))
    except (TypeError, ValueError):
        return ""
    return str(decoded.get("url", "") or "") if isinstance(decoded, dict) else ""


def _set_message_content(message: Any, content: str) -> None:
    if isinstance(message, dict):
        message["content"] = content
    else:
        message.content = content


def _set_part_text(part: Any, text: str) -> None:
    if isinstance(part, dict):
        part["text"] = text
    else:
        part.text = text


def _set_message_tool_calls(message: Any, tool_calls: list[Any]) -> None:
    if isinstance(message, dict):
        if tool_calls:
            message["tool_calls"] = tool_calls
        else:
            message.pop("tool_calls", None)
    else:
        message.tool_calls = tool_calls or None
