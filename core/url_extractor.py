from __future__ import annotations

import re

# Supported platforms: Douyin, Bilibili.
# Each entry: (platform_name, compiled_regex)
_PLATFORM_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "bilibili",
        re.compile(
            r"https?://(?:www\.|m\.)?bilibili\.com/(?:video/|bangumi/play/)[^\s\"'<>]+",
            re.IGNORECASE,
        ),
    ),
    (
        "bilibili_short",
        re.compile(r"https?://b23\.tv/[^\s\"'<>]+", re.IGNORECASE),
    ),
    (
        "bilibili_bv",
        re.compile(r"\b(BV[a-zA-Z0-9]{10})\b"),
    ),
    (
        "douyin",
        re.compile(
            r"https?://(?:www\.|m\.)?douyin\.com/(?:video/|note/|slides/|share/(?:video|note|slides)/)[^\s\"'<>]+",
            re.IGNORECASE,
        ),
    ),
    (
        "douyin_short",
        re.compile(r"https?://v\.douyin\.com/[^\s\"'<>]+", re.IGNORECASE),
    ),
]

# BV number → full Bilibili URL
_BV_TO_URL = "https://www.bilibili.com/video/{bv}"


def extract_video_url(text: str) -> str | None:
    """Extract the first recognizable video/post URL from arbitrary text.

    Handles share text that mixes Chinese characters with embedded URLs,
    short links, and bare BV numbers.

    Returns the cleaned URL string, or None if nothing is found.
    """
    for platform, pattern in _PLATFORM_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        raw = match.group(0)
        if platform == "bilibili_bv":
            return _BV_TO_URL.format(bv=raw)
        # Strip trailing punctuation that may have been captured
        return raw.rstrip(".,;:\"'）】》")

    return None


def is_video_url(text: str) -> bool:
    """Return True if text contains at least one recognised URL."""
    return extract_video_url(text) is not None
