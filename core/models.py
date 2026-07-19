from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HotComment:
    message: str
    likes: int = 0
    username: str = ""
    comment_id: str = ""
    media_urls: list[str] = field(default_factory=list)
    media_descriptions: list[str] = field(default_factory=list)
    replies: list[HotComment] = field(default_factory=list)
    reply_count: int = 0


@dataclass
class MediaWork:
    platform: str
    source_url: str
    work_type: str = "视频"
    work_id: str = ""
    title: str = ""
    description: str = ""
    topics: list[str] = field(default_factory=list)
    author: str = ""
    author_id: str = ""
    published_at: str = ""
    video_url: str | None = None
    audio_url: str | None = None
    image_urls: list[str] = field(default_factory=list)
    local_video_path: Path | None = None
    local_audio_path: Path | None = None
    subtitle: str = ""
    transcript: str = ""
    visual_summary: str = ""
    comments: list[HotComment] = field(default_factory=list)
