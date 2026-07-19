from __future__ import annotations

# ruff: noqa: E402, I001

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

PLUGIN_DIR = Path(__file__).resolve().parent
ASTRBOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(ASTRBOT_DIR))

from core.context_formatter import format_media_work, select_hot_comments
from core.douyin_resolver import (
    _fetch_hot_comments,
    _fill_cdp_comment_replies,
    _normalize_cdp_comment_payload,
    _request_signed_comments,
)
from core.douyin_signer import generate_a_bogus
from core.media_input import (
    VideoReference,
    cleanup_direct_video_cache,
    cleanup_owned_video_path,
    direct_video_attachment_path,
    extract_direct_video_references,
    localize_direct_videos,
    remove_direct_video_attachment_parts,
    resolve_direct_video,
)
from core.models import HotComment, MediaWork
from core.video_captioner import build_comment_media_prompt
from core.video_context import (
    VIDEO_CONTEXT_PRUNED,
    prune_video_contexts,
    wrap_video_context,
)
from astrbot_plugin_video_chat.main import (
    DEFAULT_DIRECT_VIDEO_QUESTION,
    VideoChatPlugin,
)


class _FakeVideoEvent:
    def __init__(self, message: list | None = None, message_str: str = "") -> None:
        self.message_obj = type("MessageObject", (), {"message": message or []})()
        self.message_str = message_str
        self._extras: dict[str, object] = {}

    def get_extra(self, key: str, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key: str, value: object) -> None:
        self._extras[key] = value


class _FakeProviderRequest:
    def __init__(
        self,
        *,
        prompt: str = "",
        contexts: list | None = None,
        extra_user_content_parts: list | None = None,
    ) -> None:
        self.prompt = prompt
        self.contexts = contexts or []
        self.extra_user_content_parts = extra_user_content_parts or []


class AutoVideoContextTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _plugin(config: dict | None = None) -> VideoChatPlugin:
        plugin = object.__new__(VideoChatPlugin)
        plugin.config = config or {}
        return plugin

    async def test_explicit_link_is_injected_and_tool_is_deduplicated(self) -> None:
        old = {
            "role": "user",
            "content": [{"type": "text", "text": wrap_video_context("old")}],
        }
        plugin = self._plugin({"max_video_context_details": 1})
        plugin._do_analyze = AsyncMock(return_value="new-details")
        event = _FakeVideoEvent(message_str="看看 https://www.bilibili.com/video/BV1xx")
        request = _FakeProviderRequest(prompt="这个视频讲了什么？", contexts=[old])

        await plugin.inject_video_context(event, request)
        duplicate_result = await plugin.analyze_video(
            event, "https://www.bilibili.com/video/BV2yy"
        )

        self.assertEqual(old["content"][0]["text"], VIDEO_CONTEXT_PRUNED)
        self.assertIn("new-details", request.extra_user_content_parts[-1].text)
        normalized_url = "https://www.bilibili.com/video/BV1xx"
        self.assertEqual(event.get_extra("video_chat_processed_source"), normalized_url)
        self.assertIn("当前请求已自动解析一个视频", duplicate_result)
        plugin._do_analyze.assert_awaited_once_with(event, normalized_url)

    async def test_direct_video_has_priority_and_removes_only_direct_placeholder(
        self,
    ) -> None:
        from astrbot.api.message_components import Video

        plugin = self._plugin()
        plugin._analyze_direct_video = AsyncMock(return_value="direct-details")
        plugin._do_analyze = AsyncMock(return_value="link-details")
        direct_placeholder = type(
            "Part",
            (),
            {"text": "[Video Attachment: name direct.mp4, path D:/direct.mp4]"},
        )()
        quoted_placeholder = type(
            "Part",
            (),
            {
                "text": (
                    "[Video Attachment in quoted message: "
                    "name quoted.mp4, path D:/quoted.mp4]"
                )
            },
        )()
        event = _FakeVideoEvent(
            message=[Video(file="remote.mp4")],
            message_str="https://www.bilibili.com/video/BV1xx",
        )
        request = _FakeProviderRequest(
            prompt="<attachment>",
            extra_user_content_parts=[direct_placeholder, quoted_placeholder],
        )

        await plugin.inject_video_context(event, request)

        analyzed_reference = plugin._analyze_direct_video.await_args.args[1]
        self.assertEqual(analyzed_reference.path, "D:/direct.mp4")
        plugin._do_analyze.assert_not_awaited()
        self.assertNotIn(direct_placeholder, request.extra_user_content_parts)
        self.assertIn(quoted_placeholder, request.extra_user_content_parts)
        self.assertEqual(request.prompt, DEFAULT_DIRECT_VIDEO_QUESTION)
        self.assertEqual(event.get_extra("video_chat_processed_source"), "direct-video")

    async def test_numbered_video_placeholder_uses_default_question(self) -> None:
        from astrbot.api.message_components import Video

        plugin = self._plugin()
        plugin._analyze_direct_video = AsyncMock(return_value="direct-details")
        event = _FakeVideoEvent(message=[Video(file="remote.mp4")])
        request = _FakeProviderRequest(prompt="[视频1]")

        await plugin.inject_video_context(event, request)

        self.assertEqual(request.prompt, DEFAULT_DIRECT_VIDEO_QUESTION)

    async def test_failed_analysis_does_not_consume_context_slot(self) -> None:
        old = {
            "role": "user",
            "content": [{"type": "text", "text": wrap_video_context("old")}],
        }
        plugin = self._plugin({"max_video_context_details": 1})
        plugin._do_analyze = AsyncMock(return_value="视频链接解析失败，请稍后重试。")
        event = _FakeVideoEvent(message_str="https://www.bilibili.com/video/BV1xx")
        request = _FakeProviderRequest(contexts=[old])

        await plugin.inject_video_context(event, request)

        self.assertIn("old", old["content"][0]["text"])
        self.assertNotIn(
            "astrbot-video-chat:context", request.extra_user_content_parts[0].text
        )
        self.assertIsNone(event.get_extra("video_chat_processed_source"))

    async def test_disabled_auto_parse_still_prunes_history(self) -> None:
        old = {
            "role": "user",
            "content": [{"type": "text", "text": wrap_video_context("old")}],
        }
        latest = {
            "role": "user",
            "content": [{"type": "text", "text": wrap_video_context("latest")}],
        }
        plugin = self._plugin(
            {"auto_parse_video_messages": False, "max_video_context_details": 1}
        )
        plugin._do_analyze = AsyncMock(return_value="unused")
        event = _FakeVideoEvent(message_str="https://www.bilibili.com/video/BV1xx")
        request = _FakeProviderRequest(contexts=[old, latest])

        await plugin.inject_video_context(event, request)

        self.assertEqual(old["content"][0]["text"], VIDEO_CONTEXT_PRUNED)
        self.assertIn("latest", latest["content"][0]["text"])
        plugin._do_analyze.assert_not_awaited()

    async def test_many_video_contexts_keep_only_latest_details(self) -> None:
        contexts = []
        for index in range(8):
            contexts.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"用户原话-{index}\n"
                                f"{wrap_video_context(f'视频详情-{index}')}\n"
                                f"用户尾句-{index}"
                            ),
                        }
                    ],
                }
            )
            contexts.append({"role": "assistant", "content": f"AI回复-{index}"})

        pruned = prune_video_contexts(contexts, max_details=3)

        self.assertEqual(pruned, 5)
        for index in range(8):
            user_text = contexts[index * 2]["content"][0]["text"]
            self.assertIn(f"用户原话-{index}", user_text)
            self.assertIn(f"用户尾句-{index}", user_text)
            self.assertEqual(contexts[index * 2 + 1]["content"], f"AI回复-{index}")
            if index < 5:
                self.assertIn(VIDEO_CONTEXT_PRUNED, user_text)
                self.assertNotIn(f"视频详情-{index}", user_text)
            else:
                self.assertIn(f"视频详情-{index}", user_text)

    async def test_incoming_video_reserves_one_context_slot(self) -> None:
        contexts = [
            {
                "role": "user",
                "content": [{"type": "text", "text": wrap_video_context(str(index))}],
            }
            for index in range(5)
        ]

        pruned = prune_video_contexts(
            contexts,
            max_details=3,
            incoming_details=1,
        )

        self.assertEqual(pruned, 3)
        self.assertEqual(
            [context["content"][0]["text"] for context in contexts[:3]],
            [VIDEO_CONTEXT_PRUNED] * 3,
        )
        self.assertIn("3", contexts[3]["content"][0]["text"])
        self.assertIn("4", contexts[4]["content"][0]["text"])

    async def test_video_prompt_includes_sanitized_same_turn_text(self) -> None:
        plugin = self._plugin({"video_user_context_max_chars": 20})
        event = _FakeVideoEvent()
        context = plugin._video_user_context(
            '[视频1]\n<image_context id="图1">图片说明</image_context>\n这是用户问题很长很长'
        )
        event.set_extra("video_chat_user_context", context)

        prompt = plugin._caption_prompt(event)

        self.assertNotIn("图片说明", prompt)
        self.assertNotIn("[视频1]", prompt)
        self.assertIn("这是用户问题", prompt)
        self.assertIn("不要把用户的猜测", prompt)


class DirectVideoInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_video_extension_is_rejected(self) -> None:
        invalid_path = PLUGIN_DIR / "temporary-invalid-video.txt"
        invalid_path.write_text("not a video", encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "不支持的视频文件格式"):
                await resolve_direct_video(
                    VideoReference(path=str(invalid_path)),
                    max_bytes=1024,
                )
        finally:
            invalid_path.unlink(missing_ok=True)

    def test_only_top_level_video_is_detected(self) -> None:
        from astrbot.api.message_components import Reply, Video

        direct = Video(file="file:///direct.mp4", path="direct.mp4")
        quoted = Video(file="file:///quoted.mp4", path="quoted.mp4")
        event = type(
            "Event",
            (),
            {
                "message_obj": type(
                    "MessageObject",
                    (),
                    {"message": [Reply(id="1", chain=[quoted]), direct]},
                )()
            },
        )()

        references = extract_direct_video_references(event)

        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].file, "file:///direct.mp4")
        self.assertEqual(references[0].path, "direct.mp4")

    def test_reply_video_is_not_detected_without_top_level_video(self) -> None:
        from astrbot.api.message_components import Reply, Video

        event = type(
            "Event",
            (),
            {
                "message_obj": type(
                    "MessageObject",
                    (),
                    {
                        "message": [
                            Reply(
                                id="1",
                                chain=[Video(file="file:///quoted.mp4")],
                            )
                        ]
                    },
                )()
            },
        )()

        self.assertEqual(extract_direct_video_references(event), [])

    async def test_raw_onebot_url_localizes_bare_video_file(self) -> None:
        from astrbot.api.message_components import Video

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            video = Video(file="bare-name.mp4")
            message_obj = type(
                "MessageObject",
                (),
                {
                    "message": [video],
                    "raw_message": {
                        "message": [
                            {
                                "type": "video",
                                "data": {
                                    "file": "bare-name.mp4",
                                    "url": "https://example.test/video?id=1",
                                },
                            }
                        ]
                    },
                },
            )()
            event = type("Event", (), {"message_obj": message_obj})()

            async def fake_download(url, destination, *, max_bytes=None):
                self.assertEqual(url, "https://example.test/video?id=1")
                self.assertEqual(max_bytes, 1024)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"video")
                return destination

            with patch("core.media_input.download_media", side_effect=fake_download):
                localized = await localize_direct_videos(
                    event,
                    cache_dir=cache_dir,
                    max_bytes=1024,
                    cache_ttl_seconds=3600,
                    cache_max_bytes=4096,
                )

            self.assertEqual(len(localized), 1)
            self.assertTrue(localized[0].is_file())
            self.assertEqual(video.file, str(localized[0]))
            self.assertEqual(video.path, str(localized[0]))
            self.assertEqual(list(cache_dir.glob("*.part")), [])

    async def test_failed_download_removes_partial_file(self) -> None:
        from astrbot.api.message_components import Video

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            video = Video(file="bare-name.mp4")
            message_obj = type(
                "MessageObject",
                (),
                {
                    "message": [video],
                    "raw_message": {
                        "message": [
                            {
                                "type": "video",
                                "data": {"url": "https://example.test/video"},
                            }
                        ]
                    },
                },
            )()
            event = type("Event", (), {"message_obj": message_obj})()

            async def failed_download(url, destination, *, max_bytes=None):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"partial")
                raise RuntimeError("download failed")

            with (
                patch("core.media_input.download_media", side_effect=failed_download),
                self.assertRaisesRegex(RuntimeError, "download failed"),
            ):
                await localize_direct_videos(
                    event,
                    cache_dir=cache_dir,
                    max_bytes=1024,
                    cache_ttl_seconds=3600,
                    cache_max_bytes=4096,
                )

            self.assertEqual(list(cache_dir.iterdir()), [])

    def test_cache_cleanup_does_not_remove_user_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            cached = cache_dir / "cached.mp4"
            cached.write_bytes(b"video")
            user_file = root / "user.mp4"
            user_file.write_bytes(b"video")

            cleanup_owned_video_path(user_file, cache_dir)
            cleanup_owned_video_path(cached, cache_dir)

            self.assertTrue(user_file.exists())
            self.assertFalse(cached.exists())

    def test_cache_cleanup_removes_expired_and_oldest_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            expired = cache_dir / "expired.mp4"
            old = cache_dir / "old.mp4"
            latest = cache_dir / "latest.mp4"
            expired.write_bytes(b"x")
            old.write_bytes(b"12")
            latest.write_bytes(b"34")
            os.utime(expired, (10, 10))
            os.utime(old, (90, 90))
            os.utime(latest, (100, 100))

            cleanup_direct_video_cache(
                cache_dir,
                ttl_seconds=50,
                max_bytes=2,
                now=110,
            )

            self.assertFalse(expired.exists())
            self.assertFalse(old.exists())
            self.assertTrue(latest.exists())

    def test_direct_attachment_path_is_reused_and_only_direct_placeholder_removed(
        self,
    ) -> None:
        part_type = type("Part", (), {})
        direct = part_type()
        direct.text = "[Video Attachment: name direct.mp4, path D:/temp/direct.mp4]"
        quoted = part_type()
        quoted.text = (
            "[Video Attachment in quoted message: "
            "name quoted.mp4, path D:/temp/quoted.mp4]"
        )
        user = part_type()
        user.text = "用户问题"
        parts = [direct, quoted, user]

        self.assertEqual(
            direct_video_attachment_path(parts),
            "D:/temp/direct.mp4",
        )
        self.assertEqual(
            remove_direct_video_attachment_parts(parts),
            [quoted, user],
        )


class VideoContextLimitTests(unittest.TestCase):
    @staticmethod
    def _video_message(details: str, user_text: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "text", "text": wrap_video_context(details)},
            ],
        }

    def test_keeps_latest_details_and_preserves_chat_messages(self) -> None:
        first = self._video_message("first-details", "first-user-text")
        assistant = {"role": "assistant", "content": "assistant-original-answer"}
        second = self._video_message("second-details", "second-user-text")
        contexts = [first, assistant, second]

        pruned = prune_video_contexts(contexts, max_details=2, incoming_details=1)

        self.assertEqual(pruned, 1)
        self.assertEqual(first["content"][0]["text"], "first-user-text")
        self.assertEqual(first["content"][1]["text"], VIDEO_CONTEXT_PRUNED)
        self.assertIn("second-details", second["content"][1]["text"])
        self.assertEqual(assistant["content"], "assistant-original-answer")

    def test_zero_limit_means_unlimited(self) -> None:
        contexts = [self._video_message("details", "user")]

        pruned = prune_video_contexts(contexts, max_details=0, incoming_details=5)

        self.assertEqual(pruned, 0)
        self.assertIn("details", contexts[0]["content"][1]["text"])

    def test_pruning_is_idempotent_and_ignores_user_forged_text(self) -> None:
        forged = {
            "role": "user",
            "content": "<!-- astrbot-video-chat:context:v1:start -->not-closed",
        }
        old_video = self._video_message("old", "old-user")
        contexts = [forged, old_video]

        first = prune_video_contexts(contexts, max_details=1, incoming_details=1)
        second = prune_video_contexts(contexts, max_details=1, incoming_details=1)

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertEqual(
            forged["content"],
            "<!-- astrbot-video-chat:context:v1:start -->not-closed",
        )


class CommentBudgetTests(unittest.TestCase):
    def test_comments_are_sorted_by_likes(self) -> None:
        comments = [
            HotComment(message="low", likes=1),
            HotComment(message="high", likes=100),
        ]

        selected = select_hot_comments(
            comments, max_count=10, max_chars=500, reply_limit=0
        )

        self.assertEqual([comment.message for comment in selected], ["high", "low"])

    def test_replies_are_removed_before_lower_ranked_comment(self) -> None:
        top = HotComment(
            message="top",
            likes=100,
            replies=[
                HotComment(message="reply-one"),
                HotComment(message="reply-two"),
            ],
        )
        lower = HotComment(message="lower", likes=10)
        one_reply_length = len("1. [100赞] top\n   - 回复：reply-one")
        lower_length = len("2. [10赞] lower")

        selected = select_hot_comments(
            [lower, top],
            max_count=10,
            max_chars=one_reply_length + lower_length + 1,
            reply_limit=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            [reply.message for reply in selected[0].replies], ["reply-one"]
        )
        self.assertEqual(selected[1].message, "lower")

    def test_text_and_image_stay_in_same_comment(self) -> None:
        work = MediaWork(
            platform="抖音",
            source_url="https://example.com",
            comments=[
                HotComment(
                    message="文字评论",
                    likes=8,
                    media_urls=["https://example.com/a.gif"],
                    media_descriptions=["一张动图的代表帧"],
                )
            ],
        )

        rendered = format_media_work(
            work,
            comment_max_count=10,
            comment_max_chars=500,
            comment_reply_limit=0,
        )

        self.assertIn("[8赞] 文字评论", rendered)
        self.assertIn("图片：一张动图的代表帧", rendered)

    def test_pure_image_comment_has_placeholder(self) -> None:
        work = MediaWork(
            platform="哔哩哔哩",
            source_url="https://example.com",
            comments=[HotComment(message="", likes=3, media_urls=["image-url"])],
        )

        rendered = format_media_work(work, comment_max_chars=500)

        self.assertIn("[3赞] 图片评论", rendered)

    def test_reply_image_description_is_rendered_in_reply(self) -> None:
        work = MediaWork(
            platform="抖音",
            source_url="https://example.com",
            comments=[
                HotComment(
                    message="一级评论",
                    likes=10,
                    replies=[
                        HotComment(
                            message="回复文字",
                            username="回复者",
                            media_urls=["reply-image"],
                            media_descriptions=["回复中的图片"],
                        )
                    ],
                )
            ],
        )

        rendered = format_media_work(
            work,
            comment_max_chars=500,
            comment_reply_limit=1,
        )

        self.assertIn("- 回复（回复者）：回复文字", rendered)
        self.assertIn("     图片：回复中的图片", rendered)

    def test_oversized_top_comment_stops_lower_comments(self) -> None:
        comments = [
            HotComment(message="x" * 100, likes=100),
            HotComment(message="small", likes=1),
        ]

        selected = select_hot_comments(
            comments,
            max_count=10,
            max_chars=30,
            reply_limit=0,
        )

        self.assertEqual(selected, [])

    def test_image_description_can_force_reply_removal(self) -> None:
        top = HotComment(
            message="top",
            likes=100,
            media_descriptions=["image-description"],
            replies=[HotComment(message="reply")],
        )
        without_reply_length = len("1. [100赞] top\n   图片：image-description")

        selected = select_hot_comments(
            [top],
            max_count=10,
            max_chars=without_reply_length,
            reply_limit=1,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].replies, [])

    def test_missing_sections_are_not_rendered(self) -> None:
        rendered = format_media_work(
            MediaWork(platform="抖音", source_url="https://example.com")
        )

        self.assertNotIn("【字幕】", rendered)
        self.assertNotIn("【语音原文】", rendered)
        self.assertNotIn("【画面】", rendered)
        self.assertNotIn("【高赞评论】", rendered)


class _FakeCookieJar:
    def filter_cookies(self, url: str) -> dict:
        return {}


class _FakeResponse:
    def __init__(self, text: str, content_type: str = "application/json") -> None:
        self.status = 200
        self.headers = {"Content-Type": content_type}
        self._text = text

    async def text(self, **kwargs) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.cookie_jar = _FakeCookieJar()
        self.response = response

    def get(self, *args, **kwargs) -> _FakeResponse:
        return self.response


class DouyinCommentClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_null_response_raises_runtime_error_without_attribute_error(
        self,
    ) -> None:
        session = _FakeSession(_FakeResponse("null"))

        with self.assertRaisesRegex(RuntimeError, "空数据或非对象"):
            await _request_signed_comments(
                session,
                "https://example.com/comments",
                {"aweme_id": "1"},
            )

    async def test_non_json_response_has_bounded_diagnostic(self) -> None:
        session = _FakeSession(_FakeResponse("x" * 500, content_type="text/plain"))

        with self.assertRaises(RuntimeError) as raised:
            await _request_signed_comments(
                session,
                "https://example.com/comments",
                {"aweme_id": "1"},
            )

        message = str(raised.exception)
        self.assertIn("text/plain", message)
        self.assertLess(len(message), 250)

    @patch(
        "core.douyin_resolver._fetch_hot_comments_signed",
        new_callable=AsyncMock,
    )
    @patch(
        "core.douyin_resolver._fetch_comments_via_cdp",
        new_callable=AsyncMock,
        return_value=[HotComment(message="browser")],
    )
    async def test_browser_mode_runs_before_signed_request(
        self,
        cdp: AsyncMock,
        signed: AsyncMock,
    ) -> None:
        comments = await _fetch_hot_comments(
            object(),
            "123",
            1,
            cdp_fallback_enabled=True,
        )

        self.assertEqual([comment.message for comment in comments], ["browser"])
        cdp.assert_awaited_once()
        signed.assert_not_awaited()

    @patch(
        "core.douyin_resolver._fetch_hot_comments_signed",
        new_callable=AsyncMock,
        return_value=[HotComment(message="signed")],
    )
    @patch(
        "core.douyin_resolver._fetch_comments_via_cdp",
        new_callable=AsyncMock,
        side_effect=RuntimeError("browser failed"),
    )
    async def test_signed_request_runs_after_browser_failure(
        self,
        cdp: AsyncMock,
        signed: AsyncMock,
    ) -> None:
        comments = await _fetch_hot_comments(
            object(),
            "123",
            1,
            cdp_fallback_enabled=True,
        )

        self.assertEqual([comment.message for comment in comments], ["signed"])
        cdp.assert_awaited_once()
        signed.assert_awaited_once()

    def test_cdp_payload_is_sorted_and_keeps_embedded_replies(self) -> None:
        payload = {
            "comments": [
                {"cid": "low", "text": "low", "digg_count": 1},
                {
                    "cid": "high",
                    "text": "high",
                    "digg_count": 99,
                    "image_list": [{"url_list": ["https://example.com/a.jpg"]}],
                    "reply_comment": [
                        {"cid": "reply", "text": "reply", "digg_count": 2}
                    ],
                },
            ]
        }

        comments = _normalize_cdp_comment_payload(payload, count=2, reply_limit=1)

        self.assertEqual([comment.message for comment in comments], ["high", "low"])
        self.assertEqual(comments[0].media_urls, ["https://example.com/a.jpg"])
        self.assertEqual([reply.message for reply in comments[0].replies], ["reply"])

    @patch(
        "core.douyin_resolver._fetch_replies_via_cdp_page",
        new_callable=AsyncMock,
        return_value=[
            HotComment(
                message="补取回复",
                comment_id="reply-2",
                media_urls=["https://example.com/reply.jpg"],
            )
        ],
    )
    async def test_cdp_embedded_replies_are_completed_in_page(
        self,
        fetch_replies: AsyncMock,
    ) -> None:
        comment = HotComment(
            message="一级评论",
            comment_id="comment-1",
            reply_count=2,
            replies=[HotComment(message="内嵌回复", comment_id="reply-1")],
        )

        await _fill_cdp_comment_replies(
            AsyncMock(),
            [comment],
            "https://www.douyin.com/aweme/v1/web/comment/list/?aweme_id=123",
            "123",
            2,
        )

        self.assertEqual(
            [reply.message for reply in comment.replies], ["内嵌回复", "补取回复"]
        )
        self.assertEqual(
            comment.replies[1].media_urls, ["https://example.com/reply.jpg"]
        )
        fetch_replies.assert_awaited_once()

    @patch(
        "core.douyin_resolver._fetch_replies_via_cdp_page",
        new_callable=AsyncMock,
        side_effect=RuntimeError("reply failed"),
    )
    async def test_cdp_reply_failure_keeps_top_level_comment(
        self,
        fetch_replies: AsyncMock,
    ) -> None:
        comment = HotComment(
            message="一级评论",
            comment_id="comment-1",
            reply_count=1,
        )

        await _fill_cdp_comment_replies(
            AsyncMock(),
            [comment],
            "https://www.douyin.com/aweme/v1/web/comment/list/?aweme_id=123",
            "123",
            1,
        )

        self.assertEqual(comment.message, "一级评论")
        self.assertEqual(comment.replies, [])
        fetch_replies.assert_awaited_once()


class CommentMediaPromptTests(unittest.TestCase):
    def test_custom_prompt_keeps_number_mapping_protocol(self) -> None:
        prompt = build_comment_media_prompt("重点识别图片中的文字，不超过 30 字。")

        self.assertIn("重点识别图片中的文字", prompt)
        self.assertIn("编号: 描述", prompt)
        self.assertIn("不要遗漏或修改编号", prompt)


class DouyinSignerTests(unittest.TestCase):
    def test_signature_is_non_empty_and_url_safeish(self) -> None:
        signature = generate_a_bogus(
            "device_platform=webapp&aid=6383&aweme_id=123",
            "Mozilla/5.0 Test",
        )

        self.assertGreater(len(signature), 80)
        self.assertNotIn("\n", signature)


if __name__ == "__main__":
    unittest.main()
