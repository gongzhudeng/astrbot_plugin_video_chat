from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.context_formatter import format_media_work, select_hot_comments
from core.douyin_resolver import (
    _fetch_hot_comments,
    _normalize_cdp_comment_payload,
    _request_signed_comments,
)
from core.douyin_signer import generate_a_bogus
from core.models import HotComment, MediaWork
from core.video_captioner import build_comment_media_prompt


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
