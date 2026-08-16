from __future__ import annotations

# ruff: noqa: E402, I001

import asyncio
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

PLUGIN_DIR = Path(__file__).resolve().parent
ASTRBOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(PLUGIN_DIR.parent))
sys.path.insert(0, str(ASTRBOT_DIR))

from core.context_formatter import format_media_work, select_hot_comments
from core.douyin_resolver import (
    DouyinResult,
    _extract_from_router_data,
    _fetch_hot_comments,
    _fill_cdp_comment_replies,
    _normalize_cdp_comment_payload,
    _request_signed_comments,
)
from core.douyin_signer import generate_a_bogus
from core.video_resolver import _extract_info_sync
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
from core.video_captioner import (
    _frame_filter,
    _image_mime,
    _jpeg_qscale,
    build_comment_media_prompt,
    caption_from_frames,
    validate_vision_response,
)
from core.image_preprocess import prepare_image_bytes
from core.video_context import (
    VIDEO_CONTEXT_PRUNED,
    delete_video_context,
    list_video_contexts,
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
        self.unified_msg_origin = "test:private:user"
        self._extras: dict[str, object] = {}

    def get_extra(self, key: str, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key: str, value: object) -> None:
        self._extras[key] = value

    @staticmethod
    def plain_result(text: str) -> str:
        return text


class _FakeProviderRequest:
    def __init__(
        self,
        *,
        prompt: str = "",
        contexts: list | None = None,
        extra_user_content_parts: list | None = None,
        conversation=None,
        func_tool=None,
    ) -> None:
        self.prompt = prompt
        self.contexts = contexts or []
        self.extra_user_content_parts = extra_user_content_parts or []
        self.conversation = conversation
        self.func_tool = func_tool


class ImagePreprocessTests(unittest.TestCase):
    @staticmethod
    def _encode_image(
        size: tuple[int, int],
        *,
        mode: str = "RGB",
        color="white",
        image_format: str = "PNG",
    ) -> bytes:
        from PIL import Image

        image = Image.new(mode, size, color)
        output = io.BytesIO()
        image.save(output, format=image_format)
        return output.getvalue()

    def test_large_image_resizes_and_small_image_is_not_upscaled(self) -> None:
        from PIL import Image

        large = self._encode_image((2400, 1200))
        prepared = prepare_image_bytes(large, max_size=1280, quality=85)
        with Image.open(io.BytesIO(prepared)) as image:
            self.assertEqual(image.size, (1280, 640))

        small = self._encode_image((640, 320))
        self.assertEqual(
            prepare_image_bytes(small, max_size=1280, quality=85),
            small,
        )
        self.assertEqual(_image_mime(small), "image/png")

    def test_transparent_png_becomes_jpeg_with_white_background(self) -> None:
        from PIL import Image

        transparent = self._encode_image(
            (1600, 800),
            mode="RGBA",
            color=(0, 0, 0, 0),
        )
        prepared = prepare_image_bytes(transparent, max_size=1280, quality=85)

        self.assertEqual(_image_mime(prepared), "image/jpeg")
        with Image.open(io.BytesIO(prepared)) as image:
            pixel = image.convert("RGB").getpixel((0, 0))
        self.assertTrue(all(channel >= 245 for channel in pixel))

    def test_ffmpeg_defaults_are_unchanged_when_disabled(self) -> None:
        self.assertEqual(_frame_filter("1.000000", 0), "fps=1.000000")
        self.assertEqual(_jpeg_qscale(0, 85), 5)

    def test_ffmpeg_preprocess_only_downscales_and_maps_quality(self) -> None:
        frame_filter = _frame_filter("1.000000", 1280)

        self.assertIn("scale=w=min(1280\\,iw):h=min(1280\\,ih)", frame_filter)
        self.assertIn("force_original_aspect_ratio=decrease", frame_filter)
        self.assertEqual(_jpeg_qscale(1280, 85), 6)
        self.assertLess(_jpeg_qscale(1280, 95), _jpeg_qscale(1280, 75))


class AutoVideoContextTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _plugin(config: dict | None = None) -> VideoChatPlugin:
        plugin = object.__new__(VideoChatPlugin)
        plugin.config = config or {}
        return plugin

    def test_ordered_fallback_config_prefers_new_list_and_supports_legacy(self):
        plugin = self._plugin(
            {
                "caption_fallback_provider_ids": ["second", "third"],
                "caption_fallback_provider_id": "legacy",
            }
        )
        self.assertEqual(
            plugin._configured_fallback_ids(
                "caption_fallback_provider_ids",
                ("caption_fallback_provider_id",),
            ),
            ["second", "third"],
        )

        plugin.config = {
            "caption_fallback_provider_ids": [],
            "caption_fallback_provider_id": "legacy",
        }
        self.assertEqual(
            plugin._configured_fallback_ids(
                "caption_fallback_provider_ids",
                ("caption_fallback_provider_id",),
            ),
            [],
        )

        plugin.config = {"caption_fallback_provider_id": "legacy"}
        self.assertEqual(
            plugin._configured_fallback_ids(
                "caption_fallback_provider_ids",
                ("caption_fallback_provider_id",),
            ),
            ["legacy"],
        )

    async def test_visual_provider_fallback_skips_empty_error_and_refusal(self):
        plugin = self._plugin(
            {
                "caption_refusal_keywords": ["i can't discuss", "我是kiro"],
            }
        )
        providers = [object(), object(), object(), object()]
        plugin._visual_providers = lambda event: providers
        results = iter(
            [
                "   ",
                RuntimeError("provider unavailable"),
                "I can't discuss that.",
                "画面中有人正在晾衣服。",
            ]
        )
        calls: list[object] = []

        async def operation(provider):
            calls.append(provider)
            result = next(results)
            if isinstance(result, Exception):
                raise result
            response = SimpleNamespace(completion_text=result)
            return validate_vision_response(
                response,
                refusal_keywords=plugin._visual_refusal_keywords(),
                route="测试路径",
            )

        result = await plugin._try_visual_providers(_FakeVideoEvent(), operation)

        self.assertEqual(result, "画面中有人正在晾衣服。")
        self.assertEqual(calls, providers)

    def test_default_visual_refusal_keywords_work_before_config_is_saved(self):
        plugin = self._plugin()

        self.assertIn("i can't discuss", plugin._visual_refusal_keywords())
        self.assertIn("我是kiro", plugin._visual_refusal_keywords())

    def test_visual_refusal_keywords_match_kiro_and_discuss_responses(self):
        plugin = self._plugin(
            {
                "caption_refusal_keywords": [
                    "I can't discuss",
                    "我是Kiro",
                    "AI 开发环境助手",
                ]
            }
        )

        for text in (
            "I can't discuss that.",
            "我是Kiro，一个AI开发环境助手。",
        ):
            with self.assertRaisesRegex(RuntimeError, "返回了拒绝内容"):
                validate_vision_response(
                    SimpleNamespace(completion_text=text),
                    refusal_keywords=plugin._visual_refusal_keywords(),
                    route="测试路径",
                )

    async def test_stale_request_refreshes_latest_video_history_and_keeps_decorations(
        self,
    ) -> None:
        stale_history = [{"role": "user", "content": "较早的消息"}]
        video_turn = {
            "role": "user",
            "content": [
                {"type": "text", "text": "看看这个视频"},
                {"type": "text", "text": wrap_video_context("最新视频详情")},
            ],
        }
        latest_history = stale_history + [
            video_turn,
            {"role": "assistant", "content": "我看到了"},
        ]
        stale_conversation = SimpleNamespace(
            cid="conversation-id",
            history=json.dumps(stale_history, ensure_ascii=False),
        )
        latest_conversation = SimpleNamespace(
            cid="conversation-id",
            history=json.dumps(latest_history, ensure_ascii=False),
        )
        manager = SimpleNamespace(
            get_conversation=AsyncMock(return_value=latest_conversation),
            update_conversation=AsyncMock(),
        )
        plugin = self._plugin({"max_video_context_details": 1})
        plugin.context = SimpleNamespace(conversation_manager=manager)
        event = _FakeVideoEvent(message_str="刚才那个视频呢")
        persona_context = {"role": "assistant", "content": "人格开场"}
        file_context = {"role": "system", "content": "文件提取结果"}
        request = _FakeProviderRequest(
            prompt="刚才那个视频呢",
            contexts=[persona_context, *stale_history, file_context],
            conversation=stale_conversation,
        )

        await plugin.inject_video_context(event, request)

        self.assertIs(request.conversation, latest_conversation)
        self.assertEqual(
            request.contexts,
            [persona_context, *latest_history, file_context],
        )
        self.assertEqual(len(list_video_contexts(request.contexts)), 1)
        manager.update_conversation.assert_not_awaited()

    async def test_refresh_failure_does_not_mutate_request(self) -> None:
        stale_history = [{"role": "user", "content": "旧消息"}]
        stale_conversation = SimpleNamespace(
            cid="conversation-id",
            history=json.dumps(stale_history, ensure_ascii=False),
        )
        event = _FakeVideoEvent(message_str="普通消息")

        for latest_conversation in (
            None,
            SimpleNamespace(cid="conversation-id", history="not-json"),
        ):
            with self.subTest(latest_conversation=latest_conversation):
                manager = SimpleNamespace(
                    get_conversation=AsyncMock(return_value=latest_conversation),
                    update_conversation=AsyncMock(),
                )
                plugin = self._plugin()
                plugin.context = SimpleNamespace(conversation_manager=manager)
                original_contexts = copy.deepcopy(stale_history)
                request = _FakeProviderRequest(
                    prompt="普通消息",
                    contexts=copy.deepcopy(original_contexts),
                    conversation=stale_conversation,
                )

                await plugin.inject_video_context(event, request)

                self.assertIs(request.conversation, stale_conversation)
                self.assertEqual(request.contexts, original_contexts)
                manager.update_conversation.assert_not_awaited()

    async def test_refresh_skips_custom_context_without_stale_snapshot(self) -> None:
        stale_history = [{"role": "user", "content": "旧消息"}]
        latest_history = stale_history + [
            {"role": "assistant", "content": "较新的回复"}
        ]
        stale_conversation = SimpleNamespace(
            cid="conversation-id",
            history=json.dumps(stale_history, ensure_ascii=False),
        )
        latest_conversation = SimpleNamespace(
            cid="conversation-id",
            history=json.dumps(latest_history, ensure_ascii=False),
        )
        manager = SimpleNamespace(
            get_conversation=AsyncMock(return_value=latest_conversation),
            update_conversation=AsyncMock(),
        )
        plugin = self._plugin()
        plugin.context = SimpleNamespace(conversation_manager=manager)
        event = _FakeVideoEvent(message_str="普通消息")
        custom_contexts = [{"role": "system", "content": "完全自定义上下文"}]
        request = _FakeProviderRequest(
            prompt="普通消息",
            contexts=copy.deepcopy(custom_contexts),
            conversation=stale_conversation,
        )

        await plugin.inject_video_context(event, request)

        self.assertIs(request.conversation, stale_conversation)
        self.assertEqual(request.contexts, custom_contexts)
        manager.update_conversation.assert_not_awaited()

    async def test_temporary_mode_uses_video_only_for_current_request(self) -> None:
        old = {
            "role": "user",
            "content": [{"type": "text", "text": wrap_video_context("old")}],
        }
        manager = SimpleNamespace(update_conversation=AsyncMock())
        conversation = SimpleNamespace(cid="conversation-id", history="[]")
        plugin = self._plugin({"max_video_context_details": 0})
        plugin.context = SimpleNamespace(conversation_manager=manager)
        plugin._do_analyze = AsyncMock(return_value="temporary-details")
        event = _FakeVideoEvent(message_str="看看 https://www.bilibili.com/video/BV1xx")
        request = _FakeProviderRequest(
            prompt="这段视频讲了什么？",
            contexts=[old],
            conversation=conversation,
        )

        await plugin.inject_video_context(event, request)

        request_context = request.extra_user_content_parts[-1]
        self.assertIn("temporary-details", request_context.text)
        self.assertTrue(request_context._no_save)
        self.assertEqual(old["content"][0]["text"], VIDEO_CONTEXT_PRUNED)
        saved_history = manager.update_conversation.await_args.kwargs["history"]
        self.assertNotIn(
            "astrbot-video-chat:context:v1:start",
            json.dumps(saved_history, ensure_ascii=False),
        )
        self.assertEqual(saved_history[-1]["role"], "user")
        self.assertEqual(
            saved_history[-1]["content"],
            [{"type": "text", "text": "这段视频讲了什么？"}],
        )
        self.assertNotIn(
            "astrbot-video-chat:context:v1:start",
            conversation.history,
        )

    async def test_persistent_mode_saves_user_turn_before_provider_finishes(
        self,
    ) -> None:
        manager = SimpleNamespace(update_conversation=AsyncMock())
        conversation = SimpleNamespace(cid="conversation-id", history="[]")
        plugin = self._plugin({"max_video_context_details": 2})
        plugin.context = SimpleNamespace(conversation_manager=manager)
        plugin._do_analyze = AsyncMock(return_value="persistent-details")
        event = _FakeVideoEvent(message_str="看看 https://www.bilibili.com/video/BV1xx")
        request = _FakeProviderRequest(
            prompt="记住这段视频",
            conversation=conversation,
        )

        await plugin.inject_video_context(event, request)

        request_context = request.extra_user_content_parts[-1]
        self.assertFalse(request_context._no_save)
        saved_history = manager.update_conversation.await_args.kwargs["history"]
        saved_context = saved_history[-1]["content"][-1]["text"]
        self.assertEqual(saved_context, request_context.text)
        self.assertIn("persistent-details", saved_context)
        self.assertEqual(saved_history[-1]["content"][0]["text"], "记住这段视频")

    async def test_zero_mode_cleans_history_without_a_new_video(self) -> None:
        old = {
            "role": "user",
            "content": [
                {"type": "text", "text": "旧问题"},
                {"type": "text", "text": wrap_video_context("old-details")},
            ],
        }
        manager = SimpleNamespace(update_conversation=AsyncMock())
        conversation = SimpleNamespace(cid="conversation-id", history="[]")
        plugin = self._plugin({"max_video_context_details": 0})
        plugin.context = SimpleNamespace(conversation_manager=manager)
        event = _FakeVideoEvent(message_str="普通追问")
        request = _FakeProviderRequest(
            prompt="普通追问",
            contexts=[old],
            conversation=conversation,
        )

        await plugin.inject_video_context(event, request)

        self.assertEqual(old["content"][0]["text"], "旧问题")
        self.assertEqual(old["content"][1]["text"], VIDEO_CONTEXT_PRUNED)
        manager.update_conversation.assert_awaited_once()
        self.assertNotIn(
            "astrbot-video-chat:context:v1:start",
            conversation.history,
        )

    async def test_clear_video_context_command_preserves_chat_messages(self) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "用户原话"},
                    {"type": "text", "text": wrap_video_context("old-details")},
                ],
            },
            {"role": "assistant", "content": "AI 回复"},
        ]
        conversation = SimpleNamespace(
            cid="conversation-id",
            history=json.dumps(history, ensure_ascii=False),
        )
        manager = SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation-id"),
            get_conversation=AsyncMock(return_value=conversation),
            update_conversation=AsyncMock(),
        )
        plugin = self._plugin()
        plugin.context = SimpleNamespace(conversation_manager=manager)
        event = _FakeVideoEvent()

        results = [result async for result in plugin.cmd_clear_video_context(event)]

        self.assertEqual(results, ["已清理 1 个完整视频解析详情。"])
        saved_history = manager.update_conversation.await_args.kwargs["history"]
        self.assertEqual(saved_history[0]["content"][0]["text"], "用户原话")
        self.assertEqual(saved_history[0]["content"][1]["text"], VIDEO_CONTEXT_PRUNED)
        self.assertEqual(saved_history[1]["content"], "AI 回复")

    async def test_explicit_link_is_injected_and_tool_is_deduplicated(self) -> None:
        old = {
            "role": "user",
            "content": [{"type": "text", "text": wrap_video_context("old")}],
        }
        plugin = self._plugin(
            {
                "max_video_context_details": 1,
                "analyze_video_tool_enabled": True,
            }
        )
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
        self.assertIn("当前请求已解析一个视频", duplicate_result)
        plugin._do_analyze.assert_awaited_once_with(event, normalized_url)

    async def test_disabled_tool_is_removed_and_handler_rejects(self) -> None:
        removed: list[str] = []
        tool_set = SimpleNamespace(remove_tool=removed.append)
        plugin = self._plugin()
        plugin._do_analyze = AsyncMock(return_value="unused")
        event = _FakeVideoEvent(message_str="普通消息")
        request = _FakeProviderRequest(func_tool=tool_set)

        await plugin.inject_video_context(event, request)
        result = await plugin.analyze_video(
            event, "https://www.bilibili.com/video/BV1xx"
        )

        self.assertEqual(removed, ["analyze_video"])
        self.assertIn("视频分析工具已关闭", result)
        plugin._do_analyze.assert_not_awaited()

    async def test_enabled_tool_allows_only_one_analysis_per_event(self) -> None:
        plugin = self._plugin({"analyze_video_tool_enabled": True})
        plugin._do_analyze = AsyncMock(return_value="[视频解析结果]\n工具详情")
        event = _FakeVideoEvent()

        first = await plugin.analyze_video(
            event, "https://www.bilibili.com/video/BV1xx"
        )
        second = await plugin.analyze_video(
            event, "https://www.bilibili.com/video/BV2yy"
        )

        self.assertIn("工具详情", first)
        self.assertIn("当前请求已解析一个视频", second)
        plugin._do_analyze.assert_awaited_once()

    async def test_list_and_delete_video_context_commands(self) -> None:
        history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": wrap_video_context(
                            "[视频解析结果]\n\n【作品】\n平台：抖音\n标题：老头奶茶"
                        ),
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "video-call",
                        "function": {
                            "name": "analyze_video",
                            "arguments": json.dumps(
                                {"url": "https://v.douyin.com/old/"}
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "video-call",
                "content": ("[视频解析结果]\n\n【作品】\n平台：抖音\n标题：小学生幻想"),
            },
        ]
        conversation = SimpleNamespace(
            cid="conversation-id",
            history=json.dumps(history, ensure_ascii=False),
        )
        manager = SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation-id"),
            get_conversation=AsyncMock(return_value=conversation),
            update_conversation=AsyncMock(),
        )
        plugin = self._plugin()
        plugin.context = SimpleNamespace(conversation_manager=manager)
        event = _FakeVideoEvent()

        listed = [result async for result in plugin.cmd_list_video_context(event)]
        deleted = [
            result async for result in plugin.cmd_delete_video_context(event, "2")
        ]

        self.assertIn("1. [自动] 抖音 · 老头奶茶", listed[0])
        self.assertIn("2. [工具] 抖音 · 小学生幻想", listed[0])
        self.assertEqual(deleted, ["已删除视频上下文：[工具] 抖音 · 小学生幻想"])
        saved = manager.update_conversation.await_args.kwargs["history"]
        self.assertEqual(len(list_video_contexts(saved)), 1)
        self.assertFalse(any(item.get("role") == "tool" for item in saved))

    async def test_douyin_video_uses_local_frames_instead_of_native_video_url(self):
        plugin = self._plugin()
        event = _FakeVideoEvent()
        resolved = DouyinResult(play_url="https://cdn.example/video.mp4")
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with (
                patch(
                    "astrbot_plugin_video_chat.main.resolve_douyin",
                    new=AsyncMock(return_value=resolved),
                ),
                patch(
                    "astrbot_plugin_video_chat.main.download_media",
                    new=AsyncMock(),
                ) as download_media,
                patch.object(
                    plugin,
                    "_caption_frames_with_fallback",
                    new=AsyncMock(return_value="frame summary"),
                ) as caption_frames,
                patch.object(
                    plugin,
                    "_caption_url_with_fallback",
                    new=AsyncMock(),
                ) as caption_url,
            ):
                work = await plugin._analyze_douyin(
                    event,
                    "https://v.douyin.com/example/",
                    temp_dir,
                    0,
                    0,
                    120,
                    "",
                    1024,
                )

        finally:
            temp_dir.rmdir()

        download_media.assert_awaited_once()
        caption_frames.assert_awaited_once()
        caption_url.assert_not_awaited()
        self.assertEqual(work.visual_summary, "frame summary")
        self.assertEqual(work.local_video_path, temp_dir / "douyin_video.mp4")

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
        user_context = plugin._caption_user_context(event)
        video_info = plugin._video_info_for_caption(
            MediaWork(
                platform="抖音",
                source_url="https://v.douyin.com/example/",
                title="测试标题",
                description="测试描述",
                topics=["#测试", "#舞蹈"],
                author="测试作者",
                author_id="author-1",
                published_at="2026-08-10",
            )
        )

        self.assertNotIn("图片说明", user_context)
        self.assertNotIn("[视频1]", user_context)
        self.assertIn("这是用户问题", user_context)
        self.assertIn("用户聊天记录", prompt)
        self.assertIn("视频信息", prompt)
        self.assertIn("标题：测试标题", video_info)
        self.assertIn("作者：测试作者（author-1）", video_info)
        self.assertIn("话题：#测试 #舞蹈", video_info)
        self.assertIn("链接：https://v.douyin.com/example/", video_info)

    async def test_caption_payload_orders_frames_prompt_chat_and_video_info(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        class Provider:
            async def text_chat(self, *, contexts):
                captured["contexts"] = contexts
                return SimpleNamespace(completion_text="转述结果")

        with patch(
            "core.video_captioner._extract_frames_sync",
            return_value=["data:image/jpeg;base64,frame-1"],
        ):
            result = await caption_from_frames(
                Path("video.mp4"),
                provider=Provider(),
                prompt="转述提示词",
                user_context="<user_context>用户聊天记录</user_context>",
                video_info="<video_info>标题：测试标题</video_info>",
            )

        content = captured["contexts"][0]["content"]
        self.assertEqual(result, "转述结果")
        self.assertEqual(
            [part["type"] for part in content],
            ["image_url", "text", "text", "text"],
        )
        self.assertEqual(content[1]["text"], "转述提示词")
        self.assertIn("用户聊天记录", content[2]["text"])
        self.assertIn("标题：测试标题", content[3]["text"])


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

    def test_zero_limit_clears_all_video_details(self) -> None:
        contexts = [self._video_message("details", "user")]

        pruned = prune_video_contexts(contexts, max_details=0, incoming_details=5)

        self.assertEqual(pruned, 1)
        self.assertEqual(contexts[0]["content"][0]["text"], "user")
        self.assertEqual(contexts[0]["content"][1]["text"], VIDEO_CONTEXT_PRUNED)

    def test_mixed_marker_and_tool_contexts_share_one_limit(self) -> None:
        contexts = [
            self._video_message(
                "[视频解析结果]\n\n【作品】\n平台：抖音\n标题：自动视频",
                "自动链接",
            ),
            {
                "role": "assistant",
                "content": "保留这段正文",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "video-call",
                        "function": {
                            "name": "analyze_video",
                            "arguments": '{"url":"https://v.douyin.com/tool/"}',
                        },
                    },
                    {
                        "type": "function",
                        "id": "other-call",
                        "function": {"name": "other_tool", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "video-call",
                "content": (
                    "[视频解析结果]\n\n【作品】\n平台：哔哩哔哩\n标题：工具视频"
                ),
            },
            {
                "role": "tool",
                "tool_call_id": "other-call",
                "content": "其他工具结果",
            },
        ]

        entries = list_video_contexts(contexts)
        pruned = prune_video_contexts(contexts, max_details=1)

        self.assertEqual([entry.kind for entry in entries], ["自动", "工具"])
        self.assertEqual([entry.title for entry in entries], ["自动视频", "工具视频"])
        self.assertEqual(pruned, 1)
        self.assertEqual(len(list_video_contexts(contexts)), 1)
        self.assertIn(VIDEO_CONTEXT_PRUNED, contexts[0]["content"][1]["text"])
        self.assertEqual(contexts[1]["tool_calls"][0]["id"], "video-call")

    def test_deleting_tool_context_preserves_unrelated_tool_call(self) -> None:
        contexts = [
            {
                "role": "assistant",
                "content": "保留正文",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "video-call",
                        "function": {
                            "name": "analyze_video",
                            "arguments": '{"url":"https://v.douyin.com/tool/"}',
                        },
                    },
                    {
                        "type": "function",
                        "id": "other-call",
                        "function": {"name": "other_tool", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "video-call",
                "content": "[视频解析结果]\n\n【作品】\n平台：抖音",
            },
            {
                "role": "tool",
                "tool_call_id": "other-call",
                "content": "其他工具结果",
            },
        ]

        deleted = delete_video_context(contexts, 1)

        self.assertIsNotNone(deleted)
        self.assertEqual(contexts[0]["content"], "保留正文")
        self.assertEqual(
            [call["id"] for call in contexts[0]["tool_calls"]], ["other-call"]
        )
        self.assertEqual(len(contexts), 2)
        self.assertEqual(contexts[1]["tool_call_id"], "other-call")

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


class SttTimeoutTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _plugin(timeout_seconds: float) -> VideoChatPlugin:
        plugin = object.__new__(VideoChatPlugin)
        plugin.config = {
            "stt_enabled": True,
            "stt_timeout_seconds": timeout_seconds,
        }
        return plugin

    async def _run_media(
        self,
        *,
        timeout_seconds: float,
        visual_delay: float,
        stt_delay: float,
    ) -> tuple[MediaWork, bool, float]:
        plugin = self._plugin(timeout_seconds)
        work = MediaWork(platform="测试", source_url="")
        stt_cancelled = False

        async def visual_operation() -> str:
            await asyncio.sleep(visual_delay)
            return "画面结果"

        async def transcribe(*args, **kwargs) -> str:
            nonlocal stt_cancelled
            try:
                await asyncio.sleep(stt_delay)
                return "语音结果"
            except asyncio.CancelledError:
                stt_cancelled = True
                raise

        plugin._transcribe_with_fallback = transcribe
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        await plugin._analyze_local_video_media(
            _FakeVideoEvent(),
            work,
            Path("video.mp4"),
            Path("."),
            120,
            "",
            visual_operation=visual_operation,
        )
        return work, stt_cancelled, loop.time() - started_at

    async def test_stt_result_is_kept_when_visual_runs_past_deadline(self) -> None:
        work, cancelled, _ = await self._run_media(
            timeout_seconds=0.01,
            visual_delay=0.04,
            stt_delay=0.02,
        )

        self.assertEqual(work.visual_summary, "画面结果")
        self.assertEqual(work.transcript, "语音结果")
        self.assertFalse(cancelled)

    async def test_stt_is_cancelled_after_visual_finishes_and_deadline_expires(
        self,
    ) -> None:
        work, cancelled, elapsed = await self._run_media(
            timeout_seconds=0.03,
            visual_delay=0.005,
            stt_delay=1,
        )

        self.assertEqual(work.visual_summary, "画面结果")
        self.assertEqual(work.transcript, "")
        self.assertTrue(cancelled)
        self.assertLess(elapsed, 0.2)

    async def test_visual_may_run_past_deadline_before_stt_is_cancelled(self) -> None:
        work, cancelled, elapsed = await self._run_media(
            timeout_seconds=0.01,
            visual_delay=0.04,
            stt_delay=1,
        )

        self.assertEqual(work.visual_summary, "画面结果")
        self.assertEqual(work.transcript, "")
        self.assertTrue(cancelled)
        self.assertGreaterEqual(elapsed, 0.035)
        self.assertLess(elapsed, 0.2)

    async def test_zero_timeout_waits_for_stt_after_visual_finishes(self) -> None:
        work, cancelled, elapsed = await self._run_media(
            timeout_seconds=0,
            visual_delay=0.005,
            stt_delay=0.03,
        )

        self.assertEqual(work.transcript, "语音结果")
        self.assertFalse(cancelled)
        self.assertGreaterEqual(elapsed, 0.025)


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


class DouyinMediaTypeTests(unittest.TestCase):
    @staticmethod
    def _router_html(page_type: str, item: dict) -> str:
        payload = {
            "loaderData": {
                f"{page_type}_(id)/page": {"videoInfoRes": {"item_list": [item]}}
            }
        }
        return f"<script>window._ROUTER_DATA = {json.dumps(payload)}</script>"

    def test_note_prefers_images_when_compatibility_video_exists(self) -> None:
        item = {
            "desc": "图文内容",
            "images": [{"url_list": ["https://example.com/note.jpg"]}],
            "video": {"play_addr": {"url_list": ["https://example.com/compat.mp4"]}},
        }

        result = _extract_from_router_data(
            self._router_html("note", item),
            "7664179859198843889",
            preferred_type="note",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.image_urls, ["https://example.com/note.jpg"])
        self.assertIsNone(result.play_url)

    def test_video_keeps_play_url_when_images_also_exist(self) -> None:
        item = {
            "desc": "视频内容",
            "images": [{"url_list": ["https://example.com/cover.jpg"]}],
            "video": {"play_addr": {"url_list": ["https://example.com/video.mp4"]}},
        }

        result = _extract_from_router_data(
            self._router_html("video", item),
            "7664179859198843889",
            preferred_type="video",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.play_url, "https://example.com/video.mp4")
        self.assertEqual(result.image_urls, [])

    def test_new_nested_page_payload_supports_camel_case_media_fields(self) -> None:
        item = {
            "description": "新版页面的视频内容",
            "authorInfo": {"name": "测试作者", "uniqueId": "author-id"},
            "createTime": "1786536070",
            "textExtra": [{"hashtagName": "测试"}],
            "video": {"playAddr": {"urlList": ["https://example.com/video.mp4"]}},
        }
        payload = {
            "loaderData": {
                "video_(id)/page": {
                    "payload": {"aweme_detail": item},
                }
            }
        }
        html = f"<script>window._ROUTER_DATA = {json.dumps(payload)}</script>"

        result = _extract_from_router_data(
            html,
            "7664179859198843889",
            preferred_type="video",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.play_url, "https://example.com/video.mp4")
        self.assertEqual(result.title, "新版页面的视频内容")
        self.assertEqual(result.author, "测试作者")
        self.assertEqual(result.author_id, "抖音号 author-id")
        self.assertEqual(result.topics, ["#测试"])


class VideoResolverCookieRetryTests(unittest.TestCase):
    def test_stale_cookie_error_retries_once_without_cookie_file(self) -> None:
        calls: list[str | None] = []

        class FakeYoutubeDL:
            def __init__(self, options: dict) -> None:
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def extract_info(self, url: str, download: bool) -> dict:
                cookiefile = self.options.get("cookiefile")
                calls.append(cookiefile)
                if cookiefile:
                    raise RuntimeError(
                        "Fresh cookies (not necessarily logged in) are needed"
                    )
                return {"title": "retry succeeds"}

        fake_module = type("FakeYtDlp", (), {"YoutubeDL": FakeYoutubeDL})()
        with (
            patch.dict(sys.modules, {"yt_dlp": fake_module}),
            patch(
                "core.video_resolver._sanitize_cookies_file",
                return_value="stale-cookies.txt",
            ),
        ):
            result = _extract_info_sync(
                "https://v.douyin.com/example/",
                proxy=None,
                cookies_file="stale-cookies.txt",
            )

        self.assertEqual(result, {"title": "retry succeeds"})
        self.assertEqual(calls, ["stale-cookies.txt", None])


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
