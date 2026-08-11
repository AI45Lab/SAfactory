"""Pins the shape contract between session_steps.response and the RL mask builder.

The training-side mask builder locates a trajectory by matching stored messages
against the turns it generated, and it compares every non-``content`` key for
exact equality. That makes the shape of ``session_steps.response`` a correctness
boundary rather than a storage detail: a mismatch drops whole rollout groups
behind a single log line. These tests exist so a future change to either side
fails here instead of silently starving training.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

# rl/buffer_server.py is a service entrypoint: it resolves its log directory at
# import time, so the root has to exist before the module is loaded. Keep the
# logs out of the working tree.
_LOG_ROOT = tempfile.mkdtemp(prefix="safactory-test-")
os.environ.setdefault("AIEVOBOX_ROOT", _LOG_ROOT)

from rl.buffer_server import _assistant_message_from_stored_response  # noqa: E402

# The text sglang /generate returns, which rl/llm_proxy.py uses both as the mask
# builder's tree key and as choices[0].message.content of the response it builds.
ASSISTANT_TEXT = "<think>hypotenuse is 5</think>The answer is 5."


def _llm_proxy_response(text: str = ASSISTANT_TEXT) -> str:
    """The chat-completion body rl/llm_proxy.py synthesizes for a rollout turn."""
    return json.dumps(
        {
            "id": "chatcmpl-session-1",
            "object": "chat.completion",
            "model": "proxy",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
            "metadata": {"weight_version": 17},
        }
    )


class StoredResponseRebuildTest(unittest.TestCase):
    """rl/buffer_server rebuilds this step's assistant turn from the stored row."""

    def test_sqlite_row_resolves_to_the_generated_turn(self) -> None:
        """The sqlite backend stores the whole envelope in this column."""
        self.assertEqual(
            _assistant_message_from_stored_response(_llm_proxy_response()),
            {"role": "assistant", "content": ASSISTANT_TEXT},
        )

    def test_cloud_row_resolves_to_the_generated_turn(self) -> None:
        """The cloud backend stores the extracted message instead."""
        from gateway.storage import _chat_completion_output

        column = json.dumps(_chat_completion_output(_llm_proxy_response()))

        self.assertEqual(
            _assistant_message_from_stored_response(column),
            {"role": "assistant", "content": ASSISTANT_TEXT},
        )

    def test_legacy_plain_text_row_is_still_understood(self) -> None:
        self.assertEqual(
            _assistant_message_from_stored_response(ASSISTANT_TEXT),
            {"role": "assistant", "content": ASSISTANT_TEXT},
        )

    def test_multi_choice_output_picks_a_role_bearing_message(self) -> None:
        column = json.dumps(
            [
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        )
        self.assertEqual(
            _assistant_message_from_stored_response(column),
            {"role": "assistant", "content": "first"},
        )

    def test_nothing_is_appended_when_there_is_no_turn_to_rebuild(self) -> None:
        for column in ("", None, "null", "{}", json.dumps({"content": "no role"})):
            with self.subTest(column=column):
                self.assertIsNone(_assistant_message_from_stored_response(column))

    def test_the_envelope_never_becomes_the_assistant_content(self) -> None:
        """Regression guard for the bug this contract was written for.

        Appending the column as-is produced an assistant message whose content
        was a JSON blob, which the mask builder rejected -- dropping the whole
        rollout group behind one log line.
        """
        rebuilt = _assistant_message_from_stored_response(_llm_proxy_response())

        self.assertNotIn("chatcmpl", rebuilt["content"])
        self.assertNotIn("choices", rebuilt["content"])


class MaskBuilderShapeContractTest(unittest.TestCase):
    """Which assistant shapes the mask builder's prefix match accepts."""

    def setUp(self) -> None:
        try:
            from rl.mask.trajectory_mask_builder import TrajectoryMaskBuilder
        except ImportError as exc:  # torch/transformers are training-side deps
            self.skipTest(f"mask builder dependencies are not installed: {exc}")
        # _message_matches needs no tokenizer, so skip __init__.
        self.builder = TrajectoryMaskBuilder.__new__(TrajectoryMaskBuilder)
        # The node the builder stores at generation time.
        self.node = {"role": "assistant", "content": ASSISTANT_TEXT}

    def test_rebuilt_message_matches_the_generated_turn(self) -> None:
        rebuilt = _assistant_message_from_stored_response(_llm_proxy_response())

        self.assertTrue(self.builder._message_matches(self.node, rebuilt))

    def test_think_blocks_are_ignored_when_comparing_content(self) -> None:
        self.assertTrue(
            self.builder._message_matches(
                self.node, {"role": "assistant", "content": "The answer is 5."}
            )
        )

    def test_extra_keys_break_the_match(self) -> None:
        """Why the rebuilt message must be passed through verbatim.

        Neither dropping these fields nor synthesizing them is safe, so a shape
        carrying them has to fail here rather than match by luck.
        """
        for extra in ("reasoning_content", "tool_calls", "refusal", "think"):
            with self.subTest(extra=extra):
                candidate = {
                    "role": "assistant",
                    "content": "The answer is 5.",
                    extra: "anything",
                }
                self.assertFalse(self.builder._message_matches(self.node, candidate))

    def test_raw_envelope_does_not_match(self) -> None:
        candidate = {"role": "assistant", "content": _llm_proxy_response()}
        self.assertFalse(self.builder._message_matches(self.node, candidate))


if __name__ == "__main__":
    unittest.main()
