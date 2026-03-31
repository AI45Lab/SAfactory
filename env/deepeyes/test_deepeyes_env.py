import tempfile
import unittest
import io
from pathlib import Path
import sys

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.deepeyes.deepeyes_env import DeepEyesEnv
from env.deepeyes.reward import compute_common_reasoning_score, compute_math_score


class StaticJudgeClient:
    def __init__(self, response: str, *, available: bool = True) -> None:
        self._response = response
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages, *, temperature: float):
        del messages, temperature
        return self._response


class SequenceJudgeClient:
    def __init__(self, responses, *, available: bool = True) -> None:
        self._responses = list(responses)
        self._available = available
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    def chat(self, messages, *, temperature: float):
        del messages, temperature
        idx = self.calls
        self.calls += 1
        if idx >= len(self._responses):
            return None
        return self._responses[idx]


class DeepEyesEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tmpdir.name) / "source.png"
        Image.new("RGB", (128, 128), color=(255, 0, 0)).save(self.image_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_parquet_ref_env(self) -> DeepEyesEnv:
        parquet_path = Path(self.tmpdir.name) / "deepeyes_row.parquet"
        row = {
            "data_source": "vl_agent",
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "<image>\nWhat color is the square?"},
            ],
            "images": [str(self.image_path)],
            "reward_model": {"ground_truth": "red"},
            "extra_info": {"question": "What color is the square?"},
            "ability": "vl_chart",
            "env_name": "visual_toolbox_v2",
        }
        pq.write_table(pa.Table.from_pylist([row]), parquet_path)
        dataset_ref = {
            "__dataset_ref__": {
                "kind": "parquet_row",
                "path": str(parquet_path),
                "row_group": 0,
                "row_in_group": 0,
                "row_idx": 0,
            }
        }
        return DeepEyesEnv(dataset=dataset_ref)

    def _make_env(self) -> DeepEyesEnv:
        dataset = {
            "data_source": "vl_agent",
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "<image>\nWhat color is the square?"}
            ],
            "images": [str(self.image_path)],
            "reward_model": {"ground_truth": "red"},
            "extra_info": {"question": "What color is the square?"},
        }
        return DeepEyesEnv(dataset=dataset)

    def _make_embedded_image_env(self) -> DeepEyesEnv:
        buf = io.BytesIO()
        Image.new("RGB", (128, 128), color=(0, 255, 0)).save(buf, format="PNG")
        dataset = {
            "data_source": "vl_agent",
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "<image>\nWhat color is the square?"},
            ],
            "images": [{"bytes": buf.getvalue(), "path": None}],
            "reward_model": {"ground_truth": "green"},
            "extra_info": {"question": "What color is the square?"},
        }
        return DeepEyesEnv(dataset=dataset)

    def test_reset_builds_multimodal_prompt(self) -> None:
        env = self._make_env()
        env.reset()
        messages = env.get_task_prompt()
        self.assertEqual(messages[0]["role"], "system")
        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "image_url")

    def test_zoom_tool_appends_tool_feedback(self) -> None:
        env = self._make_env()
        env.reset()
        out = env.step(
            '<think>need zoom</think><tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,0,64,64]}}</tool_call>'
        )
        self.assertFalse(out.terminated)
        self.assertEqual(out.reward, 0.0)
        self.assertEqual(env.total_tool_calls, 1)
        last_message = env.get_task_prompt()[-1]
        self.assertEqual(last_message["role"], "user")

    def test_final_answer_scores_with_judge(self) -> None:
        env = self._make_env()
        env.judge_client = StaticJudgeClient("Judgement: 1")
        env.reset()
        env.step(
            '<think>need zoom</think><tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,0,64,64]}}</tool_call>'
        )
        out = env.step("<think>done</think><answer>red</answer>")
        self.assertTrue(out.terminated)
        self.assertGreater(out.reward, 1.0)

    def test_embedded_image_bytes_are_supported(self) -> None:
        env = self._make_embedded_image_env()
        env.judge_client = StaticJudgeClient("Judgement: 1")
        env.reset()
        messages = env.get_task_prompt()
        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "image_url")
        out = env.step("<think>done</think><answer>green</answer>")
        self.assertTrue(out.terminated)
        self.assertGreater(out.reward, 0.0)

    def test_parquet_row_reference_is_supported(self) -> None:
        env = self._make_parquet_ref_env()
        env.judge_client = StaticJudgeClient("Judgement: 1")
        env.reset()
        out = env.step("<think>done</think><answer>red</answer>")
        self.assertTrue(out.terminated)
        self.assertGreater(out.reward, 0.0)

    def test_max_turns_exhausted_after_tool_call_returns_zero(self) -> None:
        env = self._make_env()
        env.max_turns = 1
        env.reset()
        out = env.step(
            '<think>need zoom</think><tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,0,64,64]}}</tool_call>'
        )
        self.assertTrue(out.terminated)
        self.assertEqual(out.reward, 0.0)
        self.assertEqual(out.info["extra"]["termination_reason"], "max_turns_reached")
        self.assertIsNone(out.info["reward_result"])

    def test_malformed_tool_call_returns_error_and_continues(self) -> None:
        env = self._make_env()
        env.reset()
        out = env.step('<think>need zoom</think><tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[0,0,64,64]}</tool_call>')
        self.assertFalse(out.terminated)
        self.assertEqual(out.reward, 0.0)
        self.assertTrue(out.info["extra"]["malformed_tool_call"])
        last_message = env.get_task_prompt()[-1]
        self.assertEqual(last_message["role"], "user")
        self.assertIn("Invalid tool call format", str(last_message["content"]))

    def test_no_judge_returns_zero_visual_reward(self) -> None:
        env = self._make_env()
        env.reset()
        out = env.step("<think>done</think><answer>red</answer>")
        self.assertTrue(out.terminated)
        self.assertEqual(out.reward, 0.0)

    def test_missing_tool_call_or_answer_returns_zero(self) -> None:
        env = self._make_env()
        env.judge_client = StaticJudgeClient("Judgement: 1")
        env.reset()
        out = env.step("<think>I can answer soon</think>plain text without tags")
        self.assertTrue(out.terminated)
        self.assertEqual(out.reward, 0.0)
        self.assertEqual(out.info["extra"]["termination_reason"], "missing_tool_call_or_answer")
        self.assertIsNone(out.info["reward_result"])

    def test_common_reasoning_retries_invalid_judge_outputs(self) -> None:
        judge_client = SequenceJudgeClient(
            [
                "unexpected output",
                "## Equivalence Judgement\nTRUE",
            ]
        )
        result = compute_common_reasoning_score(
            "<think>done</think><answer>Paris</answer>",
            "Paris",
            question="Where is this place?",
            tool_used=False,
            judge_client=judge_client,
        )
        self.assertEqual(judge_client.calls, 2)
        self.assertEqual(result.acc_reward, 1.0)
        self.assertTrue(result.judge_used)

    def test_math_retries_transport_failures_only(self) -> None:
        judge_client = SequenceJudgeClient(
            [
                None,
                "unparseable output",
                "## Equivalence Judgement\nTRUE",
            ]
        )
        result = compute_math_score(
            "<think>calc</think>\\boxed{42}",
            "24",
            question="What is 6 * 4?",
            judge_client=judge_client,
        )
        self.assertEqual(judge_client.calls, 2)
        self.assertEqual(result.acc_reward, 0.0)
        self.assertFalse(result.judge_used)
        self.assertEqual(result.judge_response, "unparseable output")


if __name__ == "__main__":
    unittest.main()
