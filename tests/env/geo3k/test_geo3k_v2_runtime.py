from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.data_manager.load_yaml import (
    _build_parquet_row_refs,
    _cached_parquet_file,
    _cached_parquet_row_group,
    materialize_dataset_item,
)


class ParquetRowReferenceTest(unittest.TestCase):
    def tearDown(self) -> None:
        _cached_parquet_row_group.cache_clear()
        _cached_parquet_file.cache_clear()

    def test_selected_columns_are_materialized_and_json_safe(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.parquet"
            pq.write_table(
                pa.table(
                    {
                        "problem": ["question"],
                        "answer": ["42"],
                        "images": [["data:image/png;base64,abc"]],
                        "preprocessed_images": [b"not-json-safe"],
                    }
                ),
                path,
            )

            item = _build_parquet_row_refs(
                str(path),
                columns=["problem", "answer", "images"],
            )[0]
            row = materialize_dataset_item(item)

            self.assertEqual(set(row), {"problem", "answer", "images"})
            self.assertEqual(row["answer"], "42")
            json.dumps(row)


class Geo3kTurnLimitTest(unittest.TestCase):
    def test_turn_limit_is_a_normal_termination(self) -> None:
        try:
            from env.geo3k.runner import _RunState
        except ImportError as exc:
            self.skipTest(f"Geo3K grading dependencies are not installed: {exc}")

        state = _RunState(
            base_url="http://unused",
            model="model",
            temperature=1.0,
            timeout_s=1.0,
            question="question",
            ground_truth="5",
            image_urls=[],
            max_turns=3,
            max_images=0,
            echo_images_on_feedback=False,
        )
        actions = iter(
            [
                '<tool_call>{"name":"calc_score","arguments":{"answer":"4"}}</tool_call>',
            ]
            * 3
        )
        state._call_gateway = lambda: next(actions)  # type: ignore[method-assign]

        result = state.run()

        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["turns"], 3)


class GatewayWeightVersionTest(unittest.TestCase):
    def test_slime_weight_version_is_retained_in_gateway_metadata(self) -> None:
        from gateway.app import _merge_stream_event
        from gateway.storage import _response_weight_version

        response = json.dumps({"metadata": {"weight_version": 17}})
        self.assertEqual(_response_weight_version(response), 17)
        self.assertIsNone(_response_weight_version("{}"))

        stream_summary: dict = {}
        _merge_stream_event(
            stream_summary,
            {"metadata": {"weight_version": 18}},
        )
        self.assertEqual(stream_summary["metadata"]["weight_version"], 18)


if __name__ == "__main__":
    unittest.main()
