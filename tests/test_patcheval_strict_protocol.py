from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from env.patcheval import strict_runner


ROOT = Path(__file__).resolve().parents[1]
PATCHEVAL_ROOT = ROOT / "env" / "patcheval" / "PatchEval" / "patcheval"


class PatchEvalStrictPromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(PATCHEVAL_ROOT))
        from exp_llm.helper.llm_suite import LLMClient

        cls.official_client = LLMClient
        records = json.loads((PATCHEVAL_ROOT / "datasets" / "input.json").read_text(encoding="utf-8"))
        cls.record = next(record for record in records if record.get("is_poc"))

    def test_each_s1_prompt_matches_official_builder(self) -> None:
        functions = [
            {
                "id": item["id"],
                "original_code": strict_runner._process_original_code(item["snippet"]),
            }
            for item in self.record["vul_func"]
        ]
        settings = {
            "s1.1": ("Default.txt", True),
            "s1.2": ("Ablation_without_CoT.txt", False),
            "s1.3": ("Ablation_without_Knowledge.txt", True),
            "s1.4": ("Default.txt", True),
        }
        for setting, (filename, use_cot) in settings.items():
            with self.subTest(setting=setting):
                template = (PATCHEVAL_ROOT / "exp_llm" / "prompt_templates" / filename).read_text(
                    encoding="utf-8"
                )
                expected = self.official_client.build_prompt(
                    None,
                    functions,
                    self.record,
                    {},
                    template,
                    use_cot,
                    self.record["cve_id"],
                    [],
                )
                actual = strict_runner._build_prompt(
                    record=self.record,
                    vul_functions=self.record["vul_func"],
                    feedbacks={},
                    prompt_template=template,
                    use_cot=use_cot,
                )
                self.assertEqual(actual, expected)

    def test_official_function_json_is_parsed(self) -> None:
        response = '```json\n[{"id":"vul_py_1","patch":"def fixed():\\n    return True"}]\n```'
        self.assertEqual(
            strict_runner._parse_response(response),
            {"vul_py_1": "def fixed():\n    return True"},
        )


if __name__ == "__main__":
    unittest.main()
