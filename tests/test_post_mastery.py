import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline.post_mastery_generate import (
    _build_prompt,
    _classify_failure,
    _parse_generated_output,
    _similarity_score,
    generate_post_mastery_questions,
)
from kb_pipeline.post_mastery_plan import (
    build_post_mastery_plan,
    replan_failed_plan,
)


class GeneratedOutputParsingTests(unittest.TestCase):
    def test_direct_json(self) -> None:
        parsed, error = _parse_generated_output(
            '{"question":"How many?","steps":["2 + 3 = 5"],"answer":"5"}'
        )
        self.assertEqual(error, "")
        self.assertEqual(parsed["answer"], "5")

    def test_fenced_json(self) -> None:
        parsed, error = _parse_generated_output(
            '```json\n{"question":"How many?","steps":["4 * 2 = 8"],"answer":"8"}\n```'
        )
        self.assertEqual(error, "")
        self.assertEqual(parsed["steps"], ["4 * 2 = 8"])

    def test_double_encoded_json(self) -> None:
        payload = json.dumps(
            json.dumps(
                {
                    "question": "How many?",
                    "steps": ["9 - 4 = 5"],
                    "answer": "5",
                }
            )
        )
        parsed, error = _parse_generated_output(payload)
        self.assertEqual(error, "")
        self.assertEqual(parsed["question"], "How many?")

    def test_solution_fallback_and_numeric_cleanup(self) -> None:
        parsed, error = _parse_generated_output(
            '{"question":"How many?","solution":"10 * 1000 = 10000\\n10000 - 1 = 9999","answer":"9,999"}'
        )
        self.assertEqual(error, "")
        self.assertEqual(len(parsed["steps"]), 2)
        self.assertEqual(parsed["answer"], "9999")

    def test_number_only_variants_are_similar(self) -> None:
        score = _similarity_score(
            "A shop packs 12 items into 3 boxes. How many items are in each box?",
            "A shop packs 20 items into 5 boxes. How many items are in each box?",
        )
        self.assertGreaterEqual(score, 0.88)

    def test_prompt_treats_kb_as_optional(self) -> None:
        prompt = _build_prompt(
            "A seed problem.",
            "Equal",
            {
                "math": {"skill_tags": ["addition"]},
                "diversity": {
                    "primary_scene": {"domain": "library"},
                    "alternative_scenes": [{"domain": "bakery"}],
                    "variation_mode": "same mathematical template in a new scene",
                },
                "kb_inspiration": {"scene_keywords": ["books"]},
            },
        )
        self.assertIn("optional inspiration rather than a fixed template", prompt[1]["content"])
        self.assertIn("same mathematical template", prompt[1]["content"])

    def test_failure_classification(self) -> None:
        self.assertEqual(
            _classify_failure("question is too similar to an earlier generated question"),
            "similarity",
        )
        self.assertEqual(
            _classify_failure("response is not a valid JSON object"),
            "invalid_json",
        )
        self.assertEqual(
            _classify_failure("TimeoutError: timed out"),
            "request_error",
        )

    def test_batch_round_retries_only_failed_item(self) -> None:
        class FakeCompletions:
            calls = 0

            async def create(self, **kwargs):
                self.__class__.calls += 1
                if self.__class__.calls == 1:
                    raise TimeoutError("temporary timeout")
                message = SimpleNamespace(
                    content='{"question":"A library has 6 shelves with 4 books on each shelf. How many books are there?","steps":["6 * 4 = 24"],"answer":"24"}'
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)]
                )

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

            async def close(self):
                return None

        plans = [
            {
                "source_task_id": 1,
                "plan_id": "1_0",
                "knowledge": {
                    "math": {"skill_tags": ["multiplication"]},
                    "diversity": {
                        "primary_scene": {"domain": "community_library"},
                        "alternative_scenes": [],
                    },
                    "kb_inspiration": {},
                },
            }
        ]
        mastery = [
            {
                "task_id": 1,
                "question": "A seed problem.",
                "target_difficulty": "Equal",
            }
        ]
        fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "generated.jsonl"
            raw_output = Path(temp_dir) / "generated.raw.jsonl"
            failed_output = Path(temp_dir) / "generated.failed.jsonl"
            with patch.dict(sys.modules, {"openai": fake_openai}):
                generated, _, failed = generate_post_mastery_questions(
                    plans,
                    mastery,
                    max_retries=1,
                    concurrency=1,
                    round_retry_delay=0,
                    output_path=output,
                    raw_output_path=raw_output,
                    failed_output_path=failed_output,
                )
            self.assertEqual(FakeCompletions.calls, 2)
            self.assertEqual(len(generated), 1)
            self.assertEqual(failed, [])
            self.assertTrue(
                (Path(temp_dir) / "generated.rounds" / "round_000.failed.jsonl").exists()
            )
            self.assertTrue(
                (Path(temp_dir) / "generated.rounds" / "round_001.success.jsonl").exists()
            )
            self.assertEqual(failed_output.read_text(encoding="utf-8"), "")


class SynthesisPlanTests(unittest.TestCase):
    def test_compact_schema_and_count(self) -> None:
        mastery = [
            {
                "task_id": 1,
                "question": "A seed question",
                "target_count": 4,
                "target_difficulty": "Equal",
                "target_difficulty_bucket": "medium",
            }
        ]
        kb_records = [
            {
                "task_id": 1,
                "knowledge": {
                    "skill_tags": ["addition"],
                    "operation_sequence": ["addition"],
                    "knowledge_signature": "addition::addition",
                    "step_count": 1,
                    "difficulty_bucket": "easy",
                },
                "concepts": {},
            },
            {
                "task_id": 2,
                "surface_template": "<PERSON> has <NUM> books.",
                "scene_template": "<PERSON> has <NUM> books.",
                "scenario_template": "<PERSON> has <NUM> <TERM>.",
                "knowledge": {
                    "skill_tags": ["addition"],
                    "operation_sequence": ["addition"],
                    "knowledge_signature": "addition::addition",
                    "step_count": 1,
                    "difficulty_bucket": "medium",
                },
                "concepts": {
                    "persons": ["Alex"],
                    "focus_terms": ["books"],
                    "units": ["books"],
                },
            },
        ]
        plan = build_post_mastery_plan(mastery, kb_records, [])
        self.assertEqual(len(plan), 4)
        self.assertEqual(
            set(plan[0]),
            {"source_task_id", "plan_id", "knowledge"},
        )
        self.assertEqual(plan[0]["source_task_id"], 1)
        signatures = {
            record["knowledge"]["diversity"]["plan_signature"]
            for record in plan
        }
        domains = {
            record["knowledge"]["diversity"]["primary_scene"]["domain"]
            for record in plan
        }
        self.assertEqual(len(signatures), 4)
        self.assertEqual(len(domains), 4)
        self.assertNotIn("question_template", plan[0]["knowledge"])

    def test_only_similarity_failure_replans(self) -> None:
        mastery = [
            {
                "task_id": 1,
                "question": "A seed question",
                "target_count": 1,
                "target_difficulty": "Equal",
                "target_difficulty_bucket": "medium",
            }
        ]
        kb_records = [
            {
                "task_id": 1,
                "knowledge": {
                    "skill_tags": ["addition"],
                    "operation_sequence": ["addition"],
                    "knowledge_signature": "addition::addition",
                    "step_count": 1,
                    "difficulty_bucket": "easy",
                },
                "concepts": {},
            }
        ]
        plan = build_post_mastery_plan(mastery, kb_records, [])[0]
        same_plan = replan_failed_plan(plan, "request_error", 1)
        new_plan = replan_failed_plan(plan, "similarity", 1)
        self.assertEqual(
            same_plan["knowledge"]["diversity"]["plan_signature"],
            plan["knowledge"]["diversity"]["plan_signature"],
        )
        self.assertNotEqual(
            new_plan["knowledge"]["diversity"]["plan_signature"],
            plan["knowledge"]["diversity"]["plan_signature"],
        )
        self.assertEqual(
            new_plan["knowledge"]["diversity"]["replan_reason"],
            "similarity",
        )

    def test_stubborn_validation_failure_replans(self) -> None:
        plan = {
            "plan_id": "1_0",
            "knowledge": {
                "diversity": {
                    "primary_scene": {"domain": "warehouse"},
                    "alternative_scenes": [],
                    "variation_mode": "same core operations",
                    "narrative_style": "short everyday story",
                    "number_strategy": "use fresh small integers",
                    "plan_signature": "warehouse|original",
                },
                "kb_inspiration": {
                    "scene_keywords": ["boxes"],
                    "possible_roles": ["worker"],
                    "possible_units": ["items"],
                },
            },
        }
        new_plan = replan_failed_plan(plan, "stubborn_validation", 2)
        diversity = new_plan["knowledge"]["diversity"]
        self.assertNotEqual(
            diversity["plan_signature"],
            plan["knowledge"]["diversity"]["plan_signature"],
        )
        self.assertEqual(diversity["replan_reason"], "stubborn_validation")
        self.assertEqual(diversity["replanned_round"], 2)
        self.assertEqual(
            new_plan["knowledge"]["kb_inspiration"]["scene_keywords"],
            [],
        )
        response_plan = replan_failed_plan(
            plan,
            "validation_response_failure",
            3,
        )
        self.assertEqual(
            response_plan["knowledge"]["diversity"]["replan_reason"],
            "validation_response_failure",
        )


if __name__ == "__main__":
    unittest.main()
