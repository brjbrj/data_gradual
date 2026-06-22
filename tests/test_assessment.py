import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline.assessment import (
    _parse_step_evaluation_response,
    evaluate_answers,
)


def _answer(task_id: str, attempt_index: int = 0) -> dict:
    return {
        "task_id": task_id,
        "source_task_id": 1,
        "attempt_index": attempt_index,
        "question": "A box has 2 rows with 3 items in each row. How many items?",
        "reference_answer": "6",
        "steps": ["2 * 3 = 6"],
        "final_answer": "6",
        "extracted_answer": "6",
        "is_correct": True,
    }


def _score_response() -> str:
    return json.dumps(
        {
            "step_count": 1,
            "correctness": [1.0],
            "logical": [1.0],
            "standardization": [1.0],
            "completeness": [1.0],
            "efficiency": [1.0],
            "final_answer_correct": True,
            "overall_reason": "correct",
        }
    )


class AssessmentOptimizationTests(unittest.TestCase):
    def test_identical_answers_are_scored_once(self) -> None:
        class FakeCompletions:
            calls = []

            async def create(self, **kwargs):
                self.__class__.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=_score_response()
                            )
                        )
                    ]
                )

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=FakeCompletions()
                )

            async def close(self):
                return None

        fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "scores.partial"
            with patch.dict(
                sys.modules,
                {"openai": fake_openai},
            ), patch.dict(
                "os.environ",
                {
                    "SCORE_DEDUPLICATE": "1",
                    "SCORE_ENABLE_THINKING": "0",
                    "SCORE_MAX_RETRIES": "0",
                },
            ):
                reports = evaluate_answers(
                    [_answer("1_0", 0), _answer("1_1", 1)],
                    max_concurrency=8,
                    checkpoint_path=checkpoint,
                )

        self.assertEqual(len(FakeCompletions.calls), 1)
        self.assertEqual(len(reports), 2)
        self.assertEqual(
            {report["task_id"] for report in reports},
            {"1_0", "1_1"},
        )
        request = FakeCompletions.calls[0]
        self.assertEqual(
            request["extra_body"]["chat_template_kwargs"][
                "enable_thinking"
            ],
            False,
        )
        self.assertLess(request["max_tokens"], 900)

    def test_failed_json_is_retried_without_fallback(self) -> None:
        class FakeCompletions:
            calls = 0

            async def create(self, **kwargs):
                self.__class__.calls += 1
                content = (
                    "not json"
                    if self.__class__.calls == 1
                    else _score_response()
                )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content)
                        )
                    ]
                )

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=FakeCompletions()
                )

            async def close(self):
                return None

        fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        with patch.dict(
            sys.modules,
            {"openai": fake_openai},
        ), patch.dict(
            "os.environ",
            {
                "SCORE_MAX_RETRIES": "1",
                "SCORE_RETRY_DELAY": "0",
            },
        ):
            reports = evaluate_answers([_answer("1_0")])

        self.assertEqual(FakeCompletions.calls, 2)
        self.assertEqual(reports[0]["overall_reason"], "correct")

    def test_checkpoint_resumes_and_expands_duplicate(self) -> None:
        first = _answer("1_0", 0)
        second = _answer("1_1", 1)
        existing = _parse_step_evaluation_response(
            first,
            _score_response(),
        )

        class FailIfCalledCompletions:
            async def create(self, **kwargs):
                raise AssertionError("checkpoint should avoid API calls")

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(
                    completions=FailIfCalledCompletions()
                )

            async def close(self):
                return None

        fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "scores.partial"
            checkpoint.write_text(
                json.dumps(existing) + "\n",
                encoding="utf-8",
            )
            with patch.dict(
                sys.modules,
                {"openai": fake_openai},
            ), patch.dict(
                "os.environ",
                {
                    "SCORE_RESUME": "1",
                    "SCORE_DEDUPLICATE": "1",
                },
            ):
                reports = evaluate_answers(
                    [first, second],
                    checkpoint_path=checkpoint,
                )

        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[1]["task_id"], "1_1")
        self.assertEqual(reports[1]["step_score_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
