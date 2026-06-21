import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline.validation import (
    _needs_tiebreak,
    _repair_prompt,
    decide_validation,
    precheck_candidate,
    summarize_blind_votes,
    validate_generated_questions,
)


class ValidationLogicTests(unittest.TestCase):
    def test_precheck_detects_arithmetic_error(self) -> None:
        report = precheck_candidate(
            {
                "question": "A box has 2 rows of 3 items. How many items are there?",
                "steps": ["2 * 3 = 7"],
                "answer": "7",
            }
        )
        self.assertFalse(report["passed"])
        self.assertIn("arithmetic_error", report["issues"])

    def test_precheck_handles_percentage_chain_without_false_positive(self) -> None:
        report = precheck_candidate(
            {
                "question": "A worker earns a percentage bonus. What is the bonus?",
                "steps": [
                    "Calculate the bonus: 20% of $120 = 0.20 * 120 = $24."
                ],
                "answer": "24",
            }
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["arithmetic_errors"], [])

    def test_precheck_validates_clean_chained_equation(self) -> None:
        report = precheck_candidate(
            {
                "question": "How many blocks are there?",
                "steps": [
                    "Compute the blocks: (400 - 300) / 50 = 100 / 50 = 3 blocks."
                ],
                "answer": "3",
            }
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["arithmetic_errors"][0]["expression"],
            "100 / 50",
        )

    def test_blind_vote_consensus(self) -> None:
        votes = [
            {
                "solvable": True,
                "unique_answer": True,
                "answer": "6",
                "confidence": 0.9,
            },
            {
                "solvable": True,
                "unique_answer": True,
                "answer": "6.0",
                "confidence": 0.8,
            },
        ]
        summary = summarize_blind_votes(votes)
        self.assertTrue(summary["consensus"])
        self.assertEqual(summary["consensus_count"], 2)
        self.assertFalse(_needs_tiebreak(votes))

    def test_disagreement_requests_tiebreak(self) -> None:
        votes = [
            {"solvable": True, "unique_answer": True, "answer": "6"},
            {"solvable": True, "unique_answer": True, "answer": "7"},
        ]
        self.assertTrue(_needs_tiebreak(votes))

    def test_difficulty_mismatch_requires_question_repair(self) -> None:
        decision = decide_validation(
            {"answer": "6"},
            {"passed": True, "issues": []},
            [
                {"solvable": True, "unique_answer": True, "answer": "6", "confidence": 1.0},
                {"solvable": True, "unique_answer": True, "answer": "6", "confidence": 1.0},
            ],
            {
                "question_valid": True,
                "solvable": True,
                "unique_answer": True,
                "answer_correct": True,
                "steps_correct": True,
                "difficulty_match": False,
                "estimated_difficulty": "Equal",
                "error_type": "difficulty_mismatch",
                "repair_action": "repair_question",
            },
            "Hard",
        )
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["repair_action"], "repair_question")

    def test_repair_prompt_includes_audit_details(self) -> None:
        messages = _repair_prompt(
            {
                "question": "How many?",
                "steps": ["2 + 2 = 5"],
                "answer": "5",
            },
            {
                "error_type": "arithmetic_error",
                "reasons": ["incorrect_answer"],
                "blind_summary": {"consensus_answer": "4"},
                "precheck": {"arithmetic_errors": [{"step_index": 0}]},
                "audit": {
                    "short_reason": "The addition is incorrect.",
                    "correct_answer": "4",
                    "first_error_step": 0,
                },
            },
            {"knowledge": {}},
            "repair_solution",
            "Easy",
            {"question": "A seed.", "solution_steps": "2 + 2 = 4"},
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["validation_report"]["correct_answer"], "4")
        self.assertEqual(payload["validation_report"]["first_error_step"], 0)
        self.assertEqual(payload["validation_report"]["blind_consensus_answer"], "4")


class ValidationEndToEndTests(unittest.TestCase):
    def test_wrong_answer_is_repaired_then_revalidated(self) -> None:
        class FakeCompletions:
            async def create(self, **kwargs):
                messages = kwargs["messages"]
                system = messages[0]["content"]
                payload = json.loads(messages[1]["content"])
                if "blind verifier" in system:
                    content = json.dumps(
                        {
                            "solvable": True,
                            "unique_answer": True,
                            "steps": ["2 * 3 = 6"],
                            "answer": "6",
                            "confidence": 0.98,
                        }
                    )
                elif "strict mathematical problem auditor" in system:
                    candidate_answer = payload["candidate"]["answer"]
                    passed = candidate_answer == "6"
                    content = json.dumps(
                        {
                            "question_valid": True,
                            "solvable": True,
                            "unique_answer": True,
                            "answer_correct": passed,
                            "steps_correct": passed,
                            "difficulty_match": True,
                            "estimated_difficulty": "Easy",
                            "estimated_step_count": 1,
                            "first_error_step": -1 if passed else 0,
                            "error_type": "none" if passed else "arithmetic_error",
                            "repair_action": "pass" if passed else "repair_solution",
                            "correct_answer": "6",
                            "short_reason": "correct" if passed else "wrong arithmetic",
                        }
                    )
                else:
                    content = json.dumps(
                        {
                            "question": payload["candidate"]["question"],
                            "steps": ["2 * 3 = 6"],
                            "answer": "6",
                        }
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
                self.chat = SimpleNamespace(completions=FakeCompletions())

            async def close(self):
                return None

        candidates = [
            {
                "source_task_id": 1,
                "plan_id": "1_0",
                "difficulty": "Easy",
                "question": "A box has 2 rows of 3 items. How many items are there?",
                "steps": ["2 * 3 = 5"],
                "answer": "5",
            }
        ]
        plans = [
            {
                "source_task_id": 1,
                "plan_id": "1_0",
                "knowledge": {
                    "math": {"skill_tags": ["multiplication"]},
                    "diversity": {
                        "primary_scene": {"domain": "warehouse"},
                        "alternative_scenes": [],
                    },
                },
            }
        ]
        mastery = [
            {
                "task_id": 1,
                "target_difficulty": "Easy",
            }
        ]
        fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.dict(sys.modules, {"openai": fake_openai}):
                validated, reports, failed = validate_generated_questions(
                    candidates,
                    plans,
                    mastery,
                    concurrency=4,
                    max_rounds=2,
                    round_retry_delay=0,
                    validated_path=root / "validated.jsonl",
                    reports_path=root / "validation_reports.jsonl",
                    failed_path=root / "validation.failed.jsonl",
                    repair_history_path=root / "repair_history.jsonl",
                )
            self.assertEqual(len(validated), 1)
            self.assertEqual(validated[0]["answer"], "6")
            self.assertEqual(failed, [])
            self.assertGreaterEqual(len(reports), 2)
            self.assertTrue((root / "validation.rounds" / "round_000.repairs.jsonl").exists())
            self.assertTrue((root / "validation.rounds" / "round_001.reports.jsonl").exists())

    def test_stubborn_failure_forces_replan_and_full_regeneration(self) -> None:
        class FakeCompletions:
            async def create(self, **kwargs):
                messages = kwargs["messages"]
                system = messages[0]["content"]
                payload = json.loads(messages[1]["content"])
                if "blind verifier" in system:
                    content = json.dumps(
                        {
                            "solvable": True,
                            "unique_answer": True,
                            "steps": ["2 * 3 = 6"],
                            "answer": "6",
                            "confidence": 0.98,
                        }
                    )
                elif "strict mathematical problem auditor" in system:
                    passed = payload["candidate"]["question"].startswith("Fresh")
                    content = json.dumps(
                        {
                            "question_valid": True,
                            "solvable": True,
                            "unique_answer": True,
                            "answer_correct": passed,
                            "steps_correct": passed,
                            "difficulty_match": True,
                            "estimated_difficulty": "Easy",
                            "estimated_step_count": 1,
                            "first_error_step": -1 if passed else 0,
                            "error_type": "none" if passed else "reasoning_error",
                            "repair_action": "pass" if passed else "repair_question",
                            "correct_answer": "6",
                            "short_reason": "correct" if passed else "still invalid",
                        }
                    )
                else:
                    replanned = (
                        payload["generation_plan"]
                        .get("diversity", {})
                        .get("replan_reason")
                        == "stubborn_validation"
                    )
                    content = json.dumps(
                        {
                            "question": (
                                "Fresh problem: A box has 2 rows of 3 items. How many items?"
                                if replanned
                                else "Changed problem: A box has 2 rows of 3 items. How many items?"
                            ),
                            "steps": ["2 * 3 = 6" if replanned else "2 * 3 = 5"],
                            "answer": "6" if replanned else "5",
                        }
                    )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

            async def close(self):
                return None

        candidates = [
            {
                "source_task_id": 1,
                "plan_id": "1_0",
                "difficulty": "Easy",
                "question": "A box has 2 rows of 3 items. How many items?",
                "steps": ["2 * 3 = 5"],
                "answer": "5",
            }
        ]
        plans = [
            {
                "source_task_id": 1,
                "plan_id": "1_0",
                "knowledge": {
                    "math": {"skill_tags": ["multiplication"]},
                    "diversity": {
                        "primary_scene": {"domain": "warehouse"},
                        "alternative_scenes": [],
                        "plan_signature": "warehouse|original",
                    },
                    "kb_inspiration": {},
                },
            }
        ]
        mastery = [{"task_id": 1, "target_difficulty": "Easy"}]
        fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.dict(
                sys.modules,
                {"openai": fake_openai},
            ), patch.dict(
                "os.environ",
                {"QC_REPLAN_AFTER": "2"},
            ):
                validated, _, failed = validate_generated_questions(
                    candidates,
                    plans,
                    mastery,
                    concurrency=4,
                    max_rounds=4,
                    round_retry_delay=0,
                    validated_path=root / "validated.jsonl",
                    reports_path=root / "validation_reports.jsonl",
                    failed_path=root / "validation.failed.jsonl",
                    repair_history_path=root / "repair_history.jsonl",
                )
            history = [
                json.loads(line)
                for line in (root / "repair_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(validated), 1)
            self.assertEqual(failed, [])
            self.assertTrue(history[1]["repair"]["forced_replan"])
            self.assertEqual(
                history[1]["repair"]["replan_reason"],
                "stubborn_validation",
            )

    def test_repeated_invalid_validation_json_forces_replan(self) -> None:
        class FakeCompletions:
            async def create(self, **kwargs):
                messages = kwargs["messages"]
                system = messages[0]["content"]
                payload = json.loads(messages[1]["content"])
                if "blind verifier" in system:
                    if not payload["question"].startswith("Fresh"):
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content="not json")
                                )
                            ]
                        )
                    content = json.dumps(
                        {
                            "solvable": True,
                            "unique_answer": True,
                            "steps": ["2 * 3 = 6"],
                            "answer": "6",
                            "confidence": 0.98,
                        }
                    )
                elif "strict mathematical problem auditor" in system:
                    passed = payload["candidate"]["question"].startswith("Fresh")
                    content = json.dumps(
                        {
                            "question_valid": passed,
                            "solvable": passed,
                            "unique_answer": passed,
                            "answer_correct": passed,
                            "steps_correct": passed,
                            "difficulty_match": passed,
                            "estimated_difficulty": "Easy",
                            "estimated_step_count": 1,
                            "first_error_step": -1,
                            "error_type": "none" if passed else "invalid_format",
                            "repair_action": "pass" if passed else "regenerate_question",
                            "correct_answer": "6",
                            "short_reason": "correct" if passed else "invalid response",
                        }
                    )
                else:
                    content = json.dumps(
                        {
                            "question": "Fresh problem: 2 groups have 3 items each. How many items?",
                            "steps": ["2 * 3 = 6"],
                            "answer": "6",
                        }
                    )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=FakeCompletions())

            async def close(self):
                return None

        candidates = [
            {
                "source_task_id": 1,
                "plan_id": "1_0",
                "difficulty": "Easy",
                "question": "Original problem: 2 groups have 3 items each. How many items?",
                "steps": ["2 * 3 = 6"],
                "answer": "6",
            }
        ]
        plans = [
            {
                "source_task_id": 1,
                "plan_id": "1_0",
                "knowledge": {
                    "diversity": {
                        "primary_scene": {"domain": "warehouse"},
                        "alternative_scenes": [],
                        "plan_signature": "warehouse|original",
                    },
                    "kb_inspiration": {},
                },
            }
        ]
        fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.dict(
                sys.modules,
                {"openai": fake_openai},
            ), patch.dict(
                "os.environ",
                {
                    "QC_REPLAN_AFTER": "99",
                    "QC_REPLAN_AFTER_RETRY_ERRORS": "2",
                },
            ):
                validated, _, failed = validate_generated_questions(
                    candidates,
                    plans,
                    [{"task_id": 1, "target_difficulty": "Easy"}],
                    concurrency=4,
                    max_rounds=4,
                    round_retry_delay=0,
                    validated_path=root / "validated.jsonl",
                    reports_path=root / "validation_reports.jsonl",
                    failed_path=root / "validation.failed.jsonl",
                    repair_history_path=root / "repair_history.jsonl",
                )
            history = [
                json.loads(line)
                for line in (root / "repair_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(validated), 1)
            self.assertEqual(failed, [])
            self.assertEqual(len(history), 1)
            self.assertTrue(
                history[0]["repair"]["forced_retry_replan"]
            )
            self.assertEqual(
                history[0]["repair"]["replan_reason"],
                "validation_response_failure",
            )


if __name__ == "__main__":
    unittest.main()
