from __future__ import annotations

"""Mathematical validation, repair, and backtracking for generated questions.

This stage checks whether generated questions are solvable, have a unique
numeric answer, and have mathematically correct candidate steps. It may repair
solutions/questions or trigger regeneration/replan. Training-style step
polishing is intentionally handled later by ``kb_pipeline.step_refine``.
"""

import argparse
import ast
import asyncio
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .post_mastery_generate import (
    _normalize_answer,
    _normalize_steps,
    _parse_bool_env,
    _parse_float_env,
    _parse_generated_output,
    _parse_int_env,
    _format_seconds,
    _decode_json_candidate,
    _unwrap_payload,
)
from .post_mastery_plan import replan_failed_plan
from .utils import normalize_whitespace, read_jsonl, write_json, write_jsonl


DIFFICULTY_STEP_RANGES = {
    "Easy": (1, 2),
    "Slightly Easy": (2, 3),
    "Equal": (2, 4),
    "Slightly Hard": (4, 6),
    "Hard": (6, 10),
}

ALLOWED_DIFFICULTIES = set(DIFFICULTY_STEP_RANGES)
DIFFICULTY_RANK = {
    "Easy": 0,
    "Slightly Easy": 1,
    "Equal": 2,
    "Slightly Hard": 3,
    "Hard": 4,
}
NUMERIC_TOKEN_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:/\d+(?:\.\d+)?)?"
)
ARITHMETIC_SUFFIX_RE = re.compile(r"(?P<expression>[.\d(][\d\s.+\-*/()]*)$")
CLAIMED_RESULT_RE = re.compile(
    r"^\s*(?P<result>[-+]?(?:\d+(?:\.\d+)?|\.\d+))"
)
CALCULATE_STEP_RE = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]\s*)?calculate\b", re.IGNORECASE)
TRAINING_UNFRIENDLY_SCENE_RE = re.compile(
    r"\b(?:"
    r"computer\s+lab|gigabytes?|software|database|server|storage|"
    r"solar\s+panels?|kilowatts?|reservoir|water\s+distribution|"
    r"warehouse|pallets?|cartons?|logistics|delivery\s+center|"
    r"construction\s+site|bricks?|laboratory|science\s+lab|"
    r"airport|regional\s+airport|recycling\s+center|technician|engineer"
    r")\b",
    re.IGNORECASE,
)
OVERUSED_FINAL_ANSWERS = {"0", "10", "20", "30", "40", "50", "60", "100", "120"}
TRAINING_STYLE_ISSUES = {
    "question_too_long_for_training",
    "solution_too_long_for_training",
    "too_many_steps_for_training",
    "template_calculate_steps",
    "training_unfriendly_scene",
}


def _json_message(system: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build a compact two-message JSON-oriented chat request."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _blind_solve_prompt(question: str, vote_index: int) -> List[Dict[str, str]]:
    """Prompt a verifier to solve the question without seeing candidate output."""
    return _json_message(
        (
            "You independently solve one math problem. You are a blind verifier: "
            "you do not know any candidate answer. Return only valid JSON."
        ),
        {
            "task": "Independently solve the question.",
            "vote_index": vote_index,
            "question": question,
            "rules": [
                "Judge whether the question is mathematically solvable as written.",
                "Judge whether it has exactly one numeric answer.",
                "Use concise necessary steps only.",
                "The answer must be a numeric string without units, commas, currency symbols, or extra words.",
                "Do not infer missing conditions.",
            ],
            "output_schema": {
                "solvable": "boolean",
                "unique_answer": "boolean",
                "steps": ["string"],
                "answer": "numeric string",
                "confidence": "number from 0 to 1",
            },
        },
    )


def _audit_prompt(
    candidate: Dict[str, Any],
    blind_summary: Dict[str, Any],
    target_difficulty: str,
    seed_reference: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Prompt the auditor to compare candidate output against blind consensus."""
    low, high = DIFFICULTY_STEP_RANGES.get(target_difficulty, (2, 4))
    return _json_message(
        (
            "You are a strict mathematical problem auditor. Check correctness, "
            "solvability, uniqueness, reasoning steps, and target difficulty. "
            "Return only valid JSON."
        ),
        {
            "task": "Audit the generated problem and proposed solution.",
            "target_difficulty": target_difficulty,
            "expected_reasoning_steps": [low, high],
            "seed_reference_for_relative_difficulty_only": {
                "question": seed_reference.get("question", ""),
                "solution_steps": seed_reference.get("solution_steps", ""),
            },
            "candidate": {
                "question": candidate.get("question", ""),
                "steps": candidate.get("steps", []),
                "answer": candidate.get("answer", ""),
            },
            "independent_blind_solution_summary": blind_summary,
            "difficulty_rules": {
                "Easy": "Clearly easier than the seed; about 1-2 necessary reasoning steps.",
                "Slightly Easy": "Slightly easier than the seed; about 2-3 necessary reasoning steps.",
                "Equal": "Comparable reasoning depth; usually 2-4 necessary reasoning steps.",
                "Slightly Hard": "One meaningful added dependency; usually 4-6 necessary reasoning steps.",
                "Hard": "Several dependent constraints; usually 6-10 necessary reasoning steps.",
            },
            "rules": [
                "Do not accept an answer merely because it matches the candidate answer.",
                "Check each candidate step for arithmetic and logical correctness.",
                "A question with missing information, multiple valid answers, or contradictory conditions fails.",
                "Difficulty must be judged from necessary reasoning, not from verbose wording.",
                "Use repair_solution when the question is valid and should remain unchanged.",
                "Use repair_question for a localized ambiguity, missing condition, or difficulty mismatch.",
                "Use regenerate_question when the mathematical structure is seriously invalid or cannot be safely repaired.",
            ],
            "output_schema": {
                "question_valid": "boolean",
                "solvable": "boolean",
                "unique_answer": "boolean",
                "answer_correct": "boolean",
                "steps_correct": "boolean",
                "difficulty_match": "boolean",
                "estimated_difficulty": "Easy|Slightly Easy|Equal|Slightly Hard|Hard",
                "estimated_step_count": "integer",
                "first_error_step": "integer, -1 if none",
                "error_type": "none|arithmetic_error|reasoning_error|answer_mismatch|missing_condition|ambiguous_question|unsolvable|multiple_answers|difficulty_mismatch|invalid_format",
                "repair_action": "pass|repair_solution|repair_question|regenerate_question",
                "correct_answer": "numeric string or empty",
                "short_reason": "short string",
            },
        },
    )


def _repair_prompt(
    candidate: Dict[str, Any],
    report: Dict[str, Any],
    plan: Dict[str, Any],
    action: str,
    target_difficulty: str,
    seed_reference: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Prompt repair/regeneration according to the decision action."""
    audit = report.get("audit") or {}
    if action == "repair_solution":
        task = (
            "Keep the question text exactly unchanged. Re-solve it independently "
            "and replace only the steps and answer."
        )
    elif action == "repair_question":
        task = (
            "Make the smallest necessary change to the question to fix ambiguity, "
            "missing conditions, uniqueness, correctness, or difficulty. Then solve it."
        )
    else:
        task = (
            "Regenerate the entire problem from the plan. Use a fresh scene and wording, "
            "preserve the intended mathematical skill and target difficulty, and solve it."
        )

    return _json_message(
        (
            "You repair one generated math problem. Return only one valid JSON "
            "object with question, steps, and answer."
        ),
        {
            "task": task,
            "target_difficulty": target_difficulty,
            "seed_reference_for_relative_difficulty_only": {
                "question": seed_reference.get("question", ""),
                "solution_steps": seed_reference.get("solution_steps", ""),
            },
            "candidate": candidate,
            "validation_report": {
                "error_type": report.get("error_type"),
                "reasons": report.get("reasons", []),
                "short_reason": audit.get("short_reason", ""),
                "correct_answer": audit.get("correct_answer", ""),
                "first_error_step": audit.get("first_error_step", -1),
                "blind_consensus_answer": (
                    report.get("blind_summary", {}).get("consensus_answer", "")
                ),
                "precheck_arithmetic_errors": (
                    report.get("precheck", {}).get("arithmetic_errors", [])
                ),
            },
            "generation_plan": plan.get("knowledge", {}),
            "rules": [
                "The result must have exactly one numeric answer.",
                "Every step must be concise, necessary, and mathematically correct.",
                "Every step should explain the reasoning goal and what information supports it, not merely name a calculation.",
                "Start every step with an ordinal label such as 'Step 1:', 'Step 2:', and so on.",
                "Use connective wording such as First, Next, Then, After that, So, Therefore, or Finally to make dependencies explicit.",
                "A step may use a problem condition, one previous result, or several independently computed quantities; state the correct dependency without pretending every step only follows from the immediately previous one.",
                "Each equation should be accompanied by a short explanation of why it is relevant and what intermediate value it obtains.",
                "The repaired solution should read like a direct GSM8K-style dependency-aware reasoning chain, not like a calculator checklist.",
                "Do not add trial-and-error, alternative approaches, or meta reasoning; only keep the direct correct solution path.",
                "Prefer one main inference or equation per step; split packed semicolon calculations into separate steps.",
                "The answer must be a numeric string without units, commas, or symbols.",
                "Match the target difficulty through necessary reasoning depth.",
                "If the previous issue was a style warning such as template_calculate_steps, keep the math simple and rewrite steps with varied natural wording instead of adding complexity.",
                "Prefer a compact GSM8K-style everyday setting unless the current question must be kept unchanged.",
                "Do not mention validation, repair, audits, or previous mistakes.",
                "Return exactly one JSON object and no markdown.",
            ],
            "output_schema": {
                "question": "string",
                "steps": ["string"],
                "answer": "numeric string",
            },
        },
    )


def _safe_arithmetic_eval(expression: str) -> Optional[float]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported expression")

    try:
        return evaluate(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _simple_equations(step: str) -> List[Tuple[str, float]]:
    """Extract only unambiguous arithmetic equations from one reasoning step.

    This is deliberately conservative. Natural-language expressions such as
    ``20% of $120 = 0.20 * 120 = $24`` must not be misread as ``120 = 0.20``.
    The model-based audit handles equations whose units or prose make them
    unsafe to parse programmatically.
    """
    normalized = step.replace(",", "").replace("$", "")
    parts = normalized.split("=")
    equations: List[Tuple[str, float]] = []
    for index in range(len(parts) - 1):
        left = parts[index].strip()
        if index == 0:
            match = ARITHMETIC_SUFFIX_RE.search(left)
            if match is None:
                continue
            expression = match.group("expression").strip()
        else:
            expression = left

        if (
            not expression
            or re.fullmatch(r"[\d\s.+\-*/()]+", expression) is None
            or not re.search(r"\d\s*[+\-*/]\s*[\d(]", expression)
        ):
            continue

        result_match = CLAIMED_RESULT_RE.match(parts[index + 1])
        if result_match is None:
            continue
        remainder = parts[index + 1][result_match.end():].lstrip()
        if remainder.startswith(("+", "-", "*", "/")):
            continue
        equations.append(
            (expression, float(result_match.group("result")))
        )
    return equations


def precheck_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Run deterministic checks before spending model calls on validation.

    Hard structural/math issues become ``issues``. Training friendliness items
    default to ``warnings`` so validation remains focused on correctness unless
    the environment explicitly requests hard style failures.
    """
    question = normalize_whitespace(candidate.get("question", ""))
    steps = _normalize_steps(candidate.get("steps", []))
    answer = _normalize_answer(candidate.get("answer", ""))
    issues: List[str] = []
    warnings: List[str] = []
    arithmetic_errors: List[Dict[str, Any]] = []

    if not question:
        issues.append("missing_question")
    if not steps:
        issues.append("missing_steps")
    if not answer:
        issues.append("invalid_answer")
    if question.count("?") > 1:
        issues.append("multiple_subquestions")
    if re.search(
        r"\b(?:and|also)\s+(?:how many|how much|what is|what are)\b",
        question,
        flags=re.IGNORECASE,
    ):
        issues.append("multiple_subquestions")

    max_question_chars = _parse_int_env("QC_MAX_QUESTION_CHARS", 700)
    max_solution_chars = _parse_int_env("QC_MAX_SOLUTION_CHARS", 900)
    max_step_count = _parse_int_env("QC_MAX_STEP_COUNT", 10)
    calculate_max_steps = _parse_int_env("QC_TEMPLATE_CALCULATE_MAX_STEPS", 1)
    block_unfriendly_scene = _parse_bool_env("QC_BLOCK_TRAINING_UNFRIENDLY_SCENES", True)
    warn_overused_answer = _parse_bool_env("QC_WARN_OVERUSED_FINAL_ANSWERS", True)
    style_hard_fail = _parse_bool_env("QC_TRAINING_STYLE_HARD_FAIL", False)
    severe_question_chars = _parse_int_env("QC_SEVERE_MAX_QUESTION_CHARS", 1200)
    severe_solution_chars = _parse_int_env("QC_SEVERE_MAX_SOLUTION_CHARS", 1800)
    severe_step_count = _parse_int_env("QC_SEVERE_MAX_STEP_COUNT", 16)

    def add_style_issue(issue: str) -> None:
        if style_hard_fail:
            issues.append(issue)
        else:
            warnings.append(issue)

    if max_question_chars > 0 and len(question) > max_question_chars:
        add_style_issue("question_too_long_for_training")
    solution_text = normalize_whitespace(" ".join(steps))
    if max_solution_chars > 0 and len(solution_text) > max_solution_chars:
        add_style_issue("solution_too_long_for_training")
    if max_step_count > 0 and len(steps) > max_step_count:
        add_style_issue("too_many_steps_for_training")
    calculate_starts = sum(1 for step in steps if CALCULATE_STEP_RE.search(step))
    if (
        calculate_max_steps >= 0
        and calculate_starts > calculate_max_steps
        and calculate_starts / max(1, len(steps)) >= 0.4
    ):
        add_style_issue("template_calculate_steps")
    if block_unfriendly_scene and TRAINING_UNFRIENDLY_SCENE_RE.search(question):
        add_style_issue("training_unfriendly_scene")
    if warn_overused_answer and answer in OVERUSED_FINAL_ANSWERS:
        warnings.append("overused_final_answer")
    if severe_question_chars > 0 and len(question) > severe_question_chars:
        issues.append("severely_long_question")
    if severe_solution_chars > 0 and len(solution_text) > severe_solution_chars:
        issues.append("severely_long_solution")
    if severe_step_count > 0 and len(steps) > severe_step_count:
        issues.append("severely_many_steps")

    normalized_step_keys = [
        re.sub(r"\s+", "", step.lower())
        for step in steps
    ]
    if len(normalized_step_keys) != len(set(normalized_step_keys)):
        issues.append("duplicate_steps")

    for step_index, step in enumerate(steps):
        for expression, claimed in _simple_equations(step):
            calculated = _safe_arithmetic_eval(expression)
            if calculated is None:
                continue
            if not math.isclose(calculated, claimed, rel_tol=1e-9, abs_tol=1e-9):
                arithmetic_errors.append(
                    {
                        "step_index": step_index,
                        "expression": expression,
                        "claimed": claimed,
                        "calculated": calculated,
                    }
                )
    if arithmetic_errors:
        issues.append("arithmetic_error")

    if steps and answer:
        final_numbers = NUMERIC_TOKEN_RE.findall(steps[-1].replace(",", ""))
        if final_numbers:
            final_step_answer = _normalize_answer(final_numbers[-1])
            if final_step_answer and not _answers_equal(final_step_answer, answer):
                warnings.append("last_step_answer_mismatch")

    return {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "arithmetic_errors": arithmetic_errors,
        "normalized": {
            "question": question,
            "steps": steps,
            "answer": answer,
        },
    }


def _numeric_value(value: Any) -> Optional[float]:
    text = _normalize_answer(value)
    if not text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def _answers_equal(left: Any, right: Any) -> bool:
    left_value = _numeric_value(left)
    right_value = _numeric_value(right)
    if left_value is None or right_value is None:
        return normalize_whitespace(left) == normalize_whitespace(right)
    return math.isclose(left_value, right_value, rel_tol=1e-7, abs_tol=1e-7)


def _parse_blind_vote(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    payload = _unwrap_payload(_decode_json_candidate(raw))
    if not payload:
        return None, "blind response is not valid JSON"
    answer = _normalize_answer(payload.get("answer"))
    solvable = _coerce_bool(payload.get("solvable", False))
    unique_answer = _coerce_bool(payload.get("unique_answer", False))
    steps = _normalize_steps(payload.get("steps", []))
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0
    if solvable and unique_answer and not answer:
        return None, "blind response missing numeric answer"
    return {
        "solvable": solvable,
        "unique_answer": unique_answer,
        "steps": steps,
        "answer": answer,
        "confidence": confidence,
        "raw_model_output": raw,
    }, ""


def _parse_audit(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    payload = _unwrap_payload(_decode_json_candidate(raw))
    if not payload:
        return None, "audit response is not valid JSON"

    estimated_difficulty = str(payload.get("estimated_difficulty") or "")
    if estimated_difficulty not in ALLOWED_DIFFICULTIES:
        estimated_difficulty = "Equal"
    action = str(payload.get("repair_action") or "regenerate_question")
    if action not in {
        "pass",
        "repair_solution",
        "repair_question",
        "regenerate_question",
    }:
        action = "regenerate_question"
    try:
        estimated_step_count = int(payload.get("estimated_step_count") or 0)
    except Exception:
        estimated_step_count = 0
    try:
        first_error_step = int(payload.get("first_error_step", -1))
    except Exception:
        first_error_step = -1
    return {
        "question_valid": _coerce_bool(payload.get("question_valid", False)),
        "solvable": _coerce_bool(payload.get("solvable", False)),
        "unique_answer": _coerce_bool(payload.get("unique_answer", False)),
        "answer_correct": _coerce_bool(payload.get("answer_correct", False)),
        "steps_correct": _coerce_bool(payload.get("steps_correct", False)),
        "difficulty_match": _coerce_bool(payload.get("difficulty_match", False)),
        "estimated_difficulty": estimated_difficulty,
        "estimated_step_count": estimated_step_count,
        "first_error_step": first_error_step,
        "error_type": str(payload.get("error_type") or "invalid_format"),
        "repair_action": action,
        "correct_answer": _normalize_answer(payload.get("correct_answer")),
        "short_reason": normalize_whitespace(payload.get("short_reason", "")),
        "raw_model_output": raw,
    }, ""


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value or "").strip().lower()
    return normalized in {"true", "1", "yes", "y", "pass"}


def _answer_groups(votes: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    for vote in votes:
        if not vote.get("solvable") or not vote.get("unique_answer") or not vote.get("answer"):
            continue
        matching = next(
            (
                group
                for group in groups
                if _answers_equal(group[0].get("answer"), vote.get("answer"))
            ),
            None,
        )
        if matching is None:
            groups.append([vote])
        else:
            matching.append(vote)
    return sorted(groups, key=len, reverse=True)


def summarize_blind_votes(votes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups = _answer_groups(votes)
    consensus_group = groups[0] if groups and len(groups[0]) >= 2 else []
    consensus_answer = consensus_group[0]["answer"] if consensus_group else ""
    representative_vote = (
        max(
            consensus_group,
            key=lambda vote: float(vote.get("confidence", 0.0)),
        )
        if consensus_group
        else None
    )
    solvable_votes = sum(1 for vote in votes if vote.get("solvable"))
    unique_votes = sum(1 for vote in votes if vote.get("unique_answer"))
    return {
        "vote_count": len(votes),
        "solvable_votes": solvable_votes,
        "unique_answer_votes": unique_votes,
        "consensus": bool(consensus_group),
        "consensus_count": len(consensus_group),
        "consensus_answer": consensus_answer,
        "representative_steps": (
            representative_vote.get("steps", [])
            if representative_vote
            else []
        ),
        "answers": [vote.get("answer", "") for vote in votes],
        "average_confidence": round(
            sum(float(vote.get("confidence", 0.0)) for vote in votes)
            / max(1, len(votes)),
            4,
        ),
    }


def _needs_tiebreak(votes: Sequence[Dict[str, Any]]) -> bool:
    if len(votes) < 2:
        return True
    summary = summarize_blind_votes(votes)
    return not summary["consensus"]


def decide_validation(
    candidate: Dict[str, Any],
    precheck: Dict[str, Any],
    votes: Sequence[Dict[str, Any]],
    audit: Optional[Dict[str, Any]],
    target_difficulty: str,
) -> Dict[str, Any]:
    """Combine precheck, blind votes, and audit into one repair decision."""
    blind = summarize_blind_votes(votes)
    candidate_answer = _normalize_answer(candidate.get("answer", ""))
    reasons: List[str] = []
    warnings: List[str] = list(precheck.get("warnings", []))
    difficulty_tolerance = max(0, _parse_int_env("QC_DIFFICULTY_TOLERANCE", 1))
    require_exact_difficulty = _parse_bool_env("QC_REQUIRE_EXACT_DIFFICULTY", False)

    if not precheck.get("passed"):
        hard_issues = [
            issue
            for issue in precheck.get("issues", [])
            if issue not in TRAINING_STYLE_ISSUES
        ]
        warnings.extend(
            issue
            for issue in precheck.get("issues", [])
            if issue in TRAINING_STYLE_ISSUES
        )
        reasons.extend(hard_issues)
    if not blind["consensus"]:
        reasons.append("no_blind_consensus")
    elif not _answers_equal(blind["consensus_answer"], candidate_answer):
        reasons.append("answer_mismatch")
    if blind["solvable_votes"] < 2:
        reasons.append("unsolvable")
    if blind["unique_answer_votes"] < 2:
        reasons.append("non_unique_answer")
    if audit is None:
        reasons.append("missing_audit")
    else:
        for field, issue in (
            ("question_valid", "invalid_question"),
            ("solvable", "unsolvable"),
            ("unique_answer", "non_unique_answer"),
            ("answer_correct", "incorrect_answer"),
            ("steps_correct", "invalid_steps"),
        ):
            if not audit.get(field):
                reasons.append(issue)
        estimated_difficulty = str(audit.get("estimated_difficulty") or "")
        difficulty_gap = abs(
            DIFFICULTY_RANK.get(estimated_difficulty, DIFFICULTY_RANK.get(target_difficulty, 2))
            - DIFFICULTY_RANK.get(target_difficulty, 2)
        )
        if not audit.get("difficulty_match"):
            if require_exact_difficulty or difficulty_gap > difficulty_tolerance:
                reasons.append("difficulty_mismatch")
            else:
                warnings.append("soft_difficulty_mismatch")
        if (
            estimated_difficulty in ALLOWED_DIFFICULTIES
            and estimated_difficulty != target_difficulty
        ):
            if require_exact_difficulty or difficulty_gap > difficulty_tolerance:
                reasons.append("difficulty_mismatch")
            else:
                warnings.append("estimated_adjacent_difficulty")

    reasons = list(dict.fromkeys(reasons))
    warnings = list(dict.fromkeys(warnings))
    passed = not reasons
    if passed:
        action = "pass"
        error_type = "none"
    elif any(
        reason in reasons
        for reason in ("unsolvable", "non_unique_answer", "invalid_question", "no_blind_consensus")
    ):
        action = (
            audit.get("repair_action", "regenerate_question")
            if audit
            else "regenerate_question"
        )
        if action == "repair_solution":
            action = "repair_question"
        error_type = audit.get("error_type", "unsolvable") if audit else "unsolvable"
    elif "difficulty_mismatch" in reasons:
        action = "repair_question"
        error_type = "difficulty_mismatch"
    else:
        action = "repair_solution"
        if "arithmetic_error" in reasons:
            error_type = "arithmetic_error"
        elif any(
            reason in reasons
            for reason in (
                "missing_question",
                "missing_steps",
                "invalid_answer",
                "multiple_subquestions",
                "duplicate_steps",
            )
        ):
            error_type = "invalid_format"
        else:
            audit_error = (
                str(audit.get("error_type") or "")
                if audit
                else ""
            )
            error_type = (
                audit_error
                if audit_error and audit_error != "none"
                else "answer_mismatch"
            )

    return {
        "passed": passed,
        "repair_action": action,
        "error_type": error_type,
        "reasons": reasons,
        "warnings": warnings,
        "target_difficulty": target_difficulty,
        "blind_summary": blind,
    }


def _candidate_fingerprint(candidate: Dict[str, Any]) -> str:
    question = re.sub(r"\s+", " ", str(candidate.get("question", "")).lower()).strip()
    answer = _normalize_answer(candidate.get("answer", ""))
    return f"{question}::{answer}"


def _project_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_task_id": candidate.get("source_task_id"),
        "plan_id": candidate.get("plan_id"),
        "difficulty": candidate.get("difficulty"),
        "question": normalize_whitespace(candidate.get("question", "")),
        "steps": _normalize_steps(candidate.get("steps", [])),
        "answer": _normalize_answer(candidate.get("answer", "")),
    }


def _model_aliases(model: str) -> set[str]:
    normalized = str(model).strip().rstrip("/")
    if not normalized:
        return set()
    aliases = {normalized}
    basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    if basename:
        aliases.add(basename)
    return aliases


async def _resolve_served_model_name(client: Any, model: str) -> str:
    try:
        served = await client.models.list()
    except Exception:
        return model

    expected = _model_aliases(model)
    for item in getattr(served, "data", []):
        served_id = getattr(item, "id", None)
        if served_id and _model_aliases(str(served_id)) & expected:
            return str(served_id)
    return model


async def _run_validation_async(
    candidates: Sequence[Dict[str, Any]],
    plans: Sequence[Dict[str, Any]],
    mastery_records: Sequence[Dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    concurrency: int,
    timeout: int,
    max_rounds: int,
    blind_votes: int,
    tiebreak_votes: int,
    max_tokens: int,
    enable_thinking: bool,
    force_json: bool,
    round_retry_delay: float,
    validated_path: Optional[Path],
    reports_path: Optional[Path],
    failed_path: Optional[Path],
    repair_history_path: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate, repair, regenerate, and replan candidates until settled.

    The loop writes checkpoints after each round so interrupted validation can
    preserve accepted records and make failures inspectable. If repeated repair
    attempts stall, the active plan is refreshed before generating again.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("The brj environment requires the openai package.") from exc

    client = AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    model = await _resolve_served_model_name(client, model)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    plan_lookup = {str(item.get("plan_id")): item for item in plans}
    mastery_lookup = {str(item.get("task_id")): item for item in mastery_records}
    original_order = {
        str(item.get("plan_id")): index
        for index, item in enumerate(candidates)
    }
    accepted: Dict[str, Dict[str, Any]] = {}
    reports: List[Dict[str, Any]] = []
    repair_history: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = [
        {
            "candidate": _project_candidate(item),
            "history": [],
            "repeated_count": 0,
            "quality_failures": 0,
            "retry_failures": 0,
            "replan_count": 0,
            "active_plan": plan_lookup.get(str(item.get("plan_id")), {}),
        }
        for item in candidates
    ]
    final_failed: List[Dict[str, Any]] = []
    round_index = 0
    infinite_rounds = max_rounds < 0
    progress_interval = max(
        1.0,
        _parse_float_env("QC_PROGRESS_INTERVAL", 10.0),
    )
    progress_every = max(
        1,
        _parse_int_env("QC_PROGRESS_EVERY", 10),
    )
    replan_after = _parse_int_env("QC_REPLAN_AFTER", 3)
    retry_replan_after = _parse_int_env(
        "QC_REPLAN_AFTER_RETRY_ERRORS",
        3,
    )
    print(
        f"[validate] config model={model} concurrency={concurrency} "
        f"blind_votes={blind_votes} tiebreak_votes={tiebreak_votes} "
        f"max_rounds={'infinite' if infinite_rounds else max_rounds} "
        f"replan_after={'disabled' if replan_after < 0 else replan_after} "
        f"retry_replan_after="
        f"{'disabled' if retry_replan_after < 0 else retry_replan_after} "
        f"timeout={timeout}s progress_every={progress_every} "
        f"progress_interval={progress_interval:g}s",
        flush=True,
    )

    async def collect_stage(
        tasks: Sequence[asyncio.Task],
        *,
        stage: str,
        on_result: Any,
    ) -> None:
        total_tasks = len(tasks)
        if total_tasks == 0:
            print(
                f"[validate] round={round_index} stage={stage} skipped requests=0",
                flush=True,
            )
            return

        stage_started = time.time()
        last_log_at = stage_started
        completed = 0
        succeeded = 0
        errors = 0
        pending_tasks = set(tasks)
        print(
            f"[validate] round={round_index} stage={stage} "
            f"start requests={total_tasks} concurrency={concurrency}",
            flush=True,
        )

        while pending_tasks:
            done_tasks, pending_tasks = await asyncio.wait(
                pending_tasks,
                timeout=progress_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
            now = time.time()
            if not done_tasks:
                elapsed = now - stage_started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (
                    _format_seconds((total_tasks - completed) / rate)
                    if rate > 0
                    else "--:--"
                )
                print(
                    f"[validate] round={round_index} stage={stage} "
                    f"heartbeat {completed}/{total_tasks} "
                    f"({completed / total_tasks * 100:5.1f}%) "
                    f"ok={succeeded} error={errors} active={len(pending_tasks)} "
                    f"rate={rate:.2f}/s elapsed={_format_seconds(elapsed)} "
                    f"eta={eta}",
                    flush=True,
                )
                last_log_at = now
                continue

            for task in done_tasks:
                completed += 1
                try:
                    result = task.result()
                    on_result(result)
                    meta = result[2] if len(result) > 2 else {}
                    if result[1] is not None and not meta.get("error"):
                        succeeded += 1
                    else:
                        errors += 1
                except Exception as exc:
                    errors += 1
                    print(
                        f"[validate] round={round_index} stage={stage} "
                        f"task_error={type(exc).__name__}: {exc}",
                        flush=True,
                    )

            should_log = (
                completed <= 5
                or completed % progress_every == 0
                or completed == total_tasks
                or now - last_log_at >= progress_interval
            )
            if should_log:
                elapsed = now - stage_started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (
                    _format_seconds((total_tasks - completed) / rate)
                    if rate > 0
                    else "--:--"
                )
                print(
                    f"[validate] round={round_index} stage={stage} "
                    f"{completed}/{total_tasks} "
                    f"({completed / total_tasks * 100:5.1f}%) "
                    f"ok={succeeded} error={errors} active={len(pending_tasks)} "
                    f"rate={rate:.2f}/s elapsed={_format_seconds(elapsed)} "
                    f"eta={eta}",
                    flush=True,
                )
                last_log_at = now

    async def request_json(
        messages: List[Dict[str, str]],
        *,
        temperature: float,
    ) -> str:
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        }
        if force_json:
            request["response_format"] = {"type": "json_object"}
        if not enable_thinking:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        async with semaphore:
            response = await client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    async def blind_vote(
        item_index: int,
        candidate: Dict[str, Any],
        vote_index: int,
    ) -> Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]]:
        try:
            raw = await request_json(
                _blind_solve_prompt(candidate["question"], vote_index),
                temperature=0.1 + vote_index * 0.05,
            )
            parsed, error = _parse_blind_vote(raw)
            return item_index, parsed, {
                "vote_index": vote_index,
                "error": error,
                "raw_model_output": raw,
            }
        except Exception as exc:
            return item_index, None, {
                "vote_index": vote_index,
                "error": f"{type(exc).__name__}: {exc}",
                "raw_model_output": "",
            }

    async def audit_item(
        item_index: int,
        candidate: Dict[str, Any],
        blind_summary: Dict[str, Any],
        target_difficulty: str,
        seed_reference: Dict[str, Any],
    ) -> Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]]:
        try:
            raw = await request_json(
                _audit_prompt(
                    candidate,
                    blind_summary,
                    target_difficulty,
                    seed_reference,
                ),
                temperature=0.1,
            )
            parsed, error = _parse_audit(raw)
            return item_index, parsed, {
                "error": error,
                "raw_model_output": raw,
            }
        except Exception as exc:
            return item_index, None, {
                "error": f"{type(exc).__name__}: {exc}",
                "raw_model_output": "",
            }

    async def repair_item(
        item_index: int,
        item: Dict[str, Any],
        report: Dict[str, Any],
    ) -> Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]]:
        candidate = item["candidate"]
        plan = (
            item.get("active_plan")
            or plan_lookup.get(str(candidate.get("plan_id")), {})
        )
        seed_reference = mastery_lookup.get(
            str(candidate.get("source_task_id")),
            {},
        )
        action = str(report.get("repair_action") or "regenerate_question")
        repeated_count = int(item.get("repeated_count") or 0)
        quality_failure_count = int(item.get("quality_failures") or 0) + 1
        retry_failure_count = int(item.get("retry_failures") or 0) + 1
        force_stubborn_replan = (
            replan_after >= 0
            and quality_failure_count >= max(1, replan_after)
        )
        force_retry_replan = (
            report.get("repair_action") == "retry_validation"
            and retry_replan_after >= 0
            and retry_failure_count >= max(1, retry_replan_after)
        )
        if (
            repeated_count >= 1
            or force_stubborn_replan
            or force_retry_replan
        ):
            action = "regenerate_question"
        repair_plan = plan
        if action == "regenerate_question":
            if force_retry_replan:
                replan_reason = "validation_response_failure"
            elif force_stubborn_replan:
                replan_reason = "stubborn_validation"
            else:
                replan_reason = "validation_regeneration"
            repair_plan = replan_failed_plan(
                plan,
                replan_reason,
                retry_round=int(item.get("replan_count") or 0) + 1,
            )
        else:
            replan_reason = ""
        try:
            raw = await request_json(
                _repair_prompt(
                    candidate,
                    report,
                    repair_plan,
                    action,
                    str(candidate.get("difficulty") or "Equal"),
                    seed_reference,
                ),
                temperature=0.2 if action == "repair_solution" else 0.5,
            )
            parsed, error = _parse_generated_output(raw)
            if parsed is None:
                return item_index, None, {
                    "action": action,
                    "error": error,
                    "raw_model_output": raw,
                    "repair_plan": repair_plan,
                    "forced_replan": force_stubborn_replan,
                    "forced_retry_replan": force_retry_replan,
                    "quality_failure_count": quality_failure_count,
                    "retry_failure_count": retry_failure_count,
                    "replan_reason": replan_reason,
                }
            repaired = {
                **candidate,
                "question": parsed["question"],
                "steps": parsed["steps"],
                "answer": parsed["answer"],
            }
            if action == "repair_solution":
                repaired["question"] = candidate["question"]
            return item_index, repaired, {
                "action": action,
                "error": "",
                "raw_model_output": raw,
                "repair_plan": repair_plan,
                "forced_replan": force_stubborn_replan,
                "forced_retry_replan": force_retry_replan,
                "quality_failure_count": quality_failure_count,
                "retry_failure_count": retry_failure_count,
                "replan_reason": replan_reason,
            }
        except Exception as exc:
            return item_index, None, {
                "action": action,
                "error": f"{type(exc).__name__}: {exc}",
                "raw_model_output": "",
                "repair_plan": repair_plan,
                "forced_replan": force_stubborn_replan,
                "forced_retry_replan": force_retry_replan,
                "quality_failure_count": quality_failure_count,
                "retry_failure_count": retry_failure_count,
                "replan_reason": replan_reason,
            }

    try:
        while pending and (infinite_rounds or round_index <= max_rounds):
            total = len(pending)
            started_at = time.time()
            print(f"[validate] round={round_index} batch_size={total}", flush=True)
            prechecks = [
                precheck_candidate(item["candidate"])
                for item in pending
            ]
            precheck_failed = sum(
                1 for precheck in prechecks if not precheck.get("passed")
            )
            print(
                f"[validate] round={round_index} stage=precheck complete "
                f"passed={total - precheck_failed} failed={precheck_failed}",
                flush=True,
            )

            votes_by_item: List[List[Dict[str, Any]]] = [[] for _ in pending]
            vote_meta_by_item: List[List[Dict[str, Any]]] = [[] for _ in pending]
            blind_tasks = [
                asyncio.create_task(
                    blind_vote(index, item["candidate"], vote_index)
                )
                for index, item in enumerate(pending)
                if prechecks[index]["normalized"]["question"]
                for vote_index in range(blind_votes)
            ]

            def record_blind_vote(
                result: Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]],
            ) -> None:
                index, vote, meta = result
                if vote is not None:
                    votes_by_item[index].append(vote)
                vote_meta_by_item[index].append(meta)

            await collect_stage(
                blind_tasks,
                stage="blind_solve",
                on_result=record_blind_vote,
            )

            tiebreak_tasks = [
                asyncio.create_task(
                    blind_vote(
                        index,
                        item["candidate"],
                        blind_votes + tie_index,
                    )
                )
                for index, item in enumerate(pending)
                if prechecks[index]["normalized"]["question"]
                and _needs_tiebreak(votes_by_item[index])
                for tie_index in range(tiebreak_votes)
            ]
            await collect_stage(
                tiebreak_tasks,
                stage="tiebreak_solve",
                on_result=record_blind_vote,
            )

            blind_summaries = [
                summarize_blind_votes(votes)
                for votes in votes_by_item
            ]
            audits: List[Optional[Dict[str, Any]]] = [None] * total
            audit_meta: List[Dict[str, Any]] = [{} for _ in pending]
            audit_tasks = [
                asyncio.create_task(
                    audit_item(
                        index,
                        item["candidate"],
                        blind_summaries[index],
                        str(item["candidate"].get("difficulty") or "Equal"),
                        mastery_lookup.get(
                            str(item["candidate"].get("source_task_id")),
                            {},
                        ),
                    )
                )
                for index, item in enumerate(pending)
                if prechecks[index]["normalized"]["question"]
            ]

            def record_audit(
                result: Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]],
            ) -> None:
                index, audit, meta = result
                audits[index] = audit
                audit_meta[index] = meta

            await collect_stage(
                audit_tasks,
                stage="strict_audit",
                on_result=record_audit,
            )

            round_reports: List[Dict[str, Any]] = []
            failed_items: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
            for index, item in enumerate(pending):
                candidate = item["candidate"]
                decision = decide_validation(
                    candidate,
                    prechecks[index],
                    votes_by_item[index],
                    audits[index],
                    str(candidate.get("difficulty") or "Equal"),
                )
                if any(meta.get("error") for meta in vote_meta_by_item[index]) and not votes_by_item[index]:
                    decision.update(
                        {
                            "passed": False,
                            "repair_action": "retry_validation",
                            "error_type": "request_error",
                        }
                    )
                    decision["reasons"] = ["blind_request_error"]
                if audit_meta[index].get("error") and audits[index] is None:
                    decision.update(
                        {
                            "passed": False,
                            "repair_action": "retry_validation",
                            "error_type": "request_error",
                        }
                    )
                    decision["reasons"] = ["audit_request_error"]

                report = {
                    "source_task_id": candidate.get("source_task_id"),
                    "plan_id": candidate.get("plan_id"),
                    "round": round_index,
                    "quality_failures_before": int(
                        item.get("quality_failures") or 0
                    ),
                    "retry_failures_before": int(
                        item.get("retry_failures") or 0
                    ),
                    "replan_count": int(item.get("replan_count") or 0),
                    "candidate": candidate,
                    "precheck": prechecks[index],
                    "blind_votes": votes_by_item[index],
                    "blind_request_meta": vote_meta_by_item[index],
                    "blind_summary": blind_summaries[index],
                    "audit": audits[index],
                    "audit_request_meta": audit_meta[index],
                    **decision,
                }
                round_reports.append(report)
                reports.append(report)
                if decision["passed"]:
                    accepted[str(candidate.get("plan_id"))] = _project_candidate(candidate)
                else:
                    failed_items.append((index, item, report))

            can_retry = infinite_rounds or round_index < max_rounds
            reason_counter = Counter(
                reason
                for report in round_reports
                for reason in report.get("reasons", [])
            )
            warning_counter = Counter(
                warning
                for report in round_reports
                for warning in report.get("warnings", [])
            )
            print(
                f"[validate] round={round_index} stage=decision complete "
                f"passed={total - len(failed_items)} failed={len(failed_items)} "
                f"can_retry={str(can_retry).lower()} "
                f"top_reasons={dict(reason_counter.most_common(8))} "
                f"top_warnings={dict(warning_counter.most_common(8))}",
                flush=True,
            )
            next_pending: List[Dict[str, Any]] = []
            round_repairs: List[Dict[str, Any]] = []
            if can_retry and failed_items:
                def should_repair_retry_error(
                    item: Dict[str, Any],
                    report: Dict[str, Any],
                ) -> bool:
                    return (
                        report.get("repair_action") == "retry_validation"
                        and retry_replan_after >= 0
                        and int(item.get("retry_failures") or 0) + 1
                        >= max(1, retry_replan_after)
                    )

                repair_tasks = [
                    asyncio.create_task(repair_item(index, item, report))
                    for index, item, report in failed_items
                    if (
                        report.get("repair_action") != "retry_validation"
                        or should_repair_retry_error(item, report)
                    )
                ]
                repairs_by_index: Dict[int, Tuple[Optional[Dict[str, Any]], Dict[str, Any]]] = {}

                def record_repair(
                    result: Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]],
                ) -> None:
                    index, repaired, meta = result
                    repairs_by_index[index] = (repaired, meta)

                await collect_stage(
                    repair_tasks,
                    stage="repair",
                    on_result=record_repair,
                )
                repair_actions = Counter(
                    str(meta.get("action") or "unknown")
                    for _, meta in repairs_by_index.values()
                )
                forced_replans = sum(
                    1
                    for _, meta in repairs_by_index.values()
                    if (
                        meta.get("forced_replan")
                        or meta.get("forced_retry_replan")
                    )
                )
                print(
                    f"[validate] round={round_index} stage=repair_summary "
                    f"actions={dict(repair_actions)} "
                    f"forced_replans={forced_replans}",
                    flush=True,
                )

                for index, item, report in failed_items:
                    if (
                        report.get("repair_action") == "retry_validation"
                        and index not in repairs_by_index
                    ):
                        next_pending.append(
                            {
                                **item,
                                "retry_failures": (
                                    int(item.get("retry_failures") or 0) + 1
                                ),
                            }
                        )
                        continue
                    repaired, meta = repairs_by_index.get(index, (None, {"error": "missing repair result"}))
                    history_entry = {
                        "source_task_id": item["candidate"].get("source_task_id"),
                        "plan_id": item["candidate"].get("plan_id"),
                        "round": round_index,
                        "before": item["candidate"],
                        "validation_report": report,
                        "repair": meta,
                        "after": repaired,
                    }
                    round_repairs.append(history_entry)
                    repair_history.append(history_entry)
                    quality_failure_count = int(
                        item.get("quality_failures") or 0
                    ) + 1
                    if repaired is None:
                        next_pending.append(
                            {
                                **item,
                                "quality_failures": quality_failure_count,
                                "retry_failures": (
                                    int(item.get("retry_failures") or 0) + 1
                                    if report.get("repair_action")
                                    == "retry_validation"
                                    else 0
                                ),
                            }
                        )
                        continue
                    did_replan = (
                        str(meta.get("action") or "")
                        == "regenerate_question"
                    )
                    repeated = (
                        _candidate_fingerprint(repaired)
                        == _candidate_fingerprint(item["candidate"])
                    )
                    next_pending.append(
                        {
                            "candidate": _project_candidate(repaired),
                            "history": [*item.get("history", []), history_entry],
                            "quality_failures": (
                                0 if did_replan else quality_failure_count
                            ),
                            "retry_failures": 0,
                            "replan_count": (
                                int(item.get("replan_count") or 0) + 1
                                if did_replan
                                else int(item.get("replan_count") or 0)
                            ),
                            "active_plan": (
                                meta.get("repair_plan")
                                if did_replan
                                else item.get("active_plan")
                            ),
                            "repeated_count": (
                                int(item.get("repeated_count") or 0) + 1
                                if repeated
                                else 0
                            ),
                        }
                    )

            final_failed = [
                {
                    "source_task_id": item["candidate"].get("source_task_id"),
                    "plan_id": item["candidate"].get("plan_id"),
                    "difficulty": item["candidate"].get("difficulty"),
                    "question": item["candidate"].get("question"),
                    "steps": item["candidate"].get("steps"),
                    "answer": item["candidate"].get("answer"),
                    "round": round_index,
                    "active_plan": item.get("active_plan"),
                    "validation_report": report,
                    "next_candidate": next(
                        (
                            pending_item["candidate"]
                            for pending_item in next_pending
                            if str(pending_item["candidate"].get("plan_id"))
                            == str(item["candidate"].get("plan_id"))
                        ),
                        None,
                    ),
                    "next_plan": next(
                        (
                            pending_item.get("active_plan")
                            for pending_item in next_pending
                            if str(pending_item["candidate"].get("plan_id"))
                            == str(item["candidate"].get("plan_id"))
                        ),
                        None,
                    ),
                }
                for _, item, report in failed_items
            ]

            if validated_path is not None:
                round_dir = validated_path.parent / "validation.rounds"
                round_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"round_{round_index:03d}"
                write_jsonl(round_dir / f"{prefix}.reports.jsonl", round_reports)
                write_jsonl(round_dir / f"{prefix}.repairs.jsonl", round_repairs)
                write_jsonl(round_dir / f"{prefix}.failed.jsonl", final_failed)
                write_json(
                    round_dir / f"{prefix}.summary.json",
                    {
                        "round": round_index,
                        "input": total,
                        "passed": sum(1 for report in round_reports if report["passed"]),
                        "failed": len(failed_items),
                        "next_round": len(next_pending),
                        "error_types": dict(
                            Counter(
                                report.get("error_type", "unknown")
                                for report in round_reports
                                if not report.get("passed")
                            )
                        ),
                    },
                )
                ordered_validated = sorted(
                    accepted.values(),
                    key=lambda item: original_order.get(str(item.get("plan_id")), 10**12),
                )
                write_jsonl(validated_path, ordered_validated)
                if reports_path is not None:
                    write_jsonl(reports_path, reports)
                if repair_history_path is not None:
                    write_jsonl(repair_history_path, repair_history)
                if failed_path is not None:
                    write_jsonl(failed_path, final_failed)

            elapsed = time.time() - started_at
            print(
                f"[validate] round={round_index} complete "
                f"passed={sum(1 for report in round_reports if report['passed'])} "
                f"failed={len(failed_items)} next_batch={len(next_pending)} "
                f"elapsed={_format_seconds(elapsed)}",
                flush=True,
            )
            pending = next_pending
            round_index += 1
            if pending and round_retry_delay > 0:
                await asyncio.sleep(round_retry_delay)
    finally:
        await client.close()

    ordered_validated = sorted(
        accepted.values(),
        key=lambda item: original_order.get(str(item.get("plan_id")), 10**12),
    )
    return ordered_validated, reports, final_failed


def validate_generated_questions(
    candidates: Sequence[Dict[str, Any]],
    plans: Sequence[Dict[str, Any]],
    mastery_records: Sequence[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    concurrency: Optional[int] = None,
    timeout: Optional[int] = None,
    max_rounds: Optional[int] = None,
    blind_votes: Optional[int] = None,
    tiebreak_votes: Optional[int] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    force_json: Optional[bool] = None,
    round_retry_delay: Optional[float] = None,
    validated_path: Optional[Path] = None,
    reports_path: Optional[Path] = None,
    failed_path: Optional[Path] = None,
    repair_history_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Synchronous wrapper used by the validation stage script."""
    return asyncio.run(
        _run_validation_async(
            candidates,
            plans,
            mastery_records,
            model=model
            or os.environ.get("QC_MODEL")
            or os.environ.get("GEN_MODEL")
            or "/root/brjverl/models/Qwen3.6-27B",
            base_url=base_url
            or os.environ.get("VLLM_BASE_URL")
            or "http://127.0.0.1:8911/v1",
            api_key=api_key
            or os.environ.get("VLLM_API_KEY")
            or "EMPTY",
            concurrency=max(
                1,
                concurrency
                if concurrency is not None
                else _parse_int_env("QC_CONCURRENCY", 256),
            ),
            timeout=timeout
            if timeout is not None
            else _parse_int_env("VLLM_TIMEOUT", 600),
            max_rounds=max_rounds
            if max_rounds is not None
            else _parse_int_env("QC_MAX_ROUNDS", 3),
            blind_votes=max(
                2,
                blind_votes
                if blind_votes is not None
                else _parse_int_env("QC_BLIND_VOTES", 2),
            ),
            tiebreak_votes=max(
                0,
                tiebreak_votes
                if tiebreak_votes is not None
                else _parse_int_env("QC_TIEBREAK_VOTES", 1),
            ),
            max_tokens=max_tokens
            if max_tokens is not None
            else _parse_int_env("QC_MAX_TOKENS", 900),
            enable_thinking=enable_thinking
            if enable_thinking is not None
            else _parse_bool_env("QC_ENABLE_THINKING", False),
            force_json=force_json
            if force_json is not None
            else _parse_bool_env("QC_FORCE_JSON", False),
            round_retry_delay=round_retry_delay
            if round_retry_delay is not None
            else _parse_float_env("QC_ROUND_RETRY_DELAY", 1.0),
            validated_path=validated_path,
            reports_path=reports_path,
            failed_path=failed_path,
            repair_history_path=repair_history_path,
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for ``run/06_validate_generated.sh``."""
    parser = argparse.ArgumentParser(
        description="Blind-solve, audit, repair, and revalidate generated math problems."
    )
    parser.add_argument("--generated", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mastery", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reports-output", required=False)
    parser.add_argument("--failed-output", required=False)
    parser.add_argument("--repair-history-output", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("--concurrency", type=int, required=False)
    parser.add_argument("--max-rounds", type=int, required=False)
    parser.add_argument("--blind-votes", type=int, required=False)
    parser.add_argument("--tiebreak-votes", type=int, required=False)
    args = parser.parse_args(argv)

    generated_path = Path(args.generated)
    output_path = Path(args.output)
    reports_path = (
        Path(args.reports_output)
        if args.reports_output
        else output_path.with_name("validation_reports.jsonl")
    )
    failed_path = (
        Path(args.failed_output)
        if args.failed_output
        else output_path.with_name("validation.failed.jsonl")
    )
    repair_history_path = (
        Path(args.repair_history_output)
        if args.repair_history_output
        else output_path.with_name("repair_history.jsonl")
    )

    validated, reports, failed = validate_generated_questions(
        read_jsonl(generated_path),
        read_jsonl(Path(args.plan)),
        read_jsonl(Path(args.mastery)),
        model=args.model,
        concurrency=args.concurrency,
        max_rounds=args.max_rounds,
        blind_votes=args.blind_votes,
        tiebreak_votes=args.tiebreak_votes,
        validated_path=output_path,
        reports_path=reports_path,
        failed_path=failed_path,
        repair_history_path=repair_history_path,
    )
    write_jsonl(output_path, validated)
    write_jsonl(reports_path, reports)
    write_jsonl(failed_path, failed)
    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "input": len(read_jsonl(generated_path)),
            "validated": len(validated),
            "failed": len(failed),
            "output": str(output_path),
            "reports": str(reports_path),
            "failed_output": str(failed_path),
            "repair_history": str(repair_history_path),
            "round_output_dir": str(output_path.parent / "validation.rounds"),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "validated": len(validated),
                "failed": len(failed),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
