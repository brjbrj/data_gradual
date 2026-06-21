from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import VLLMClient
from .prompts import build_evaluation_prompt
from .utils import lookup_key, normalize_whitespace, read_json, read_jsonl, safe_json_from_text, write_json, write_jsonl


@dataclass
class EvaluationResult:
    task_id: Any
    source_task_id: Any
    verdict: str
    scores: Dict[str, int]
    issues: List[str]
    short_reason: str
    overall_score: float
    passed: bool


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except Exception:
        score = 1
    return max(1, min(5, score))


def _score_from_length(text: str, target_min: int, target_max: int) -> int:
    n = len(normalize_whitespace(text))
    if target_min <= n <= target_max:
        return 5
    if n < target_min * 0.6 or n > target_max * 1.4:
        return 2
    return 3


def _simple_difficulty_score(candidate: Dict[str, Any], target: Dict[str, Any]) -> int:
    step_count = int(candidate.get("step_count", 0) or 0)
    expected = target.get("step_count_range") or [1, 6]
    if isinstance(expected, Sequence) and len(expected) >= 2:
        low, high = int(expected[0]), int(expected[1])
        if low <= step_count <= high:
            return 5
        if low - 1 <= step_count <= high + 1:
            return 4
    return 2


def _simple_correctness_score(candidate: Dict[str, Any], source: Dict[str, Any]) -> int:
    answer = normalize_whitespace(candidate.get("answer", ""))
    if not answer:
        return 1
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
        return 5
    if source and normalize_whitespace(source.get("answer", "")) == answer:
        return 5
    if len(answer.split()) <= 6:
        return 4
    return 3


def _simple_brevity_score(candidate: Dict[str, Any]) -> int:
    answer = normalize_whitespace(candidate.get("answer", ""))
    solution = normalize_whitespace(candidate.get("solution", ""))
    if len(answer) <= 30 and len(solution) <= 800:
        return 5
    if len(answer) <= 60 and len(solution) <= 1200:
        return 4
    return 2


def _simple_non_redundancy_score(candidate: Dict[str, Any]) -> int:
    text = f"{candidate.get('question', '')}\n{candidate.get('solution', '')}"
    parts = [normalize_whitespace(p) for p in re.split(r"[.?!]\s*", text) if normalize_whitespace(p)]
    if not parts:
        return 1
    unique = len(set(parts))
    ratio = unique / len(parts)
    if ratio >= 0.95:
        return 5
    if ratio >= 0.85:
        return 4
    if ratio >= 0.7:
        return 3
    return 2


def _heuristic_evaluate(candidate: Dict[str, Any], source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    scores = {
        "correctness": _simple_correctness_score(candidate, source),
        "difficulty_match": _simple_difficulty_score(candidate, target),
        "brevity": _simple_brevity_score(candidate),
        "non_redundancy": _simple_non_redundancy_score(candidate),
        "answer_quality": _score_from_length(candidate.get("answer", ""), 1, 80),
    }
    overall = sum(scores.values()) / len(scores)
    issues: List[str] = []
    if scores["correctness"] <= 2:
        issues.append("possible_incorrect_answer")
    if scores["difficulty_match"] <= 2:
        issues.append("difficulty_mismatch")
    if scores["brevity"] <= 2:
        issues.append("too_verbose")
    if scores["non_redundancy"] <= 2:
        issues.append("redundant_text")
    verdict = "pass" if overall >= 4.0 and not issues else "fail"
    return {
        "verdict": verdict,
        "scores": scores,
        "issues": issues,
        "short_reason": "heuristic_evaluation",
        "overall_score": round(overall, 3),
        "passed": verdict == "pass",
    }


class QuestionEvaluator:
    def __init__(self, client: Optional[VLLMClient] = None) -> None:
        self.client = client or VLLMClient()

    def evaluate_one(self, candidate: Dict[str, Any], source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        rubric = {
            "correctness": "Must be mathematically consistent and the answer should be verifiable.",
            "difficulty_match": "The step count and reasoning depth should match the target bucket.",
            "brevity": "The answer and solution should be concise, without unnecessary explanation.",
            "non_redundancy": "Do not repeat facts, sentences, or filler wording.",
            "answer_quality": "The final answer should be short, direct, and clean.",
        }
        try:
            raw = self.client.chat(build_evaluation_prompt(candidate, rubric), temperature=0.0, top_p=1.0, max_tokens=700)
            parsed = safe_json_from_text(raw) or {}
            scores = {k: _clamp_score(parsed.get("scores", {}).get(k)) for k in rubric}
            verdict = str(parsed.get("verdict", "fail")).strip().lower()
            issues = [str(item) for item in parsed.get("issues", []) if str(item).strip()]
            short_reason = normalize_whitespace(parsed.get("short_reason", ""))
            overall = sum(scores.values()) / len(scores)
            if not verdict:
                verdict = "pass" if overall >= 4.0 else "fail"
            passed = verdict == "pass" and overall >= 4.0 and scores["correctness"] >= 4 and scores["difficulty_match"] >= 3
            if not short_reason:
                short_reason = "model_evaluation"
            if not issues and not passed:
                issues = ["model_rejected"]
            return {
                "verdict": verdict,
                "scores": scores,
                "issues": issues,
                "short_reason": short_reason,
                "overall_score": round(overall, 3),
                "passed": passed,
            }
        except Exception:
            return _heuristic_evaluate(candidate, source, target)


def evaluate_questions(records: Sequence[Dict[str, Any]], source_lookup: Dict[Any, Dict[str, Any]], target_lookup: Dict[Any, Dict[str, Any]], client: Optional[VLLMClient] = None) -> List[Dict[str, Any]]:
    evaluator = QuestionEvaluator(client=client)
    outputs: List[Dict[str, Any]] = []
    for record in records:
        source = lookup_key(source_lookup, record.get("source_task_id"), {})
        target = lookup_key(target_lookup, record.get("source_task_id"), {})
        result = evaluator.evaluate_one(record, source, target)
        outputs.append(
            {
                "task_id": record.get("task_id"),
                "source_task_id": record.get("source_task_id"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "solution": record.get("solution"),
                "difficulty_bucket": record.get("difficulty_bucket"),
                "step_count": record.get("step_count"),
                "evaluation": result,
            }
        )
    return outputs


def _load_candidate_records(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_json(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated questions.")
    parser.add_argument("--input", required=True, help="Candidate JSONL or JSON path")
    parser.add_argument("--source-map", required=True, help="JSON map keyed by source task_id")
    parser.add_argument("--target-map", required=True, help="JSON map keyed by source task_id")
    parser.add_argument("--output", required=False, help="Output JSONL path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    records = _load_candidate_records(input_path)
    source_lookup = read_json(Path(args.source_map))
    target_lookup = read_json(Path(args.target_map))
    evaluated = evaluate_questions(records, source_lookup, target_lookup)

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}.evaluated.jsonl")
    write_jsonl(output_path, evaluated)
    print(json.dumps({"output": str(output_path), "count": len(evaluated)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
