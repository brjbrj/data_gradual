from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .client import VLLMClient
from .prompts import build_repair_prompt
from .utils import lookup_key, normalize_whitespace, read_json, read_jsonl, safe_json_from_text, write_json, write_jsonl


REPEAT_SENTENCE_RE = re.compile(r"([^.?!]+[.?!])")
NUMERIC_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")


@dataclass
class CandidateResult:
    task_id: Any
    source_task_id: Any
    question: str
    solution: str
    answer: str
    difficulty_bucket: str
    step_count: int
    eval: Dict[str, Any]
    noise_report: Dict[str, Any]
    repaired: bool = False


def _sentence_set(text: str) -> List[str]:
    parts = [normalize_whitespace(p) for p in REPEAT_SENTENCE_RE.findall(text or "")]
    return [p for p in parts if p]


def _repetition_penalty(text: str) -> float:
    sentences = _sentence_set(text)
    if not sentences:
        return 0.0
    unique = len(set(sentences))
    return 1.0 - unique / len(sentences)


def _answer_leakage(candidate: Dict[str, Any], source: Dict[str, Any]) -> bool:
    answer = normalize_whitespace(candidate.get("answer", ""))
    question = normalize_whitespace(candidate.get("question", ""))
    source_question = normalize_whitespace(source.get("question", ""))
    source_answer = normalize_whitespace(source.get("answer", ""))
    if source_answer and answer == source_answer:
        return True
    if source_question and question == source_question:
        return True
    return False


def _overlap_ratio(text_a: str, text_b: str) -> float:
    a = set(re.findall(r"[A-Za-z0-9]+", text_a.lower()))
    b = set(re.findall(r"[A-Za-z0-9]+", text_b.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _detect_noise(candidate: Dict[str, Any], source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    question = normalize_whitespace(candidate.get("question", ""))
    solution = normalize_whitespace(candidate.get("solution", ""))
    answer = normalize_whitespace(candidate.get("answer", ""))
    source_question = normalize_whitespace(source.get("question", ""))
    source_scene = normalize_whitespace(source.get("scene_template", ""))

    issues: List[str] = []
    if not question or len(question) < 20:
        issues.append("question_too_short")
    if len(question) > 220:
        issues.append("question_too_long")
    if len(solution) > 1000:
        issues.append("solution_too_long")
    if _answer_leakage(candidate, source):
        issues.append("answer_or_question_leaks_source")
    if _repetition_penalty(question) > 0.35 or _repetition_penalty(solution) > 0.4:
        issues.append("repetitive_text")
    if _overlap_ratio(question, source_question) > 0.72:
        issues.append("near_duplicate_question")
    if _overlap_ratio(question, source_scene) > 0.8:
        issues.append("scene_copied_too_closely")

    target_bucket = str(target.get("bucket", candidate.get("difficulty_bucket", "medium")))
    candidate_bucket = str(candidate.get("difficulty_bucket", "medium"))
    if candidate_bucket and target_bucket and candidate_bucket != target_bucket:
        issues.append("difficulty_drift")

    step_count = int(candidate.get("step_count", 0) or 0)
    expected = target.get("step_count_range") or [1, 6]
    if isinstance(expected, Sequence) and len(expected) >= 2:
        low, high = int(expected[0]), int(expected[1])
        if step_count < low or step_count > high:
            issues.append("step_count_out_of_range")

    return {
        "issues": issues,
        "is_clean": not issues,
        "overlap_ratio": round(_overlap_ratio(question, source_question), 3),
        "repetition_penalty": round(max(_repetition_penalty(question), _repetition_penalty(solution)), 3),
    }


class NoiseRepairer:
    def __init__(self, client: Optional[VLLMClient] = None) -> None:
        self.client = client or VLLMClient()

    def repair_one(self, candidate: Dict[str, Any], source: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        noise = _detect_noise(candidate, source, target)
        if noise["is_clean"]:
            return {**candidate, "noise_report": noise, "repaired": False}

        messages = build_repair_prompt(candidate, noise["issues"], target)
        raw = self.client.chat(messages, temperature=0.1, top_p=0.8, max_tokens=900)
        parsed = safe_json_from_text(raw) or {}
        repaired = {
            "question": normalize_whitespace(parsed.get("question") or candidate.get("question", "")),
            "solution": normalize_whitespace(parsed.get("solution") or candidate.get("solution", "")),
            "answer": normalize_whitespace(parsed.get("answer") or candidate.get("answer", "")),
            "difficulty_bucket": normalize_whitespace(parsed.get("difficulty_bucket") or target.get("bucket") or candidate.get("difficulty_bucket", "medium")),
            "step_count": int(parsed.get("step_count") or candidate.get("step_count") or target.get("reference_step_count") or 1),
            "repair_notes": normalize_whitespace(parsed.get("repair_notes") or ""),
        }
        post_noise = _detect_noise(repaired, source, target)
        repaired["noise_report"] = post_noise
        repaired["repaired"] = True
        if post_noise["is_clean"]:
            return repaired
        return {**repaired, "repair_failed": True}


def repair_questions(records: Sequence[Dict[str, Any]], source_lookup: Dict[Any, Dict[str, Any]], targets: Dict[Any, Dict[str, Any]], client: Optional[VLLMClient] = None) -> List[Dict[str, Any]]:
    repairer = NoiseRepairer(client=client)
    output: List[Dict[str, Any]] = []
    for record in records:
        source = lookup_key(source_lookup, record.get("source_task_id"), {})
        target = lookup_key(targets, record.get("source_task_id"), {})
        output.append(repairer.repair_one(record, source, target))
    return output


def _load_candidate_records(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_json(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Repair noisy generated questions.")
    parser.add_argument("--input", required=True, help="Candidate JSONL or JSON path")
    parser.add_argument("--source-map", required=True, help="JSON map keyed by source task_id")
    parser.add_argument("--target-map", required=True, help="JSON map keyed by source task_id")
    parser.add_argument("--output", required=False, help="Output JSONL path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    records = _load_candidate_records(input_path)
    source_lookup = read_json(Path(args.source_map))
    target_lookup = read_json(Path(args.target_map))
    repaired = repair_questions(records, source_lookup, target_lookup)

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}.repaired.jsonl")
    write_jsonl(output_path, repaired)
    print(json.dumps({"output": str(output_path), "count": len(repaired)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
