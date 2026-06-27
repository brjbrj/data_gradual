from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .client import VLLMClient
from .distribute import distribute_mastery_records
from .prompts import build_step_evaluation_prompt, build_victim_answer_prompt
from .utils import normalize_whitespace, read_json, read_jsonl, safe_json_from_text, write_json, write_jsonl


NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
FINAL_MARK_RE = re.compile(r"####\s*(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"[.?!;]\s+|\n+")


@dataclass
class VictimAnswerRecord:
    task_id: Any
    source_task_id: Any
    attempt_index: int
    question: str
    reference_answer: str
    parsed_output: Dict[str, Any]
    raw_output: str
    extracted_answer: str
    is_correct: bool


@dataclass
class StepEvaluationRecord:
    task_id: Any
    source_task_id: Any
    attempt_index: int
    question: str
    reference_answer: str
    candidate_answer: Dict[str, Any]
    step_count: int
    steps: List[Dict[str, Any]]
    step_score_mean: float
    step_score_sum: float
    final_answer_correct: bool
    overall_reason: str


def project_victim_answer_record(record: Dict[str, Any]) -> Dict[str, Any]:
    parsed_output = record.get("parsed_output", {})
    steps = parsed_output.get("steps", []) if isinstance(parsed_output, dict) else []
    return {
        "task_id": record.get("task_id"),
        "source_task_id": record.get("source_task_id"),
        "attempt_index": record.get("attempt_index", 0),
        "question": record.get("question", ""),
        "reference_answer": record.get("reference_answer", ""),
        "steps": steps if isinstance(steps, list) else [],
        "final_answer": normalize_whitespace(str(parsed_output.get("final_answer", ""))) if isinstance(parsed_output, dict) else "",
        "extracted_answer": record.get("extracted_answer", ""),
        "is_correct": bool(record.get("is_correct")),
    }


def project_victim_answer_raw_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": record.get("task_id"),
        "source_task_id": record.get("source_task_id"),
        "attempt_index": record.get("attempt_index", 0),
        "question": record.get("question", ""),
        "reference_answer": record.get("reference_answer", ""),
        "parsed_output": record.get("parsed_output", {}),
        "raw_output": record.get("raw_output", ""),
        "extracted_answer": record.get("extracted_answer", ""),
        "is_correct": bool(record.get("is_correct")),
    }


def project_step_evaluation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(record)


def _normalize_number_token(text: str) -> str:
    cleaned = normalize_whitespace(text).lower()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("%", "")
    return cleaned


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("final_answer", "answer"):
                value = parsed.get(key)
                if value is not None:
                    return normalize_whitespace(str(value))
        if isinstance(parsed, list) and parsed:
            last_item = parsed[-1]
            if isinstance(last_item, dict):
                for key in ("final_answer", "answer"):
                    value = last_item.get(key)
                    if value is not None:
                        return normalize_whitespace(str(value))
    except Exception:
        pass
    match = FINAL_MARK_RE.search(text)
    if match:
        return normalize_whitespace(match.group(1))
    numbers = NUMBER_RE.findall(text.replace(",", ""))
    if numbers:
        return normalize_whitespace(numbers[-1])
    return normalize_whitespace(text.splitlines()[-1] if text.splitlines() else text)


def _normalize_steps(steps: Any) -> List[str]:
    if not isinstance(steps, list):
        return []
    outputs: List[str] = []
    for step in steps:
        if step is None:
            continue
        if isinstance(step, str):
            normalized = normalize_whitespace(step)
            if normalized:
                outputs.append(normalized)
        else:
            normalized = normalize_whitespace(str(step))
            if normalized:
                outputs.append(normalized)
    return outputs


def _decode_json_string_fragment(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except Exception:
        return value.replace('\\"', '"').replace("\\n", " ")


def _extract_keyed_string(raw_output: str, keys: Sequence[str]) -> str:
    for key in keys:
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            raw_output,
            flags=re.DOTALL,
        )
        if match:
            return normalize_whitespace(_decode_json_string_fragment(match.group(1)))
        loose_match = re.search(
            rf'"{re.escape(key)}"\s*:\s*([^,\}}\]]+)',
            raw_output,
            flags=re.DOTALL,
        )
        if loose_match:
            return normalize_whitespace(loose_match.group(1).strip().strip('"'))
    return ""


def _extract_steps_from_malformed_json(raw_output: str) -> List[str]:
    if not raw_output:
        return []
    start_match = re.search(r'"steps"\s*:\s*\[', raw_output, flags=re.DOTALL)
    if not start_match:
        return []

    start = start_match.end()
    tail = raw_output[start:]
    end_candidates = []
    for pattern in (r'\]\s*,\s*"final_answer"', r'\]\s*,\s*"answer"', r'\]\s*\}'):
        match = re.search(pattern, tail, flags=re.DOTALL)
        if match:
            end_candidates.append(match.start())
    if end_candidates:
        steps_fragment = tail[: min(end_candidates)]
    else:
        final_key = re.search(r',\s*"(?:final_answer|answer)"\s*:', tail, flags=re.DOTALL)
        steps_fragment = tail[: final_key.start()] if final_key else tail

    quoted_items = re.findall(r'"((?:\\.|[^"\\])*)"', steps_fragment, flags=re.DOTALL)
    recovered = [
        normalize_whitespace(_decode_json_string_fragment(item))
        for item in quoted_items
    ]
    recovered = [item for item in recovered if item]
    if recovered:
        return recovered

    # Last resort for badly truncated arrays that no longer contain quoted strings.
    lines = re.split(r"\n+|(?<=[.;])\s+", steps_fragment)
    return [
        normalize_whitespace(line.strip().strip(",[]\"'"))
        for line in lines
        if normalize_whitespace(line.strip().strip(",[]\"'"))
    ]


def _parse_victim_output(raw_output: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    try:
        decoded = json.loads(raw_output)
        if isinstance(decoded, dict):
            parsed = decoded
    except Exception:
        parsed = {}

    steps = _normalize_steps(parsed.get("steps"))
    if not steps:
        steps = _extract_steps_from_malformed_json(raw_output)
    final_answer = normalize_whitespace(str(parsed.get("final_answer", parsed.get("answer", "")))) if parsed else ""
    if not final_answer:
        final_answer = _extract_keyed_string(raw_output, ("final_answer", "answer"))
    if not final_answer:
        final_answer = _extract_final_answer(raw_output)
    return {
        "steps": steps,
        "final_answer": final_answer,
    }


ALLOWED_STEP_SCORES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _normalize_step_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score in ALLOWED_STEP_SCORES:
        return score
    if 0.0 <= score <= 1.0:
        closest = min(ALLOWED_STEP_SCORES, key=lambda candidate: abs(candidate - score))
        return float(closest)
    if 1.0 <= score <= 5.0:
        mapped = round((score - 1.0) / 4.0, 1)
        if mapped in ALLOWED_STEP_SCORES:
            return float(mapped)
        closest = min(ALLOWED_STEP_SCORES, key=lambda candidate: abs(candidate - mapped))
        return float(closest)
    return 0.0


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _build_step_records(step_texts: Sequence[str], score_map: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    step_count = len(step_texts)
    for idx, step_text in enumerate(step_texts):
        scores = {
            key: _normalize_step_score(values[idx] if idx < len(values) else 0.0)
            for key, values in score_map.items()
        }
        records.append(
            {
                "step_id": idx + 1,
                "text": step_text,
                "scores": scores,
                "short_reason": "",
            }
        )
    return records


def _numeric_match(a: str, b: str) -> bool:
    na = _normalize_number_token(a)
    nb = _normalize_number_token(b)
    if not na or not nb:
        return False
    try:
        return math.isclose(float(na), float(nb), rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return na == nb


def _is_correct_answer(candidate: str, reference: str) -> bool:
    if not candidate or not reference:
        return False
    if _numeric_match(candidate, reference):
        return True
    cand_norm = _normalize_number_token(candidate)
    ref_norm = _normalize_number_token(reference)
    if cand_norm == ref_norm:
        return True
    return candidate.strip().lower() == reference.strip().lower()


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_line(prefix: str, index: int, total: int, started_at: float) -> str:
    total = max(1, total)
    index = min(index, total)
    elapsed = time.time() - started_at
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else 0.0
    percent = index * 100.0 / total
    return f"{prefix} {index}/{total} ({percent:5.1f}%) elapsed={_format_seconds(elapsed)} eta={_format_seconds(remaining)}"


def _should_log_progress(done: int, total: int) -> bool:
    if done <= 20:
        return True
    if total <= 200:
        return done % 5 == 0 or done == total
    interval = max(1, total // 100)
    return done % interval == 0 or done == total


def _resolve_workers(requested: Optional[int], env_name: str, default: int = 256) -> int:
    if requested is not None:
        return max(1, requested)
    env_value = os.environ.get(env_name)
    if env_value:
        try:
            return max(1, int(env_value))
        except Exception:
            pass
    return max(1, default)


def _parse_int_env(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return int(default)


def _parse_float_env(*names: str, default: float) -> float:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _fallback_step_scores(step_texts: Sequence[str], question: str, reference_answer: str, candidate_answer: str) -> Dict[str, Any]:
    steps = []
    candidate_correct = _is_correct_answer(candidate_answer, reference_answer)
    for idx, step_text in enumerate(step_texts, start=1):
        length = len(step_text.split())
        score = 4 if length >= 3 else 3
        if any(ch.isdigit() for ch in step_text):
            score += 1
        if _normalize_number_token(reference_answer) and _normalize_number_token(reference_answer) in _normalize_number_token(step_text):
            score += 1
        score = max(1, min(5, score))
        steps.append(
            {
                "step_id": idx,
                "text": step_text,
                "scores": {
                    "correctness": min(1.0, max(0.0, (score - 1) / 4.0)),
                    "logical": min(1.0, max(0.0, (score - 1) / 4.0)),
                    "standardization": min(1.0, max(0.0, (score - 1) / 4.0)),
                    "completeness": min(1.0, max(0.0, (score - 1) / 4.0)),
                    "efficiency": min(1.0, max(0.0, (max(1, min(5, 6 - max(1, length // 6))) - 1) / 4.0)),
                },
                "short_reason": "heuristic_fallback",
            }
        )
    if not steps:
        steps = [
            {
                "step_id": 1,
                "text": normalize_whitespace(candidate_answer),
                "scores": {
                    "correctness": 1.0 if candidate_correct else 0.4,
                    "logical": 0.6,
                    "standardization": 0.8,
                    "completeness": 0.4,
                    "efficiency": 0.8,
                },
                "short_reason": "heuristic_fallback",
            }
        ]
    step_score_sum = sum(sum(step["scores"].values()) / 5.0 for step in steps)
    step_score_mean = step_score_sum / len(steps)
    return {
        "steps": steps,
        "step_count": len(steps),
        "step_score_mean": step_score_mean,
        "step_score_sum": step_score_sum,
        "final_answer_correct": candidate_correct,
        "overall_reason": "heuristic_fallback",
    }


class VictimAnswerer:
    def __init__(
        self,
        client: Optional[VLLMClient] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> None:
        self.client = client or VLLMClient()
        self.temperature = 0.3 if temperature is None else float(temperature)
        self.top_p = 0.95 if top_p is None else float(top_p)

    def answer_one(self, record: Dict[str, Any], attempt_index: int) -> VictimAnswerRecord:
        question = normalize_whitespace(record.get("question", ""))
        reference_answer = normalize_whitespace(record.get("answer", ""))
        prompt = build_victim_answer_prompt(question, attempt_index)
        raw = self.client.chat(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=1024)
        parsed_output = _parse_victim_output(raw)
        extracted = normalize_whitespace(parsed_output.get("final_answer", "")) or _extract_final_answer(raw)
        return VictimAnswerRecord(
            task_id=f"{record.get('task_id')}_{attempt_index}",
            source_task_id=record.get("task_id"),
            attempt_index=attempt_index,
            question=question,
            reference_answer=reference_answer,
            parsed_output=parsed_output,
            raw_output=raw,
            extracted_answer=extracted,
            is_correct=_is_correct_answer(extracted, reference_answer),
        )


def answer_questions(
    records: Sequence[Dict[str, Any]],
    n_answers: int,
    client: Optional[VLLMClient] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    existing_records: Optional[Sequence[Dict[str, Any]]] = None,
    answer_output_path: Optional[Path] = None,
    answer_raw_output_path: Optional[Path] = None,
    checkpoint_every: Optional[int] = None,
) -> List[Dict[str, Any]]:
    answerer = VictimAnswerer(client=client, temperature=temperature, top_p=top_p)
    expected_total = len(records) * max(1, n_answers)
    started_at = time.time()
    workers = _resolve_workers(max_concurrency, "ANSWER_CONCURRENCY", default=256)
    checkpoint_every = (
        max(1, _parse_int_env("ANSWER_CHECKPOINT_EVERY", default=50))
        if checkpoint_every is None
        else max(1, int(checkpoint_every))
    )
    existing_by_task_id: Dict[str, Dict[str, Any]] = {}
    for record in existing_records or []:
        task_id = str(record.get("task_id") or "")
        if task_id:
            existing_by_task_id[task_id] = dict(record)
    tasks: List[Dict[str, Any]] = []
    for record in records:
        for attempt_index in range(n_answers):
            task_id = f"{record.get('task_id')}_{attempt_index}"
            if task_id not in existing_by_task_id:
                tasks.append({"record": record, "attempt_index": attempt_index})

    outputs_by_task_id: Dict[str, Dict[str, Any]] = dict(existing_by_task_id)
    total = len(tasks)
    if existing_by_task_id:
        print(
            f"[answer] resume enabled: loaded={len(existing_by_task_id)} "
            f"remaining={total} expected={expected_total}",
            flush=True,
        )

    def ordered_outputs() -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for source_record in records:
            for attempt_index in range(n_answers):
                task_id = f"{source_record.get('task_id')}_{attempt_index}"
                record = outputs_by_task_id.get(task_id)
                if record is not None:
                    outputs.append(record)
        return outputs

    def save_checkpoint(reason: str) -> None:
        if answer_output_path is None and answer_raw_output_path is None:
            return
        outputs = ordered_outputs()
        if answer_output_path is not None:
            write_jsonl(answer_output_path, [project_victim_answer_record(record) for record in outputs])
        if answer_raw_output_path is not None:
            write_jsonl(answer_raw_output_path, [project_victim_answer_raw_record(record) for record in outputs])
        print(
            f"[answer] checkpoint saved reason={reason} "
            f"records={len(outputs)}/{expected_total}",
            flush=True,
        )

    if total == 0:
        save_checkpoint("nothing_pending")
        return ordered_outputs()

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
        future_to_index = {
            executor.submit(answerer.answer_one, task["record"], task["attempt_index"]): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_index):
            record = future.result().__dict__
            outputs_by_task_id[str(record.get("task_id"))] = record
            done += 1
            if done % checkpoint_every == 0:
                save_checkpoint("periodic")
            if _should_log_progress(done, total):
                print(_progress_line("[answer]", done, total, started_at), flush=True)
    save_checkpoint("complete")
    return ordered_outputs()


class StepEvaluator:
    def __init__(self, client: Optional[VLLMClient] = None) -> None:
        self.client = client or VLLMClient()

    def evaluate_one(self, answer_record: Dict[str, Any]) -> StepEvaluationRecord:
        question = normalize_whitespace(answer_record.get("question", ""))
        reference_answer = normalize_whitespace(answer_record.get("reference_answer", ""))
        parsed_output = answer_record.get("parsed_output", {})
        if not isinstance(parsed_output, dict):
            parsed_output = _parse_victim_output(normalize_whitespace(answer_record.get("raw_output", "")))
        step_texts = _normalize_steps(parsed_output.get("steps"))
        candidate_short = normalize_whitespace(parsed_output.get("final_answer", "")) or answer_record.get("extracted_answer") or ""

        prompt = build_step_evaluation_prompt(question, reference_answer, step_texts, candidate_short)
        try:
            raw = self.client.chat(prompt, temperature=0.0, top_p=1.0, max_tokens=1200)
            parsed = safe_json_from_text(raw) or {}
            score_keys = ("correctness", "logical", "standardization", "completeness", "efficiency")
            score_arrays = {}
            for key in score_keys:
                values = parsed.get(key, [])
                if not isinstance(values, list):
                    raise ValueError(f"missing score array: {key}")
                if len(values) != len(step_texts):
                    raise ValueError(f"score array length mismatch for {key}")
                score_arrays[key] = [_normalize_step_score(value) for value in values]
            normalized_steps = _build_step_records(step_texts, score_arrays)
            if not normalized_steps:
                raise ValueError("no steps to score")
            step_score_sum = sum(sum(step["scores"].values()) for step in normalized_steps) / 5.0
            step_score_mean = step_score_sum / len(normalized_steps)
            final_answer_correct = _coerce_bool(parsed.get("final_answer_correct"))
            if not final_answer_correct:
                final_answer_correct = _is_correct_answer(candidate_short, reference_answer)
            overall_reason = normalize_whitespace(parsed.get("overall_reason", ""))
            if not overall_reason:
                overall_reason = "model_evaluation"
            return StepEvaluationRecord(
                task_id=answer_record.get("task_id"),
                source_task_id=answer_record.get("source_task_id"),
                attempt_index=int(answer_record.get("attempt_index", 0) or 0),
                question=question,
                reference_answer=reference_answer,
                candidate_answer={
                    "steps": step_texts,
                    "final_answer": candidate_short,
                },
                step_count=len(normalized_steps),
                steps=normalized_steps,
                step_score_mean=round(step_score_mean, 4),
                step_score_sum=round(step_score_sum, 4),
                final_answer_correct=final_answer_correct,
                overall_reason=overall_reason,
            )
        except Exception:
            fallback = _fallback_step_scores(
                step_texts,
                question,
                reference_answer,
                candidate_short,
            )
            return StepEvaluationRecord(
                task_id=answer_record.get("task_id"),
                source_task_id=answer_record.get("source_task_id"),
                attempt_index=int(answer_record.get("attempt_index", 0) or 0),
                question=question,
                reference_answer=reference_answer,
                candidate_answer={
                    "steps": step_texts,
                    "final_answer": candidate_short,
                },
                step_count=fallback["step_count"],
                steps=fallback["steps"],
                step_score_mean=round(float(fallback["step_score_mean"]), 4),
                step_score_sum=round(float(fallback["step_score_sum"]), 4),
                final_answer_correct=bool(fallback["final_answer_correct"]),
                overall_reason=fallback["overall_reason"],
            )


def _step_evaluation_inputs(
    answer_record: Dict[str, Any],
) -> Tuple[str, str, List[str], str]:
    question = normalize_whitespace(answer_record.get("question", ""))
    reference_answer = normalize_whitespace(
        answer_record.get("reference_answer", "")
    )
    parsed_output = answer_record.get("parsed_output", {})
    if not isinstance(parsed_output, dict):
        parsed_output = _parse_victim_output(
            normalize_whitespace(answer_record.get("raw_output", ""))
        )
    step_texts = _normalize_steps(
        parsed_output.get("steps", answer_record.get("steps", []))
    )
    candidate_short = (
        normalize_whitespace(
            parsed_output.get(
                "final_answer",
                answer_record.get("final_answer", ""),
            )
        )
        or answer_record.get("extracted_answer")
        or ""
    )
    return question, reference_answer, step_texts, candidate_short


def _evaluation_signature(answer_record: Dict[str, Any]) -> str:
    question, reference_answer, step_texts, candidate_short = (
        _step_evaluation_inputs(answer_record)
    )
    return json.dumps(
        [question, reference_answer, step_texts, candidate_short],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _report_signature(report: Dict[str, Any]) -> str:
    candidate = report.get("candidate_answer", {})
    if not isinstance(candidate, dict):
        candidate = {}
    return json.dumps(
        [
            normalize_whitespace(report.get("question", "")),
            normalize_whitespace(report.get("reference_answer", "")),
            _normalize_steps(candidate.get("steps", [])),
            normalize_whitespace(candidate.get("final_answer", "")),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_step_evaluation_response(
    answer_record: Dict[str, Any],
    raw: str,
) -> Dict[str, Any]:
    question, reference_answer, step_texts, candidate_short = (
        _step_evaluation_inputs(answer_record)
    )
    parsed = safe_json_from_text(raw) or {}
    score_keys = (
        "correctness",
        "logical",
        "standardization",
        "completeness",
        "efficiency",
    )
    score_arrays: Dict[str, List[float]] = {}
    for key in score_keys:
        values = parsed.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"missing score array: {key}")
        if len(values) != len(step_texts):
            raise ValueError(f"score array length mismatch for {key}")
        score_arrays[key] = [
            _normalize_step_score(value)
            for value in values
        ]
    normalized_steps = _build_step_records(step_texts, score_arrays)
    if not normalized_steps:
        raise ValueError("no steps to score")

    step_score_sum = (
        sum(sum(step["scores"].values()) for step in normalized_steps)
        / 5.0
    )
    step_score_mean = step_score_sum / len(normalized_steps)
    final_answer_correct = _coerce_bool(
        parsed.get("final_answer_correct")
    )
    if not final_answer_correct:
        final_answer_correct = _is_correct_answer(
            candidate_short,
            reference_answer,
        )
    overall_reason = normalize_whitespace(
        parsed.get("overall_reason", "")
    )
    if not overall_reason:
        overall_reason = "model_evaluation"

    return StepEvaluationRecord(
        task_id=answer_record.get("task_id"),
        source_task_id=answer_record.get("source_task_id"),
        attempt_index=int(
            answer_record.get("attempt_index", 0) or 0
        ),
        question=question,
        reference_answer=reference_answer,
        candidate_answer={
            "steps": step_texts,
            "final_answer": candidate_short,
        },
        step_count=len(normalized_steps),
        steps=normalized_steps,
        step_score_mean=round(step_score_mean, 4),
        step_score_sum=round(step_score_sum, 4),
        final_answer_correct=final_answer_correct,
        overall_reason=overall_reason,
    ).__dict__


def _fallback_step_evaluation(
    answer_record: Dict[str, Any],
) -> Dict[str, Any]:
    question, reference_answer, step_texts, candidate_short = (
        _step_evaluation_inputs(answer_record)
    )
    fallback = _fallback_step_scores(
        step_texts,
        question,
        reference_answer,
        candidate_short,
    )
    return StepEvaluationRecord(
        task_id=answer_record.get("task_id"),
        source_task_id=answer_record.get("source_task_id"),
        attempt_index=int(
            answer_record.get("attempt_index", 0) or 0
        ),
        question=question,
        reference_answer=reference_answer,
        candidate_answer={
            "steps": step_texts,
            "final_answer": candidate_short,
        },
        step_count=fallback["step_count"],
        steps=fallback["steps"],
        step_score_mean=round(
            float(fallback["step_score_mean"]),
            4,
        ),
        step_score_sum=round(
            float(fallback["step_score_sum"]),
            4,
        ),
        final_answer_correct=bool(
            fallback["final_answer_correct"]
        ),
        overall_reason=fallback["overall_reason"],
    ).__dict__


def _clone_step_evaluation(
    report: Dict[str, Any],
    answer_record: Dict[str, Any],
) -> Dict[str, Any]:
    cloned = json.loads(json.dumps(report, ensure_ascii=False))
    question, reference_answer, step_texts, candidate_short = (
        _step_evaluation_inputs(answer_record)
    )
    cloned.update(
        {
            "task_id": answer_record.get("task_id"),
            "source_task_id": answer_record.get("source_task_id"),
            "attempt_index": int(
                answer_record.get("attempt_index", 0) or 0
            ),
            "question": question,
            "reference_answer": reference_answer,
            "candidate_answer": {
                "steps": step_texts,
                "final_answer": candidate_short,
            },
        }
    )
    return cloned


def _read_jsonl_tolerant(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
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


async def _resolve_served_model_name(async_client: Any, model: str) -> str:
    try:
        served = await async_client.models.list()
    except Exception:
        return model

    expected = _model_aliases(model)
    for item in getattr(served, "data", []):
        served_id = getattr(item, "id", None)
        if served_id and _model_aliases(str(served_id)) & expected:
            return str(served_id)
    return model


async def _evaluate_answers_async(
    answer_records: Sequence[Dict[str, Any]],
    *,
    client: Optional[VLLMClient],
    max_concurrency: Optional[int],
    checkpoint_path: Optional[Path],
    resume: bool,
) -> List[Dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The pipeline environment requires the openai package."
        ) from exc

    total = len(answer_records)
    if total == 0:
        return []

    workers = _resolve_workers(
        max_concurrency,
        "SCORE_CONCURRENCY",
        default=256,
    )
    enable_thinking = _parse_bool_env(
        "SCORE_ENABLE_THINKING",
        False,
    )
    force_json = _parse_bool_env("SCORE_FORCE_JSON", False)
    deduplicate = _parse_bool_env("SCORE_DEDUPLICATE", True)
    fallback_on_exhausted = _parse_bool_env(
        "SCORE_FALLBACK_ON_EXHAUSTED",
        True,
    )
    max_retries = _parse_int_env(
        "SCORE_MAX_RETRIES",
        default=3,
    )
    retry_delay = _parse_float_env(
        "SCORE_RETRY_DELAY",
        default=1.0,
    )
    progress_interval = max(
        1.0,
        _parse_float_env(
            "SCORE_PROGRESS_INTERVAL",
            default=5.0,
        ),
    )
    progress_every = max(
        1,
        _parse_int_env(
            "SCORE_PROGRESS_EVERY",
            default=5,
        ),
    )
    min_tokens = max(
        64,
        _parse_int_env("SCORE_MIN_TOKENS", default=256),
    )
    tokens_per_step = max(
        8,
        _parse_int_env(
            "SCORE_TOKENS_PER_STEP",
            default=24,
        ),
    )
    max_tokens_cap = max(
        min_tokens,
        _parse_int_env("SCORE_MAX_TOKENS", default=900),
    )

    base_url = (
        client.base_url
        if client is not None
        else os.environ.get("VLLM_BASE_URL")
        or "http://127.0.0.1:8911/v1"
    )
    model = (
        client.model
        if client is not None
        else os.environ.get("STEP_MODEL")
        or os.environ.get("VLLM_MODEL")
        or "local-model"
    )
    api_key = (
        client.api_key
        if client is not None
        else os.environ.get("VLLM_API_KEY")
        or "EMPTY"
    )
    timeout = (
        client.timeout
        if client is not None
        else _parse_int_env("VLLM_TIMEOUT", default=600)
    )

    async_client = AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    model = await _resolve_served_model_name(async_client, model)
    semaphore = asyncio.Semaphore(max(1, workers))
    outputs: List[Optional[Dict[str, Any]]] = [None] * total
    index_by_task = {
        str(record.get("task_id")): index
        for index, record in enumerate(answer_records)
    }
    signature_groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(answer_records):
        signature = _evaluation_signature(record)
        if not deduplicate:
            signature = f"{signature}::record:{index}"
        signature_groups[signature].append(index)

    resumed = 0
    report_by_signature: Dict[str, Dict[str, Any]] = {}
    if resume and checkpoint_path is not None:
        for report in _read_jsonl_tolerant(checkpoint_path):
            index = index_by_task.get(str(report.get("task_id")))
            if index is None or outputs[index] is not None:
                continue
            outputs[index] = report
            report_by_signature[_report_signature(report)] = report
            resumed += 1

    checkpoint_handle = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        checkpoint_handle = checkpoint_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

    failures_path_env = os.environ.get("SCORE_FAILURES_PATH", "").strip()
    failures_path = (
        Path(failures_path_env)
        if failures_path_env
        else (
            checkpoint_path.with_name(checkpoint_path.name + ".failures.jsonl")
            if checkpoint_path is not None
            else None
        )
    )
    failures_handle = None
    if failures_path is not None:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_handle = failures_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def save_report(report: Dict[str, Any]) -> None:
        if checkpoint_handle is None:
            return
        checkpoint_handle.write(
            json.dumps(report, ensure_ascii=False) + "\n"
        )

    def failure_record(
        *,
        signature: str,
        round_index: int,
        event: str,
        error: str,
    ) -> Dict[str, Any]:
        representative_index = signature_groups[signature][0]
        record = answer_records[representative_index]
        question, reference_answer, step_texts, candidate_short = (
            _step_evaluation_inputs(record)
        )
        parsed_output = (
            record.get("parsed_output", {})
            if isinstance(record.get("parsed_output"), dict)
            else {}
        )
        return {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "stage": "score",
            "event": event,
            "round": round_index,
            "signature": signature,
            "group_size": len(signature_groups[signature]),
            "task_id": record.get("task_id"),
            "source_task_id": record.get("source_task_id"),
            "attempt_index": record.get("attempt_index", 0),
            "error": error,
            "question": question,
            "reference_answer": reference_answer,
            "candidate_final_answer": candidate_short,
            "step_count": len(step_texts),
            "steps": step_texts,
            "parsed_output": parsed_output,
            "extracted_answer": record.get("extracted_answer", ""),
            "is_correct": bool(record.get("is_correct")),
            "raw_output_prefix": normalize_whitespace(
                str(record.get("raw_output", ""))[:2000]
            ),
            "model": model,
        }

    def save_failure(
        *,
        signature: str,
        round_index: int,
        event: str,
        error: str,
    ) -> None:
        if failures_handle is None:
            return
        failures_handle.write(
            json.dumps(
                failure_record(
                    signature=signature,
                    round_index=round_index,
                    event=event,
                    error=error,
                ),
                ensure_ascii=False,
            )
            + "\n"
        )

    completed = resumed
    dedup_reused = 0
    pending_signatures: List[str] = []
    for signature, indices in signature_groups.items():
        existing = report_by_signature.get(signature)
        if existing is None:
            existing = next(
                (
                    outputs[index]
                    for index in indices
                    if outputs[index] is not None
                ),
                None,
            )
        if existing is not None and deduplicate:
            for index in indices:
                if outputs[index] is not None:
                    continue
                cloned = _clone_step_evaluation(
                    existing,
                    answer_records[index],
                )
                outputs[index] = cloned
                save_report(cloned)
                completed += 1
                dedup_reused += 1
            continue
        unresolved = [
            index for index in indices
            if outputs[index] is None
        ]
        if unresolved:
            pending_signatures.append(signature)

    request_total = len(pending_signatures)
    saved_requests = total - resumed - request_total
    print(
        f"[score] records={total} unique_requests={request_total} "
        f"resumed={resumed} dedup_saved={saved_requests} "
        f"concurrency={workers} thinking={enable_thinking} "
        f"force_json={force_json} "
        f"fallback={fallback_on_exhausted} "
        f"max_tokens_cap={max_tokens_cap} "
        f"progress_every={progress_every} "
        f"progress_interval={progress_interval:g}s "
        f"failures_path={failures_path if failures_path is not None else '<none>'}",
        flush=True,
    )

    async def evaluate_signature(
        signature: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        representative_index = signature_groups[signature][0]
        record = answer_records[representative_index]
        question, reference_answer, step_texts, candidate_short = (
            _step_evaluation_inputs(record)
        )
        if not step_texts:
            report = _fallback_step_evaluation(record)
            report["overall_reason"] = "heuristic_fallback_no_steps"
            return signature, report, ""
        prompt = build_step_evaluation_prompt(
            question,
            reference_answer,
            step_texts,
            candidate_short,
        )
        request: Dict[str, Any] = {
            "model": model,
            "messages": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": min(
                max_tokens_cap,
                max(
                    min_tokens,
                    96 + tokens_per_step * len(step_texts),
                ),
            ),
        }
        if force_json:
            request["response_format"] = {
                "type": "json_object"
            }
        if not enable_thinking:
            request["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        try:
            async with semaphore:
                response = (
                    await async_client.chat.completions.create(
                        **request
                    )
                )
            raw = response.choices[0].message.content or ""
            return (
                signature,
                _parse_step_evaluation_response(record, raw),
                "",
            )
        except ValueError as exc:
            if "no steps to score" in str(exc):
                report = _fallback_step_evaluation(record)
                report["overall_reason"] = (
                    "heuristic_fallback_after_empty_step_scores"
                )
                return signature, report, ""
            return (
                signature,
                None,
                f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:
            return (
                signature,
                None,
                f"{type(exc).__name__}: {exc}",
            )

    started_at = time.time()
    pending = pending_signatures
    round_index = 0
    infinite_retries = max_retries < 0
    last_errors: Dict[str, str] = {}
    try:
        while pending and (
            infinite_retries or round_index <= max_retries
        ):
            print(
                f"[score] round={round_index} "
                f"requests={len(pending)}",
                flush=True,
            )
            tasks = {
                asyncio.create_task(
                    evaluate_signature(signature)
                )
                for signature in pending
            }
            next_pending: List[str] = []
            round_done = 0
            round_ok = 0
            round_errors = 0
            round_started_at = time.time()
            last_log_at = round_started_at
            pending_tasks = tasks
            while pending_tasks:
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    timeout=progress_interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                now = time.time()
                if not done_tasks:
                    elapsed = now - round_started_at
                    rate = (
                        round_done / elapsed
                        if elapsed > 0
                        else 0.0
                    )
                    eta = (
                        _format_seconds(
                            (len(tasks) - round_done) / rate
                        )
                        if rate > 0
                        else "--:--"
                    )
                    print(
                        f"[score] round={round_index} "
                        f"heartbeat {round_done}/{len(tasks)} "
                        f"({round_done / len(tasks) * 100:5.1f}%) "
                        f"records={completed}/{total} "
                        f"ok={round_ok} error={round_errors} "
                        f"active={len(pending_tasks)} "
                        f"rate={rate:.2f}/s "
                        f"elapsed={_format_seconds(elapsed)} "
                        f"eta={eta}",
                        flush=True,
                    )
                    last_log_at = now
                    continue

                for task in done_tasks:
                    signature, report, error = task.result()
                    round_done += 1
                    if report is None:
                        last_errors[signature] = error
                        save_failure(
                            signature=signature,
                            round_index=round_index,
                            event="retry_pending",
                            error=error,
                        )
                        next_pending.append(signature)
                        round_errors += 1
                    else:
                        report_by_signature[signature] = report
                        round_ok += 1
                        for index in signature_groups[signature]:
                            if outputs[index] is not None:
                                continue
                            cloned = _clone_step_evaluation(
                                report,
                                answer_records[index],
                            )
                            outputs[index] = cloned
                            save_report(cloned)
                            completed += 1

                should_log = (
                    round_done <= 5
                    or round_done % progress_every == 0
                    or round_done == len(tasks)
                    or now - last_log_at >= progress_interval
                )
                if should_log:
                    elapsed = now - round_started_at
                    rate = (
                        round_done / elapsed
                        if elapsed > 0
                        else 0.0
                    )
                    eta = (
                        _format_seconds(
                            (len(tasks) - round_done) / rate
                        )
                        if rate > 0
                        else "--:--"
                    )
                    print(
                        f"[score] round={round_index} "
                        f"{round_done}/{len(tasks)} "
                        f"({round_done / len(tasks) * 100:5.1f}%) "
                        f"records={completed}/{total} "
                        f"ok={round_ok} error={round_errors} "
                        f"active={len(pending_tasks)} "
                        f"rate={rate:.2f}/s "
                        f"elapsed={_format_seconds(elapsed)} "
                        f"eta={eta}",
                        flush=True,
                    )
                    last_log_at = now

            pending = next_pending
            if pending:
                print(
                    f"[score] round={round_index} complete "
                    f"retry={len(pending)}",
                    flush=True,
                )
                round_index += 1
                if (
                    infinite_retries
                    or round_index <= max_retries
                ) and retry_delay > 0:
                    await asyncio.sleep(retry_delay)

        if pending:
            if not fallback_on_exhausted:
                examples = [
                    last_errors.get(signature, "unknown error")
                    for signature in pending[:3]
                ]
                raise RuntimeError(
                    "step scoring retries exhausted for "
                    f"{len(pending)} unique responses; completed "
                    f"records are preserved in {checkpoint_path}. "
                    f"Examples: {examples}"
                )
            print(
                f"[score] retries exhausted; applying configured "
                f"fallback to {len(pending)} unique responses",
                flush=True,
            )
            for signature in pending:
                representative_index = signature_groups[signature][0]
                final_error = last_errors.get(signature, "unknown error")
                save_failure(
                    signature=signature,
                    round_index=round_index,
                    event="fallback_after_retries",
                    error=final_error,
                )
                report = _fallback_step_evaluation(
                    answer_records[representative_index]
                )
                report["overall_reason"] = (
                    "heuristic_fallback_after_retries: "
                    + final_error
                )
                for index in signature_groups[signature]:
                    if outputs[index] is not None:
                        continue
                    cloned = _clone_step_evaluation(
                        report,
                        answer_records[index],
                    )
                    outputs[index] = cloned
                    save_report(cloned)
                    completed += 1
    finally:
        if checkpoint_handle is not None:
            checkpoint_handle.close()
        if failures_handle is not None:
            failures_handle.close()
        await async_client.close()

    print(
        f"[score] complete records={completed}/{total} "
        f"requests={request_total} dedup_reused={dedup_reused} "
        f"elapsed={_format_seconds(time.time() - started_at)}",
        flush=True,
    )
    return [
        record
        for record in outputs
        if record is not None
    ]


def evaluate_answers(
    answer_records: Sequence[Dict[str, Any]],
    client: Optional[VLLMClient] = None,
    max_concurrency: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    return asyncio.run(
        _evaluate_answers_async(
            answer_records,
            client=client,
            max_concurrency=max_concurrency,
            checkpoint_path=checkpoint_path,
            resume=(
                _parse_bool_env("SCORE_RESUME", True)
                if resume is None
                else bool(resume)
            ),
        )
    )


def _compute_weight(score: float, method: str = "linear_cutoff") -> float:
    """Match the old pipeline: convert a 0-10 score into an evidence weight."""
    q = score / 10.0
    if method == "linear_cutoff":
        return max(0.0, 2.0 * q - 1.0)
    if method == "sigmoid":
        return 1.0 / (1.0 + math.exp(-10.0 * (q - 0.5)))
    raise ValueError(f"Unknown method: {method}")


def _compute_mastery(
    answer_scores: Sequence[float],
    is_correct: Sequence[bool],
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    weight_method: str = "linear_cutoff",
) -> Dict[str, Any]:
    if len(answer_scores) != len(is_correct):
        raise ValueError("answer_scores and is_correct must have the same length")

    total_weighted_success = 0.0
    total_weight = 0.0
    weights: List[float] = []
    for score, correct in zip(answer_scores, is_correct):
        weight = _compute_weight(score, method=weight_method)
        weights.append(weight)
        total_weighted_success += weight * (1.0 if correct else 0.0)
        total_weight += weight

    alpha_post = alpha_prior + total_weighted_success
    beta_post = beta_prior + (total_weight - total_weighted_success)
    mastery = alpha_post / (alpha_post + beta_post)
    raw_accuracy = sum(1 for value in is_correct if value) / max(1, len(is_correct))
    return {
        "mastery": mastery,
        "details": {
            "total_weighted_success": total_weighted_success,
            "total_weight": total_weight,
            "alpha_post": alpha_post,
            "beta_post": beta_post,
            "weights": weights,
            "raw_accuracy": raw_accuracy,
        },
    }


def _group_by_source(records: Sequence[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("source_task_id")].append(record)
    return grouped


def _difficulty_bucket_from_mastery(mastery: float) -> str:
    if mastery < 0.25:
        return "easy"
    if mastery < 0.5:
        return "medium"
    if mastery < 0.75:
        return "hard"
    return "very_hard"


def _difficulty_distribution(mastery: float, source_bucket: str) -> Dict[str, int]:
    source_rank_map = {"easy": 0, "medium": 1, "hard": 2, "very_hard": 3}
    rank_to_bucket = {0: "easy", 1: "medium", 2: "hard", 3: "very_hard"}
    source_rank = source_rank_map.get(source_bucket, 1)
    if mastery < 0.33:
        weights = [0.55, 0.30, 0.15]
        center = max(0, source_rank - 1)
    elif mastery < 0.66:
        weights = [0.20, 0.55, 0.25]
        center = source_rank
    else:
        weights = [0.15, 0.35, 0.50]
        center = min(3, source_rank + 1)

    buckets = [max(0, min(3, center - 1)), center, max(0, min(3, center + 1))]
    counts = {}
    total = 0
    base_total = 2 + int(round(mastery * 3))
    for idx, rank in enumerate(buckets):
        count = int(round(base_total * weights[idx]))
        if count > 0:
            counts[rank_to_bucket[rank]] = counts.get(rank_to_bucket[rank], 0) + count
            total += count

    # Ensure at least two samples and keep totals modest.
    if total < 2:
        counts[rank_to_bucket[center]] = counts.get(rank_to_bucket[center], 0) + (2 - total)
    elif total > 6:
        overflow = total - 6
        for bucket in sorted(counts.keys(), key=lambda b: counts[b], reverse=True):
            if overflow <= 0:
                break
            take = min(overflow, max(0, counts[bucket] - 1))
            counts[bucket] -= take
            overflow -= take
    return counts


def build_mastery_records(records: Sequence[Dict[str, Any]], source_lookup: Dict[Any, Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = _group_by_source(records)
    outputs: List[Dict[str, Any]] = []
    for source_task_id, items in grouped.items():
        source = source_lookup.get(source_task_id, {})
        source_bucket = source.get("knowledge", {}).get("difficulty_bucket", "medium")
        answer_accuracy = sum(1 for item in items if item.get("final_answer_correct")) / max(1, len(items))
        step_score_mean = sum(float(item.get("step_score_mean", 0.0)) for item in items) / max(1, len(items))
        step_quality = max(0.0, min(1.0, step_score_mean))
        answer_scores = [max(0.0, min(10.0, float(item.get("step_score_mean", 0.0)) * 10.0)) for item in items]
        is_correct = [bool(item.get("final_answer_correct")) for item in items]
        mastery_pack = _compute_mastery(answer_scores, is_correct)
        mastery = max(0.0, min(1.0, float(mastery_pack["mastery"])))
        source_solution_steps = (
            source.get("solution_text")
            or source.get("solution_steps")
            or source.get("knowledge", {}).get("solution_text")
            or ""
        )
        outputs.append(
            {
                "task_id": source_task_id,
                "accuracy": round(answer_accuracy, 4),
                "mastery": round(mastery, 4),
                "question": source.get("question", ""),
                "solution_steps": source_solution_steps,
                "answer": source.get("answer", ""),
            }
        )
    return sorted(outputs, key=lambda x: x["task_id"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Answer seed questions and score step quality.")
    parser.add_argument("--mode", choices=["answer", "score", "all"], default="answer", help="Pipeline stage to run")
    parser.add_argument("--input", required=True, help="Seed JSONL path for answer mode, or answer JSONL path for score mode")
    parser.add_argument("--seed-input", required=False, help="Seed JSONL path for score mode")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-answers", type=int, default=4, help="Number of answers per question")
    parser.add_argument("--model", required=False, help="Model id/path to use for this assessment stage")
    parser.add_argument("--temperature", type=float, default=None, help="Victim answer sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Victim answer nucleus sampling top_p")
    parser.add_argument("--answer-output", required=False, help="Victim answer JSONL path")
    parser.add_argument("--answer-raw-output", required=False, help="Raw victim answer JSONL path")
    parser.add_argument("--step-output", required=False, help="Step evaluation JSONL path")
    parser.add_argument("--mastery-record-output", required=False, help="Detailed mastery JSONL path")
    parser.add_argument("--mastery-output", required=False, help="Mastery JSON path")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--checkpoint-every", type=int, required=False)
    parser.add_argument("--synthesis-target-multiplier", type=int, required=False)
    parser.add_argument("--synthesis-min-per-seed", type=int, required=False)
    parser.add_argument("--synthesis-max-per-seed", type=int, required=False)
    parser.add_argument("--synthesis-balance-lambda", type=float, required=False)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = Path(args.answer_output) if args.answer_output else output_dir / "victim_answers.jsonl"
    answer_raw_path = Path(args.answer_raw_output) if args.answer_raw_output else output_dir / "victim_answers.raw.jsonl"
    step_path = Path(args.step_output) if args.step_output else output_dir / "step_evaluations.jsonl"
    mastery_record_path = Path(args.mastery_record_output) if args.mastery_record_output else output_dir / "mastery_records.jsonl"
    mastery_path = Path(args.mastery_output) if args.mastery_output else output_dir / "mastery.json"
    if args.mode in {"answer", "all"}:
        records = read_jsonl(input_path)
        answer_model = args.model or os.environ.get("VICTIM_MODEL") or os.environ.get("VLLM_MODEL")
        answer_client = VLLMClient(model=answer_model) if answer_model else None
        answer_resume = (
            _parse_bool_env("ANSWER_RESUME", True)
            if args.resume is None
            else bool(args.resume)
        )
        existing_answers = (
            _read_jsonl_tolerant(answer_raw_path)
            if answer_resume and answer_raw_path.exists()
            else []
        )
        answers = answer_questions(
            records,
            n_answers=args.n_answers,
            client=answer_client,
            temperature=args.temperature,
            top_p=args.top_p,
            existing_records=existing_answers,
            answer_output_path=answer_path,
            answer_raw_output_path=answer_raw_path,
            checkpoint_every=args.checkpoint_every,
        )
        write_jsonl(answer_path, [project_victim_answer_record(record) for record in answers])
        write_jsonl(answer_raw_path, [project_victim_answer_raw_record(record) for record in answers])
        if args.mode == "answer":
            print(
                json.dumps(
                    {
                        "answer_output": str(answer_path),
                        "answer_raw_output": str(answer_raw_path),
                        "seed_count": len(records),
                        "answer_count": len(answers),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

    if args.mode in {"score", "all"}:
        if not args.seed_input:
            raise ValueError("--seed-input is required in score/all mode")
        seed_records = read_jsonl(Path(args.seed_input))
        answer_records = read_jsonl(input_path)
        score_model = args.model or os.environ.get("STEP_MODEL") or os.environ.get("VLLM_MODEL")
        score_client = VLLMClient(model=score_model) if score_model else None
        step_checkpoint_path = step_path.with_name(
            step_path.name + ".partial"
        )
        step_reports = evaluate_answers(
            answer_records,
            client=score_client,
            checkpoint_path=step_checkpoint_path,
        )
        write_jsonl(step_path, step_reports)
        step_checkpoint_path.unlink(missing_ok=True)

        source_lookup = {record.get("task_id"): record for record in seed_records}
        mastery_records = build_mastery_records(step_reports, source_lookup)
        target_multiplier = args.synthesis_target_multiplier if args.synthesis_target_multiplier is not None else _parse_int_env("SYNTHESIS_TARGET_MULTIPLIER", default=26)
        n_min = args.synthesis_min_per_seed if args.synthesis_min_per_seed is not None else _parse_int_env("SYNTHESIS_MIN_PER_SEED", default=10)
        n_max = args.synthesis_max_per_seed if args.synthesis_max_per_seed is not None else _parse_int_env("SYNTHESIS_MAX_PER_SEED", default=50)
        lambda_balance = args.synthesis_balance_lambda if args.synthesis_balance_lambda is not None else _parse_float_env("SYNTHESIS_BALANCE_LAMBDA", default=0.3)
        mastery_records = distribute_mastery_records(
            mastery_records,
            source_lookup,
            target_multiplier=target_multiplier,
            n_min=n_min,
            n_max=n_max,
            lambda_balance=lambda_balance,
        )
        write_jsonl(mastery_record_path, mastery_records)
        write_json(mastery_path, mastery_records)

        print(
            json.dumps(
                {
                    "step_output": str(step_path),
                    "mastery_record_output": str(mastery_record_path),
                    "mastery_output": str(mastery_path),
                    "seed_count": len(seed_records),
                    "answer_count": len(answer_records),
                    "mastery_count": len(mastery_records),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
