from __future__ import annotations

"""Step-refinement stage for validated synthetic math data.

This module is intentionally downstream of mathematical validation. It assumes
that ``question``, ``answer``, and the solution path are already correct, and it
asks the model to rewrite only ``steps`` into dependency-aware training targets.
All public entrypoints preserve every non-step field from the input record.
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .post_mastery_generate import (
    _decode_json_candidate,
    _format_seconds,
    _normalize_steps,
    _parse_bool_env,
    _parse_float_env,
    _parse_int_env,
    _unwrap_payload,
)
from .utils import normalize_whitespace, read_jsonl, write_json, write_jsonl


STEP_LABEL_RE = re.compile(r"^\s*(?:step\s*)?\d+\s*[:.)-]\s*", re.IGNORECASE)
MECHANICAL_START_RE = re.compile(
    r"^\s*(?:step\s*)?\d*\s*[:.)-]?\s*(?:calculate|compute|find)\b",
    re.IGNORECASE,
)
CONNECTIVE_RE = re.compile(
    r"\b(?:from the problem|given|since|because|so|therefore|this means|"
    r"combining|using|after|remaining|total|needed|left|altogether|"
    r"determine|problem states|to find|let|this gives|resulting|gives|"
    r"with|subtract|divide|multiply|adding|add|by subtracting|"
    r"by multiplying|by dividing|by adding)\b",
    re.IGNORECASE,
)
EXPLANATORY_MECHANICAL_RE = re.compile(
    r"\b(?:by|because|since|from|using|with|given|to find|to determine|"
    r"so|therefore|resulting|gives)\b",
    re.IGNORECASE,
)


def _model_aliases(model: str) -> set[str]:
    """Return path, basename, and normalized aliases for model matching."""
    normalized = str(model).strip().rstrip("/")
    if not normalized:
        return set()
    aliases = {normalized}
    basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    if basename:
        aliases.add(basename)
    return aliases


async def _resolve_served_model_name(client: Any, model: str) -> str:
    """Use the actually served model id when it matches the configured model."""
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


def _json_message(system: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build a compact two-message JSON-oriented chat request."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _refine_prompt(record: Dict[str, Any], retry_reason: str = "") -> List[Dict[str, str]]:
    """Prompt the model to rewrite only solution steps.

    The prompt locks question and answer and repeatedly states the non-step
    fields must not change. The parser below also ignores all returned fields
    except ``steps`` as a second layer of protection.
    """
    payload = {
        "task": (
            "Rewrite only the solution steps so they are better supervised-fine-tuning "
            "targets. The math is already validated; preserve the same solution path."
        ),
        "locked_fields": {
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
        },
        "current_steps": _normalize_steps(record.get("steps", [])),
        "rules": [
            "Return only JSON with a single key: steps.",
            "Do not change the question, answer, difficulty, numbers, or mathematical solution path.",
            "Do not add new assumptions, new quantities, alternative methods, or trial-and-error.",
            "Each output step must start with Step 1:, Step 2:, and so on.",
            "Each step must explain why its equation is relevant: it may use a condition from the problem, one previous result, or several independently computed quantities.",
            "Do not pretend every step depends only on the immediately previous step. Independent intermediate quantities are allowed before they are combined.",
            "Name the intermediate value produced by each calculation, especially if it is used later.",
            "Use concise GSM8K-style connectors such as From the problem, Since, So, Therefore, or Combining these quantities.",
            "Avoid mechanical step openings like only 'Calculate ...'.",
            "Prefer one main inference or equation per step. Split packed semicolon computations when needed.",
            "Keep the steps concise; improve clarity without adding verbose commentary.",
        ],
        "output_schema": {"steps": ["Step 1: ...", "Step 2: ..."]},
    }
    if retry_reason:
        payload["previous_refine_error"] = retry_reason
        payload["retry_instruction"] = (
            "Fix only the step wording/format problem. Keep the same validated math."
        )
    return _json_message(
        (
            "You improve math solution steps for training data. You must preserve "
            "the validated problem and answer. Return only valid JSON."
        ),
        payload,
    )


def _parse_refined_steps(raw: str) -> Tuple[Optional[List[str]], str]:
    """Extract a normalized ``steps`` list from a model response."""
    payload = _unwrap_payload(_decode_json_candidate(raw))
    if not payload:
        return None, "response is not a valid JSON object"
    steps = _normalize_steps(payload.get("steps") or payload.get("solution_steps"))
    if not steps:
        return None, "missing steps"
    return steps, ""


def _strip_step_label(step: str) -> str:
    """Remove a leading Step-N label before style checks."""
    return STEP_LABEL_RE.sub("", step, count=1).strip()


def _step_quality_issue(steps: Sequence[str]) -> str:
    """Return a retry reason when refined steps are still poor training targets."""
    normalized = [normalize_whitespace(step) for step in steps if normalize_whitespace(step)]
    if not normalized:
        return "missing steps"
    for index, step in enumerate(normalized, start=1):
        if not re.match(rf"^\s*Step\s+{index}\s*:", step, flags=re.IGNORECASE):
            return f"step {index} missing expected Step {index}: label"

    mechanical = 0
    for step in normalized:
        body = _strip_step_label(step)
        if MECHANICAL_START_RE.search(step) and not EXPLANATORY_MECHANICAL_RE.search(body):
            mechanical += 1
    if mechanical / max(1, len(normalized)) > 0.5:
        return "too many steps still start mechanically"

    connective = sum(1 for step in normalized if CONNECTIVE_RE.search(_strip_step_label(step)))
    if len(normalized) >= 2 and connective / len(normalized) < 0.5:
        return "too few steps explain dependencies or reasoning purpose"

    packed = sum(1 for step in normalized if step.count(";") >= 2)
    if packed:
        return "some steps still pack multiple calculations with semicolons"
    return ""


def _ensure_step_labels(steps: Sequence[str]) -> List[str]:
    """Normalize numbering locally so label-only mistakes do not waste retries."""
    labeled: List[str] = []
    for index, step in enumerate(steps, start=1):
        body = _strip_step_label(normalize_whitespace(step))
        if body:
            labeled.append(f"Step {index}: {body}")
    return labeled


def _project_record(record: Dict[str, Any], steps: Sequence[str]) -> Dict[str, Any]:
    """Copy an input record while replacing only the ``steps`` field."""
    output = dict(record)
    output["steps"] = _ensure_step_labels(steps)
    return output


def _record_key(record: Dict[str, Any], index: int) -> str:
    """Stable resume key, preferring plan_id and falling back to source/index."""
    plan_id = record.get("plan_id")
    if plan_id not in (None, ""):
        return str(plan_id)
    source_task_id = record.get("source_task_id", "")
    question = normalize_whitespace(record.get("question", ""))[:80]
    return f"{source_task_id}:{index}:{question}"


def _round_directory(output_path: Path) -> Path:
    """Directory for per-round refinement logs."""
    return output_path.parent / "refine.rounds"


def _ordered_accepted(
    records: Sequence[Dict[str, Any]],
    accepted: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return accepted records in the same order as the validated input."""
    ordered: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        key = _record_key(record, index)
        if key in accepted:
            ordered.append(accepted[key])
    return ordered


def _write_round_log(
    *,
    output_path: Path,
    round_index: int,
    round_inputs: Sequence[Dict[str, Any]],
    round_successes: Sequence[Dict[str, Any]],
    round_raw_outputs: Sequence[Dict[str, Any]],
    round_failures: Sequence[Dict[str, Any]],
    next_pending_count: int,
) -> None:
    """Persist inspectable artifacts for one completed refinement round."""
    round_dir = _round_directory(output_path)
    round_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"round_{round_index:03d}"
    write_jsonl(round_dir / f"{prefix}.input.jsonl", round_inputs)
    write_jsonl(round_dir / f"{prefix}.success.jsonl", round_successes)
    write_jsonl(round_dir / f"{prefix}.raw.jsonl", round_raw_outputs)
    write_jsonl(round_dir / f"{prefix}.failed.jsonl", round_failures)
    write_json(
        round_dir / f"{prefix}.summary.json",
        {
            "round": round_index,
            "attempted": len(round_inputs),
            "success": len(round_successes),
            "failed": len(round_failures),
            "next_round": next_pending_count,
            "failure_types": {
                error: sum(
                    1
                    for item in round_failures
                    if str(item.get("error") or "unknown") == error
                )
                for error in sorted(
                    {
                        str(item.get("error") or "unknown")
                        for item in round_failures
                    }
                )
            },
        },
    )


async def refine_solution_steps_async(
    records: Sequence[Dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    output_path: Path,
    failed_path: Path,
    raw_path: Path,
    summary_path: Path,
    concurrency: int,
    timeout: int,
    max_rounds: int,
    max_tokens: int,
    enable_thinking: bool,
    force_json: bool,
    checkpoint_every: int,
    resume: bool,
    progress_every: int,
    progress_interval: float,
) -> List[Dict[str, Any]]:
    """Refine records asynchronously with checkpoint/resume support.

    The function checkpoints accepted outputs and raw responses periodically.
    ``failed_path`` is cleared at the start of each round and rewritten with
    only the current round's failures so logs do not grow without bound. A
    separate ``refine.rounds`` directory keeps full per-round input, success,
    raw, failed, and summary files for later debugging.
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

    accepted: Dict[str, Dict[str, Any]] = {}
    raw_records: List[Dict[str, Any]] = []
    if resume and output_path.exists():
        for index, record in enumerate(read_jsonl(output_path)):
            accepted[_record_key(record, index)] = record
    if resume and raw_path.exists():
        raw_records = read_jsonl(raw_path)

    recovered_from_failed = 0
    if resume and failed_path.exists():
        for failure in read_jsonl(failed_path):
            if not isinstance(failure, dict):
                continue
            key = str(failure.get("key") or "")
            record = failure.get("record")
            raw = str(failure.get("raw_model_output") or "")
            if not key or key in accepted or not isinstance(record, dict) or not raw:
                continue
            steps, error = _parse_refined_steps(raw)
            if steps is not None:
                steps = _ensure_step_labels(steps)
                error = _step_quality_issue(steps)
            if error:
                continue
            accepted[key] = _project_record(record, steps or [])
            raw_records.append(
                {
                    "key": key,
                    "index": failure.get("index"),
                    "round": failure.get("round"),
                    "error": "",
                    "raw_model_output": raw,
                    "local_skip_reason": "recovered_from_failed_raw_output",
                }
            )
            recovered_from_failed += 1

    pending: List[Dict[str, Any]] = []
    locally_accepted = 0
    for index, record in enumerate(records):
        key = _record_key(record, index)
        if key in accepted:
            continue
        existing_steps = _ensure_step_labels(_normalize_steps(record.get("steps", [])))
        if existing_steps and not _step_quality_issue(existing_steps):
            accepted[key] = _project_record(record, existing_steps)
            raw_records.append(
                {
                    "key": key,
                    "index": index,
                    "round": -1,
                    "error": "",
                    "raw_model_output": "",
                    "local_skip_reason": "existing_steps_already_pass_quality_rules",
                }
            )
            locally_accepted += 1
            continue
        pending.append(
            {
                "index": index,
                "key": key,
                "record": record,
                "round": 0,
                "last_error": "",
            }
        )

    print(
        f"[refine_steps] config model={model} input={len(records)} "
        f"resume_done={len(accepted)} pending={len(pending)} "
        f"local_skip={locally_accepted} recovered_failed={recovered_from_failed} "
        f"concurrency={concurrency} max_rounds={max_rounds} "
        f"max_tokens={max_tokens}",
        flush=True,
    )

    async def request_refine(item: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
        request: Dict[str, Any] = {
            "model": model,
            "messages": _refine_prompt(item["record"], item.get("last_error", "")),
            "temperature": _parse_float_env("REFINE_TEMPERATURE", 0.2),
            "top_p": _parse_float_env("REFINE_TOP_P", 0.9),
            "max_tokens": max_tokens,
        }
        if force_json:
            request["response_format"] = {"type": "json_object"}
        if not enable_thinking:
            request["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        try:
            async with semaphore:
                response = await client.chat.completions.create(**request)
            raw = response.choices[0].message.content or ""
            steps, error = _parse_refined_steps(raw)
            if steps is not None:
                steps = _ensure_step_labels(steps)
                error = _step_quality_issue(steps)
            if error:
                return item["key"], None, {
                    "index": item["index"],
                    "round": item["round"],
                    "error": error,
                    "raw_model_output": raw,
                }
            assert steps is not None
            refined = _project_record(item["record"], steps)
            return item["key"], refined, {
                "index": item["index"],
                "round": item["round"],
                "error": "",
                "raw_model_output": raw,
            }
        except Exception as exc:
            return item["key"], None, {
                "index": item["index"],
                "round": item["round"],
                "error": f"{type(exc).__name__}: {exc}",
                "raw_model_output": "",
            }

    def checkpoint() -> None:
        write_jsonl(output_path, _ordered_accepted(records, accepted))
        write_jsonl(raw_path, raw_records)

    round_index = 0
    final_failed: List[Dict[str, Any]] = []
    infinite = max_rounds < 0
    while pending and (infinite or round_index <= max_rounds):
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.write_text("", encoding="utf-8")
        total = len(pending)
        pending_by_key = {item["key"]: item for item in pending}
        started = time.time()
        print(f"[refine_steps] round={round_index} pending={total}", flush=True)
        tasks = [asyncio.create_task(request_refine(item)) for item in pending]
        round_inputs = [
            {
                "key": item["key"],
                "index": item["index"],
                "round": round_index,
                "last_error": item.get("last_error", ""),
                "record": item["record"],
            }
            for item in pending
        ]
        round_successes: List[Dict[str, Any]] = []
        round_raw_outputs: List[Dict[str, Any]] = []
        round_failures: List[Dict[str, Any]] = []
        next_pending: List[Dict[str, Any]] = []
        completed = 0
        ok = 0
        errors = 0
        last_log = started
        for task in asyncio.as_completed(tasks):
            key, refined, meta = await task
            completed += 1
            raw_records.append(
                {
                    "key": key,
                    "index": meta.get("index"),
                    "round": meta.get("round"),
                    "error": meta.get("error", ""),
                    "raw_model_output": meta.get("raw_model_output", ""),
                }
            )
            round_raw_record = {
                "key": key,
                "index": meta.get("index"),
                "round": meta.get("round"),
                "error": meta.get("error", ""),
                "raw_model_output": meta.get("raw_model_output", ""),
            }
            round_raw_outputs.append(round_raw_record)
            if refined is not None:
                accepted[key] = refined
                round_successes.append(refined)
                ok += 1
            else:
                errors += 1
                item = pending_by_key[key]
                retry_item = {
                    **item,
                    "round": round_index + 1,
                    "last_error": str(meta.get("error") or "refine failed"),
                }
                next_pending.append(retry_item)
                failure_record = {
                    "key": key,
                    "index": item["index"],
                    "round": round_index,
                    "error": retry_item["last_error"],
                    "record": item["record"],
                    "raw_model_output": meta.get("raw_model_output", ""),
                }
                round_failures.append(failure_record)
                with failed_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(failure_record, ensure_ascii=False) + "\n")
            if checkpoint_every > 0 and completed % checkpoint_every == 0:
                checkpoint()
            now = time.time()
            if (
                completed <= 5
                or completed == total
                or completed % max(1, progress_every) == 0
                or now - last_log >= progress_interval
            ):
                elapsed = now - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = _format_seconds((total - completed) / rate) if rate > 0 else "--:--"
                print(
                    f"[refine_steps] round={round_index} {completed}/{total} "
                    f"ok={ok} error={errors} rate={rate:.2f}/s "
                    f"elapsed={_format_seconds(elapsed)} eta={eta}",
                    flush=True,
                )
                last_log = now
        checkpoint()
        _write_round_log(
            output_path=output_path,
            round_index=round_index,
            round_inputs=round_inputs,
            round_successes=round_successes,
            round_raw_outputs=round_raw_outputs,
            round_failures=round_failures,
            next_pending_count=len(next_pending),
        )
        if not next_pending:
            final_failed = []
            break
        final_failed = [
            {
                "key": item["key"],
                "index": item["index"],
                "round": round_index,
                "error": item.get("last_error", ""),
                "record": item["record"],
            }
            for item in next_pending
        ]
        pending = next_pending
        round_index += 1

    if final_failed:
        write_jsonl(failed_path, final_failed)
    else:
        failed_path.write_text("", encoding="utf-8")
    checkpoint()
    ordered = [
        accepted[_record_key(record, index)]
        for index, record in enumerate(records)
        if _record_key(record, index) in accepted
    ]
    summary = {
        "input_count": len(records),
        "output_count": len(ordered),
        "failed_count": len(final_failed),
        "pass_rate": round(len(ordered) / max(1, len(records)), 4),
        "output": str(output_path),
        "failed_output": str(failed_path),
        "raw_output": str(raw_path),
        "round_output_dir": str(_round_directory(output_path)),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return ordered


def refine_solution_steps(
    records: Sequence[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    output_path: Path,
    failed_path: Path,
    raw_path: Path,
    summary_path: Path,
) -> List[Dict[str, Any]]:
    """Synchronous wrapper used by the stage script and tests."""
    return asyncio.run(
        refine_solution_steps_async(
            records,
            model=model
            or os.environ.get("REFINE_MODEL")
            or os.environ.get("REPAIR_MODEL")
            or os.environ.get("QC_MODEL")
            or os.environ.get("VLLM_MODEL")
            or "",
            base_url=base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8911/v1"),
            api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
            output_path=output_path,
            failed_path=failed_path,
            raw_path=raw_path,
            summary_path=summary_path,
            concurrency=_parse_int_env("REFINE_CONCURRENCY", 128),
            timeout=_parse_int_env("VLLM_TIMEOUT", 600),
            max_rounds=_parse_int_env("REFINE_MAX_ROUNDS", -1),
            max_tokens=_parse_int_env("REFINE_MAX_TOKENS", 900),
            enable_thinking=_parse_bool_env("REFINE_ENABLE_THINKING", False),
            force_json=_parse_bool_env("REFINE_FORCE_JSON", True),
            checkpoint_every=_parse_int_env("REFINE_CHECKPOINT_EVERY", 50),
            resume=_parse_bool_env("REFINE_RESUME", True),
            progress_every=_parse_int_env("REFINE_PROGRESS_EVERY", 20),
            progress_interval=_parse_float_env("REFINE_PROGRESS_INTERVAL", 10.0),
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for ``run/refine_solution_steps.py``."""
    parser = argparse.ArgumentParser(description="Refine validated solution steps for training.")
    parser.add_argument("--input", required=True, help="Validated JSONL path")
    parser.add_argument("--output", required=True, help="Refined JSONL path")
    parser.add_argument("--failed-output", required=True, help="Per-round failed JSONL path")
    parser.add_argument("--raw-output", required=True, help="Raw refine responses JSONL path")
    parser.add_argument("--summary-output", required=True, help="Summary JSON path")
    parser.add_argument("--model", required=False, help="OpenAI-compatible model name")
    args = parser.parse_args(argv)

    records = read_jsonl(Path(args.input))
    refine_solution_steps(
        records,
        model=args.model,
        output_path=Path(args.output),
        failed_path=Path(args.failed_output),
        raw_path=Path(args.raw_output),
        summary_path=Path(args.summary_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
