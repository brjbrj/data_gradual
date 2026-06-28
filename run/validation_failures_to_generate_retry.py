from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_pipeline.post_mastery_plan import replan_failed_plan
from kb_pipeline.utils import read_jsonl, write_jsonl


def _plan_id(record: Dict[str, Any]) -> str:
    return str(record.get("plan_id") or "")


def _failure_type(record: Dict[str, Any]) -> str:
    report = record.get("validation_report")
    if isinstance(report, dict):
        error_type = str(report.get("error_type") or "").strip()
        if error_type:
            return f"validation_{error_type}"
        action = str(report.get("repair_action") or "").strip()
        if action:
            return f"validation_{action}"
    return "validation_failed"


def _retry_round(record: Dict[str, Any]) -> int:
    try:
        return int(record.get("round") or 0) + 1
    except Exception:
        return 1


def _select_plan(
    record: Dict[str, Any],
    plan_lookup: Dict[str, Dict[str, Any]],
    retry_round: int,
) -> Optional[Dict[str, Any]]:
    for key in ("next_plan", "active_plan"):
        value = record.get(key)
        if isinstance(value, dict) and value:
            return value
    plan = plan_lookup.get(_plan_id(record))
    if not isinstance(plan, dict) or not plan:
        return None
    return replan_failed_plan(
        plan,
        _failure_type(record),
        retry_round=retry_round,
    )


def convert_validation_failures(
    validation_failed: Path,
    plan_path: Path,
) -> list[Dict[str, Any]]:
    failures = read_jsonl(validation_failed) if validation_failed.exists() else []
    plans = read_jsonl(plan_path)
    plan_lookup = {_plan_id(plan): plan for plan in plans if _plan_id(plan)}
    outputs: list[Dict[str, Any]] = []
    for record in failures:
        plan_id = _plan_id(record)
        if not plan_id:
            continue
        retry_round = _retry_round(record)
        next_plan = _select_plan(record, plan_lookup, retry_round)
        if not isinstance(next_plan, dict) or not next_plan:
            continue
        outputs.append(
            {
                "source_task_id": record.get("source_task_id"),
                "plan_id": plan_id,
                "difficulty": record.get("difficulty"),
                "round": int(record.get("round") or 0),
                "attempts": retry_round,
                "failure_type": _failure_type(record),
                "next_action": "reuse_plan",
                "error": "validation failed; retrying generation",
                "candidate_question": record.get("question", ""),
                "raw_model_output": "",
                "validation_failed": record,
                "plan": plan_lookup.get(plan_id, record.get("active_plan") or {}),
                "next_plan": next_plan,
                "next_round": retry_round,
            }
        )
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert validation failures into generate-stage retry records."
    )
    parser.add_argument("--validation-failed", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    outputs = convert_validation_failures(
        Path(args.validation_failed),
        Path(args.plan),
    )
    write_jsonl(Path(args.output), outputs)
    print(
        json.dumps(
            {
                "validation_failed": args.validation_failed,
                "output": args.output,
                "retry_count": len(outputs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
