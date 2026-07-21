from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kb_pipeline.distribute import (  # noqa: E402
    DIFFICULTY_STEP_RANGES,
    DIFFICULTY_TO_BUCKET,
    distribute_mastery_records,
)
from kb_pipeline.utils import read_jsonl, write_json, write_jsonl  # noqa: E402


DIFFICULTY_CHOICES = ["Easy", "Slightly Easy", "Equal", "Slightly Hard", "Hard"]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if not math.isfinite(result):
        return default
    return result


def _is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "correct"}


def _source_lookup(seed_records: Sequence[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    lookup: Dict[Any, Dict[str, Any]] = {}
    for record in seed_records:
        task_id = record.get("task_id")
        lookup[task_id] = dict(record)
        lookup[str(task_id)] = dict(record)
    return lookup


def _answers_by_source(answer_records: Sequence[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for record in answer_records:
        grouped[str(record.get("source_task_id", record.get("task_id")))].append(record)
    return grouped


def build_answer_accuracy_mastery(
    answer_records: Sequence[Dict[str, Any]],
    seed_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build mastery records from final-answer correctness only.

    This intentionally ignores step-level judge scores. It is used for the
    ablation that asks whether step-based mastery contributes useful signal.
    """
    sources = {record.get("task_id"): dict(record) for record in seed_records}
    grouped = _answers_by_source(answer_records)
    outputs: List[Dict[str, Any]] = []
    for task_id, source in sources.items():
        items = grouped.get(str(task_id), [])
        if not items:
            accuracy = 0.0
        else:
            accuracy = sum(1 for item in items if _is_correct(item.get("is_correct"))) / len(items)
        source_solution_steps = (
            source.get("solution_text")
            or source.get("solution_steps")
            or source.get("knowledge", {}).get("solution_text")
            or ""
        )
        outputs.append(
            {
                "task_id": task_id,
                "accuracy": round(accuracy, 4),
                "answer_accuracy": round(accuracy, 4),
                "mastery": round(accuracy, 4),
                "mastery_source": "answer_accuracy_only",
                "step_score_mean": None,
                "step_quality": None,
                "question": source.get("question", ""),
                "solution_steps": source_solution_steps,
                "answer": source.get("answer", ""),
            }
        )
    return sorted(outputs, key=lambda item: str(item.get("task_id")))


def _with_difficulty(record: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
    updated = dict(record)
    updated["target_difficulty"] = difficulty
    updated["target_difficulty_bucket"] = DIFFICULTY_TO_BUCKET[difficulty]
    updated["target_step_count_range"] = DIFFICULTY_STEP_RANGES[difficulty]
    return updated


def _uniform_count(records: Sequence[Dict[str, Any]], requested: Optional[int]) -> int:
    if requested is not None:
        return max(0, int(requested))
    counts = [max(0, int(_to_float(item.get("target_count"), 0.0))) for item in records]
    if not counts:
        return 0
    return max(0, int(round(sum(counts) / len(counts))))


def apply_variant(
    mastery_records: Sequence[Dict[str, Any]],
    *,
    variant: str,
    uniform_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    records = [dict(item) for item in mastery_records]
    if variant in {"hard_all", "equal_all", "easy_all"}:
        difficulty = {
            "hard_all": "Hard",
            "equal_all": "Equal",
            "easy_all": "Easy",
        }[variant]
        return [
            {
                **_with_difficulty(item, difficulty),
                "ablation_variant": variant,
                "ablation_note": "target_count preserved; target difficulty overridden",
            }
            for item in records
        ]

    if variant == "uniform_count":
        count = _uniform_count(records, uniform_count)
        outputs = []
        for item in records:
            updated = dict(item)
            updated["target_count"] = count
            updated["ablation_variant"] = variant
            updated["ablation_note"] = "target difficulty preserved; target_count set equal for every seed"
            outputs.append(updated)
        return outputs

    if variant == "identity":
        return [
            {
                **item,
                "ablation_variant": variant,
                "ablation_note": "original mastery distribution copied unchanged",
            }
            for item in records
        ]

    raise ValueError(f"Unsupported variant: {variant}")


def _summarize(records: Sequence[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    difficulty_counts: Dict[str, int] = {}
    bucket_counts: Dict[str, int] = {}
    total_target_count = 0
    mastery_values: List[float] = []
    for item in records:
        count = max(0, int(_to_float(item.get("target_count"), 0.0)))
        total_target_count += count
        difficulty = str(item.get("target_difficulty") or "")
        bucket = str(item.get("target_difficulty_bucket") or "")
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + count
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + count
        mastery_values.append(max(0.0, min(1.0, _to_float(item.get("mastery"), 0.0))))
    return {
        "variant": variant,
        "seed_count": len(records),
        "total_target_count": total_target_count,
        "difficulty_counts_by_generated_count": difficulty_counts,
        "bucket_counts_by_generated_count": bucket_counts,
        "mastery_min": min(mastery_values) if mastery_values else None,
        "mastery_max": max(mastery_values) if mastery_values else None,
        "mastery_mean": (sum(mastery_values) / len(mastery_values)) if mastery_values else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build isolated ablation mastery variants.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "answer_accuracy_only",
            "hard_all",
            "equal_all",
            "easy_all",
            "uniform_count",
            "identity",
        ],
    )
    parser.add_argument("--mastery", help="Original mastery JSONL path for distribution ablations.")
    parser.add_argument("--answers", help="Victim answer raw JSONL path for answer_accuracy_only.")
    parser.add_argument("--seed-input", required=True, help="KB records JSONL path.")
    parser.add_argument("--output", required=True, help="Output mastery JSONL path.")
    parser.add_argument("--summary-output", help="Optional summary JSON path.")
    parser.add_argument("--uniform-count", type=int, help="Count to use for uniform_count; default is rounded original mean.")
    parser.add_argument("--synthesis-target-multiplier", type=int, default=26)
    parser.add_argument("--synthesis-min-per-seed", type=int, default=10)
    parser.add_argument("--synthesis-max-per-seed", type=int, default=50)
    parser.add_argument("--synthesis-balance-lambda", type=float, default=0.3)
    parser.add_argument("--synthesis-allocation-policy", default=None)
    parser.add_argument("--synthesis-active-threshold", type=int, default=None)
    parser.add_argument("--synthesis-marginal-alpha", type=float, default=None)
    parser.add_argument("--synthesis-threshold-boost", type=float, default=None)
    parser.add_argument("--synthesis-cold-start-factor", type=float, default=None)
    args = parser.parse_args(argv)

    seed_records = read_jsonl(Path(args.seed_input))
    source_lookup = _source_lookup(seed_records)

    if args.variant == "answer_accuracy_only":
        if not args.answers:
            raise ValueError("--answers is required for answer_accuracy_only")
        base_records = build_answer_accuracy_mastery(read_jsonl(Path(args.answers)), seed_records)
        outputs = distribute_mastery_records(
            base_records,
            source_lookup,
            target_multiplier=args.synthesis_target_multiplier,
            n_min=args.synthesis_min_per_seed,
            n_max=args.synthesis_max_per_seed,
            lambda_balance=args.synthesis_balance_lambda,
            allocation_policy=args.synthesis_allocation_policy or os.environ.get("SYNTHESIS_ALLOCATION_POLICY", "legacy"),
            active_threshold=args.synthesis_active_threshold if args.synthesis_active_threshold is not None else int(os.environ.get("SYNTHESIS_ACTIVE_THRESHOLD", "0") or 0),
            marginal_alpha=args.synthesis_marginal_alpha if args.synthesis_marginal_alpha is not None else float(os.environ.get("SYNTHESIS_MARGINAL_ALPHA", "0.7") or 0.7),
            threshold_boost=args.synthesis_threshold_boost if args.synthesis_threshold_boost is not None else float(os.environ.get("SYNTHESIS_THRESHOLD_BOOST", "2.0") or 2.0),
            cold_start_factor=args.synthesis_cold_start_factor if args.synthesis_cold_start_factor is not None else float(os.environ.get("SYNTHESIS_COLD_START_FACTOR", "0.0") or 0.0),
        )
        outputs = [
            {
                **item,
                "ablation_variant": args.variant,
                "ablation_note": "mastery and allocation computed from final-answer accuracy only",
            }
            for item in outputs
        ]
    else:
        if not args.mastery:
            raise ValueError("--mastery is required for this variant")
        outputs = apply_variant(
            read_jsonl(Path(args.mastery)),
            variant=args.variant,
            uniform_count=args.uniform_count,
        )

    output_path = Path(args.output)
    write_jsonl(output_path, outputs)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    write_json(summary_path, _summarize(outputs, args.variant))
    print(
        json.dumps(
            {
                "variant": args.variant,
                "output": str(output_path),
                "summary": str(summary_path),
                "seed_count": len(outputs),
                "total_target_count": sum(max(0, int(_to_float(item.get("target_count"), 0.0))) for item in outputs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
