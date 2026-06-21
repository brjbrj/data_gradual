from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .utils import read_json, read_jsonl, write_json, write_jsonl


BUCKET_ORDER = {"easy": 0, "medium": 1, "hard": 2, "very_hard": 3}
BUCKET_BY_INDEX = {0: "easy", 1: "medium", 2: "hard", 3: "very_hard"}


def _clamp_rank(rank: int) -> int:
    return max(0, min(3, rank))


def _shift_bucket(bucket: str, delta: int) -> str:
    return BUCKET_BY_INDEX[_clamp_rank(BUCKET_ORDER.get(bucket, 1) + delta)]


def _difficulty_profile(mastery_score: float, source_bucket: str) -> Dict[str, int]:
    """Return a compact distribution of target buckets per seed item.

    The distribution is intentionally modest to avoid duplicate-heavy expansion.
    """
    source_rank = _clamp_rank(BUCKET_ORDER.get(source_bucket, 1))
    lower_rank = _clamp_rank(source_rank - 1)
    upper_rank = _clamp_rank(source_rank + 1)
    upper2_rank = _clamp_rank(source_rank + 2)
    if mastery_score < 0.33:
        targets = [lower_rank, source_rank, source_rank]
    elif mastery_score < 0.66:
        targets = [lower_rank, source_rank, upper_rank]
    else:
        targets = [source_rank, upper_rank, upper_rank]
        if mastery_score > 0.83:
            targets.append(upper2_rank)

    counts: Dict[str, int] = {}
    for rank in targets:
        bucket = BUCKET_BY_INDEX[_clamp_rank(rank)]
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _difficulty_label_from_mastery(mastery_score: float) -> str:
    if mastery_score <= 0.2:
        return "Easy"
    if mastery_score <= 0.4:
        return "Slightly Easy"
    if mastery_score <= 0.6:
        return "Equal"
    if mastery_score <= 0.8:
        return "Slightly Hard"
    return "Hard"


def _candidate_modes_for_bucket(bucket: str) -> List[str]:
    if bucket == "easy":
        return ["surface_swap", "scene_swap"]
    if bucket == "medium":
        return ["same_structure", "scene_swap", "surface_swap"]
    if bucket == "hard":
        return ["same_structure", "entity_swap", "surface_swap"]
    return ["same_structure", "entity_swap"]


def _mode_for_bucket(bucket: str, index: int) -> str:
    modes = _candidate_modes_for_bucket(bucket)
    return modes[index % len(modes)]


def _stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _target_bucket_priority(target_difficulty: str, target_bucket: str) -> List[str]:
    label = str(target_difficulty or "").strip().lower()
    bucket = str(target_bucket or "medium").strip().lower()
    label_map = {
        "easy": ["easy", "medium"],
        "slightly easy": ["easy", "medium"],
        "equal": ["medium", "easy", "hard"],
        "slightly hard": ["hard", "medium", "very_hard"],
        "hard": ["very_hard", "hard"],
    }
    bucket_map = {
        "easy": ["easy", "medium"],
        "medium": ["medium", "easy", "hard"],
        "hard": ["hard", "medium", "very_hard"],
        "very_hard": ["very_hard", "hard"],
    }
    priorities = label_map.get(label) or bucket_map.get(bucket) or ["medium", "easy", "hard", "very_hard"]
    seen: List[str] = []
    for item in priorities:
        if item not in seen:
            seen.append(item)
    return seen


def _select_plan_source(
    synthesis_by_bucket: Dict[str, List[Dict[str, Any]]],
    mastery_task_id: Any,
    target_difficulty: str,
    target_bucket: str,
    variant_index: int,
) -> Dict[str, Any]:
    priorities = _target_bucket_priority(target_difficulty, target_bucket)
    seed_text = f"{mastery_task_id}|{target_difficulty}|{target_bucket}|{variant_index}"
    for bucket in priorities:
        candidates = synthesis_by_bucket.get(bucket, [])
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda item: (
                str(item.get("task_id")) == str(mastery_task_id),
                str(item.get("task_id")),
            ),
        )
        start = _stable_int(seed_text + f"|{bucket}") % len(ranked)
        for offset in range(len(ranked)):
            card = ranked[(start + offset) % len(ranked)]
            if str(card.get("task_id")) == str(mastery_task_id) and len(ranked) > 1:
                continue
            return card
    all_candidates: List[Dict[str, Any]] = []
    for bucket in priorities:
        all_candidates.extend(synthesis_by_bucket.get(bucket, []))
    if not all_candidates:
        return {}
    ranked = sorted(
        all_candidates,
        key=lambda item: (
            str(item.get("task_id")) == str(mastery_task_id),
            str(item.get("task_id")),
        ),
    )
    start = _stable_int(seed_text + "|fallback") % len(ranked)
    return ranked[start]


def build_generation_plan(
    mastery_records: Sequence[Dict[str, Any]],
    synthesis_plan: Sequence[Dict[str, Any]],
    source_lookup: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    synthesis_by_task = {item.get("task_id"): item for item in synthesis_plan}
    synthesis_by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for item in synthesis_plan:
        bucket = str(item.get("difficulty_bucket") or item.get("target_difficulty_bucket") or item.get("knowledge", {}).get("difficulty_bucket") or "medium")
        synthesis_by_bucket.setdefault(bucket, []).append(item)
    expanded: List[Dict[str, Any]] = []

    for mastery in mastery_records:
        task_id = mastery.get("task_id")
        synthesis = synthesis_by_task.get(task_id, {})
        source_record = source_lookup.get(task_id, {}) if source_lookup else {}
        seed_question = mastery.get("question") or source_record.get("question", "") or synthesis.get("question", "")
        seed_answer = mastery.get("answer") or source_record.get("answer", "") or synthesis.get("answer", "")
        seed_solution_steps = mastery.get("solution_steps") or source_record.get("solution_steps") or source_record.get("solution_text") or ""
        seed_knowledge = source_record.get("knowledge", {}) if isinstance(source_record, dict) else {}
        source_bucket = str(source_record.get("difficulty_bucket") or seed_knowledge.get("difficulty_bucket") or synthesis.get("difficulty_bucket") or "medium")
        mastery_score = float(mastery.get("mastery", mastery.get("mastery_score", 0.5)))
        target_count = int(mastery.get("target_count", 0) or 0)
        target_difficulty = mastery.get("target_difficulty") or _difficulty_label_from_mastery(mastery_score)
        target_bucket = mastery.get("target_difficulty_bucket")
        if not target_bucket:
            target_bucket = {
                "Easy": "easy",
                "Slightly Easy": "easy",
                "Equal": "medium",
                "Slightly Hard": "hard",
                "Hard": "very_hard",
            }.get(str(target_difficulty), source_bucket)
        step_range = mastery.get("target_step_count_range") or {
            "Easy": [1, 2],
            "Slightly Easy": [2, 3],
            "Equal": [2, 4],
            "Slightly Hard": [4, 6],
            "Hard": [6, 10],
        }.get(str(target_difficulty), [2, 4])

        variant_total = target_count if target_count > 0 else max(1, sum(_difficulty_profile(mastery_score, source_bucket).values()))
        for variant_index in range(variant_total):
            selected_card = _select_plan_source(
                synthesis_by_bucket,
                mastery_task_id=task_id,
                target_difficulty=str(target_difficulty),
                target_bucket=str(target_bucket),
                variant_index=variant_index,
            )
            selected_concepts = selected_card.get("concepts", {}) if isinstance(selected_card, dict) else {}
            selected_knowledge = selected_card.get("knowledge", {}) if isinstance(selected_card, dict) else {}
            selected_bucket = str(selected_card.get("difficulty_bucket") or selected_card.get("target_difficulty_bucket") or selected_knowledge.get("difficulty_bucket") or target_bucket)
            expanded.append(
                {
                    "task_id": f"{task_id}_{variant_index}",
                    "source_task_id": task_id,
                    "seed_task_id": task_id,
                    "variant_index": variant_index,
                    "source_question": seed_question,
                    "source_answer": seed_answer,
                    "source_solution_steps": seed_solution_steps,
                    "source_knowledge": seed_knowledge,
                    "source_difficulty_bucket": source_bucket,
                    "mastery_score": mastery_score,
                    "answer_accuracy": mastery.get("accuracy", mastery.get("answer_accuracy", 0.0)),
                    "step_score_mean": mastery.get("step_score_mean", mastery.get("step_mean_score", 0.0)),
                    "step_quality": mastery.get("step_quality", mastery.get("step_score_mean", 0.0)),
                    "target_count": variant_total,
                    "target_difficulty": target_difficulty,
                    "target_difficulty_bucket": target_bucket,
                    "target_step_count_range": step_range,
                    "mode": _mode_for_bucket(target_bucket, variant_index),
                    "plan_source_task_id": selected_card.get("task_id", ""),
                    "plan_source_question": selected_card.get("question", ""),
                    "plan_source_answer": selected_card.get("answer", ""),
                    "plan_source_scene_text": selected_card.get("scene_text", ""),
                    "plan_source_surface_template": selected_card.get("surface_template", ""),
                    "plan_source_scene_template": selected_card.get("scene_template", ""),
                    "plan_source_scenario_template": selected_card.get("scenario_template", ""),
                    "plan_source_concepts": selected_concepts,
                    "plan_source_knowledge": selected_knowledge,
                    "plan_source_difficulty_bucket": selected_bucket,
                    "question": selected_card.get("question", ""),
                    "answer": selected_card.get("answer", ""),
                    "solution_steps": selected_card.get("solution_text", ""),
                    "concepts": selected_concepts,
                    "knowledge": selected_knowledge,
                    "scene_text": selected_card.get("scene_text", ""),
                    "surface_template": selected_card.get("surface_template", ""),
                    "scene_template": selected_card.get("scene_template", ""),
                    "scenario_template": selected_card.get("scenario_template", ""),
                    "difficulty_bucket": selected_bucket,
                    "anchor_concepts": selected_concepts.get("all_terms", [])[:8] if isinstance(selected_concepts, dict) else [],
                    "anchor_persons": selected_concepts.get("persons", [])[:4] if isinstance(selected_concepts, dict) else [],
                    "anchor_terms": selected_concepts.get("focus_terms", [])[:8] if isinstance(selected_concepts, dict) else [],
                    "anchor_units": selected_concepts.get("units", [])[:6] if isinstance(selected_concepts, dict) else [],
                    "knowledge_signature": selected_knowledge.get("knowledge_signature", ""),
                    "operation_sequence": selected_knowledge.get("operation_sequence", []),
                    "selection_reason": {
                        "candidate_bucket_priority": _target_bucket_priority(str(target_difficulty), str(target_bucket)),
                        "selected_bucket": selected_bucket,
                    },
                }
            )

    return expanded


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a generation plan from mastery records.")
    parser.add_argument("--mastery", required=True, help="Mastery JSON path")
    parser.add_argument("--synthesis-plan", required=True, help="Synthesis plan JSONL path")
    parser.add_argument("--source-map", required=False, help="Optional source KB map JSON path")
    parser.add_argument("--output", required=True, help="Expanded generation plan JSONL path")
    parser.add_argument("--summary-output", required=False, help="Optional JSON summary path")
    args = parser.parse_args(argv)

    mastery_records = read_json(Path(args.mastery))
    synthesis_plan = read_jsonl(Path(args.synthesis_plan))
    source_lookup = read_json(Path(args.source_map)) if args.source_map else None
    expanded = build_generation_plan(mastery_records, synthesis_plan, source_lookup=source_lookup)

    output_path = Path(args.output)
    write_jsonl(output_path, expanded)

    summary = {
        "source_count": len(mastery_records),
        "expanded_count": len(expanded),
        "bucket_counts": {},
    }
    for item in expanded:
        bucket = item["target_difficulty_bucket"]
        summary["bucket_counts"][bucket] = summary["bucket_counts"].get(bucket, 0) + 1

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "summary": str(summary_path),
                "expanded_count": len(expanded),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
