from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import normalize_whitespace, read_json, read_jsonl, write_json, write_jsonl


@dataclass
class KBRecord:
    task_id: Any
    question: str
    answer: str
    solution_text: str
    scene_text: str
    surface_template: str
    scene_template: str
    scenario_template: str
    concepts: Dict[str, List[str]]
    knowledge: Dict[str, Any]
    source_schema: str
    question_type: Optional[str] = None


def _ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def _step_bucket(step_count: int) -> str:
    if step_count <= 1:
        return "very_short"
    if step_count <= 2:
        return "short"
    if step_count <= 4:
        return "medium"
    return "long"


def _difficulty_rank(label: str) -> int:
    order = {"easy": 0, "medium": 1, "hard": 2, "very_hard": 3}
    return order.get(label, 1)


class SynthesisPlanBuilder:
    def __init__(self, kb_dir: str, output_dir: Optional[str] = None, dataset_name: Optional[str] = None, max_peers: int = 6) -> None:
        self.kb_dir = Path(kb_dir)
        self.dataset_name = dataset_name or self.kb_dir.name
        self.output_dir = Path(output_dir) if output_dir else self.kb_dir / "synthesis"
        self.max_peers = max_peers

    def _load_records(self) -> Tuple[List[KBRecord], Dict[str, Any]]:
        manifest_path = self.kb_dir / "manifest.json"
        records_path = self.kb_dir / "records.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing KB manifest: {manifest_path}")
        if not records_path.exists():
            raise FileNotFoundError(f"Missing KB records: {records_path}")
        manifest = read_json(manifest_path)
        raw_records = read_jsonl(records_path)
        records = [KBRecord(**record) for record in raw_records]
        return records, manifest

    def _build_indexes(self, records: Sequence[KBRecord]) -> Dict[str, Dict[str, List[KBRecord]]]:
        by_signature: Dict[str, List[KBRecord]] = {}
        by_skill_bucket: Dict[str, List[KBRecord]] = {}
        by_scene_template: Dict[str, List[KBRecord]] = {}
        by_difficulty: Dict[str, List[KBRecord]] = {}

        for record in records:
            signature = str(record.knowledge.get("knowledge_signature", "")).strip() or "unknown"
            skill_tags = tuple(sorted(_ensure_list(record.knowledge.get("skill_tags"))))
            step_count = int(record.knowledge.get("step_count", 0) or 0)
            scene_key = normalize_whitespace(record.scene_template).lower()
            difficulty = str(record.knowledge.get("difficulty_bucket", "medium"))

            by_signature.setdefault(signature, []).append(record)
            by_skill_bucket.setdefault("|".join(skill_tags) or f"steps:{_step_bucket(step_count)}", []).append(record)
            by_scene_template.setdefault(scene_key, []).append(record)
            by_difficulty.setdefault(difficulty, []).append(record)

        return {
            "by_signature": by_signature,
            "by_skill_bucket": by_skill_bucket,
            "by_scene_template": by_scene_template,
            "by_difficulty": by_difficulty,
        }

    def _sorted_peer_ids(self, base: KBRecord, pool: Sequence[KBRecord]) -> List[Any]:
        peers = [item for item in pool if item.task_id != base.task_id]
        peers.sort(
            key=lambda item: (
                normalize_whitespace(item.scene_template).lower() == normalize_whitespace(base.scene_template).lower(),
                normalize_whitespace(item.scenario_template).lower() == normalize_whitespace(base.scenario_template).lower(),
                item.task_id,
            )
        )
        return [peer.task_id for peer in peers[: self.max_peers]]

    def _difficulty_target(self, record: KBRecord) -> Dict[str, Any]:
        step_count = int(record.knowledge.get("step_count", 0) or 0)
        difficulty = str(record.knowledge.get("difficulty_bucket", "medium"))
        target_steps = {
            "easy": [1, 2],
            "medium": [2, 4],
            "hard": [4, 6],
            "very_hard": [6, 10],
        }.get(difficulty, [2, 4])
        return {
            "bucket": difficulty,
            "step_count_range": target_steps,
            "reference_step_count": step_count,
            "rank": _difficulty_rank(difficulty),
        }

    def _build_mode(self, base: KBRecord, mode: str, peer_ids: Sequence[Any]) -> Dict[str, Any]:
        concept_names = _ensure_list(base.concepts.get("all_terms"))
        focus_terms = _ensure_list(base.concepts.get("focus_terms"))
        persons = _ensure_list(base.concepts.get("persons"))
        units = _ensure_list(base.concepts.get("units"))
        skill_tags = _ensure_list(base.knowledge.get("skill_tags"))
        operation_sequence = _ensure_list(base.knowledge.get("operation_sequence"))
        target = self._difficulty_target(base)

        return {
            "mode": mode,
            "source_task_id": base.task_id,
            "source_question": base.question,
            "source_answer": base.answer,
            "source_scene_template": base.scene_template,
            "source_scenario_template": base.scenario_template,
            "peer_task_ids": list(peer_ids),
            "target_difficulty": target,
            "constraints": {
                "preserve_skill_tags": skill_tags,
                "preserve_operation_sequence": operation_sequence,
                "preserve_answer_type": "numeric",
                "anchor_concepts": concept_names[:8],
                "anchor_persons": persons[:4],
                "anchor_terms": focus_terms[:8],
                "anchor_units": units[:6],
            },
            "generation_brief": {
                "purpose": "Generate candidates that keep the same structure while improving surface diversity and scene variety.",
                "requirements": [
                    "Preserve the core math relation and the approximate step structure.",
                    "Prefer changing the scene, people, objects, or narrative angle.",
                    "Avoid duplicating existing surface forms, especially the outer wording.",
                    "Keep the answer format consistent with the source problem so it remains verifiable.",
                    "Match the requested difficulty bucket closely.",
                ],
                "mode_hint": self._mode_hint(mode),
            },
        }

    def _mode_hint(self, mode: str) -> str:
        hints = {
            "same_structure": "Keep the same arithmetic skeleton and only swap the wording and a few entities.",
            "scene_swap": "Prioritize changing the scene and background setting while keeping the computation chain intact.",
            "entity_swap": "Replace people, objects, units, and quantity context while preserving the logic.",
            "surface_swap": "Preserve the semantics and steps, but rewrite the surface wording.",
        }
        return hints.get(mode, "Keep the logic correct and emphasize diversity.")

    def _build_plan_card(self, base: KBRecord, signature_peers: Sequence[KBRecord], skill_peers: Sequence[KBRecord]) -> Dict[str, Any]:
        signature_peer_ids = self._sorted_peer_ids(base, signature_peers)
        skill_peer_ids = self._sorted_peer_ids(base, skill_peers)
        step_count = int(base.knowledge.get("step_count", 0) or 0)
        difficulty = str(base.knowledge.get("difficulty_bucket", "medium"))

        return {
            "task_id": base.task_id,
            "source_schema": base.source_schema,
            "question_type": base.question_type,
            "question": base.question,
            "answer": base.answer,
            "scene_text": base.scene_text,
            "surface_template": base.surface_template,
            "scene_template": base.scene_template,
            "scenario_template": base.scenario_template,
            "concepts": base.concepts,
            "knowledge": base.knowledge,
            "difficulty_bucket": difficulty,
            "step_bucket": _step_bucket(step_count),
            "diversity_profile": {
                "knowledge_signature": base.knowledge.get("knowledge_signature", ""),
                "skill_tags": _ensure_list(base.knowledge.get("skill_tags")),
                "operation_sequence": _ensure_list(base.knowledge.get("operation_sequence")),
                "candidate_modes": [
                    self._build_mode(base, "same_structure", signature_peer_ids),
                    self._build_mode(base, "scene_swap", skill_peer_ids),
                    self._build_mode(base, "entity_swap", signature_peer_ids[: max(1, len(signature_peer_ids) // 2)]),
                    self._build_mode(base, "surface_swap", skill_peer_ids[: max(1, len(skill_peer_ids) // 2)]),
                ],
            },
        }

    def build(self) -> Dict[str, Path]:
        records, manifest = self._load_records()
        indexes = self._build_indexes(records)
        by_signature = indexes["by_signature"]
        by_skill_bucket = indexes["by_skill_bucket"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = self.output_dir / "synthesis_plan.jsonl"
        summary_path = self.output_dir / "summary.json"
        manifest_path = self.output_dir / "manifest.json"

        plan_cards: List[Dict[str, Any]] = []
        with plan_path.open("w", encoding="utf-8") as f:
            for record in records:
                signature = str(record.knowledge.get("knowledge_signature", "")).strip() or "unknown"
                skill_tags = tuple(sorted(_ensure_list(record.knowledge.get("skill_tags"))))
                step_bucket = _step_bucket(int(record.knowledge.get("step_count", 0) or 0))
                skill_key = "|".join(skill_tags) or f"steps:{step_bucket}"
                signature_peers = by_signature.get(signature, [])
                skill_peers = by_skill_bucket.get(skill_key, [])
                card = self._build_plan_card(record, signature_peers, skill_peers)
                plan_cards.append(card)
                f.write(json.dumps(card, ensure_ascii=False) + "\n")

        summary = {
            "dataset_name": self.dataset_name,
            "source_kb_dir": str(self.kb_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "plan_count": len(plan_cards),
            "mode_count": 4 if plan_cards else 0,
            "step_bucket_counts": {},
            "difficulty_counts": {},
        }
        for card in plan_cards:
            bucket = card["step_bucket"]
            summary["step_bucket_counts"][bucket] = summary["step_bucket_counts"].get(bucket, 0) + 1
            difficulty = card["difficulty_bucket"]
            summary["difficulty_counts"][difficulty] = summary["difficulty_counts"].get(difficulty, 0) + 1

        write_json(summary_path, summary)
        write_json(manifest_path, {
            "dataset_name": self.dataset_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_manifest": manifest,
            "source_kb_dir": str(self.kb_dir),
            "output_dir": str(self.output_dir),
            "plan_path": str(plan_path),
            "summary_path": str(summary_path),
            "plan_count": len(plan_cards),
        })

        return {"output_dir": self.output_dir, "plan": plan_path, "summary": summary_path, "manifest": manifest_path}


def build_synthesis_plan(kb_dir: str, output_dir: Optional[str] = None, dataset_name: Optional[str] = None, max_peers: int = 6) -> Dict[str, Path]:
    return SynthesisPlanBuilder(kb_dir=kb_dir, output_dir=output_dir, dataset_name=dataset_name, max_peers=max_peers).build()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a synthesis plan from KB artifacts.")
    parser.add_argument("--kb-dir", required=True, help="KB output directory, e.g. outputs/kb/gsm8k")
    parser.add_argument("--output-dir", required=False, help="Synthesis plan output directory")
    parser.add_argument("--dataset-name", required=False, help="Optional dataset name")
    parser.add_argument("--max-peers", type=int, default=6, help="Max reference peers per card")
    args = parser.parse_args(argv)

    outputs = build_synthesis_plan(
        kb_dir=args.kb_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_peers=args.max_peers,
    )
    print(json.dumps({"outputs": {k: str(v) for k, v in outputs.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

