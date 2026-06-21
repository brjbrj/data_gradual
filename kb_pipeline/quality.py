from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import VLLMClient
from .prompts import build_quality_prompt, build_repair_prompt, build_retry_generation_prompt
from .utils import lookup_key, normalize_whitespace, read_json, read_jsonl, safe_json_from_text, write_json, write_jsonl


CRITICAL_ISSUES = {
    "difficulty_mismatch",
    "incorrect_answer",
    "unsolvable",
    "non_unique_answer",
    "invalid_steps",
    "answer_without_solution",
}


@dataclass
class QualityVote:
    verdict: str
    scores: Dict[str, int]
    issues: List[str]
    repair_suggestions: List[str]
    short_reason: str
    raw_output: str


def _clamp_score(value: Any) -> int:
    try:
        score = float(value)
    except Exception:
        return 3
    if 0.0 <= score <= 1.0:
        return max(1, min(5, int(round(score * 4.0)) + 1))
    return max(1, min(5, int(round(score))))


def _normalize_vote(raw_output: str, fallback_issue: Optional[str] = None) -> QualityVote:
    parsed = safe_json_from_text(raw_output) or {}
    scores_obj = parsed.get("scores", {}) if isinstance(parsed.get("scores", {}), dict) else {}
    scores = {
        key: _clamp_score(scores_obj.get(key, 3))
        for key in ("difficulty_match", "correctness", "answer_uniqueness", "step_validity", "brevity")
    }
    verdict = str(parsed.get("verdict", "fail")).strip().lower()
    if verdict not in {"pass", "fail"}:
        verdict = "fail"
    issues = [normalize_whitespace(item) for item in parsed.get("issues", []) if normalize_whitespace(item)]
    if fallback_issue and fallback_issue not in issues:
        issues.append(fallback_issue)
    suggestions = [normalize_whitespace(item) for item in parsed.get("repair_suggestions", []) if normalize_whitespace(item)]
    short_reason = normalize_whitespace(parsed.get("short_reason", ""))
    return QualityVote(
        verdict=verdict,
        scores=scores,
        issues=issues,
        repair_suggestions=suggestions,
        short_reason=short_reason,
        raw_output=raw_output,
    )


def _aggregate_votes(votes: Sequence[QualityVote]) -> Dict[str, Any]:
    pass_count = sum(1 for vote in votes if vote.verdict == "pass")
    issue_counts: Dict[str, int] = {}
    suggestion_counts: Dict[str, int] = {}
    score_totals: Dict[str, float] = {key: 0.0 for key in ("difficulty_match", "correctness", "answer_uniqueness", "step_validity", "brevity")}

    for vote in votes:
        for key, value in vote.scores.items():
            score_totals[key] += value
        for issue in vote.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for suggestion in vote.repair_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

    vote_count = max(1, len(votes))
    avg_scores = {key: round(total / vote_count, 3) for key, total in score_totals.items()}
    sorted_issues = sorted(issue_counts.items(), key=lambda item: (-item[1], item[0]))
    sorted_suggestions = sorted(suggestion_counts.items(), key=lambda item: (-item[1], item[0]))
    top_issues = [item[0] for item in sorted_issues[:5]]
    top_suggestions = [item[0] for item in sorted_suggestions[:5]]

    critical = [issue for issue in top_issues if issue in CRITICAL_ISSUES]
    overall_score = sum(avg_scores.values()) / len(avg_scores)
    verdict = "pass" if pass_count >= (vote_count // 2 + 1) and not critical and overall_score >= 3.8 else "fail"
    return {
        "verdict": verdict,
        "pass_count": pass_count,
        "vote_count": vote_count,
        "avg_scores": avg_scores,
        "issues": top_issues,
        "critical_issues": critical,
        "repair_suggestions": top_suggestions,
        "overall_score": round(overall_score, 3),
    }


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _should_log_progress(done: int, total: int) -> bool:
    if done <= 20:
        return True
    if total <= 200:
        return done % 5 == 0 or done == total
    interval = max(1, total // 100)
    return done % interval == 0 or done == total


class QualityInspector:
    def __init__(self, client: Optional[VLLMClient] = None, votes: int = 3) -> None:
        self.client = client or VLLMClient()
        self.votes = max(1, votes)

    def inspect_once(
        self,
        candidate: Dict[str, Any],
        source: Dict[str, Any],
        target: Dict[str, Any],
        hint: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
    ) -> QualityVote:
        raw = self.client.chat(
            build_quality_prompt(candidate, source, target, hint=hint),
            temperature=temperature,
            top_p=0.9,
            max_tokens=900,
        )
        return _normalize_vote(raw)

    def inspect(self, candidate: Dict[str, Any], source: Dict[str, Any], target: Dict[str, Any], hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        votes = [
            self.inspect_once(candidate, source, target, hint=hint, temperature=0.15 + 0.05 * i)
            for i in range(self.votes)
        ]
        aggregated = _aggregate_votes(votes)
        aggregated["votes"] = [vote.__dict__ for vote in votes]
        aggregated["short_reason"] = "multi_vote_quality_check"
        return aggregated


def _repair_candidate(
    candidate: Dict[str, Any],
    source: Dict[str, Any],
    target: Dict[str, Any],
    issues: Sequence[str],
    suggestions: Sequence[str],
    client: VLLMClient,
) -> Dict[str, Any]:
    prompt = build_repair_prompt(candidate, list(issues), target)
    feedback = {
        "issues": list(issues),
        "suggestions": list(suggestions),
        "source": source,
        "candidate": candidate,
    }
    raw = client.chat(prompt, temperature=0.15, top_p=0.9, max_tokens=1100)
    parsed = safe_json_from_text(raw) or {}
    solution_steps = normalize_whitespace(
        parsed.get("solution")
        or parsed.get("solution_steps")
        or candidate.get("solution_steps", "")
        or candidate.get("solution", "")
    )
    repaired = {
        "question": normalize_whitespace(parsed.get("question") or candidate.get("question", "")),
        "solution": solution_steps,
        "solution_steps": solution_steps,
        "answer": normalize_whitespace(parsed.get("answer") or candidate.get("answer", "")),
        "difficulty_bucket": normalize_whitespace(parsed.get("difficulty_bucket") or target.get("bucket") or candidate.get("difficulty_bucket", "medium")),
        "step_count": int(parsed.get("step_count") or candidate.get("step_count") or target.get("reference_step_count") or 1),
        "repair_notes": normalize_whitespace(parsed.get("repair_notes") or ""),
        "raw_model_output": raw,
        "repair_feedback": feedback,
    }
    return repaired


def repair_candidate(
    candidate: Dict[str, Any],
    source: Dict[str, Any],
    target: Dict[str, Any],
    issues: Sequence[str],
    suggestions: Sequence[str],
    client: VLLMClient,
) -> Dict[str, Any]:
    return _repair_candidate(candidate, source, target, issues, suggestions, client)


def _regenerate_candidate(
    plan_card: Dict[str, Any],
    target: Dict[str, Any],
    feedback: Dict[str, Any],
    client: VLLMClient,
) -> Dict[str, Any]:
    prompt = build_retry_generation_prompt(plan_card, target, feedback=feedback)
    raw = client.chat(prompt, temperature=0.55, top_p=0.9, max_tokens=1200)
    parsed = safe_json_from_text(raw) or {}
    solution_steps = normalize_whitespace(parsed.get("solution_steps") or parsed.get("solution") or "")
    return {
        "question": normalize_whitespace(parsed.get("question") or ""),
        "solution": solution_steps,
        "solution_steps": solution_steps,
        "answer": normalize_whitespace(parsed.get("answer") or ""),
        "difficulty_bucket": normalize_whitespace(parsed.get("difficulty_bucket") or target.get("bucket") or "medium"),
        "step_count": int(parsed.get("step_count") or target.get("reference_step_count") or 1),
        "raw_model_output": raw,
        "regen_feedback": feedback,
    }


def regenerate_candidate(
    plan_card: Dict[str, Any],
    target: Dict[str, Any],
    feedback: Dict[str, Any],
    client: VLLMClient,
) -> Dict[str, Any]:
    return _regenerate_candidate(plan_card, target, feedback, client)


def repair_until_pass(
    candidate: Dict[str, Any],
    plan_card: Dict[str, Any],
    source: Dict[str, Any],
    target: Dict[str, Any],
    qc_client: VLLMClient,
    repair_client: VLLMClient,
    gen_client: VLLMClient,
    votes: int = 3,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    inspector = QualityInspector(client=qc_client, votes=votes)
    current = dict(candidate)
    history: List[Dict[str, Any]] = []

    for round_index in range(max_rounds):
        qc_report = inspector.inspect(current, source, target)
        history.append(
            {
                "round": round_index,
                "candidate": current,
                "qc_report": qc_report,
            }
        )
        if qc_report["verdict"] == "pass":
            return {
                "passed": True,
                "candidate": current,
                "qc_report": qc_report,
                "history": history,
            }

        feedback = {
            "round": round_index,
            "issues": qc_report["issues"],
            "critical_issues": qc_report["critical_issues"],
            "repair_suggestions": qc_report["repair_suggestions"],
            "avg_scores": qc_report["avg_scores"],
        }

        if round_index % 2 == 0:
            next_candidate = _repair_candidate(current, source, target, qc_report["issues"], qc_report["repair_suggestions"], repair_client)
        else:
            next_candidate = _regenerate_candidate(plan_card, target, feedback, gen_client)

        next_candidate.setdefault("source_task_id", plan_card.get("source_task_id", plan_card.get("task_id")))
        next_candidate.setdefault("task_id", plan_card.get("task_id"))
        next_candidate.setdefault("source_question", plan_card.get("question"))
        next_candidate.setdefault("source_answer", plan_card.get("answer"))
        next_candidate.setdefault("source_knowledge", plan_card.get("knowledge"))
        next_candidate.setdefault("generation_target", target)
        next_candidate.setdefault("generation_feedback", feedback)
        current = next_candidate

    final_qc = inspector.inspect(current, source, target)
    return {
        "passed": final_qc["verdict"] == "pass",
        "candidate": current,
        "qc_report": final_qc,
        "history": history,
    }


def _load_records(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_json(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-vote quality checking and repair loop.")
    parser.add_argument("--generated", required=True, help="Generated JSONL path")
    parser.add_argument("--plan", required=True, help="Generation plan JSONL path")
    parser.add_argument("--source-map", required=True, help="Source map JSON path")
    parser.add_argument("--target-map", required=True, help="Target map JSON path")
    parser.add_argument("--output", required=True, help="Quality-checked JSONL path")
    parser.add_argument("--votes", type=int, default=3, help="Number of judge votes")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max repair rounds")
    parser.add_argument("--qc-model", required=False, help="Judge model name")
    parser.add_argument("--repair-model", required=False, help="Repair model name")
    parser.add_argument("--gen-model", required=False, help="Regeneration model name")
    args = parser.parse_args(argv)

    generated = read_jsonl(Path(args.generated))
    plan_cards = read_jsonl(Path(args.plan))
    source_map = read_json(Path(args.source_map))
    target_map = read_json(Path(args.target_map))
    plan_by_task = {card.get("task_id"): card for card in plan_cards}

    qc_model = args.qc_model or os.environ.get("QC_MODEL") or os.environ.get("QUALITY_MODEL") or os.environ.get("STEP_MODEL") or os.environ.get("JUDGE_MODEL") or os.environ.get("VLLM_MODEL") or "/root/brjverl/models/Qwen3.6-27B"
    repair_model = args.repair_model or os.environ.get("REPAIR_MODEL") or os.environ.get("VLLM_REPAIR_MODEL") or os.environ.get("GEN_MODEL") or os.environ.get("VLLM_GEN_MODEL") or os.environ.get("VLLM_MODEL") or qc_model
    gen_model = args.gen_model or os.environ.get("GEN_MODEL") or os.environ.get("VLLM_GEN_MODEL") or os.environ.get("VLLM_MODEL") or repair_model

    qc_client = VLLMClient(model=qc_model)
    repair_client = VLLMClient(model=repair_model)
    gen_client = VLLMClient(model=gen_model)

    total = len(generated)
    started_at = time.time()

    def _process(item: Dict[str, Any]) -> Dict[str, Any]:
        source = lookup_key(source_map, item.get("source_task_id"), {})
        target = lookup_key(target_map, item.get("source_task_id"), {})
        plan_card = lookup_key(plan_by_task, item.get("task_id"), {}) or lookup_key(plan_by_task, item.get("source_task_id"), {})
        inspector = QualityInspector(client=qc_client, votes=args.votes)
        current = dict(item)
        history: List[Dict[str, Any]] = []
        passed = False
        final_report: Dict[str, Any] = {}
        for round_idx in range(args.max_rounds):
            report = inspector.inspect(current, source, target)
            history.append({"round": round_idx, "candidate": current, "qc_report": report})
            final_report = report
            if report["verdict"] == "pass":
                passed = True
                break
            feedback = {
                "round": round_idx,
                "issues": report["issues"],
                "critical_issues": report["critical_issues"],
                "repair_suggestions": report["repair_suggestions"],
                "avg_scores": report["avg_scores"],
            }
            if round_idx % 2 == 0:
                current = repair_candidate(current, source, target, report["issues"], report["repair_suggestions"], repair_client)
            else:
                current = regenerate_candidate(plan_card or item, {
                    "bucket": target.get("bucket", "medium"),
                    "step_count_range": target.get("step_count_range", [2, 4]),
                    "reference_step_count": target.get("reference_step_count", 0),
                }, feedback, gen_client)
            current.setdefault("task_id", item.get("task_id"))
            current.setdefault("source_task_id", item.get("source_task_id"))
            current.setdefault("source_question", item.get("source_question"))
            current.setdefault("source_answer", item.get("source_answer"))
            current.setdefault("source_knowledge", item.get("source_knowledge"))
            current.setdefault("generation_target", item.get("generation_target"))
        if not passed:
            report = inspector.inspect(current, source, target)
            history.append({"round": args.max_rounds, "candidate": current, "qc_report": report})
            final_report = report
            passed = report["verdict"] == "pass"
        return {
            "task_id": item.get("task_id"),
            "source_task_id": item.get("source_task_id"),
            "passed": passed,
            "question": current.get("question", ""),
            "answer": current.get("answer", ""),
            "solution": current.get("solution", ""),
            "difficulty_bucket": current.get("difficulty_bucket", ""),
            "step_count": current.get("step_count", 0),
            "qc_report": final_report,
            "history": history,
            "generation_target": item.get("generation_target", {}),
            "source_question": item.get("source_question", ""),
            "source_answer": item.get("source_answer", ""),
            "source_knowledge": item.get("source_knowledge", {}),
            "mode": item.get("generation_target", {}).get("mode", ""),
        }

    outputs: List[Optional[Dict[str, Any]]] = [None] * total
    try:
        workers = max(1, int(os.environ.get("QC_CONCURRENCY", "256")))
    except Exception:
        workers = 256
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, total))) as executor:
        future_to_index = {executor.submit(_process, item): idx for idx, item in enumerate(generated)}
        done = 0
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            outputs[idx] = future.result()
            done += 1
            if _should_log_progress(done, total):
                elapsed = time.time() - started_at
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (total - done) / rate if rate > 0 else 0.0
                print(
                    f"[quality] {done}/{total} ({done * 100.0 / max(1, total):5.1f}%) "
                    f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(remaining)}",
                    flush=True,
                )

    final_outputs = [item for item in outputs if item is not None]
    write_jsonl(Path(args.output), final_outputs)
    print(json.dumps({"output": args.output, "count": len(final_outputs)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
