from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import VLLMClient
from .prompts import build_generation_prompt
from .utils import normalize_whitespace, read_json, read_jsonl, safe_json_from_text, write_json, write_jsonl


def _difficulty_from_step_count(step_count: int) -> str:
    if step_count <= 1:
        return "easy"
    if step_count <= 3:
        return "medium"
    if step_count <= 5:
        return "hard"
    return "very_hard"


def _target_from_plan_card(plan_card: Dict[str, Any]) -> Dict[str, Any]:
    knowledge = plan_card.get("knowledge", {})
    if "generation_target" in plan_card and isinstance(plan_card.get("generation_target"), dict):
        target = dict(plan_card["generation_target"])
        target.setdefault(
            "bucket",
            plan_card.get("target_difficulty_bucket")
            or plan_card.get("difficulty_bucket")
            or knowledge.get("difficulty_bucket")
            or _difficulty_from_step_count(int(knowledge.get("step_count", 0) or 0)),
        )
        target.setdefault("step_count_range", {
            "easy": [1, 2],
            "medium": [2, 4],
            "hard": [4, 6],
            "very_hard": [6, 10],
        }.get(target["bucket"], [2, 4]))
        target.setdefault("difficulty_level", plan_card.get("target_difficulty") or plan_card.get("difficulty_level") or target["bucket"])
        return target
    return {
        "bucket": plan_card.get("difficulty_bucket")
        or plan_card.get("target_difficulty_bucket")
        or knowledge.get("difficulty_bucket")
        or _difficulty_from_step_count(int(knowledge.get("step_count", 0) or 0)),
        "difficulty_level": plan_card.get("target_difficulty") or plan_card.get("difficulty_level") or plan_card.get("difficulty_bucket")
        or plan_card.get("target_difficulty_bucket")
        or knowledge.get("difficulty_bucket")
        or _difficulty_from_step_count(int(knowledge.get("step_count", 0) or 0)),
        "step_count_range": {
            "easy": [1, 2],
            "medium": [2, 4],
            "hard": [4, 6],
            "very_hard": [6, 10],
        }.get(plan_card.get("difficulty_bucket") or knowledge.get("difficulty_bucket"), [2, 4]),
        "reference_step_count": int(knowledge.get("step_count", 0) or 0),
        "skill_tags": knowledge.get("skill_tags", []),
    }


DEFAULT_TEMPERATURE_BY_BUCKET = {
    "easy": 0.3,
    "medium": 0.4,
    "hard": 0.6,
    "very_hard": 0.7,
}

DEFAULT_TOP_P_BY_BUCKET = {
    "easy": 0.3,
    "medium": 0.4,
    "hard": 0.6,
    "very_hard": 0.7,
}


def _parse_float_map(value: Optional[Any]) -> Dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        source = value
    else:
        try:
            source = json.loads(str(value))
        except Exception:
            return {}
    outputs: Dict[str, float] = {}
    for key, raw in source.items():
        try:
            outputs[str(key)] = float(raw)
        except Exception:
            continue
    return outputs


def _parse_retry_limit(value: Optional[str], default: int = 2) -> int:
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_json_map_env(*names: str) -> Dict[str, float]:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            parsed = json.loads(value)
        except Exception:
            continue
        if isinstance(parsed, dict):
            result: Dict[str, float] = {}
            for key, raw in parsed.items():
                try:
                    result[str(key)] = float(raw)
                except Exception:
                    continue
            if result:
                return result
    return {}


def _parse_float_env(*names: str, default: Optional[float] = None) -> Optional[float]:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except Exception:
            continue
    return default


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


def _parse_bool_env(*names: str, default: bool = False) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _normalize_question_text(text: str) -> str:
    return normalize_whitespace(text).lower()


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize_question_text(text)))


def _is_too_similar(candidate: str, seen: Sequence[str], threshold: float = 0.85) -> bool:
    candidate_text = _normalize_question_text(candidate)
    if not candidate_text:
        return False
    candidate_tokens = _token_set(candidate_text)
    if not candidate_tokens:
        return False
    for other in seen:
        other_text = _normalize_question_text(other)
        if not other_text:
            continue
        if candidate_text == other_text:
            return True
        other_tokens = _token_set(other_text)
        if not other_tokens:
            continue
        overlap = len(candidate_tokens & other_tokens) / max(1, len(candidate_tokens | other_tokens))
        if overlap >= threshold:
            return True
    return False


def _resolve_generation_params(
    target: Dict[str, Any],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    temperature_map: Optional[Dict[str, float]] = None,
    top_p_map: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    bucket = str(target.get("bucket") or "medium")
    difficulty_level = str(target.get("difficulty_level") or bucket)
    resolved_temperature = temperature
    resolved_top_p = top_p
    if resolved_temperature is None:
        resolved_temperature = (temperature_map or {}).get(difficulty_level, (temperature_map or {}).get(bucket, DEFAULT_TEMPERATURE_BY_BUCKET.get(bucket, 0.5)))
    if resolved_top_p is None:
        resolved_top_p = (top_p_map or {}).get(difficulty_level, (top_p_map or {}).get(bucket, DEFAULT_TOP_P_BY_BUCKET.get(bucket, 0.9)))
    return {
        "temperature": max(0.0, float(resolved_temperature)),
        "top_p": max(0.0, min(1.0, float(resolved_top_p))),
    }


def _normalize_generation_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), dict):
            return payload["data"]
        if isinstance(payload.get("result"), dict):
            return payload["result"]
        return payload
    if isinstance(payload, list) and payload:
        for item in reversed(payload):
            if isinstance(item, dict):
                return item
    return {}


def _extract_generation_fields(parsed: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    question = normalize_whitespace(parsed.get("question") or "")
    solution_steps_text = normalize_whitespace(
        parsed.get("solution") or parsed.get("solution_steps") or parsed.get("reasoning_brief") or ""
    )
    steps = parsed.get("steps", [])
    if not isinstance(steps, list):
        steps = []
    normalized_steps = [normalize_whitespace(str(step)) for step in steps if normalize_whitespace(str(step))]
    if not normalized_steps and solution_steps_text:
        normalized_steps = [line.strip() for line in solution_steps_text.splitlines() if line.strip()]
    solution = normalize_whitespace(parsed.get("solution") or parsed.get("solution_steps") or solution_steps_text or "")
    answer = normalize_whitespace(parsed.get("final_answer") or parsed.get("answer") or "")
    difficulty_bucket = normalize_whitespace(parsed.get("difficulty_bucket") or target["bucket"])
    try:
        step_count = int(parsed.get("step_count") or target.get("reference_step_count") or 1)
    except Exception:
        step_count = int(target.get("reference_step_count") or 1)
    if not normalized_steps and solution:
        normalized_steps = [line.strip() for line in solution.splitlines() if line.strip()]
    if not solution and normalized_steps:
        solution = "\n".join(normalized_steps)
    if not solution_steps_text and normalized_steps:
        solution_steps_text = "\n".join(normalized_steps)
    return {
        "question": question,
        "steps": normalized_steps,
        "solution": solution,
        "solution_steps": solution_steps_text,
        "answer": answer,
        "final_answer": answer,
        "difficulty_bucket": difficulty_bucket,
        "step_count": step_count,
    }


def _generation_max_tokens() -> int:
    return max(128, _parse_int_env("GEN_MAX_TOKENS", default=640))


def project_generation_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": record.get("task_id"),
        "source_task_id": record.get("source_task_id"),
        "variant_index": record.get("variant_index", 0),
        "question": record.get("question", ""),
        "steps": record.get("steps", []),
        "solution": record.get("solution", ""),
        "solution_steps": record.get("solution_steps", ""),
        "answer": record.get("answer", ""),
        "final_answer": record.get("final_answer", ""),
        "difficulty_bucket": record.get("difficulty_bucket", ""),
        "step_count": record.get("step_count", 0),
        "source_question": record.get("source_question", ""),
        "source_answer": record.get("source_answer", ""),
        "source_knowledge": record.get("source_knowledge", {}),
        "generation_target": record.get("generation_target", {}),
        "generation_failed": bool(record.get("generation_failed", False)),
        "generation_attempts": record.get("generation_attempts", 0),
        "generation_error": record.get("generation_error", ""),
    }


def project_generation_intermediate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": record.get("task_id"),
        "source_task_id": record.get("source_task_id"),
        "seed_task_id": record.get("source_task_id"),
        "variant_index": record.get("variant_index", 0),
        "question": record.get("question", ""),
        "steps": record.get("steps", []),
        "solution": record.get("solution", ""),
        "solution_steps": record.get("solution_steps", ""),
        "answer": record.get("answer", ""),
        "final_answer": record.get("final_answer", ""),
        "difficulty_bucket": record.get("difficulty_bucket", ""),
        "step_count": record.get("step_count", 0),
        "source_question": record.get("source_question", ""),
        "source_answer": record.get("source_answer", ""),
        "source_knowledge": record.get("source_knowledge", {}),
        "generation_target": record.get("generation_target", {}),
        "plan_source_task_id": record.get("plan_source_task_id", ""),
        "plan_source_question": record.get("plan_source_question", ""),
        "plan_source_answer": record.get("plan_source_answer", ""),
        "plan_source_scene_text": record.get("plan_source_scene_text", ""),
        "plan_source_surface_template": record.get("plan_source_surface_template", ""),
        "plan_source_scene_template": record.get("plan_source_scene_template", ""),
        "plan_source_scenario_template": record.get("plan_source_scenario_template", ""),
        "plan_source_concepts": record.get("plan_source_concepts", {}),
        "plan_source_knowledge": record.get("plan_source_knowledge", {}),
        "plan_source_difficulty_bucket": record.get("plan_source_difficulty_bucket", ""),
        "generation_failed": bool(record.get("generation_failed", False)),
        "generation_attempts": record.get("generation_attempts", 0),
        "generation_error": record.get("generation_error", ""),
    }


def project_generation_raw_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": record.get("task_id"),
        "source_task_id": record.get("source_task_id"),
        "variant_index": record.get("variant_index", 0),
        "question": record.get("question", ""),
        "source_question": record.get("source_question", ""),
        "source_answer": record.get("source_answer", ""),
        "generation_target": record.get("generation_target", {}),
        "raw_model_output": record.get("raw_model_output", ""),
        "generation_failed": bool(record.get("generation_failed", False)),
        "generation_error": record.get("generation_error", ""),
    }


class QuestionGenerator:
    def __init__(self, client: Optional[VLLMClient] = None) -> None:
        self.client = client or VLLMClient()

    def generate_one(
        self,
        plan_card: Dict[str, Any],
        target_mode: Optional[str] = None,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        temperature_map: Optional[Dict[str, float]] = None,
        top_p_map: Optional[Dict[str, float]] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        target = _target_from_plan_card(plan_card)
        if target_mode:
            target["mode"] = target_mode
        messages = build_generation_prompt(plan_card, target, feedback=feedback)
        sampling = _resolve_generation_params(target, temperature=temperature, top_p=top_p, temperature_map=temperature_map, top_p_map=top_p_map)
        retries = max_retries
        if retries is None:
            retries = _parse_retry_limit(os.environ.get("GEN_MAX_RETRIES"), default=2)
        else:
            retries = int(retries)

        last_error: Optional[str] = None
        infinite_retries = retries < 0
        attempt = 0
        while infinite_retries or attempt <= retries:
            try:
                response_format = {"type": "json_object"} if _parse_bool_env("GEN_FORCE_JSON", default=True) else None
                raw = self.client.chat(
                    messages,
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    max_tokens=_generation_max_tokens(),
                    response_format=response_format,
                )
                parsed = _normalize_generation_payload(safe_json_from_text(raw) or {})
                extracted = _extract_generation_fields(parsed, target)
                question = extracted["question"]
                normalized_steps = extracted["steps"]
                solution = extracted["solution"]
                solution_steps_text = extracted["solution_steps"]
                answer = extracted["answer"]
                difficulty_bucket = extracted["difficulty_bucket"]
                step_count = extracted["step_count"]
                missing_fields = []
                if not question:
                    missing_fields.append("question")
                if not solution and not normalized_steps:
                    missing_fields.append("solution")
                if not answer:
                    missing_fields.append("answer")
                if missing_fields:
                    last_error = f"missing_required_generation_fields:{','.join(missing_fields)}"
                    if infinite_retries or attempt < retries:
                        time.sleep(min(2.0 * (attempt + 1), 6.0))
                        attempt += 1
                        continue
                    break
                return {
                    "task_id": plan_card.get("task_id"),
                    "source_task_id": plan_card.get("source_task_id", plan_card.get("task_id")),
                    "variant_index": plan_card.get("variant_index", 0),
                    "question": question,
                    "steps": normalized_steps,
                    "solution": solution,
                    "solution_steps": solution_steps_text,
                    "answer": answer,
                    "final_answer": answer,
                    "difficulty_bucket": difficulty_bucket,
                    "step_count": step_count,
                    "source_question": plan_card.get("question"),
                    "source_answer": plan_card.get("answer"),
                    "plan_source_task_id": plan_card.get("plan_source_task_id", ""),
                    "plan_source_question": plan_card.get("plan_source_question", ""),
                    "plan_source_answer": plan_card.get("plan_source_answer", ""),
                    "plan_source_scene_text": plan_card.get("plan_source_scene_text", ""),
                    "plan_source_surface_template": plan_card.get("plan_source_surface_template", ""),
                    "plan_source_scene_template": plan_card.get("plan_source_scene_template", ""),
                    "plan_source_scenario_template": plan_card.get("plan_source_scenario_template", ""),
                    "plan_source_concepts": plan_card.get("plan_source_concepts", {}),
                    "plan_source_knowledge": plan_card.get("plan_source_knowledge", {}),
                    "plan_source_difficulty_bucket": plan_card.get("plan_source_difficulty_bucket", ""),
                    "source_scene_template": plan_card.get("scene_template"),
                    "source_scenario_template": plan_card.get("scenario_template"),
                    "source_knowledge": plan_card.get("knowledge"),
                    "generation_target": target,
                    "raw_model_output": raw,
                    "generation_failed": False,
                    "generation_attempts": attempt + 1,
                }
            except Exception as exc:
                last_error = str(exc)
                if infinite_retries or attempt < retries:
                    time.sleep(min(2.0 * (attempt + 1), 6.0))
                    attempt += 1
                    continue
                break

        attempts_made = attempt + 1 if attempt >= 0 else 0
        return {
            "task_id": plan_card.get("task_id"),
            "source_task_id": plan_card.get("source_task_id", plan_card.get("task_id")),
            "variant_index": plan_card.get("variant_index", 0),
            "question": "",
            "steps": [],
            "solution": "",
            "solution_steps": "",
            "answer": "",
            "final_answer": "",
            "difficulty_bucket": target["bucket"],
            "step_count": int(target.get("reference_step_count") or 1),
            "source_question": plan_card.get("question"),
            "source_answer": plan_card.get("answer"),
            "plan_source_task_id": plan_card.get("plan_source_task_id", ""),
            "plan_source_question": plan_card.get("plan_source_question", ""),
            "plan_source_answer": plan_card.get("plan_source_answer", ""),
            "plan_source_scene_text": plan_card.get("plan_source_scene_text", ""),
            "plan_source_surface_template": plan_card.get("plan_source_surface_template", ""),
            "plan_source_scene_template": plan_card.get("plan_source_scene_template", ""),
            "plan_source_scenario_template": plan_card.get("plan_source_scenario_template", ""),
            "plan_source_concepts": plan_card.get("plan_source_concepts", {}),
            "plan_source_knowledge": plan_card.get("plan_source_knowledge", {}),
            "plan_source_difficulty_bucket": plan_card.get("plan_source_difficulty_bucket", ""),
            "source_scene_template": plan_card.get("scene_template"),
            "source_scenario_template": plan_card.get("scenario_template"),
            "source_knowledge": plan_card.get("knowledge"),
            "generation_target": target,
            "raw_model_output": "",
            "generation_failed": True,
            "generation_error": last_error or "unknown_generation_failure",
            "generation_attempts": attempts_made,
        }


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


def _should_log_progress(done: int, total: int) -> bool:
    if done <= 20:
        return True
    if total <= 200:
        return done % 5 == 0 or done == total
    interval = max(1, total // 100)
    return done % interval == 0 or done == total


def generate_questions(
    plan_cards: Sequence[Dict[str, Any]],
    client: Optional[VLLMClient] = None,
    max_concurrency: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    temperature_map: Optional[Dict[str, float]] = None,
    top_p_map: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    generator = QuestionGenerator(client=client)
    total = len(plan_cards)
    started_at = time.time()
    workers = _resolve_workers(max_concurrency, "GEN_CONCURRENCY", default=256)
    workers = min(workers, _parse_int_env("GEN_CONCURRENCY_CAP", default=256))
    if total == 0:
        return []
    outputs: List[Optional[Dict[str, Any]]] = [None] * total
    seen_questions: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
        future_to_index = {}
        for index, card in enumerate(plan_cards):
            mode = card.get("mode")
            if not mode:
                modes = card.get("diversity_profile", {}).get("candidate_modes", [])
                mode = modes[0].get("mode") if modes else None
            future = executor.submit(
                generator.generate_one,
                card,
                mode,
                None,
                temperature,
                top_p,
                temperature_map,
                top_p_map,
            )
            future_to_index[future] = index

        done = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if _is_too_similar(result.get("question", ""), seen_questions):
                    card = plan_cards[index]
                    mode = card.get("mode")
                    if not mode:
                        modes = card.get("diversity_profile", {}).get("candidate_modes", [])
                        mode = modes[0].get("mode") if modes else None
                    target = _target_from_plan_card(card)
                    if mode:
                        target["mode"] = mode
                    feedback = {
                        "reason": "too_similar_to_previous_candidate",
                        "previous_questions": seen_questions[-5:],
                    }
                    retry_result = generator.generate_one(
                        card,
                        mode,
                        None,
                        temperature,
                        top_p,
                        temperature_map,
                        top_p_map,
                        feedback=feedback,
                    )
                    if retry_result.get("question"):
                        result = retry_result
                if result.get("question"):
                    seen_questions.append(result["question"])
                outputs[index] = result
            except Exception as exc:
                card = plan_cards[index]
                mode = card.get("mode")
                if not mode:
                    modes = card.get("diversity_profile", {}).get("candidate_modes", [])
                    mode = modes[0].get("mode") if modes else None
                target = _target_from_plan_card(card)
                if mode:
                    target["mode"] = mode
                outputs[index] = {
                    "task_id": card.get("task_id"),
                    "source_task_id": card.get("source_task_id", card.get("task_id")),
                    "variant_index": card.get("variant_index", 0),
                    "question": "",
                    "solution": "",
                    "answer": "",
                    "final_answer": "",
                    "difficulty_bucket": target["bucket"],
                    "step_count": int(target.get("reference_step_count") or 1),
                    "source_question": card.get("question"),
                    "source_answer": card.get("answer"),
                    "plan_source_task_id": card.get("plan_source_task_id", ""),
                    "plan_source_question": card.get("plan_source_question", ""),
                    "plan_source_answer": card.get("plan_source_answer", ""),
                    "plan_source_scene_text": card.get("plan_source_scene_text", ""),
                    "plan_source_surface_template": card.get("plan_source_surface_template", ""),
                    "plan_source_scene_template": card.get("plan_source_scene_template", ""),
                    "plan_source_scenario_template": card.get("plan_source_scenario_template", ""),
                    "plan_source_concepts": card.get("plan_source_concepts", {}),
                    "plan_source_knowledge": card.get("plan_source_knowledge", {}),
                    "plan_source_difficulty_bucket": card.get("plan_source_difficulty_bucket", ""),
                    "source_scene_template": card.get("scene_template"),
                    "source_scenario_template": card.get("scenario_template"),
                    "source_knowledge": card.get("knowledge"),
                    "generation_target": target,
                    "raw_model_output": "",
                    "generation_failed": True,
                    "generation_error": str(exc),
                    "generation_attempts": 0,
                }
            done += 1
            if _should_log_progress(done, total):
                elapsed = time.time() - started_at
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (total - done) / rate if rate > 0 else 0.0
                print(
                    f"[generate] {done}/{total} ({done * 100.0 / max(1, total):5.1f}%) "
                    f"elapsed={int(elapsed)//60:02d}:{int(elapsed)%60:02d} "
                    f"eta={int(max(0, remaining))//60:02d}:{int(max(0, remaining))%60:02d}",
                    flush=True,
                )
    failed = sum(1 for item in outputs if item and item.get("generation_failed"))
    if failed:
        print(f"[generate] failed={failed}/{total}", flush=True)
    return [item for item in outputs if item is not None]


def _load_plan_cards(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    return read_json(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate candidate questions from a synthesis plan.")
    parser.add_argument("--plan", required=True, help="Synthesis plan JSONL or JSON path")
    parser.add_argument("--output", required=False, help="Output JSONL path")
    parser.add_argument("--intermediate-output", required=False, help="Intermediate structured JSONL path")
    parser.add_argument("--failed-output", required=False, help="Failed generation JSONL path")
    parser.add_argument("--temperature", type=float, required=False, help="Global generation temperature override")
    parser.add_argument("--top-p", dest="top_p", type=float, required=False, help="Global generation top_p override")
    parser.add_argument("--temperature-map", required=False, help="JSON map of difficulty bucket to temperature")
    parser.add_argument("--top-p-map", required=False, help="JSON map of difficulty bucket to top_p")
    parser.add_argument("--raw-output", required=False, help="Raw generation log JSONL path")
    parser.add_argument("--failed-raw-output", required=False, help="Raw failed generation log JSONL path")
    args = parser.parse_args(argv)

    plan_path = Path(args.plan)
    plan_cards = _load_plan_cards(plan_path)
    generated = generate_questions(
        plan_cards,
        temperature=args.temperature if args.temperature is not None else _parse_float_env("GEN_TEMPERATURE", "VLLM_GEN_TEMPERATURE", default=0.5),
        top_p=args.top_p if args.top_p is not None else _parse_float_env("GEN_TOP_P", "VLLM_GEN_TOP_P", default=0.5),
        temperature_map=_parse_float_map(args.temperature_map) or _parse_json_map_env("GEN_TEMPERATURE_MAP", "VLLM_GEN_TEMPERATURE_MAP"),
        top_p_map=_parse_float_map(args.top_p_map) or _parse_json_map_env("GEN_TOP_P_MAP", "VLLM_GEN_TOP_P_MAP"),
    )

    output_path = Path(args.output) if args.output else plan_path.with_name(f"{plan_path.stem}.generated.jsonl")
    intermediate_output_path = Path(args.intermediate_output) if args.intermediate_output else output_path.with_name(f"{output_path.stem}.intermediate.jsonl")
    raw_output_path = Path(args.raw_output) if args.raw_output else output_path.with_name(f"{output_path.stem}.raw.jsonl")
    failed_output_path = Path(args.failed_output) if args.failed_output else output_path.with_name(f"{output_path.stem}.failed.jsonl")
    failed_raw_output_path = Path(args.failed_raw_output) if args.failed_raw_output else output_path.with_name(f"{output_path.stem}.failed.raw.jsonl")
    success = [record for record in generated if not record.get("generation_failed") and record.get("question") and record.get("steps") and record.get("answer")]
    failed = [record for record in generated if record not in success]
    write_jsonl(output_path, [project_generation_record(record) for record in success])
    write_jsonl(intermediate_output_path, [project_generation_intermediate_record(record) for record in success])
    write_jsonl(raw_output_path, [project_generation_raw_record(record) for record in success])
    write_jsonl(failed_output_path, [project_generation_record(record) for record in failed])
    write_jsonl(failed_raw_output_path, [project_generation_raw_record(record) for record in failed])
    print(json.dumps({"output": str(output_path), "intermediate_output": str(intermediate_output_path), "raw_output": str(raw_output_path), "failed_output": str(failed_output_path), "failed_raw_output": str(failed_raw_output_path), "success_count": len(success), "failed_count": len(failed)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
