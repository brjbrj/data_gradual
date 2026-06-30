from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .post_mastery_plan import replan_failed_plan
from .utils import normalize_whitespace, read_jsonl, write_json, write_jsonl


DEFAULT_TEMPERATURES = {
    "Easy": 0.3,
    "Slightly Easy": 0.4,
    "Equal": 0.5,
    "Slightly Hard": 0.6,
    "Hard": 0.7,
}

DEFAULT_TOP_PS = {
    "Easy": 0.3,
    "Slightly Easy": 0.4,
    "Equal": 0.5,
    "Slightly Hard": 0.6,
    "Hard": 0.7,
}

DIFFICULTY_RULES = {
    "Easy": "Make it clearly easier than the seed and solvable in about 1-2 necessary steps.",
    "Slightly Easy": "Make it slightly easier than the seed with fewer dependencies and about 2-3 necessary steps.",
    "Equal": "Keep approximately the same reasoning depth as the seed, usually 2-4 necessary steps.",
    "Slightly Hard": "Make it slightly harder than the seed by adding one meaningful dependency, usually 4-6 necessary steps.",
    "Hard": "Make it substantially harder than the seed with several dependent conditions, usually 6-10 necessary steps.",
}

NUMERIC_ANSWER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:/\d+(?:\.\d+)?)?"
)


def _parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _parse_float_map(value: Optional[str], default: Dict[str, float]) -> Dict[str, float]:
    if not value:
        return dict(default)
    try:
        parsed = json.loads(value)
    except Exception:
        return dict(default)
    if not isinstance(parsed, dict):
        return dict(default)
    result = dict(default)
    for key, item in parsed.items():
        try:
            result[str(key)] = float(item)
        except Exception:
            continue
    return result


def _model_aliases(model: str) -> set[str]:
    normalized = str(model).strip().rstrip("/")
    if not normalized:
        return set()
    aliases = {normalized}
    basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    if basename:
        aliases.add(basename)
    return aliases


def _mastery_lookup(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(record.get("task_id")): record for record in records}


def _build_prompt(
    seed_question: str,
    difficulty: str,
    knowledge: Dict[str, Any],
    attempt_index: int = 0,
    retry_reason: str = "",
) -> List[Dict[str, str]]:
    math_knowledge = knowledge.get("math", {})
    diversity = knowledge.get("diversity", {})
    primary_scene = diversity.get("primary_scene", {})
    alternative_scenes = diversity.get("alternative_scenes", [])
    scene_options = [primary_scene] + [
        scene
        for scene in alternative_scenes
        if isinstance(scene, dict) and scene
    ]
    active_scene = (
        scene_options[attempt_index % len(scene_options)]
        if scene_options
        else {}
    )
    diversity_brief = {
        "active_scene": active_scene,
        "variation_mode": diversity.get("variation_mode"),
        "narrative_style": diversity.get("narrative_style"),
        "number_strategy": diversity.get("number_strategy"),
    }
    kb_inspiration = knowledge.get("kb_inspiration", {})
    system = (
        "You create one mathematically correct word problem. "
        "Return only one valid JSON object and no other text."
    )
    user_lines = [
        f"Create one new math problem with relative difficulty: {difficulty}.",
        DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["Equal"]),
        f"Seed problem for mathematical reference only: {seed_question}",
        f"Mathematical guidance: {json.dumps(math_knowledge, ensure_ascii=False, separators=(',', ':'))}",
        f"Diversity brief: {json.dumps(diversity_brief, ensure_ascii=False, separators=(',', ':'))}",
        f"Optional knowledge-base inspiration: {json.dumps(kb_inspiration, ensure_ascii=False, separators=(',', ':'))}",
        "",
        "Requirements:",
        "- Create exactly one self-contained question with one uniquely determined answer.",
        "- Follow the mathematical skill and target difficulty, but treat all knowledge-base material as optional inspiration rather than a fixed template.",
        "- Use the active scene as the main diversity direction. You may freely invent suitable people, objects, and details inside that scene.",
        "- A familiar mathematical template in a genuinely different scene is acceptable.",
        "- Do not copy a full sentence, name, number pattern, or story setup from the seed problem or knowledge-base examples.",
        "- The steps must be concise, ordered, necessary, and mathematically correct.",
        "- Each step must contain an actual calculation or inference, not a copied fact.",
        "- Optimize for supervised fine-tuning on GSM8K-style reasoning: natural everyday wording, compact question text, and compact solution steps.",
        "- Prefer familiar settings such as school, shopping, chores, sports, books, food, money, time, distance, simple work schedules, games, or small community events.",
        "- Avoid industrial, laboratory, software, storage, solar, reservoir, airport, warehouse, logistics-center, or other technical settings unless the seed mathematically requires them.",
        "- Keep the question short and direct, usually within one paragraph and under about 90 English words. Do not add decorative background.",
        "- Keep solution steps short and natural: most problems should use 2-6 steps, while Hard problems may use up to 8 concise steps when necessary.",
        "- Do not start multiple steps with the same formulaic phrase such as 'Calculate the'. Use varied concise reasoning and equations.",
        "- Prefer clean integer arithmetic, realistic small or medium values, exact intermediate results, and a diverse final answer.",
        "- Avoid huge numbers, unnecessary decimals, answer 0 unless truly natural, and overused final answers such as 10, 20, 30, 40, 60, or 120.",
        "- The answer must be only a numeric string, without units, currency symbols, commas, or extra words.",
        '- Output exactly: {"question":"...","steps":["...","..."],"answer":"42"}',
    ]
    if retry_reason:
        user_lines.extend(
            [
                "",
                f"Previous output was invalid because: {retry_reason}.",
                "Correct the format and return only the required JSON object.",
            ]
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _decode_json_candidate(text: str) -> Optional[Any]:
    candidate = text.strip().lstrip("\ufeff")
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    for _ in range(2):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, str):
            candidate = parsed.strip()
            continue
        return parsed

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
            return parsed
        except json.JSONDecodeError:
            continue

    relaxed = (
        candidate.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    relaxed = re.sub(r",\s*([}\]])", r"\1", relaxed)
    try:
        parsed = ast.literal_eval(relaxed)
        return parsed
    except (ValueError, SyntaxError):
        return None


def _unwrap_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("data", "result", "output"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
        return payload
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                return item
    return {}


def _normalize_steps(value: Any) -> List[str]:
    if isinstance(value, list):
        return [
            normalize_whitespace(item)
            for item in value
            if normalize_whitespace(item)
        ]
    if isinstance(value, dict):
        return [
            normalize_whitespace(item)
            for item in value.values()
            if normalize_whitespace(item)
        ]
    text = str(value or "").strip()
    if not text:
        return []
    lines = [
        re.sub(r"^\s*(?:step\s*)?\d+\s*[\).:-]\s*", "", line, flags=re.IGNORECASE)
        for line in text.splitlines()
        if line.strip()
    ]
    if len(lines) <= 1:
        lines = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
            if part.strip()
        ]
    return [normalize_whitespace(line) for line in lines if normalize_whitespace(line)]


def _normalize_answer(value: Any) -> str:
    text = normalize_whitespace(value).replace(",", "").replace("$", "")
    if not text:
        return ""
    if NUMERIC_ANSWER_RE.fullmatch(text):
        return text
    matches = NUMERIC_ANSWER_RE.findall(text)
    return matches[-1] if matches else ""


def _parse_generated_output(raw: str) -> Tuple[Optional[Dict[str, Any]], str]:
    payload = _unwrap_payload(_decode_json_candidate(raw))
    if not payload:
        return None, "response is not a valid JSON object"

    question = normalize_whitespace(payload.get("question") or "")
    steps = _normalize_steps(
        payload.get("steps")
        or payload.get("solution_steps")
        or payload.get("solution")
    )
    answer = _normalize_answer(
        payload.get("answer")
        or payload.get("final_answer")
    )

    missing: List[str] = []
    if not question:
        missing.append("question")
    if not steps:
        missing.append("steps")
    if not answer:
        missing.append("answer")
    if missing:
        return None, f"missing or invalid fields: {','.join(missing)}"

    return {
        "question": question,
        "steps": steps,
        "answer": answer,
    }, ""


def _keyword_tokens(values: Sequence[Any]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            nested: List[Any] = []
            for nested_value in value.values():
                if isinstance(nested_value, list):
                    nested.extend(nested_value)
                else:
                    nested.append(nested_value)
            tokens.update(_keyword_tokens(nested))
            continue
        for token in re.findall(r"[a-z]+", str(value).lower()):
            if len(token) >= 4:
                tokens.add(token)
    return tokens


def _as_sequence(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _scene_tokens(scene: Dict[str, Any]) -> set[str]:
    return _keyword_tokens(
        [
            scene.get("domain", ""),
            scene.get("setting", ""),
            *_as_sequence(scene.get("roles")),
            *_as_sequence(scene.get("objects")),
            *_as_sequence(scene.get("units")),
        ]
    )


def _plan_alignment_error(
    parsed: Dict[str, Any],
    plan: Dict[str, Any],
    difficulty: str,
) -> str:
    """Check only lightweight generation-stage obligations.

    Planning owns diversity and similarity. Validation owns mathematical
    correctness, exact difficulty, solvability, and repairs. Generation only
    needs to produce parseable fields and visibly follow the assigned plan.
    """
    question = str(parsed.get("question") or "")
    steps = _normalize_steps(parsed.get("steps", []))
    answer = _normalize_answer(parsed.get("answer", ""))
    if not question or not steps or not answer:
        return "missing generated question, steps, or numeric answer"

    knowledge = plan.get("knowledge", {}) if isinstance(plan, dict) else {}
    diversity = knowledge.get("diversity", {}) if isinstance(knowledge, dict) else {}
    primary_scene = diversity.get("primary_scene", {})
    alternative_scenes = diversity.get("alternative_scenes", [])
    inspiration = knowledge.get("kb_inspiration", {}) if isinstance(knowledge, dict) else {}

    plan_tokens = set()
    if isinstance(primary_scene, dict):
        plan_tokens.update(_scene_tokens(primary_scene))
    for scene in alternative_scenes if isinstance(alternative_scenes, list) else []:
        if isinstance(scene, dict):
            plan_tokens.update(_scene_tokens(scene))
    plan_tokens.update(
        _keyword_tokens(
            [
                *_as_sequence(inspiration.get("scene_keywords")),
                *_as_sequence(inspiration.get("possible_roles")),
                *_as_sequence(inspiration.get("possible_units")),
            ]
        )
    )
    if plan_tokens:
        question_tokens = _keyword_tokens([question])
        if not (question_tokens & plan_tokens):
            return "question does not reflect the assigned plan scene or inspiration keywords"

    # A very broad structural guard only. Exact difficulty and step validity are
    # intentionally deferred to validation because plans are not rigid templates.
    if difficulty == "Hard" and len(steps) < 2:
        return "hard target produced too few reasoning steps"
    if difficulty == "Easy" and len(steps) > 5:
        return "easy target produced too many reasoning steps"
    return ""


def _should_log_progress(done: int, total: int) -> bool:
    if done <= 20:
        return True
    if total <= 500:
        return done % 5 == 0 or done == total
    interval = max(1, total // 200)
    return done % interval == 0 or done == total


def _format_seconds(value: float) -> str:
    total = max(0, int(value))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _classify_failure(error: str) -> str:
    normalized = str(error or "").lower()
    if "not a valid json" in normalized or "not valid json" in normalized:
        return "invalid_json"
    if "missing or invalid fields" in normalized:
        return "missing_fields"
    if "plan" in normalized or "scene" in normalized or "reasoning steps" in normalized:
        return "plan_mismatch"
    if any(
        marker in normalized
        for marker in (
            "timeout",
            "connection",
            "apierror",
            "internalserver",
            "serviceunavailable",
            "ratelimit",
            "http",
        )
    ):
        return "request_error"
    return "unknown"


def _next_action(failure_type: str) -> str:
    return "reuse_plan"


def _round_directory(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.rounds"


def _write_round_checkpoint(
    *,
    output_path: Optional[Path],
    raw_output_path: Optional[Path],
    failed_output_path: Optional[Path],
    round_index: int,
    round_plans: Sequence[Dict[str, Any]],
    round_successes: Sequence[Dict[str, Any]],
    round_raw_outputs: Sequence[Dict[str, Any]],
    round_failures: Sequence[Dict[str, Any]],
    cumulative_successes: Sequence[Dict[str, Any]],
    cumulative_raw_outputs: Sequence[Dict[str, Any]],
) -> None:
    if output_path is None or raw_output_path is None or failed_output_path is None:
        return

    round_dir = _round_directory(output_path)
    round_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"round_{round_index:03d}"
    write_jsonl(round_dir / f"{prefix}.plan.jsonl", round_plans)
    write_jsonl(round_dir / f"{prefix}.success.jsonl", round_successes)
    write_jsonl(round_dir / f"{prefix}.raw.jsonl", round_raw_outputs)
    write_jsonl(round_dir / f"{prefix}.failed.jsonl", round_failures)
    write_json(
        round_dir / f"{prefix}.summary.json",
        {
            "round": round_index,
            "attempted": len(round_plans),
            "success": len(round_successes),
            "failed": len(round_failures),
            "failure_types": {
                failure_type: sum(
                    1
                    for item in round_failures
                    if item.get("failure_type") == failure_type
                )
                for failure_type in sorted(
                    {
                        str(item.get("failure_type") or "unknown")
                        for item in round_failures
                    }
                )
            },
        },
    )

    # Canonical files are refreshed after every completed batch round.
    write_jsonl(output_path, cumulative_successes)
    write_jsonl(raw_output_path, cumulative_raw_outputs)
    write_jsonl(failed_output_path, round_failures)


def _plan_id(record: Dict[str, Any]) -> str:
    return str(record.get("plan_id") or "")


def _ordered_by_plan(
    records: Sequence[Dict[str, Any]],
    plan_order: Dict[str, int],
) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: plan_order.get(_plan_id(item), 10**12),
    )


def _load_existing_generation_state(
    *,
    output_path: Optional[Path],
    raw_output_path: Optional[Path],
    plan_order: Dict[str, int],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    successes: Dict[str, Dict[str, Any]] = {}
    raw_outputs: Dict[str, Dict[str, Any]] = {}

    if output_path is not None and output_path.exists():
        for record in read_jsonl(output_path):
            plan_id = _plan_id(record)
            if plan_id and plan_id in plan_order:
                successes[plan_id] = record

    if raw_output_path is not None and raw_output_path.exists():
        for record in read_jsonl(raw_output_path):
            plan_id = _plan_id(record)
            if plan_id and plan_id in plan_order:
                raw_outputs[plan_id] = record

    return successes, raw_outputs


def _load_resume_failures(
    *,
    failed_output_path: Optional[Path],
    completed_plan_ids: set[str],
    max_retries: int,
    retry_completed_failures: bool = False,
) -> Tuple[List[Dict[str, Any]], set[str], int]:
    if failed_output_path is None or not failed_output_path.exists():
        return [], set(), 0

    pending: List[Dict[str, Any]] = []
    failed_plan_ids: set[str] = set()
    next_round = 0
    infinite_rounds = max_retries < 0
    for failure in read_jsonl(failed_output_path):
        if not isinstance(failure, dict):
            continue
        failed_plan_id = str(failure.get("plan_id") or "")
        if failed_plan_id:
            failed_plan_ids.add(failed_plan_id)
        if (
            failed_plan_id
            and failed_plan_id in completed_plan_ids
            and not retry_completed_failures
        ):
            continue
        attempts = int(failure.get("attempts") or 0)
        retry_round = max(1, attempts)
        if not infinite_rounds and retry_round > max_retries:
            continue
        plan = failure.get("next_plan")
        if not isinstance(plan, dict):
            original_plan = failure.get("plan")
            if not isinstance(original_plan, dict):
                continue
            plan = replan_failed_plan(
                original_plan,
                str(failure.get("failure_type") or "unknown"),
                retry_round=retry_round,
            )
            failure["next_plan"] = plan
            failure["next_round"] = retry_round
        pending.append(
            {
                "plan": plan,
                "previous_failure": failure,
            }
        )
        next_round = max(next_round, retry_round)
    return pending, failed_plan_ids, next_round


def _write_generation_checkpoint(
    *,
    output_path: Optional[Path],
    raw_output_path: Optional[Path],
    failed_output_path: Optional[Path],
    summary_path: Optional[Path],
    cumulative_successes: Dict[str, Dict[str, Any]],
    cumulative_raw_outputs: Dict[str, Dict[str, Any]],
    latest_failures: Sequence[Dict[str, Any]],
    plan_order: Dict[str, int],
    total_plans: int,
    round_index: int,
    done_in_round: int,
    checkpoint_reason: str,
) -> None:
    if output_path is None or raw_output_path is None or failed_output_path is None:
        return

    ordered_successes = _ordered_by_plan(cumulative_successes.values(), plan_order)
    ordered_raw_outputs = _ordered_by_plan(cumulative_raw_outputs.values(), plan_order)
    write_jsonl(output_path, ordered_successes)
    write_jsonl(raw_output_path, ordered_raw_outputs)
    write_jsonl(failed_output_path, latest_failures)

    if summary_path is not None:
        write_json(
            summary_path,
            {
                "planned": total_plans,
                "generated": len(ordered_successes),
                "failed_latest_checkpoint": len(latest_failures),
                "remaining": max(0, total_plans - len(ordered_successes)),
                "round": round_index,
                "done_in_round": done_in_round,
                "checkpoint_reason": checkpoint_reason,
                "output": str(output_path),
                "raw_output": str(raw_output_path),
                "failed_output": str(failed_output_path),
                "round_output_dir": str(_round_directory(output_path)),
            },
        )


async def _generate_all_async(
    plans: Sequence[Dict[str, Any]],
    mastery_records: Sequence[Dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    concurrency: int,
    timeout: int,
    max_retries: int,
    max_tokens: int,
    temperature_map: Dict[str, float],
    top_p_map: Dict[str, float],
    enable_thinking: bool,
    force_json: bool,
    output_path: Optional[Path],
    raw_output_path: Optional[Path],
    failed_output_path: Optional[Path],
    round_retry_delay: float,
    resume: bool,
    checkpoint_every: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The brj environment must provide the openai package for async generation."
        ) from exc

    mastery = _mastery_lookup(mastery_records)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    client = AsyncOpenAI(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )

    try:
        served = await client.models.list()
        served_models = [
            str(item.id)
            for item in getattr(served, "data", [])
            if getattr(item, "id", None)
        ]
        model_aliases = _model_aliases(model)
        for served_model in served_models:
            if _model_aliases(served_model) & model_aliases:
                model = served_model
                break
    except Exception:
        # Generation will surface the original API error if the endpoint is not usable.
        pass

    async def generate_one(
        index: int,
        pending_item: Dict[str, Any],
        round_index: int,
    ) -> Tuple[int, Optional[Dict[str, Any]], Dict[str, Any]]:
        plan = pending_item["plan"]
        previous_failure = pending_item.get("previous_failure") or {}
        source_task_id = plan.get("source_task_id")
        plan_id = plan.get("plan_id")
        seed = mastery.get(str(source_task_id), {})
        difficulty = str(seed.get("target_difficulty") or "Equal")
        seed_question = normalize_whitespace(seed.get("question") or "")
        knowledge = plan.get("knowledge", {})

        retry_reason = str(previous_failure.get("error") or "")
        messages = _build_prompt(
            seed_question,
            difficulty,
            knowledge,
            attempt_index=round_index,
            retry_reason=retry_reason,
        )
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature_map.get(difficulty, 0.5),
            "top_p": top_p_map.get(difficulty, 0.5),
            "max_tokens": max_tokens,
        }
        if force_json:
            request["response_format"] = {"type": "json_object"}

        extra_body: Dict[str, Any] = {}
        if not enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if extra_body:
            request["extra_body"] = extra_body

        raw_output = ""
        candidate_question = ""
        try:
            async with semaphore:
                response = await client.chat.completions.create(**request)
            raw_output = response.choices[0].message.content or ""
            parsed, error = _parse_generated_output(raw_output)
            if parsed:
                candidate_question = parsed["question"]
                alignment_error = _plan_alignment_error(parsed, plan, difficulty)
                if alignment_error:
                    parsed = None
                    error = alignment_error
            if parsed:
                record = {
                    "source_task_id": source_task_id,
                    "plan_id": plan_id,
                    "difficulty": difficulty,
                    "question": parsed["question"],
                    "steps": parsed["steps"],
                    "answer": parsed["answer"],
                }
                raw_record = {
                    "source_task_id": source_task_id,
                    "plan_id": plan_id,
                    "difficulty": difficulty,
                    "round": round_index,
                    "attempts": round_index + 1,
                    "raw_model_output": raw_output,
                }
                return index, record, raw_record
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        failure_type = _classify_failure(error)
        failure = {
            "source_task_id": source_task_id,
            "plan_id": plan_id,
            "difficulty": difficulty,
            "round": round_index,
            "attempts": round_index + 1,
            "failure_type": failure_type,
            "next_action": _next_action(failure_type),
            "error": error or "unknown generation failure",
            "candidate_question": candidate_question,
            "raw_model_output": raw_output,
            "plan": plan,
        }
        return index, None, failure

    plan_order = {
        str(plan.get("plan_id")): index
        for index, plan in enumerate(plans)
    }
    if resume:
        cumulative_successes, cumulative_raw_outputs = _load_existing_generation_state(
            output_path=output_path,
            raw_output_path=raw_output_path,
            plan_order=plan_order,
        )
    else:
        cumulative_successes = {}
        cumulative_raw_outputs = {}
    completed_plan_ids = set(cumulative_successes)
    resume_failed_pending: List[Dict[str, Any]] = []
    resume_failed_plan_ids: set[str] = set()
    resume_round_index = 0
    if resume:
        (
            resume_failed_pending,
            resume_failed_plan_ids,
            resume_round_index,
        ) = _load_resume_failures(
            failed_output_path=failed_output_path,
            completed_plan_ids=completed_plan_ids,
            max_retries=max_retries,
            retry_completed_failures=_parse_bool_env(
                "GEN_RETRY_COMPLETED_FAILURES",
                False,
            ),
        )
    pending: List[Dict[str, Any]] = [
        {"plan": plan, "previous_failure": None}
        for plan in plans
        if str(plan.get("plan_id")) not in completed_plan_ids
        and str(plan.get("plan_id")) not in resume_failed_plan_ids
    ]
    if resume_failed_pending:
        pending = resume_failed_pending + pending
    final_failures: List[Dict[str, Any]] = []
    round_index = resume_round_index if resume_failed_pending else 0
    infinite_rounds = max_retries < 0
    summary_path = output_path.with_suffix(".summary.json") if output_path is not None else None
    if completed_plan_ids:
        print(
            f"[generate] resume enabled: loaded_success={len(completed_plan_ids)} "
            f"loaded_failed_retry={len(resume_failed_pending)} "
            f"remaining={len(pending)}",
            flush=True,
        )
        _write_generation_checkpoint(
            output_path=output_path,
            raw_output_path=raw_output_path,
            failed_output_path=failed_output_path,
            summary_path=summary_path,
            cumulative_successes=cumulative_successes,
            cumulative_raw_outputs=cumulative_raw_outputs,
            latest_failures=[],
            plan_order=plan_order,
            total_plans=len(plans),
            round_index=round_index,
            done_in_round=0,
            checkpoint_reason="resume_loaded",
        )
    print(
        f"[generate] config max_retries={max_retries} "
        f"max_tokens={max_tokens} concurrency={concurrency} "
        f"force_json={force_json} resume={resume}",
        flush=True,
    )

    try:
        while pending and (infinite_rounds or round_index <= max_retries):
            round_started_at = time.time()
            print(
                f"[generate] round={round_index} batch_size={len(pending)}",
                flush=True,
            )
            tasks = [
                asyncio.create_task(
                    generate_one(index, pending_item, round_index)
                )
                for index, pending_item in enumerate(pending)
            ]
            round_successes: List[Dict[str, Any]] = []
            round_raw_outputs: List[Dict[str, Any]] = []
            round_failures: List[Dict[str, Any]] = []
            done = 0

            for task in asyncio.as_completed(tasks):
                _, record, attempt_record = await task
                if record is not None:
                    plan_id = str(record.get("plan_id"))
                    cumulative_successes[plan_id] = record
                    cumulative_raw_outputs[plan_id] = attempt_record
                    round_successes.append(record)
                    round_raw_outputs.append(attempt_record)
                else:
                    round_failures.append(attempt_record)
                done += 1
                if checkpoint_every > 0 and done % checkpoint_every == 0:
                    _write_generation_checkpoint(
                        output_path=output_path,
                        raw_output_path=raw_output_path,
                        failed_output_path=failed_output_path,
                        summary_path=summary_path,
                        cumulative_successes=cumulative_successes,
                        cumulative_raw_outputs=cumulative_raw_outputs,
                        latest_failures=round_failures,
                        plan_order=plan_order,
                        total_plans=len(plans),
                        round_index=round_index,
                        done_in_round=done,
                        checkpoint_reason="periodic",
                    )
                    print(
                        f"[generate] checkpoint saved round={round_index} "
                        f"done={done}/{len(tasks)} total_success={len(cumulative_successes)}",
                        flush=True,
                    )
                if _should_log_progress(done, len(tasks)):
                    elapsed = time.time() - round_started_at
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (len(tasks) - done) / rate if rate > 0 else 0.0
                    print(
                        f"[generate] round={round_index} {done}/{len(tasks)} "
                        f"({done * 100.0 / max(1, len(tasks)):5.1f}%) "
                        f"elapsed={_format_seconds(elapsed)} "
                        f"eta={_format_seconds(eta)} "
                        f"success={len(round_successes)} "
                        f"failed={len(round_failures)}",
                        flush=True,
                    )

            can_retry = infinite_rounds or round_index < max_retries
            next_pending: List[Dict[str, Any]] = []
            for failure in round_failures:
                if can_retry:
                    next_plan = replan_failed_plan(
                        failure["plan"],
                        str(failure.get("failure_type") or "unknown"),
                        retry_round=round_index + 1,
                    )
                    failure["next_plan"] = next_plan
                    failure["next_round"] = round_index + 1
                    next_pending.append(
                        {
                            "plan": next_plan,
                            "previous_failure": failure,
                        }
                    )
                else:
                    failure["next_plan"] = None
                    failure["next_round"] = None

            ordered_successes = _ordered_by_plan(
                cumulative_successes.values(),
                plan_order,
            )
            ordered_raw_outputs = _ordered_by_plan(
                cumulative_raw_outputs.values(),
                plan_order,
            )
            _write_round_checkpoint(
                output_path=output_path,
                raw_output_path=raw_output_path,
                failed_output_path=failed_output_path,
                round_index=round_index,
                round_plans=[item["plan"] for item in pending],
                round_successes=round_successes,
                round_raw_outputs=round_raw_outputs,
                round_failures=round_failures,
                cumulative_successes=ordered_successes,
                cumulative_raw_outputs=ordered_raw_outputs,
            )
            if can_retry and failed_output_path is not None:
                # The next batch is reconstructed from the persisted failure
                # file, so an interrupted run leaves a complete retry input.
                persisted_failures = read_jsonl(failed_output_path)
                next_pending = [
                    {
                        "plan": failure["next_plan"],
                        "previous_failure": failure,
                    }
                    for failure in persisted_failures
                    if isinstance(failure.get("next_plan"), dict)
                ]

            final_failures = round_failures
            _write_generation_checkpoint(
                output_path=output_path,
                raw_output_path=raw_output_path,
                failed_output_path=failed_output_path,
                summary_path=summary_path,
                cumulative_successes=cumulative_successes,
                cumulative_raw_outputs=cumulative_raw_outputs,
                latest_failures=round_failures,
                plan_order=plan_order,
                total_plans=len(plans),
                round_index=round_index,
                done_in_round=done,
                checkpoint_reason="round_complete",
            )
            print(
                f"[generate] round={round_index} complete "
                f"success={len(round_successes)} failed={len(round_failures)} "
                f"next_batch={len(next_pending)}",
                flush=True,
            )
            pending = next_pending
            round_index += 1
            if pending and round_retry_delay > 0:
                await asyncio.sleep(round_retry_delay)
    finally:
        await client.close()

    ordered_successes = _ordered_by_plan(cumulative_successes.values(), plan_order)
    ordered_raw_outputs = _ordered_by_plan(cumulative_raw_outputs.values(), plan_order)
    return ordered_successes, ordered_raw_outputs, final_failures


def generate_post_mastery_questions(
    plans: Sequence[Dict[str, Any]],
    mastery_records: Sequence[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    concurrency: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature_map: Optional[Dict[str, float]] = None,
    top_p_map: Optional[Dict[str, float]] = None,
    enable_thinking: Optional[bool] = None,
    force_json: Optional[bool] = None,
    output_path: Optional[Path] = None,
    raw_output_path: Optional[Path] = None,
    failed_output_path: Optional[Path] = None,
    round_retry_delay: Optional[float] = None,
    resume: Optional[bool] = None,
    checkpoint_every: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    resolved_concurrency = concurrency
    if resolved_concurrency is None:
        resolved_concurrency = min(
            _parse_int_env("GEN_CONCURRENCY", 256),
            _parse_int_env("GEN_CONCURRENCY_CAP", 256),
        )
    return asyncio.run(
        _generate_all_async(
            plans,
            mastery_records,
            model=model
            or os.environ.get("GEN_MODEL")
            or os.environ.get("VLLM_MODEL")
            or "/root/brjverl/models/Qwen3.6-27B",
            base_url=base_url
            or os.environ.get("VLLM_BASE_URL")
            or "http://127.0.0.1:8911/v1",
            api_key=api_key
            or os.environ.get("VLLM_API_KEY")
            or "EMPTY",
            concurrency=max(1, resolved_concurrency),
            timeout=timeout
            if timeout is not None
            else _parse_int_env("VLLM_TIMEOUT", 600),
            max_retries=max_retries
            if max_retries is not None
            else _parse_int_env("GEN_MAX_RETRIES", 5),
            max_tokens=max_tokens
            if max_tokens is not None
            else _parse_int_env("GEN_MAX_TOKENS", 1200),
            temperature_map=temperature_map
            or _parse_float_map(
                os.environ.get("GEN_TEMPERATURE_MAP"),
                DEFAULT_TEMPERATURES,
            ),
            top_p_map=top_p_map
            or _parse_float_map(
                os.environ.get("GEN_TOP_P_MAP"),
                DEFAULT_TOP_PS,
            ),
            enable_thinking=enable_thinking
            if enable_thinking is not None
            else _parse_bool_env("GEN_ENABLE_THINKING", False),
            force_json=force_json
            if force_json is not None
            else _parse_bool_env("GEN_FORCE_JSON", False),
            output_path=output_path,
            raw_output_path=raw_output_path,
            failed_output_path=failed_output_path,
            round_retry_delay=round_retry_delay
            if round_retry_delay is not None
            else _parse_float_env("GEN_ROUND_RETRY_DELAY", 1.0),
            resume=resume
            if resume is not None
            else _parse_bool_env("GEN_RESUME", True),
            checkpoint_every=checkpoint_every
            if checkpoint_every is not None
            else max(1, _parse_int_env("GEN_CHECKPOINT_EVERY", 50)),
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic questions from the compact post-mastery plan."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mastery", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output", required=False)
    parser.add_argument("--failed-output", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("--concurrency", type=int, required=False)
    parser.add_argument("--max-retries", type=int, required=False)
    parser.add_argument("--max-tokens", type=int, required=False)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--checkpoint-every", type=int, required=False)
    args = parser.parse_args(argv)

    plan_path = Path(args.plan)
    mastery_path = Path(args.mastery)
    output_path = Path(args.output)
    raw_output_path = (
        Path(args.raw_output)
        if args.raw_output
        else output_path.with_name(f"{output_path.stem}.raw.jsonl")
    )
    failed_output_path = (
        Path(args.failed_output)
        if args.failed_output
        else output_path.with_name(f"{output_path.stem}.failed.jsonl")
    )

    outputs, raw_outputs, failures = generate_post_mastery_questions(
        read_jsonl(plan_path),
        read_jsonl(mastery_path),
        model=args.model,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        output_path=output_path,
        raw_output_path=raw_output_path,
        failed_output_path=failed_output_path,
    )
    write_jsonl(output_path, outputs)
    write_jsonl(raw_output_path, raw_outputs)
    write_jsonl(failed_output_path, failures)
    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "planned": len(outputs) + len(failures),
            "generated": len(outputs),
            "failed": len(failures),
            "rounds_completed": (
                max(
                    [
                        int(item.get("round") or 0)
                        for item in [*raw_outputs, *failures]
                    ],
                    default=-1,
                )
                + 1
            ),
            "max_retry_rounds": (
                args.max_retries
                if args.max_retries is not None
                else _parse_int_env("GEN_MAX_RETRIES", 5)
            ),
            "output": str(output_path),
            "raw_output": str(raw_output_path),
            "failed_output": str(failed_output_path),
            "round_output_dir": str(_round_directory(output_path)),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "generated": len(outputs),
                "failed": len(failures),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
