from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_MARK_RE = re.compile(r"####\s*(.+)$")
CHOICE_RE = re.compile(r"\b([A-E])\b")


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: Optional[str],
        model: Optional[str],
        api_key: Optional[str],
        timeout: int = 600,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8911/v1").rstrip("/")
        self.model = model or os.environ.get("VLLM_MODEL") or "local-model"
        self.api_key = api_key or os.environ.get("VLLM_API_KEY") or "EMPTY"
        self.timeout = max(1, int(timeout))
        self.max_retries = int(max_retries)
        self._resolved_model_once = False

    @staticmethod
    def _model_aliases(model: str) -> set[str]:
        normalized = str(model or "").strip().rstrip("/")
        if not normalized:
            return set()
        aliases = {normalized}
        basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
        if basename:
            aliases.add(basename)
        return aliases

    def _served_model_names(self) -> List[str]:
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=min(self.timeout, 10)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        names: List[str] = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name")
                if name:
                    names.append(str(name))
        return names

    def _use_matching_served_model(self) -> bool:
        expected = self._model_aliases(self.model)
        try:
            served_models = self._served_model_names()
        except Exception:
            return False
        for served_model in served_models:
            if self._model_aliases(served_model) & expected and served_model != self.model:
                self.model = served_model
                return True
        return False

    def chat(self, messages: List[Dict[str, str]], *, temperature: float, top_p: float, max_tokens: int) -> str:
        return self.chat_with_options(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def chat_with_options(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: Optional[int] = None,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Optional[BaseException] = None
        infinite_retries = self.max_retries < 0
        attempt = 0
        while infinite_retries or attempt <= self.max_retries:
            try:
                payload: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if presence_penalty:
                    payload["presence_penalty"] = presence_penalty
                if frequency_penalty:
                    payload["frequency_penalty"] = frequency_penalty
                if seed is not None:
                    payload["seed"] = int(seed)
                data = json.dumps(payload).encode("utf-8")
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=data,
                    headers=headers,
                    method="POST",
                )
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                decoded = json.loads(body)
                return decoded["choices"][0]["message"]["content"] or ""
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 404 and not self._resolved_model_once:
                    self._resolved_model_once = True
                    if self._use_matching_served_model():
                        continue
                if not infinite_retries and attempt >= self.max_retries:
                    break
                time.sleep(min(3.0 * (attempt + 1), 15.0))
                attempt += 1
        raise RuntimeError(f"evaluation request failed: {last_error}")


def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL records must be objects: {path}")
                records.append(payload)
    return records


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_answer_and_steps_from_gsm8k(answer_text: str) -> Tuple[str, str]:
    lines = [line.rstrip("\n") for line in str(answer_text or "").splitlines()]
    answer_index = next((i for i in range(len(lines) - 1, -1, -1) if "####" in lines[i]), -1)
    if answer_index == -1:
        raise ValueError("GSM8K answer line with '####' was not found")
    answer_value = lines[answer_index].split("####", 1)[-1].strip()
    steps_text = "\n".join(lines[:answer_index]).strip()
    return answer_value, steps_text


def normalize_choice_token(text: Any) -> str:
    match = re.search(r"\b([A-E])\b", str(text or ""), flags=re.IGNORECASE)
    return match.group(1).upper() if match else normalize_whitespace(text).upper()


def format_options(options: Any) -> str:
    if isinstance(options, list):
        items = [str(item).strip() for item in options if str(item).strip()]
    else:
        items = [str(options or "").strip()] if str(options or "").strip() else []
    return "[" + ", ".join(items) + "]"


def prepare_agieval_eng_qa_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    question = str(record.get("question", "") or "").strip()
    options_text = format_options(record.get("options"))
    if options_text != "[]":
        question = (
            f"{question} Choose the correct option from the given choices. "
            f"The options are as follows:{options_text}"
        )
    rationale = re.sub(
        r"^\s*Explanation\s*:\s*",
        "",
        str(record.get("rationale", record.get("solution_steps", "")) or "").strip(),
        flags=re.IGNORECASE,
    )
    return {
        "task_id": record.get("task_id", index),
        "question": question,
        "answer": normalize_choice_token(record.get("correct", record.get("answer", ""))),
        "solution_steps": rationale,
    }


def is_prepared_record(record: Dict[str, Any]) -> bool:
    for field in ("task_id", "question", "answer"):
        if field not in record or normalize_whitespace(record.get(field)) == "":
            return False
    return True


def prepare_record(record: Dict[str, Any], index: int, *, format_template: str) -> Dict[str, Any]:
    template = format_template.lower().strip()
    if template in {"auto", "passthrough", "none"} and is_prepared_record(record) and "####" not in str(record.get("answer", "")):
        prepared = dict(record)
        prepared.setdefault("task_id", index)
        prepared.setdefault("solution_steps", "")
        return prepared
    if template == "agieval_eng_qa":
        return prepare_agieval_eng_qa_record(record, index)
    if template in {"auto", "gsm8k"}:
        answer, solution_steps = extract_answer_and_steps_from_gsm8k(str(record.get("answer", "")))
        return {
            "task_id": record.get("task_id", index),
            "question": str(record.get("question", "")).strip(),
            "answer": answer,
            "solution_steps": solution_steps,
        }
    if template in {"passthrough", "none"}:
        prepared = dict(record)
        prepared.setdefault("task_id", index)
        prepared.setdefault("solution_steps", "")
        return prepared
    raise ValueError(f"unsupported EVAL_FORMAT_TEMPLATE={format_template}")


def prepare_records(records: Sequence[Dict[str, Any]], *, format_template: str, sample_limit: Optional[int]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    selected = records[:sample_limit] if sample_limit is not None else records
    for index, record in enumerate(selected):
        prepared.append(prepare_record(record, index, format_template=format_template))
    return prepared


def load_prompt(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"prompt file must contain non-empty content: {path}")
    return content


def build_model_input(prompt_text: str, question: str, *, prompt_mode: str, attempt_index: int, attempt_variation: bool) -> str:
    # The model-visible input is intentionally limited to the prompt and the
    # question. Reference answers and solution_steps are used only after the
    # model returns, for local scoring.
    body = f"{prompt_text}{question}" if prompt_mode == "legacy_concat" else f"{prompt_text}\n\n{question}"
    if attempt_variation and attempt_index > 0:
        body += f"\n\nThis is independent sample #{attempt_index + 1}. Produce a valid solution without copying earlier samples."
    return body


def build_messages(prompt_text: str, question: str, *, prompt_mode: str, attempt_index: int, attempt_variation: bool) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_model_input(
                prompt_text,
                question,
                prompt_mode=prompt_mode,
                attempt_index=attempt_index,
                attempt_variation=attempt_variation,
            ),
        }
    ]


def extract_final_answer(text: str, *, answer_extract_mode: str = "number") -> str:
    raw = str(text or "")
    boxed = BOXED_RE.findall(raw)
    if boxed:
        extracted = normalize_whitespace(boxed[-1])
        return normalize_choice_token(extracted) if answer_extract_mode == "choice" else extracted
    mark = FINAL_MARK_RE.search(raw)
    if mark:
        extracted = normalize_whitespace(mark.group(1))
        return normalize_choice_token(extracted) if answer_extract_mode == "choice" else extracted
    if answer_extract_mode == "choice":
        answer_match = re.search(
            r"(?:the\s+answer\s+is|answer)\s*[:：]?\s*(?:\$?\\boxed\{)?\s*([A-E])\b",
            raw,
            flags=re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1).upper()
        choices = CHOICE_RE.findall(raw)
        if choices:
            return choices[-1].upper()
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return normalize_choice_token(lines[-1] if lines else raw)
    numbers = NUMBER_RE.findall(raw.replace(",", ""))
    if numbers:
        return normalize_whitespace(numbers[-1])
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return normalize_whitespace(lines[-1] if lines else raw)


def extract_solution_steps(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [line.rstrip() for line in raw.splitlines()]
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "####" in stripped:
            break
        if "the answer is" in stripped.lower() or "\\boxed" in stripped:
            break
        kept.append(stripped)
    if kept:
        return "\n".join(kept).strip()
    boxed_match = BOXED_RE.search(raw)
    if boxed_match:
        return raw[: boxed_match.start()].strip()
    return raw


def normalize_number_token(text: str) -> str:
    cleaned = normalize_whitespace(text).lower()
    cleaned = cleaned.replace("$", "").replace(",", "").replace(" ", "").replace("%", "")
    return cleaned


def is_correct_answer(candidate: str, reference: str, *, answer_extract_mode: str = "number") -> bool:
    if not candidate or not reference:
        return False
    if answer_extract_mode == "choice":
        return normalize_choice_token(candidate) == normalize_choice_token(reference)
    cand = normalize_number_token(candidate)
    ref = normalize_number_token(reference)
    if not cand or not ref:
        return candidate.strip().lower() == reference.strip().lower()
    try:
        return math.isclose(float(cand), float(ref), rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return cand == ref


def answer_records(
    records: Sequence[Dict[str, Any]],
    *,
    client: OpenAICompatibleClient,
    prompt_text: str,
    n_answers: int,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    presence_penalty: float,
    frequency_penalty: float,
    seed_base: Optional[int],
    prompt_mode: str,
    attempt_variation: bool,
    answer_extract_mode: str,
    existing_predictions: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    expected_keys = {
        (str(record.get("task_id")), attempt)
        for record in records
        for attempt in range(max(1, n_answers))
    }
    existing_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for item in existing_predictions or []:
        key = (str(item.get("task_id")), int(item.get("attempt_index", 0) or 0))
        if key in expected_keys:
            existing_by_key[key] = dict(item)

    tasks: List[Tuple[Dict[str, Any], int]] = []
    for record in records:
        for attempt in range(max(1, n_answers)):
            key = (str(record.get("task_id")), attempt)
            if key not in existing_by_key:
                tasks.append((record, attempt))

    def run_one(record: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        question = normalize_whitespace(record.get("question", ""))
        reference = normalize_whitespace(record.get("answer", ""))
        seed = None
        if seed_base is not None:
            seed = int(seed_base) + (abs(hash((str(record.get("task_id")), attempt))) % 1_000_000_000)
        raw = client.chat_with_options(
            build_messages(
                prompt_text,
                question,
                prompt_mode=prompt_mode,
                attempt_index=attempt,
                attempt_variation=attempt_variation,
            ),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )
        extracted = extract_final_answer(raw, answer_extract_mode=answer_extract_mode)
        model_solution_steps = extract_solution_steps(raw)
        return {
            "task_id": record.get("task_id"),
            "attempt_index": attempt,
            "question": question,
            "reference_answer": reference,
            "model_solution_steps": model_solution_steps,
            "solution_steps": model_solution_steps,
            "raw_output": raw,
            "extracted_answer": extracted,
            "is_correct": is_correct_answer(extracted, reference, answer_extract_mode=answer_extract_mode),
            "request_visible_fields": ["question"],
            "prompt_mode": prompt_mode,
            "answer_extract_mode": answer_extract_mode,
            "seed": seed,
        }

    outputs_by_key = dict(existing_by_key)
    started = time.time()
    pending_total = len(tasks)
    expected_total = len(expected_keys)
    existing_total = len(existing_by_key)
    print(
        f"[eval] expected_predictions={expected_total} existing={existing_total} pending={pending_total}",
        flush=True,
    )
    if pending_total > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(1, concurrency), pending_total)) as executor:
            future_map = {executor.submit(run_one, record, attempt): (record, attempt) for record, attempt in tasks}
            for done, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                record, attempt = future_map[future]
                result = future.result()
                outputs_by_key[(str(record.get("task_id")), attempt)] = result
                if done <= 20 or done % 20 == 0 or done == pending_total:
                    elapsed = max(0.001, time.time() - started)
                    total_done = existing_total + done
                    print(
                        f"[eval] answered pending={done}/{pending_total} "
                        f"total={total_done}/{expected_total} rate={done / elapsed:.2f}/s",
                        flush=True,
                    )

    ordered: List[Dict[str, Any]] = []
    for record in records:
        for attempt in range(max(1, n_answers)):
            item = outputs_by_key.get((str(record.get("task_id")), attempt))
            if item is not None:
                ordered.append(item)
    missing = expected_total - len(ordered)
    if missing:
        print(f"[eval] warning: missing_predictions={missing}/{expected_total}", flush=True)
    return ordered


def build_report(records: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]], *, n_answers: int) -> Dict[str, Any]:
    by_task: Dict[str, List[Dict[str, Any]]] = {str(record.get("task_id")): [] for record in records}
    for prediction in predictions:
        by_task.setdefault(str(prediction.get("task_id")), []).append(prediction)
    for values in by_task.values():
        values.sort(key=lambda item: int(item.get("attempt_index", 0) or 0))

    total_questions = len(records)
    expected_predictions = total_questions * max(1, n_answers)
    total_predictions = len(predictions)
    missing_predictions = max(0, expected_predictions - total_predictions)
    correct_predictions = sum(1 for item in predictions if item.get("is_correct"))
    pass_at_k: Dict[str, float] = {}
    for k in range(1, max(1, n_answers) + 1):
        passed = 0
        for record in records:
            attempts = by_task.get(str(record.get("task_id")), [])[:k]
            if any(item.get("is_correct") for item in attempts):
                passed += 1
        pass_at_k[f"pass@{k}"] = round(passed / total_questions, 6) if total_questions else 0.0

    per_task = []
    for record in records:
        attempts = by_task.get(str(record.get("task_id")), [])
        correct = sum(1 for item in attempts if item.get("is_correct"))
        per_task.append(
            {
                "task_id": record.get("task_id"),
                "reference_answer": record.get("answer"),
                "attempts": len(attempts),
                "correct": correct,
                "pass": correct > 0,
            }
        )

    return {
        "total_questions": total_questions,
        "n_answers": max(1, n_answers),
        "expected_predictions": expected_predictions,
        "total_predictions": total_predictions,
        "missing_predictions": missing_predictions,
        "correct_predictions": correct_predictions,
        "sample_accuracy": round(correct_predictions / total_predictions, 6) if total_predictions else 0.0,
        "pass_at_k": pass_at_k,
        "per_task": per_task,
    }


def write_markdown_report(path: Path, report: Dict[str, Any], *, dataset_name: str, model: str, input_path: str) -> None:
    lines = [
        f"# Model Evaluation Report",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Input: `{input_path}`",
        f"- Model: `{model}`",
        f"- Questions: {report['total_questions']}",
        f"- Answers per question: {report['n_answers']}",
        f"- Expected predictions: {report['expected_predictions']}",
        f"- Total predictions: {report['total_predictions']}",
        f"- Missing predictions: {report['missing_predictions']}",
        f"- Sample accuracy: {report['sample_accuracy']:.4f}",
        "",
        "## Pass@k",
        "",
    ]
    for key, value in report["pass_at_k"].items():
        lines.append(f"- {key}: {value:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Independent model-answer accuracy evaluation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--format-template", default="auto")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--n-answers", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["chat", "legacy_concat"], default="legacy_concat")
    parser.add_argument("--answer-extract-mode", choices=["number", "choice"], default=os.environ.get("EVAL_ANSWER_EXTRACT_MODE", "number"))
    parser.add_argument("--attempt-variation", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=None)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    prepared_path = output_dir / "prepared.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "report.json"
    report_md_path = output_dir / "report.md"

    raw_records = read_jsonl(input_path)
    prepared = prepare_records(raw_records, format_template=args.format_template, sample_limit=args.sample_limit)
    write_jsonl(prepared_path, prepared)

    existing = read_jsonl(predictions_path) if args.resume and predictions_path.exists() else []
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    prompt_text = load_prompt(Path(args.prompt))
    predictions = answer_records(
        prepared,
        client=client,
        prompt_text=prompt_text,
        n_answers=max(1, args.n_answers),
        concurrency=max(1, args.concurrency),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        seed_base=args.seed_base,
        prompt_mode=args.prompt_mode,
        attempt_variation=args.attempt_variation,
        answer_extract_mode=args.answer_extract_mode,
        existing_predictions=existing,
    )
    write_jsonl(predictions_path, predictions)
    report = build_report(prepared, predictions, n_answers=max(1, args.n_answers))
    report["files"] = {
        "prepared": str(prepared_path),
        "predictions": str(predictions_path),
        "report_json": str(report_path),
        "report_md": str(report_md_path),
    }
    report["config"] = {
        "dataset_name": args.dataset_name,
        "input_path": str(input_path),
        "format_template": args.format_template,
        "model": args.model,
        "prompt": str(args.prompt),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "seed_base": args.seed_base,
        "prompt_mode": args.prompt_mode,
        "answer_extract_mode": args.answer_extract_mode,
        "attempt_variation": args.attempt_variation,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "concurrency": args.concurrency,
    }
    write_json(report_path, report)
    write_markdown_report(report_md_path, report, dataset_name=args.dataset_name, model=args.model, input_path=str(input_path))
    print(json.dumps({k: report[k] for k in ("total_questions", "n_answers", "sample_accuracy", "pass_at_k", "files")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
