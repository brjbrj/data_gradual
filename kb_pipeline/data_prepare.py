from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .client import VLLMClient


FormatHandler = Callable[[Dict[str, Any], int], Optional[Dict[str, Any]]]

FORMATTED_FIELDS = ("task_id", "question", "answer", "solution_steps", "proficiency_score")


def extract_answer_and_steps_from_gsm8k(answer_text: str) -> Tuple[str, str]:
    lines = [line.rstrip("\n") for line in str(answer_text or "").splitlines()]
    answer_index = next((i for i in range(len(lines) - 1, -1, -1) if "####" in lines[i]), -1)
    if answer_index == -1:
        raise ValueError("GSM8K answer line with '####' was not found")
    answer_value = lines[answer_index].split("####", 1)[-1].strip()
    steps_text = "\n".join(lines[:answer_index]).strip()
    return answer_value, steps_text


def gsm8k_format(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(f"record must be a JSON object, got {type(record).__name__}")
    answer, solution_steps = extract_answer_and_steps_from_gsm8k(str(record.get("answer", "")))
    return {
        "task_id": record.get("task_id", index),
        "question": str(record.get("question", "")).strip(),
        "answer": answer,
        "solution_steps": solution_steps,
        "proficiency_score": record.get("proficiency_score", 0),
        "question_type": str(record.get("question_type", "") or "").strip(),
    }


def _normalize_choice_answer(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"\b([A-Za-z])\b", text)
    return match.group(1).upper() if match else text.upper()


def _format_options(options: Any) -> str:
    if isinstance(options, list):
        items = [str(item).strip() for item in options if str(item).strip()]
    else:
        items = [str(options or "").strip()] if str(options or "").strip() else []
    return "[" + ", ".join(items) + "]"


def _clean_agieval_rationale(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*Explanation\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def agieval_eng_qa_format(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(f"record must be a JSON object, got {type(record).__name__}")
    question = str(record.get("question", "") or "").strip()
    options_text = _format_options(record.get("options"))
    if options_text != "[]":
        question = (
            f"{question} Choose the correct option from the given choices. "
            f"The options are as follows:{options_text}"
        )
    return {
        "task_id": record.get("task_id", index),
        "question": question,
        "answer": _normalize_choice_answer(record.get("correct", record.get("answer", ""))),
        "solution_steps": _clean_agieval_rationale(record.get("rationale", record.get("solution_steps", ""))),
        "proficiency_score": record.get("proficiency_score", 0),
        "question_type": str(record.get("question_type", "") or "").strip(),
    }


def passthrough_format(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    normalized = dict(record)
    normalized.setdefault("task_id", index)
    normalized.setdefault("proficiency_score", 0)
    normalized.setdefault("question_type", "")
    return normalized


FORMAT_HANDLERS: Dict[str, FormatHandler] = {
    "gsm8k": gsm8k_format,
    "agieval_eng_qa": agieval_eng_qa_format,
    "passthrough": passthrough_format,
    "none": passthrough_format,
}


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_num} is not a JSON object")
            yield line_num, payload


def inspect_jsonl_schema(input_path: Path, *, sample_limit: Optional[int] = None) -> Dict[str, Any]:
    total = 0
    formatted_count = 0
    classified_count = 0
    missing_format_fields: Dict[str, int] = {field: 0 for field in FORMATTED_FIELDS}
    missing_question_type = 0
    first_missing_format_line: Optional[int] = None
    first_missing_question_type_line: Optional[int] = None

    for line_num, record in _iter_jsonl(input_path):
        if sample_limit is not None and total >= sample_limit:
            break
        total += 1
        missing_fields = [
            field
            for field in FORMATTED_FIELDS
            if field not in record or record.get(field) is None or str(record.get(field)).strip() == ""
        ]
        if missing_fields:
            if first_missing_format_line is None:
                first_missing_format_line = line_num
            for field in missing_fields:
                missing_format_fields[field] += 1
        else:
            formatted_count += 1

        if str(record.get("question_type", "") or "").strip():
            classified_count += 1
        else:
            missing_question_type += 1
            if first_missing_question_type_line is None:
                first_missing_question_type_line = line_num

    is_formatted = total > 0 and formatted_count == total
    has_question_type = total > 0 and classified_count == total
    return {
        "input_path": str(input_path),
        "sample_limit": sample_limit,
        "total_count": total,
        "formatted_count": formatted_count,
        "classified_count": classified_count,
        "is_formatted": is_formatted,
        "needs_format": not is_formatted,
        "has_question_type": has_question_type,
        "needs_classify": not has_question_type,
        "missing_format_fields": missing_format_fields,
        "missing_question_type_count": missing_question_type,
        "first_missing_format_line": first_missing_format_line,
        "first_missing_question_type_line": first_missing_question_type_line,
    }


def _atomic_write_jsonl(path: Path, records: Iterable[Dict[str, Any]], *, backup: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    )
    temp_path = Path(temp_file.name)
    try:
        with temp_file:
            for record in records:
                temp_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if backup and path.exists():
            shutil.copy2(path, str(path) + ".backup")
        shutil.move(str(temp_path), str(path))
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def format_jsonl(input_path: Path, output_path: Path, *, template: str, sample_limit: Optional[int] = None) -> Dict[str, Any]:
    template_key = template.lower().strip()
    if template_key not in FORMAT_HANDLERS:
        supported = ", ".join(sorted(FORMAT_HANDLERS))
        raise ValueError(f"unsupported format template '{template}'; supported: {supported}")
    handler = FORMAT_HANDLERS[template_key]

    records: List[Dict[str, Any]] = []
    total = 0
    errors: List[Dict[str, Any]] = []
    for index, (line_num, payload) in enumerate(_iter_jsonl(input_path)):
        if sample_limit is not None and total >= sample_limit:
            break
        total += 1
        try:
            formatted = handler(payload, index)
            if formatted is not None:
                records.append(formatted)
        except Exception as exc:
            errors.append({"line": line_num, "error": str(exc)})

    _atomic_write_jsonl(output_path, records)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "format_template": template_key,
        "read_count": total,
        "write_count": len(records),
        "error_count": len(errors),
        "errors": errors[:20],
    }


def copy_jsonl(input_path: Path, output_path: Path, *, sample_limit: Optional[int] = None) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    total = 0
    for _, payload in _iter_jsonl(input_path):
        if sample_limit is not None and total >= sample_limit:
            break
        total += 1
        records.append(dict(payload))
    if input_path.resolve() != output_path.resolve():
        _atomic_write_jsonl(output_path, records)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "format_template": "already_formatted",
        "read_count": total,
        "write_count": total,
        "error_count": 0,
        "errors": [],
        "skipped": True,
        "reason": "input already has required formatted fields",
    }


def _load_classify_prompt(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "categories": str(payload.get("categories", "")),
        "role_description": str(payload.get("role_description", "")),
        "requirement": str(payload.get("requirement", "")),
    }


def _split_categories(categories: str) -> List[str]:
    return [item.strip() for item in categories.split(",") if item.strip()]


def match_first_category(categories: Sequence[str], response: str) -> str:
    cleaned = str(response or "").strip()
    for category in categories:
        if category == cleaned or category in cleaned:
            return category
    return ""


def _build_classify_messages(question: str, prompt: Dict[str, str]) -> List[Dict[str, str]]:
    user_prompt = (
        "Please classify the following questions.\n\n"
        f"Questions:{question}\n\n"
        f"Optional categories:{prompt['categories']}"
    )
    if prompt.get("requirement"):
        user_prompt += f"\n\nClassification requirements:{prompt['requirement']}"
    return [
        {"role": "system", "content": prompt["role_description"]},
        {"role": "user", "content": user_prompt},
    ]


def _build_reclassify_messages(question: str, prompt: Dict[str, str], invalid_response: str) -> List[Dict[str, str]]:
    messages = _build_classify_messages(question, prompt)
    messages.append(
        {
            "role": "assistant",
            "content": str(invalid_response or "").strip()[:500],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "The previous answer is invalid because it is not exactly one "
                "of the optional categories. Reclassify the same question now. "
                f"Allowed categories: {prompt['categories']}. "
                "Output exactly one allowed category and nothing else."
            ),
        }
    )
    return messages


def classify_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    prompt_path: Path,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_retries: int,
    overwrite_existing: bool = False,
) -> Dict[str, Any]:
    prompt = _load_classify_prompt(prompt_path)
    categories = _split_categories(prompt["categories"])
    records = [payload for _, payload in _iter_jsonl(input_path)]
    client = VLLMClient(base_url=base_url, model=model, api_key=api_key)

    pending: List[Tuple[int, Dict[str, Any]]] = []
    for index, record in enumerate(records):
        if not overwrite_existing and str(record.get("question_type", "") or "").strip():
            continue
        pending.append((index, record))

    stats: Dict[str, int] = {}
    errors: List[Dict[str, Any]] = []

    def classify_one(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, str, Optional[str]]:
        index, record = item
        question = str(record.get("question", "") or "").strip()
        if not question:
            return index, "", "missing question"
        attempt = 0
        infinite_retries = int(max_retries) < 0
        last_error = ""
        last_response = ""
        while infinite_retries or attempt <= int(max_retries):
            messages = (
                _build_classify_messages(question, prompt)
                if attempt == 0
                else _build_reclassify_messages(question, prompt, last_response or last_error)
            )
            try:
                response = client.chat(
                    messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as exc:
                last_error = str(exc)
                if not infinite_retries and attempt >= int(max_retries):
                    return index, "", last_error
                time.sleep(min(3.0 * (attempt + 1), 15.0))
                attempt += 1
                continue
            last_response = response
            category = match_first_category(categories, response)
            if category:
                return index, category, None
            last_error = f"classification response did not match categories: {response[:200]}"
            if not infinite_retries and attempt >= int(max_retries):
                return index, "", last_error
            time.sleep(min(1.0 * (attempt + 1), 5.0))
            attempt += 1
        return index, "", last_error or "classification failed"

    workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(classify_one, item) for item in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            index, category, error = future.result()
            records[index]["question_type"] = category
            if category:
                stats[category] = stats.get(category, 0) + 1
            if error:
                errors.append({"task_id": records[index].get("task_id"), "error": error})
            if done % 20 == 0 or done == len(futures):
                print(f"[prepare] classified {done}/{len(futures)} pending records")

    _atomic_write_jsonl(output_path, records, backup=input_path == output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "prompt_path": str(prompt_path),
        "total_count": len(records),
        "classified_count": len(pending) - len(errors),
        "skipped_existing_count": len(records) - len(pending),
        "error_count": len(errors),
        "category_stats": stats,
        "errors": errors[:20],
    }


def prepare_data(
    input_path: Path,
    output_path: Path,
    *,
    format_template: str,
    classify: bool,
    prompt_path: Path,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    sample_limit: Optional[int] = None,
    overwrite_classification: bool = False,
    force_format: bool = False,
    skip_format: bool = False,
    classify_max_retries: int = 3,
) -> Dict[str, Any]:
    source_schema = inspect_jsonl_schema(input_path, sample_limit=sample_limit)
    should_format = force_format or (not skip_format and source_schema["needs_format"])
    if should_format:
        format_stats = format_jsonl(input_path, output_path, template=format_template, sample_limit=sample_limit)
        classify_input_schema = inspect_jsonl_schema(output_path)
    else:
        format_stats = copy_jsonl(input_path, output_path, sample_limit=sample_limit)
        classify_input_schema = inspect_jsonl_schema(output_path)

    classify_stats: Optional[Dict[str, Any]] = None
    should_classify = classify and (
        overwrite_classification or classify_input_schema["needs_classify"]
    )
    if should_classify:
        classify_stats = classify_jsonl(
            output_path,
            output_path,
            prompt_path=prompt_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            concurrency=concurrency,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=classify_max_retries,
            overwrite_existing=overwrite_classification,
        )
    elif classify:
        classify_stats = {
            "input_path": str(output_path),
            "output_path": str(output_path),
            "total_count": classify_input_schema["total_count"],
            "classified_count": 0,
            "skipped_existing_count": classify_input_schema["total_count"],
            "error_count": 0,
            "category_stats": {},
            "errors": [],
            "skipped": True,
            "reason": "all records already have question_type",
        }
    return {
        "source_schema": source_schema,
        "prepared_schema_before_classification": classify_input_schema,
        "format": format_stats,
        "classification": classify_stats,
    }


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _default_project_root()
    parser = argparse.ArgumentParser(description="Prepare raw math JSONL data before KB construction.")
    parser.add_argument("--input", required=True, help="Raw input JSONL path")
    parser.add_argument("--output", required=False, help="Prepared output JSONL path")
    parser.add_argument("--format-template", default=os.environ.get("DATA_FORMAT_TEMPLATE", "gsm8k"))
    parser.add_argument("--inspect", action="store_true", help="Only inspect input schema and exit")
    parser.add_argument("--inspect-limit", type=int, default=None, help="Optional record cap for schema inspection")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--force-format", action="store_true")
    format_group.add_argument("--skip-format", action="store_true")
    classify_group = parser.add_mutually_exclusive_group()
    classify_group.add_argument("--classify", dest="classify", action="store_true")
    classify_group.add_argument("--no-classify", dest="classify", action="store_false")
    parser.set_defaults(classify=True)
    parser.add_argument("--classify-prompt", default=str(root / "prompt" / "classify.json"))
    parser.add_argument("--model", default=os.environ.get("CLASSIFY_MODEL") or os.environ.get("VLLM_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("CLASSIFY_BASE_URL") or os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("CLASSIFY_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CLASSIFY_CONCURRENCY", "16")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("CLASSIFY_TEMPERATURE", "0.1")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("CLASSIFY_TOP_P", "0.9")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("CLASSIFY_MAX_TOKENS", "50")))
    parser.add_argument("--classify-max-retries", type=int, default=int(os.environ.get("CLASSIFY_MAX_RETRIES", "3")))
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--overwrite-classification", action="store_true")
    args = parser.parse_args(argv)

    if args.inspect:
        stats = inspect_jsonl_schema(Path(args.input), sample_limit=args.inspect_limit)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    if not args.output:
        parser.error("--output is required unless --inspect is used")

    stats = prepare_data(
        Path(args.input),
        Path(args.output),
        format_template=args.format_template,
        classify=args.classify,
        prompt_path=Path(args.classify_prompt),
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        sample_limit=args.sample_limit,
        overwrite_classification=args.overwrite_classification,
        force_format=args.force_format,
        skip_format=args.skip_format,
        classify_max_retries=args.classify_max_retries,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
