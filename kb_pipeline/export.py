from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .utils import read_jsonl, write_json, write_jsonl


TRAINING_INSTRUCTION = (
    "You need to provide the key steps along with the necessary calculations. "
    "After answering the question, you must restate the answer value again, "
    "without units, giving only the numerical value like "
    "'The answer is $\\boxed{100}$.'"
)
STEP_LABEL_RE = re.compile(r"^\s*(?:step\s*)?\d+\s*[:.)-]\s*", re.IGNORECASE)
CALCULATE_PREFIX_RE = re.compile(r"^\s*calculate\s+(.+?):\s*(.+)$", re.IGNORECASE)
STEP_CONNECTORS = ["First", "Next", "Then", "After that", "Finally"]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_answer(value: Any) -> str:
    text = _normalize_text(value)
    text = text.replace(",", "")
    text = text.replace("$", "")
    return text


def _normalize_steps(value: Any) -> List[str]:
    if isinstance(value, list):
        return [
            _normalize_text(item)
            for item in value
            if _normalize_text(item)
        ]
    if isinstance(value, dict):
        return [
            _normalize_text(item)
            for item in value.values()
            if _normalize_text(item)
        ]
    text = str(value or "").strip()
    if not text:
        return []
    return [
        _normalize_text(line)
        for line in text.splitlines()
        if _normalize_text(line)
    ]


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[:1].lower() + text[1:]


def _connector(index: int, total: int) -> str:
    if total > 1 and index == total:
        return "Finally"
    if index <= len(STEP_CONNECTORS):
        return STEP_CONNECTORS[index - 1]
    return "Then"


def _step_purpose(index: int, total: int) -> str:
    if total > 1 and index == total:
        return "this gives the requested final quantity"
    if index == 1:
        return "this gives the first intermediate value needed for the solution"
    return "this intermediate value is needed for the next part of the solution"


def _make_step_explanatory(text: str, index: int, total: int) -> str:
    content = STEP_LABEL_RE.sub("", text, count=1).strip()
    match = CALCULATE_PREFIX_RE.match(content)
    if match is None:
        return content
    goal = _lower_first(match.group(1).strip())
    calculation = match.group(2).strip()
    return (
        f"{_connector(index, total)}, find {goal}, because "
        f"{_step_purpose(index, total)}: {calculation}"
    )


def _format_steps_for_training(steps: Sequence[str]) -> List[str]:
    formatted: List[str] = []
    total = len([step for step in steps if _normalize_text(step)])
    for index, step in enumerate(steps, start=1):
        text = _normalize_text(step)
        if not text:
            continue
        content = _make_step_explanatory(text, index, total)
        formatted.append(f"Step {index}: {content}")
    return formatted


def _solution_text(record: Dict[str, Any], answer: str) -> str:
    steps = _normalize_steps(
        record.get("steps")
        or record.get("solution_steps")
        or record.get("solution")
    )
    if steps:
        body = "\n".join(_format_steps_for_training(steps))
    else:
        body = _normalize_text(record.get("solution", ""))
    final = f"The answer is $\\boxed{{{answer}}}$."
    if not body:
        return final
    stripped = body.rstrip()
    if stripped.endswith(final):
        return stripped
    return f"{stripped}\n{final}"


def build_training_records(quality_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for record in quality_records:
        if "passed" in record and not record.get("passed"):
            continue
        question = _normalize_text(record.get("question", ""))
        answer = _normalize_answer(record.get("answer", ""))
        if not question or not answer:
            continue
        outputs.append(
            {
                "instruction": TRAINING_INSTRUCTION,
                "input": question,
                "output": _solution_text(record, answer),
            }
        )
    return outputs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export final training data from quality-checked records.")
    parser.add_argument("--input", required=True, help="Quality-checked JSONL path")
    parser.add_argument("--output", required=True, help="Training JSONL path")
    parser.add_argument("--summary-output", required=False, help="Summary JSON path")
    args = parser.parse_args(argv)

    records = read_jsonl(Path(args.input))
    training = build_training_records(records)
    write_jsonl(Path(args.output), training)

    summary = {
        "input_count": len(records),
        "output_count": len(training),
        "pass_rate": round(len(training) / max(1, len(records)), 4),
        "format": "instruction/input/output",
        "answer_template": "The answer is $\\boxed{XXX}$.",
    }
    summary_path = Path(args.summary_output) if args.summary_output else Path(args.output).with_suffix(".summary.json")
    write_json(summary_path, summary)

    print(json.dumps({"output": str(args.output), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
