from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .utils import read_jsonl, write_json, write_jsonl


def build_training_records(quality_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for record in quality_records:
        if not record.get("passed"):
            continue
        outputs.append(
            {
                "task_id": record.get("task_id"),
                "source_task_id": record.get("source_task_id"),
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "solution": record.get("solution", ""),
                "difficulty_bucket": record.get("difficulty_bucket", ""),
                "step_count": record.get("step_count", 0),
                "source_question": record.get("source_question", ""),
                "source_answer": record.get("source_answer", ""),
                "source_knowledge": record.get("source_knowledge", {}),
                "generation_target": record.get("generation_target", {}),
                "qc_report": record.get("qc_report", {}),
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
    }
    summary_path = Path(args.summary_output) if args.summary_output else Path(args.output).with_suffix(".summary.json")
    write_json(summary_path, summary)

    print(json.dumps({"output": str(args.output), "summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

