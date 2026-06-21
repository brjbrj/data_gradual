from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar


T = TypeVar("T")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if is_dataclass(record):
                record = asdict(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def dedupe_preserve_order(items: Sequence[T]) -> List[T]:
    seen = set()
    ordered: List[T] = []
    for item in items:
        key = item
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def safe_json_from_text(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


def safe_get(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def lookup_key(mapping: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    key_str = str(key)
    if key_str in mapping:
        return mapping[key_str]
    try:
        key_int = int(key)
        if key_int in mapping:
            return mapping[key_int]
    except Exception:
        pass
    return default
