from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .utils import dedupe_preserve_order, normalize_whitespace, read_jsonl, write_json, write_jsonl


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "he", "her", "his", "how", "i", "if", "in", "is", "it", "its", "more", "most",
    "much", "many", "my", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "there", "these", "they", "this", "those", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "than", "then", "into", "over", "under", "after", "before", "about", "between",
    "across", "each", "every", "per", "day", "days", "week", "weeks", "month",
    "months", "year", "years", "old", "new", "daily", "total", "left", "right",
    "all", "any", "some", "do", "does", "did", "doing", "done", "can", "could",
    "would", "should", "may", "might", "must", "am", "been", "being", "only",
    "also", "again", "both", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten",
}

QUESTION_WORDS = {"how", "what", "which", "who", "whom", "whose", "when", "where", "why"}

ACTION_WORDS = {
    "add", "adds", "added", "adding", "buy", "buys", "bought", "buying", "pay",
    "pays", "paid", "paying", "sell", "sells", "sold", "selling", "make", "makes",
    "made", "making", "give", "gives", "gave", "giving", "get", "gets", "got",
    "getting", "take", "takes", "took", "taking", "run", "runs", "ran", "running",
    "eat", "eats", "ate", "eating", "feed", "feeds", "fed", "feeding", "bake",
    "bakes", "baked", "baking", "decide", "decides", "decided", "deciding", "try",
    "tries", "tried", "trying", "increase", "increases", "increased", "increasing",
    "decrease", "decreases", "decreased", "decreasing", "need", "needs", "needed",
    "ask", "asks", "asked", "asking", "have", "has", "had", "having", "cost",
    "costs", "costed", "spend", "spends", "spent", "spending", "save", "saves",
    "saved", "saving", "share", "shares", "shared", "sharing", "split", "splits",
    "cut", "cuts", "cutting", "use", "uses", "used", "using", "contain", "contains",
    "contained", "containing", "remain", "remains", "remaining",
}

UNIT_WORDS = {
    "dollar", "dollars", "cent", "cents", "percent", "percentages", "egg", "eggs",
    "bolt", "bolts", "meter", "meters", "cup", "cups", "minute", "minutes", "hour",
    "hours", "day", "days", "week", "weeks", "year", "years", "item", "items",
    "box", "boxes", "bag", "bags", "book", "books", "people", "student", "students",
    "chicken", "chickens", "duck", "ducks", "flock", "flocks", "house", "houses",
    "profit", "profits", "fiber", "fibers", "marble", "marbles", "coin", "coins",
    "page", "pages",
}

NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
MONEY_RE = re.compile(r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
CAPITAL_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:['’]s)?)\b")
CAPITAL_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
GSM8K_STEP_RE = re.compile(r"<<\s*(.*?)\s*=\s*(.*?)\s*>>", re.DOTALL)


@dataclass
class KBItem:
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


def _extract_solution_text(record: dict) -> Tuple[str, str, str]:
    if isinstance(record.get("solution_steps"), str):
        return record.get("solution_steps", "").strip(), str(record.get("answer", "")).strip(), "formatted"

    raw_answer = record.get("answer", "")
    if isinstance(raw_answer, str) and "####" in raw_answer:
        lines = [line.rstrip("\n") for line in raw_answer.splitlines()]
        answer_index = next((i for i in range(len(lines) - 1, -1, -1) if "####" in lines[i]), -1)
        if answer_index == -1:
            return "", str(raw_answer).strip(), "raw_gsm8k"
        answer_value = lines[answer_index].split("####", 1)[-1].strip()
        solution_text = "\n".join(lines[:answer_index]).strip()
        return solution_text, answer_value, "raw_gsm8k"

    return "", str(raw_answer).strip(), "unknown"


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def _is_content_token(token: str) -> bool:
    low = token.lower()
    return low not in STOPWORDS and low not in QUESTION_WORDS and low not in ACTION_WORDS and low not in UNIT_WORDS and len(low) > 2


def _extract_person_entities(question: str, limit: int = 12) -> List[str]:
    phrases: List[str] = []
    seen = set()
    for regex in (CAPITAL_NAME_RE, CAPITAL_PHRASE_RE):
        for match in regex.finditer(question):
            phrase = normalize_whitespace(match.group(1) if regex is CAPITAL_NAME_RE else match.group(0))
            if not phrase:
                continue
            base = phrase.split()[0].replace("’s", "").replace("'s", "")
            if base.lower() in STOPWORDS or base.lower() in QUESTION_WORDS or base.lower() in ACTION_WORDS:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    return phrases


def _extract_focus_terms(question: str, limit: int = 18) -> List[str]:
    tokens = _tokenize_words(question)
    phrases: List[str] = []

    current: List[str] = []
    for raw in tokens:
        if _is_content_token(raw):
            current.append(raw)
            if len(current) == 3:
                phrases.append(" ".join(current))
                current = []
        else:
            if 1 <= len(current) <= 3 and any(len(tok) > 3 for tok in current):
                phrases.append(" ".join(current))
            current = []
    if 1 <= len(current) <= 3 and any(len(tok) > 3 for tok in current):
        phrases.append(" ".join(current))

    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if _is_content_token(a) and _is_content_token(b):
            phrases.append(f"{a} {b}")
    for i in range(len(tokens) - 2):
        tri = tokens[i : i + 3]
        if all(_is_content_token(tok) for tok in tri):
            phrases.append(" ".join(tri))

    return dedupe_preserve_order([normalize_whitespace(p) for p in phrases if normalize_whitespace(p)])[:limit]


def _extract_units(text: str) -> List[str]:
    unit_patterns = [
        r"\bdollars?\b", r"\bcents?\b", r"\bpercent\b", r"\bpercentages?\b",
        r"\beggs?\b", r"\bbolts?\b", r"\bmeters?\b", r"\bcups?\b", r"\bminutes?\b",
        r"\bhours?\b", r"\bdays?\b", r"\bweeks?\b", r"\byears?\b", r"\bitems?\b",
        r"\bboxes?\b", r"\bbags?\b", r"\bbooks?\b", r"\bpeople\b", r"\bstudents?\b",
        r"\bchickens?\b", r"\bducks?\b", r"\bflocks?\b", r"\bhouses?\b", r"\bprofits?\b",
        r"\bfibers?\b", r"\bmarbles?\b", r"\bcoins?\b", r"\bpages?\b",
    ]
    units: List[str] = []
    for pattern in unit_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            units.append(normalize_whitespace(match.group(0).lower()))
    return dedupe_preserve_order(units)


def _extract_concepts(question: str) -> Dict[str, List[str]]:
    persons = _extract_person_entities(question)
    focus_terms = _extract_focus_terms(question)
    units = _extract_units(question)
    all_terms = dedupe_preserve_order(persons + focus_terms + units)
    return {
        "persons": persons,
        "focus_terms": focus_terms,
        "units": units,
        "all_terms": all_terms,
    }


def _mask_numbers(text: str) -> str:
    text = MONEY_RE.sub("<MONEY>", text)
    text = PERCENT_RE.sub("<PERCENT>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    return text


def _mask_entities(text: str, entities: Sequence[str], token: str = "<ENTITY>") -> str:
    masked = text
    for entity in sorted(set(entities), key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(entity) + r"\b", re.IGNORECASE)
        masked = pattern.sub(token, masked)
    return masked


def _build_templates(question: str, concepts: Dict[str, List[str]]) -> Tuple[str, str, str]:
    surface_template = normalize_whitespace(_mask_numbers(question))
    scene_template = normalize_whitespace(_mask_entities(surface_template, concepts["persons"], "<PERSON>"))
    scenario_template = scene_template
    short_terms = [term for term in concepts["focus_terms"] if len(term.split()) <= 2]
    if short_terms:
        scenario_template = _mask_entities(scenario_template, short_terms[:8], "<TERM>")
    return surface_template, scene_template, normalize_whitespace(scenario_template)


def _extract_calculation_steps(solution_text: str) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    for expr, result in GSM8K_STEP_RE.findall(solution_text or ""):
        steps.append({"expression": normalize_whitespace(expr), "result": normalize_whitespace(result)})
    return steps


def _infer_operation_tags(question: str, solution_text: str) -> List[str]:
    text = f"{question}\n{solution_text}".lower()
    tags: List[str] = []
    rules = [
        ("addition", [r"\+", r"\badd\b", r"\bsum\b", r"\btotal\b", r"\baltogether\b"]),
        ("subtraction", [r"-", r"\bminus\b", r"\bleft\b", r"\bremaining\b", r"\bdifference\b"]),
        ("multiplication", [r"\*", r"\btimes\b", r"\beach\b", r"\bevery\b", r"\bper\b"]),
        ("division", [r"/", r"\bdivide\b", r"\bshared\b", r"\bhalf\b", r"\bsplit\b"]),
        ("percentage", [r"%", r"\bpercent\b", r"\bpercentage\b"]),
        ("fraction", [r"\bhalf\b", r"\bthird\b", r"\bquarter\b"]),
        ("ratio", [r"\bratio\b", r"\bproportion\b"]),
        ("average", [r"\baverage\b", r"\bmean\b"]),
        ("geometry", [r"\barea\b", r"\bperimeter\b", r"\bradius\b", r"\bdiameter\b", r"\bcircle\b", r"\brectangle\b", r"\btriangle\b"]),
        ("time", [r"\bminute\b", r"\bhour\b", r"\bday\b", r"\bweek\b", r"\bmonth\b", r"\byear\b"]),
        ("money", [r"\$", r"\bdollar\b", r"\bcents?\b", r"\bprice\b", r"\bcost\b", r"\bpay\b", r"\bsell\b"]),
        ("counting", [r"\bhow many\b", r"\bcount\b", r"\bnumber of\b"]),
        ("probability", [r"\bprobability\b", r"\bchance\b", r"\bdice\b", r"\bcoin\b"]),
    ]
    for tag, patterns in rules:
        if any(re.search(pattern, text) for pattern in patterns):
            tags.append(tag)
    return tags or ["general_arithmetic"]


def _estimate_difficulty(step_count: int, question: str, solution_text: str) -> str:
    complexity_marks = len(re.findall(r"[\+\-\*/%]", solution_text))
    token_count = len(_tokenize_words(question))
    score = step_count * 2 + complexity_marks + token_count / 20.0
    if score <= 2.5:
        return "easy"
    if score <= 5.0:
        return "medium"
    if score <= 8.0:
        return "hard"
    return "very_hard"


def _build_knowledge(question: str, solution_text: str, answer: str) -> Dict[str, Any]:
    calc_steps = _extract_calculation_steps(solution_text)
    op_tags = _infer_operation_tags(question, solution_text)
    op_sequence: List[str] = []
    for step in calc_steps:
        expr = step["expression"].lower()
        if "+" in expr:
            op_sequence.append("addition")
        if "-" in expr:
            op_sequence.append("subtraction")
        if "*" in expr:
            op_sequence.append("multiplication")
        if "/" in expr:
            op_sequence.append("division")
        if "%" in expr or "percent" in expr:
            op_sequence.append("percentage")
    if not op_sequence:
        op_sequence = op_tags[:]

    step_count = len(calc_steps)
    knowledge_signature = "|".join(sorted(set(op_tags)))
    if op_sequence:
        knowledge_signature = f"{knowledge_signature}::" + "->".join(op_sequence)

    difficulty_bucket = _estimate_difficulty(step_count, question, solution_text)

    return {
        "skill_tags": op_tags,
        "operation_sequence": op_sequence,
        "step_count": step_count,
        "calculation_steps": calc_steps,
        "knowledge_signature": knowledge_signature,
        "final_answer": answer,
        "difficulty_bucket": difficulty_bucket,
    }


class KnowledgeBaseBuilder:
    def __init__(self, input_path: str, output_dir: str, dataset_name: Optional[str] = None):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name or self.input_path.stem

    def _prepare_records(self, records: List[dict]) -> Tuple[List[dict], Path]:
        kb_dir = self.output_dir / "kb" / self.dataset_name
        kb_dir.mkdir(parents=True, exist_ok=True)
        formatted_path = kb_dir / "formatted_input.jsonl"

        formatted_records: List[dict] = []
        for idx, rec in enumerate(records):
            normalized = dict(rec)
            normalized.setdefault("task_id", idx)
            if not normalized.get("solution_steps"):
                solution_text, final_answer, source_schema = _extract_solution_text(normalized)
                normalized["solution_steps"] = solution_text
                normalized["answer"] = final_answer
                normalized["source_schema"] = source_schema
            formatted_records.append(normalized)

        write_jsonl(formatted_path, formatted_records)
        return formatted_records, formatted_path

    def _normalize_record(self, record: dict) -> Optional[KBItem]:
        question = normalize_whitespace(record.get("question", ""))
        if not question:
            return None

        solution_text, answer, source_schema = _extract_solution_text(record)
        concepts = _extract_concepts(question)
        surface_template, scene_template, scenario_template = _build_templates(question, concepts)
        knowledge = _build_knowledge(question, solution_text, answer)
        return KBItem(
            task_id=record.get("task_id"),
            question=question,
            answer=answer,
            solution_text=solution_text,
            scene_text=question,
            surface_template=surface_template,
            scene_template=scene_template,
            scenario_template=scenario_template,
            concepts=concepts,
            knowledge=knowledge,
            source_schema=source_schema,
            question_type=record.get("question_type"),
        )

    def _build_concept_bank(self, items: Sequence[KBItem]) -> List[dict]:
        bank: Dict[Tuple[str, str], dict] = {}
        for item in items:
            for kind in ("persons", "focus_terms", "units"):
                for term in item.concepts.get(kind, []):
                    key = (kind, term.lower())
                    entry = bank.setdefault(
                        key,
                        {
                            "term": term,
                            "kind": kind[:-1] if kind.endswith("s") else kind,
                            "count": 0,
                            "task_ids": [],
                            "question_types": [],
                            "scene_templates": [],
                            "examples": [],
                        },
                    )
                    entry["count"] += 1
                    entry["task_ids"].append(item.task_id)
                    entry["question_types"].append(item.question_type)
                    entry["scene_templates"].append(item.scene_template)
                    if len(entry["examples"]) < 3:
                        entry["examples"].append(item.question)
        return sorted(bank.values(), key=lambda x: (-x["count"], x["kind"], x["term"].lower()))

    def _build_scene_bank(self, items: Sequence[KBItem]) -> List[dict]:
        bank: Dict[str, dict] = {}
        for item in items:
            key = item.scene_template.lower()
            entry = bank.setdefault(
                key,
                {
                    "scene_template": item.scene_template,
                    "surface_template": item.surface_template,
                    "scenario_template": item.scenario_template,
                    "count": 0,
                    "task_ids": [],
                    "question_types": [],
                    "skill_tags": [],
                    "operation_sequences": [],
                    "step_counts": [],
                    "difficulty_buckets": [],
                    "examples": [],
                    "concepts": [],
                },
            )
            entry["count"] += 1
            entry["task_ids"].append(item.task_id)
            entry["question_types"].append(item.question_type)
            entry["skill_tags"].extend(item.knowledge["skill_tags"])
            entry["operation_sequences"].append(item.knowledge["operation_sequence"])
            entry["step_counts"].append(item.knowledge["step_count"])
            entry["difficulty_buckets"].append(item.knowledge["difficulty_bucket"])
            entry["concepts"].append(item.concepts["all_terms"][:10])
            if len(entry["examples"]) < 3:
                entry["examples"].append({"task_id": item.task_id, "question": item.question, "answer": item.answer})
        for entry in bank.values():
            entry["avg_step_count"] = round(sum(entry["step_counts"]) / len(entry["step_counts"]), 3) if entry["step_counts"] else 0.0
            entry["skill_tags"] = sorted({tag for tag in entry["skill_tags"] if tag})
            entry["question_types"] = sorted({qt for qt in entry["question_types"] if qt})
            entry["difficulty_buckets"] = sorted({d for d in entry["difficulty_buckets"] if d})
        return sorted(bank.values(), key=lambda x: (-x["count"], x["scene_template"]))

    def _build_template_bank(self, items: Sequence[KBItem]) -> List[dict]:
        bank: Dict[str, dict] = {}
        for item in items:
            key = item.surface_template.lower()
            entry = bank.setdefault(
                key,
                {
                    "surface_template": item.surface_template,
                    "scene_template": item.scene_template,
                    "scenario_template": item.scenario_template,
                    "count": 0,
                    "task_ids": [],
                    "question_types": [],
                    "difficulty_buckets": [],
                    "examples": [],
                },
            )
            entry["count"] += 1
            entry["task_ids"].append(item.task_id)
            entry["question_types"].append(item.question_type)
            entry["difficulty_buckets"].append(item.knowledge["difficulty_bucket"])
            if len(entry["examples"]) < 3:
                entry["examples"].append(item.question)
        for entry in bank.values():
            entry["question_types"] = sorted({qt for qt in entry["question_types"] if qt})
            entry["difficulty_buckets"] = sorted({d for d in entry["difficulty_buckets"] if d})
        return sorted(bank.values(), key=lambda x: (-x["count"], x["surface_template"]))

    def _build_knowledge_bank(self, items: Sequence[KBItem]) -> List[dict]:
        bank: Dict[str, dict] = {}
        for item in items:
            key = item.knowledge["knowledge_signature"]
            entry = bank.setdefault(
                key,
                {
                    "knowledge_signature": key,
                    "skill_tags": item.knowledge["skill_tags"],
                    "operation_sequence": item.knowledge["operation_sequence"],
                    "difficulty_bucket": item.knowledge["difficulty_bucket"],
                    "count": 0,
                    "task_ids": [],
                    "question_types": [],
                    "examples": [],
                    "step_counts": [],
                },
            )
            entry["count"] += 1
            entry["task_ids"].append(item.task_id)
            entry["question_types"].append(item.question_type)
            entry["step_counts"].append(item.knowledge["step_count"])
            if len(entry["examples"]) < 3:
                entry["examples"].append({"task_id": item.task_id, "question": item.question, "answer": item.answer})
        for entry in bank.values():
            entry["avg_step_count"] = round(sum(entry["step_counts"]) / len(entry["step_counts"]), 3) if entry["step_counts"] else 0.0
            entry["question_types"] = sorted({qt for qt in entry["question_types"] if qt})
        return sorted(bank.values(), key=lambda x: (-x["count"], x["knowledge_signature"]))

    def build(self, sample_limit: Optional[int] = None) -> Dict[str, Path]:
        raw_records = read_jsonl(self.input_path)
        if sample_limit is not None:
            raw_records = raw_records[:sample_limit]

        records, formatted_path = self._prepare_records(raw_records)

        items: List[KBItem] = []
        for record in records:
            item = self._normalize_record(record)
            if item is not None:
                items.append(item)

        kb_dir = self.output_dir / "kb" / self.dataset_name
        kb_dir.mkdir(parents=True, exist_ok=True)

        record_path = kb_dir / "records.jsonl"
        entity_path = kb_dir / "entities.json"
        scene_path = kb_dir / "scenes.json"
        template_path = kb_dir / "templates.json"
        knowledge_path = kb_dir / "knowledge.json"
        manifest_path = kb_dir / "manifest.json"

        write_jsonl(record_path, [asdict(item) for item in items])
        concepts = self._build_concept_bank(items)
        scenes = self._build_scene_bank(items)
        templates = self._build_template_bank(items)
        knowledge = self._build_knowledge_bank(items)

        write_json(entity_path, concepts)
        write_json(scene_path, scenes)
        write_json(template_path, templates)
        write_json(knowledge_path, knowledge)

        manifest = {
            "dataset_name": self.dataset_name,
            "input_path": str(self.input_path),
            "formatted_input_path": str(formatted_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "record_count": len(items),
            "concept_count": len(concepts),
            "scene_count": len(scenes),
            "template_count": len(templates),
            "knowledge_count": len(knowledge),
            "files": {
                "formatted_input": str(formatted_path),
                "records": str(record_path),
                "entities": str(entity_path),
                "scenes": str(scene_path),
                "templates": str(template_path),
                "knowledge": str(knowledge_path),
            },
        }
        write_json(manifest_path, manifest)

        return {
            "kb_dir": kb_dir,
            "formatted_input": formatted_path,
            "records": record_path,
            "entities": entity_path,
            "scenes": scene_path,
            "templates": template_path,
            "knowledge": knowledge_path,
            "manifest": manifest_path,
        }


def build_knowledge_base(input_path: str, output_dir: str, dataset_name: Optional[str] = None, sample_limit: Optional[int] = None) -> Dict[str, Path]:
    return KnowledgeBaseBuilder(input_path=input_path, output_dir=output_dir, dataset_name=dataset_name).build(sample_limit=sample_limit)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a knowledge base from JSONL math QA data.")
    parser.add_argument("--input", required=False, help="Input JSONL path")
    parser.add_argument("--output-dir", required=False, help="Output directory")
    parser.add_argument("--dataset-name", required=False, help="Dataset name for output paths")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional record cap")
    args = parser.parse_args(argv)

    root = _default_project_root()
    input_path = Path(args.input) if args.input else root / "data" / "gsm8k.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else root / "outputs"
    dataset_name = args.dataset_name or input_path.stem

    outputs = build_knowledge_base(
        input_path=str(input_path),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        sample_limit=args.sample_limit,
    )
    print(json.dumps({k: str(v) for k, v in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

