from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence


DIFFICULTY_LABELS = [
    "Easy",
    "Slightly Easy",
    "Equal",
    "Slightly Hard",
    "Hard",
]

DIFFICULTY_TO_BUCKET = {
    "Easy": "easy",
    "Slightly Easy": "easy",
    "Equal": "medium",
    "Slightly Hard": "hard",
    "Hard": "very_hard",
}

DIFFICULTY_STEP_RANGES = {
    "Easy": [1, 2],
    "Slightly Easy": [2, 3],
    "Equal": [2, 4],
    "Slightly Hard": [4, 6],
    "Hard": [6, 10],
}


def compute_learning_utility(m_norm: float) -> float:
    """Learning utility U(m) = [m(1-m)]^2."""
    return (m_norm * (1.0 - m_norm)) ** 2


def compute_category_balance_factor(category_count: int, max_category_count: int, lambda_balance: float = 0.5) -> float:
    """Few-shot category compensation."""
    ratio = max_category_count / max(category_count, 1)
    return ratio**lambda_balance


def difficulty_from_mastery(mastery: float) -> str:
    """Map mastery to a 5-level relative synthesis difficulty label."""
    if mastery <= 0.2:
        return "Easy"
    if mastery <= 0.4:
        return "Slightly Easy"
    if mastery <= 0.6:
        return "Equal"
    if mastery <= 0.8:
        return "Slightly Hard"
    return "Hard"


def _normalize_mastery(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _source_question_type(source: Dict[str, Any]) -> str:
    question_type = source.get("question_type")
    if question_type is None:
        question_type = source.get("knowledge", {}).get("question_type")
    value = str(question_type).strip() if question_type is not None else ""
    return value or "unknown"


def distribute_mastery_records(
    mastery_records: Sequence[Dict[str, Any]],
    source_lookup: Dict[Any, Dict[str, Any]],
    *,
    target_multiplier: int = 26,
    n_min: int = 10,
    n_max: int = 50,
    lambda_balance: float = 0.3,
) -> List[Dict[str, Any]]:
    """Attach synthesis count and relative difficulty to each mastery record."""
    if not mastery_records:
        return []

    records = [dict(item) for item in mastery_records]
    profs = [_normalize_mastery(item.get("mastery", item.get("mastery_score", 0.0))) for item in records]
    m_min = min(profs)
    m_max = max(profs)
    if abs(m_max - m_min) < 1e-8:
        m_max = m_min + 1e-8

    category_counter = Counter(_source_question_type(source_lookup.get(item.get("task_id"), {})) for item in records)
    max_category_count = max(category_counter.values()) if category_counter else 1

    weights: List[float] = []
    for item, mastery in zip(records, profs):
        m_norm = (mastery - m_min) / (m_max - m_min)
        utility = compute_learning_utility(m_norm)
        category = _source_question_type(source_lookup.get(item.get("task_id"), {}))
        balance_factor = compute_category_balance_factor(category_counter.get(category, 0), max_category_count, lambda_balance)
        weight = utility * balance_factor + 1e-8
        weights.append(weight)

    total_weight = sum(weights) or 1.0
    total_target = max(len(records), int(target_multiplier * len(records)))
    total_new_samples = max(0, total_target - len(records))

    raw_nums: List[int] = []
    for weight in weights:
        num = n_min + ((total_new_samples - len(records) * n_min) * weight / total_weight)
        num = round(num)
        num = max(n_min, min(int(num), n_max))
        raw_nums.append(num)

    diff = total_new_samples - sum(raw_nums)
    if diff != 0:
        ranked = sorted(range(len(records)), key=lambda idx: weights[idx], reverse=True)
        cursor = 0
        while diff != 0 and ranked:
            idx = ranked[cursor % len(ranked)]
            if diff > 0 and raw_nums[idx] < n_max:
                raw_nums[idx] += 1
                diff -= 1
            elif diff < 0 and raw_nums[idx] > n_min:
                raw_nums[idx] -= 1
                diff += 1
            cursor += 1
            if cursor > len(ranked) * 20 and diff != 0:
                break

    for item, mastery, num, weight in zip(records, profs, raw_nums, weights):
        difficulty = difficulty_from_mastery(mastery)
        item["target_count"] = int(num)
        item["target_difficulty"] = difficulty
        item["target_difficulty_bucket"] = DIFFICULTY_TO_BUCKET[difficulty]
        item["target_step_count_range"] = DIFFICULTY_STEP_RANGES[difficulty]

    return records
