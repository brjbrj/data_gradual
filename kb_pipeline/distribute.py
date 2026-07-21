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


def _rebalance_counts(
    counts: List[int],
    weights: Sequence[float],
    target_total: int,
    *,
    n_min: int,
    n_max: int,
) -> List[int]:
    """Adjust rounded counts toward the requested total without changing order."""
    diff = int(target_total) - sum(counts)
    if diff == 0:
        return counts
    ranked = sorted(range(len(counts)), key=lambda idx: weights[idx], reverse=True)
    cursor = 0
    while diff != 0 and ranked:
        idx = ranked[cursor % len(ranked)]
        if diff > 0 and counts[idx] < n_max:
            counts[idx] += 1
            diff -= 1
        elif diff < 0 and counts[idx] > n_min:
            counts[idx] -= 1
            diff += 1
        cursor += 1
        if cursor > len(ranked) * max(20, abs(diff) + 5) and diff != 0:
            break
    return counts


def _legacy_counts(
    weights: Sequence[float],
    *,
    total_new_samples: int,
    n_min: int,
    n_max: int,
) -> List[int]:
    total_weight = sum(weights) or 1.0
    raw_nums: List[int] = []
    for weight in weights:
        num = n_min + ((total_new_samples - len(weights) * n_min) * weight / total_weight)
        num = round(num)
        num = max(n_min, min(int(num), n_max))
        raw_nums.append(num)
    return _rebalance_counts(
        raw_nums,
        weights,
        total_new_samples,
        n_min=n_min,
        n_max=n_max,
    )


def _marginal_score(
    value: float,
    count: int,
    *,
    alpha: float,
) -> float:
    return value / ((max(0, count) + 1) ** max(0.0, alpha))


def _threshold_marginal_counts(
    weights: Sequence[float],
    *,
    total_new_samples: int,
    n_max: int,
    active_threshold: int,
    marginal_alpha: float,
    threshold_boost: float,
    cold_start_factor: float,
) -> List[int]:
    """Allocate budget into dense seed clusters with diminishing returns.

    The first pass estimates each seed's natural budget in ``0..n_max``.
    Counts below ``active_threshold`` are not kept directly; their budget is
    pooled. Borderline seeds can be reactivated when their value and closeness
    to the threshold justify the full activation cost. Remaining budget goes to
    activated seeds by a diminishing marginal score, which keeps the allocation
    concentrated without letting one seed monopolize the budget.
    """
    if not weights:
        return []
    if total_new_samples <= 0 or n_max <= 0:
        return [0 for _ in weights]
    threshold = max(1, min(int(active_threshold), int(n_max)))
    initial = _legacy_counts(
        weights,
        total_new_samples=total_new_samples,
        n_min=0,
        n_max=n_max,
    )
    counts = [0 for _ in initial]
    pool = 0
    active: List[int] = []
    candidates: List[tuple[float, int]] = []

    for idx, count in enumerate(initial):
        value = max(0.0, float(weights[idx]))
        if count >= threshold:
            counts[idx] = count
            active.append(idx)
            continue
        pool += count
        if count > 0:
            gap = threshold - count
            score = (
                value
                * max(0.0, threshold_boost)
                / max(1, gap)
                / ((count + 1) ** max(0.0, marginal_alpha))
            )
            candidates.append((score, idx))
        elif cold_start_factor > 0:
            score = value * cold_start_factor / (threshold ** max(0.0, marginal_alpha))
            candidates.append((score, idx))

    candidates.sort(key=lambda item: (item[0], weights[item[1]]), reverse=True)

    def best_active_score() -> float:
        best = 0.0
        for idx in active:
            if counts[idx] < n_max:
                best = max(
                    best,
                    _marginal_score(
                        max(0.0, float(weights[idx])),
                        counts[idx],
                        alpha=marginal_alpha,
                    ),
                )
        return best

    for activation_score, idx in candidates:
        if pool < threshold:
            break
        if counts[idx] > 0:
            continue
        if active and activation_score < best_active_score():
            break
        counts[idx] = threshold
        active.append(idx)
        pool -= threshold

    while pool > 0 and active:
        best_idx = -1
        best_score = -1.0
        for idx in active:
            if counts[idx] >= n_max:
                continue
            score = _marginal_score(
                max(0.0, float(weights[idx])),
                counts[idx],
                alpha=marginal_alpha,
            )
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx < 0:
            break
        counts[best_idx] += 1
        pool -= 1

    return counts


def distribute_mastery_records(
    mastery_records: Sequence[Dict[str, Any]],
    source_lookup: Dict[Any, Dict[str, Any]],
    *,
    target_multiplier: int = 26,
    n_min: int = 10,
    n_max: int = 50,
    lambda_balance: float = 0.3,
    allocation_policy: str = "legacy",
    active_threshold: int = 0,
    marginal_alpha: float = 0.7,
    threshold_boost: float = 2.0,
    cold_start_factor: float = 0.0,
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

    policy = str(allocation_policy or "legacy").strip().lower()
    if policy in {"legacy", "default"}:
        raw_nums = _legacy_counts(
            weights,
            total_new_samples=total_new_samples,
            n_min=n_min,
            n_max=n_max,
        )
    elif policy in {"threshold_marginal", "threshold-redistribute", "threshold_redistribute"}:
        raw_nums = _threshold_marginal_counts(
            weights,
            total_new_samples=total_new_samples,
            n_max=n_max,
            active_threshold=active_threshold,
            marginal_alpha=marginal_alpha,
            threshold_boost=threshold_boost,
            cold_start_factor=cold_start_factor,
        )
    else:
        raise ValueError(f"Unsupported synthesis allocation policy: {allocation_policy}")

    for item, mastery, num, weight in zip(records, profs, raw_nums, weights):
        difficulty = difficulty_from_mastery(mastery)
        item["target_count"] = int(num)
        item["target_difficulty"] = difficulty
        item["target_difficulty_bucket"] = DIFFICULTY_TO_BUCKET[difficulty]
        item["target_step_count_range"] = DIFFICULTY_STEP_RANGES[difficulty]
        item["allocation_policy"] = policy
        if policy != "legacy":
            item["allocation_weight"] = round(float(weight), 8)
            item["active_threshold"] = int(max(0, active_threshold))

    return records
