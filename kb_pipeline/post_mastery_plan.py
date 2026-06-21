from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .utils import read_json, read_jsonl, write_json, write_jsonl


DIFFICULTY_TO_BUCKET = {
    "Easy": "easy",
    "Slightly Easy": "easy",
    "Equal": "medium",
    "Slightly Hard": "hard",
    "Hard": "very_hard",
}

BUCKET_RANK = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "very_hard": 3,
}

SCENE_DOMAINS = [
    {"domain": "community_library", "setting": "a neighborhood library", "roles": ["librarian", "volunteer", "reader"], "objects": ["books", "shelves", "returns"], "units": ["books", "days", "shelves"]},
    {"domain": "school_club", "setting": "an after-school club", "roles": ["student", "coach", "organizer"], "objects": ["members", "materials", "sessions"], "units": ["students", "sessions", "items"]},
    {"domain": "bakery", "setting": "a small bakery", "roles": ["baker", "assistant", "customer"], "objects": ["trays", "pastries", "ingredients"], "units": ["trays", "pieces", "kilograms"]},
    {"domain": "farm_cooperative", "setting": "a farm cooperative", "roles": ["farmer", "manager", "driver"], "objects": ["crates", "produce", "deliveries"], "units": ["crates", "kilograms", "days"]},
    {"domain": "sports_tournament", "setting": "a local sports tournament", "roles": ["player", "coach", "referee"], "objects": ["teams", "matches", "points"], "units": ["games", "points", "minutes"]},
    {"domain": "museum_exhibit", "setting": "a museum exhibit", "roles": ["curator", "guide", "visitor"], "objects": ["tickets", "displays", "groups"], "units": ["visitors", "tickets", "hours"]},
    {"domain": "city_bus_route", "setting": "a city bus route", "roles": ["driver", "dispatcher", "passenger"], "objects": ["stops", "trips", "seats"], "units": ["miles", "minutes", "passengers"]},
    {"domain": "train_station", "setting": "a regional train station", "roles": ["conductor", "clerk", "traveler"], "objects": ["trains", "platforms", "tickets"], "units": ["minutes", "tickets", "miles"]},
    {"domain": "delivery_center", "setting": "a package delivery center", "roles": ["courier", "supervisor", "packer"], "objects": ["packages", "routes", "boxes"], "units": ["packages", "hours", "miles"]},
    {"domain": "warehouse", "setting": "a warehouse", "roles": ["worker", "manager", "supplier"], "objects": ["pallets", "cartons", "orders"], "units": ["cartons", "pallets", "days"]},
    {"domain": "workshop", "setting": "a repair workshop", "roles": ["technician", "apprentice", "customer"], "objects": ["parts", "repairs", "tools"], "units": ["parts", "hours", "jobs"]},
    {"domain": "construction_site", "setting": "a construction site", "roles": ["builder", "foreman", "supplier"], "objects": ["boards", "bricks", "rooms"], "units": ["meters", "bricks", "days"]},
    {"domain": "garden_center", "setting": "a garden center", "roles": ["gardener", "clerk", "customer"], "objects": ["plants", "pots", "soil"], "units": ["plants", "bags", "weeks"]},
    {"domain": "aquarium", "setting": "a public aquarium", "roles": ["keeper", "guide", "visitor"], "objects": ["tanks", "fish", "feed"], "units": ["fish", "liters", "days"]},
    {"domain": "animal_shelter", "setting": "an animal shelter", "roles": ["caretaker", "volunteer", "adopter"], "objects": ["animals", "meals", "rooms"], "units": ["animals", "meals", "days"]},
    {"domain": "clinic", "setting": "a community clinic", "roles": ["nurse", "scheduler", "patient"], "objects": ["appointments", "rooms", "supplies"], "units": ["patients", "minutes", "boxes"]},
    {"domain": "charity_event", "setting": "a charity fundraiser", "roles": ["volunteer", "donor", "coordinator"], "objects": ["donations", "tickets", "tables"], "units": ["dollars", "tickets", "tables"]},
    {"domain": "festival", "setting": "a town festival", "roles": ["vendor", "organizer", "visitor"], "objects": ["booths", "tickets", "meals"], "units": ["tickets", "hours", "items"]},
    {"domain": "movie_theater", "setting": "a movie theater", "roles": ["manager", "cashier", "guest"], "objects": ["seats", "tickets", "showings"], "units": ["seats", "tickets", "minutes"]},
    {"domain": "music_rehearsal", "setting": "a music rehearsal", "roles": ["musician", "conductor", "student"], "objects": ["songs", "sections", "sessions"], "units": ["minutes", "songs", "sessions"]},
    {"domain": "science_lab", "setting": "a school science lab", "roles": ["researcher", "student", "assistant"], "objects": ["samples", "containers", "tests"], "units": ["samples", "milliliters", "minutes"]},
    {"domain": "computer_lab", "setting": "a computer lab", "roles": ["technician", "student", "administrator"], "objects": ["files", "devices", "tasks"], "units": ["gigabytes", "minutes", "devices"]},
    {"domain": "solar_project", "setting": "a community solar project", "roles": ["engineer", "installer", "resident"], "objects": ["panels", "batteries", "homes"], "units": ["panels", "hours", "kilowatts"]},
    {"domain": "water_station", "setting": "a water distribution station", "roles": ["operator", "driver", "resident"], "objects": ["tanks", "containers", "deliveries"], "units": ["liters", "tanks", "days"]},
    {"domain": "recycling_center", "setting": "a recycling center", "roles": ["worker", "supervisor", "collector"], "objects": ["bins", "materials", "loads"], "units": ["bins", "kilograms", "loads"]},
    {"domain": "craft_market", "setting": "a craft market", "roles": ["maker", "seller", "customer"], "objects": ["items", "stalls", "orders"], "units": ["items", "dollars", "hours"]},
    {"domain": "restaurant", "setting": "a family restaurant", "roles": ["chef", "server", "manager"], "objects": ["meals", "tables", "ingredients"], "units": ["meals", "tables", "dollars"]},
    {"domain": "hotel", "setting": "a small hotel", "roles": ["manager", "clerk", "guest"], "objects": ["rooms", "bookings", "towels"], "units": ["rooms", "nights", "guests"]},
    {"domain": "airport", "setting": "a regional airport", "roles": ["agent", "pilot", "traveler"], "objects": ["flights", "bags", "gates"], "units": ["minutes", "bags", "passengers"]},
    {"domain": "hiking_trip", "setting": "a hiking trip", "roles": ["hiker", "guide", "ranger"], "objects": ["trails", "supplies", "checkpoints"], "units": ["miles", "hours", "liters"]},
    {"domain": "board_game_event", "setting": "a board-game event", "roles": ["player", "host", "scorekeeper"], "objects": ["rounds", "tokens", "points"], "units": ["rounds", "tokens", "points"]},
    {"domain": "online_course", "setting": "an online course", "roles": ["learner", "instructor", "moderator"], "objects": ["lessons", "quizzes", "modules"], "units": ["lessons", "minutes", "points"]},
]

VARIATION_MODES = [
    "same mathematical template in a completely different scene",
    "same core operations with a different unknown quantity",
    "same reasoning goal with conditions stated in a different order",
    "same mathematical relation represented through rates or grouped quantities",
    "same skill combination with a new dependency that still matches the target difficulty",
    "same operation skeleton with a comparison or remaining-amount question",
]

NARRATIVE_STYLES = [
    "short everyday story",
    "task and schedule scenario",
    "resource allocation scenario",
    "production or logistics scenario",
    "purchase or budget scenario",
    "progress and remaining-work scenario",
]

NUMBER_STRATEGIES = [
    "use fresh small integers unrelated to the seed",
    "use exact division and clean integer results",
    "use multiples that make every intermediate result exact",
    "use one percentage only when it naturally fits",
    "use rates and totals with realistic non-seed values",
    "use two groups with different quantities and a clean final answer",
]


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _task_lookup(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(record.get("task_id")): record for record in records}


def _target_bucket(mastery: Dict[str, Any]) -> str:
    difficulty = str(mastery.get("target_difficulty") or "Equal")
    return str(
        mastery.get("target_difficulty_bucket")
        or DIFFICULTY_TO_BUCKET.get(difficulty)
        or "medium"
    )


def _knowledge_match_score(
    candidate: Dict[str, Any],
    source: Dict[str, Any],
    target_bucket: str,
) -> int:
    candidate_knowledge = candidate.get("knowledge", {})
    source_knowledge = source.get("knowledge", {})

    candidate_bucket = str(candidate_knowledge.get("difficulty_bucket") or "medium")
    bucket_gap = abs(
        BUCKET_RANK.get(candidate_bucket, 1) - BUCKET_RANK.get(target_bucket, 1)
    )

    source_skills = set(_as_list(source_knowledge.get("skill_tags")))
    candidate_skills = set(_as_list(candidate_knowledge.get("skill_tags")))
    skill_overlap = len(source_skills & candidate_skills)

    source_operations = _as_list(source_knowledge.get("operation_sequence"))
    candidate_operations = _as_list(candidate_knowledge.get("operation_sequence"))
    operation_overlap = len(set(source_operations) & set(candidate_operations))

    return skill_overlap * 20 + operation_overlap * 8 - bucket_gap * 10


def _rank_scene_sources(
    kb_records: Sequence[Dict[str, Any]],
    source: Dict[str, Any],
    target_bucket: str,
    seed_key: str,
) -> List[Dict[str, Any]]:
    source_id = str(source.get("task_id"))
    candidates = [
        record
        for record in kb_records
        if str(record.get("task_id")) != source_id
    ]
    if not candidates:
        candidates = list(kb_records)

    return sorted(
        candidates,
        key=lambda record: (
            -_knowledge_match_score(record, source, target_bucket),
            _stable_int(f"{seed_key}|{record.get('task_id')}"),
        ),
    )


def _entity_pools(entity_bank: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {
        "person": [],
        "focus_term": [],
        "unit": [],
    }
    for entry in entity_bank:
        kind = str(entry.get("kind") or "")
        term = str(entry.get("term") or "").strip()
        if not term:
            continue
        if kind in pools and term not in pools[kind]:
            pools[kind].append(term)
    return pools


def _select_terms(pool: Sequence[str], key: str, count: int) -> List[str]:
    if not pool or count <= 0:
        return []
    start = _stable_int(key) % len(pool)
    selected: List[str] = []
    for offset in range(min(count, len(pool))):
        selected.append(pool[(start + offset) % len(pool)])
    return selected


def _scene_terms(
    concepts: Dict[str, Any],
    key: str,
    concept_key: str,
    count: int,
) -> List[str]:
    values: List[str] = []
    for value in _as_list(concepts.get(concept_key)):
        term = str(value).strip()
        if concept_key == "persons":
            term = re.sub(r"(?:'s|\u2019s)$", "", term).strip()
        if term and term not in values:
            values.append(term)
    return _select_terms(values, key, count)


def _compact_math_knowledge(record: Dict[str, Any]) -> Dict[str, Any]:
    knowledge = record.get("knowledge", {})
    return {
        "skill_tags": _as_list(knowledge.get("skill_tags")),
        "operation_sequence": _as_list(knowledge.get("operation_sequence")),
        "knowledge_signature": str(knowledge.get("knowledge_signature") or ""),
        "reference_step_count": int(knowledge.get("step_count") or 0),
    }


def _build_plan_knowledge(
    source: Dict[str, Any],
    scene_source: Dict[str, Any],
    entity_pools: Dict[str, List[str]],
    target_difficulty: str,
    seed_id: Any,
    variant_index: int,
) -> Dict[str, Any]:
    key = f"{seed_id}|{variant_index}"
    scene_concepts = scene_source.get("concepts", {})

    persons = _scene_terms(
        scene_concepts,
        f"{key}|scene-person",
        "persons",
        2,
    )
    focus_terms = _scene_terms(
        scene_concepts,
        f"{key}|scene-focus",
        "focus_terms",
        4,
    )
    units = _scene_terms(
        scene_concepts,
        f"{key}|scene-unit",
        "units",
        2,
    )

    if not persons:
        persons = _select_terms(
            entity_pools.get("person", []),
            f"{key}|fallback-person",
            1,
        )
    if not focus_terms:
        focus_terms = _select_terms(
            entity_pools.get("focus_term", []),
            f"{key}|fallback-focus",
            2,
        )
    if not units:
        units = _select_terms(
            entity_pools.get("unit", []),
            f"{key}|fallback-unit",
            1,
        )

    domain_start = _stable_int(f"{seed_id}|domain") % len(SCENE_DOMAINS)
    primary_domain = SCENE_DOMAINS[
        (domain_start + variant_index) % len(SCENE_DOMAINS)
    ]
    alternative_domains = [
        SCENE_DOMAINS[
            (domain_start + variant_index + offset) % len(SCENE_DOMAINS)
        ]
        for offset in (7, 13, 21)
    ]
    variation_mode = VARIATION_MODES[
        (_stable_int(f"{seed_id}|variation") + variant_index) % len(VARIATION_MODES)
    ]
    narrative_style = NARRATIVE_STYLES[
        (_stable_int(f"{seed_id}|narrative") + variant_index // len(SCENE_DOMAINS))
        % len(NARRATIVE_STYLES)
    ]
    number_strategy = NUMBER_STRATEGIES[
        (_stable_int(f"{seed_id}|numbers") + variant_index)
        % len(NUMBER_STRATEGIES)
    ]

    return {
        "math": {
            **_compact_math_knowledge(source),
            "target_difficulty": target_difficulty,
            "policy": (
                "Preserve the intended mathematical skill and relative difficulty. "
                "The exact operation sequence may be adapted when needed for a natural problem."
            ),
        },
        "diversity": {
            "primary_scene": primary_domain,
            "alternative_scenes": alternative_domains,
            "variation_mode": variation_mode,
            "narrative_style": narrative_style,
            "number_strategy": number_strategy,
            "same_template_new_scene_allowed": True,
            "scene_change_required": True,
            "plan_signature": (
                f"{primary_domain['domain']}|{variation_mode}|"
                f"{narrative_style}|{number_strategy}"
            ),
        },
        "kb_inspiration": {
            "scene_keywords": focus_terms,
            "possible_roles": persons,
            "possible_units": units,
            "related_skill_tags": _as_list(
                scene_source.get("knowledge", {}).get("skill_tags")
            ),
            "policy": (
                "Optional inspiration only. Do not copy the knowledge-base wording, "
                "names, numbers, or full template. Ignore this section if it conflicts "
                "with the primary diversity scene."
            ),
        },
    }


def build_post_mastery_plan(
    mastery_records: Sequence[Dict[str, Any]],
    kb_records: Sequence[Dict[str, Any]],
    entity_bank: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Expand accepted mastery records into a compact three-field synthesis plan."""
    kb_lookup = _task_lookup(kb_records)
    pools = _entity_pools(entity_bank)
    plan: List[Dict[str, Any]] = []

    for mastery in mastery_records:
        seed_id = mastery.get("task_id")
        source = kb_lookup.get(str(seed_id), {})
        if not source:
            continue

        target_count = max(0, int(mastery.get("target_count") or 0))
        target_bucket = _target_bucket(mastery)
        target_difficulty = str(mastery.get("target_difficulty") or "Equal")
        ranked_sources = _rank_scene_sources(
            kb_records,
            source,
            target_bucket,
            seed_key=str(seed_id),
        )
        if not ranked_sources:
            ranked_sources = [source]

        rotation = _stable_int(f"{seed_id}|{target_bucket}") % len(ranked_sources)
        for variant_index in range(target_count):
            scene_source = ranked_sources[
                (rotation + variant_index) % len(ranked_sources)
            ]
            plan_id = f"{seed_id}_{variant_index}"
            plan.append(
                {
                    "source_task_id": seed_id,
                    "plan_id": plan_id,
                    "knowledge": _build_plan_knowledge(
                        source,
                        scene_source,
                        pools,
                        target_difficulty,
                        seed_id,
                        variant_index,
                    ),
                }
            )

    return plan


def replan_failed_plan(
    plan: Dict[str, Any],
    failure_type: str,
    retry_round: int,
) -> Dict[str, Any]:
    """Return the plan for the next batch round.

    Similarity failures and stubborn validation failures receive a new diversity
    assignment. Transport and formatting failures keep the original plan.
    """
    next_plan = copy.deepcopy(plan)
    if failure_type not in {
        "similarity",
        "stubborn_validation",
        "validation_regeneration",
        "validation_response_failure",
    }:
        return next_plan

    plan_id = str(next_plan.get("plan_id") or "")
    knowledge = next_plan.setdefault("knowledge", {})
    diversity = knowledge.setdefault("diversity", {})

    current_domain = str(
        diversity.get("primary_scene", {}).get("domain") or ""
    )
    start = _stable_int(f"{plan_id}|replan|{retry_round}") % len(SCENE_DOMAINS)
    selected_index = start
    for offset in range(len(SCENE_DOMAINS)):
        candidate_index = (start + offset) % len(SCENE_DOMAINS)
        if SCENE_DOMAINS[candidate_index]["domain"] != current_domain:
            selected_index = candidate_index
            break

    primary_scene = SCENE_DOMAINS[selected_index]
    alternative_scenes = [
        SCENE_DOMAINS[(selected_index + offset) % len(SCENE_DOMAINS)]
        for offset in (5, 11, 19)
    ]
    variation_mode = VARIATION_MODES[
        (_stable_int(f"{plan_id}|replan-variation") + retry_round)
        % len(VARIATION_MODES)
    ]
    narrative_style = NARRATIVE_STYLES[
        (_stable_int(f"{plan_id}|replan-narrative") + retry_round)
        % len(NARRATIVE_STYLES)
    ]
    number_strategy = NUMBER_STRATEGIES[
        (_stable_int(f"{plan_id}|replan-number") + retry_round)
        % len(NUMBER_STRATEGIES)
    ]

    diversity.update(
        {
            "primary_scene": primary_scene,
            "alternative_scenes": alternative_scenes,
            "variation_mode": variation_mode,
            "narrative_style": narrative_style,
            "number_strategy": number_strategy,
            "same_template_new_scene_allowed": True,
            "scene_change_required": True,
            "replanned_round": retry_round,
            "replan_reason": failure_type,
            "plan_signature": (
                f"{primary_scene['domain']}|{variation_mode}|"
                f"{narrative_style}|{number_strategy}|retry:{retry_round}"
            ),
        }
    )

    inspiration = knowledge.setdefault("kb_inspiration", {})
    inspiration["scene_keywords"] = []
    inspiration["possible_roles"] = []
    inspiration["possible_units"] = []
    if failure_type == "similarity":
        inspiration["policy"] = (
            "The previous output was too similar. Use the new diversity scene and "
            "invent fresh names, objects, values, and wording. Knowledge-base content "
            "is optional and must not be copied."
        )
    else:
        inspiration["policy"] = (
            "Previous validation attempts remained incorrect or mismatched in "
            "difficulty. Build a fresh problem from the target mathematical skills "
            "and the new diversity assignment. Do not preserve the failed question's "
            "wording, values, or dependency structure. Knowledge-base content is "
            "optional inspiration rather than a restriction."
        )
    return next_plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the compact post-mastery synthesis plan."
    )
    parser.add_argument("--mastery", required=True, help="Mastery records JSONL path")
    parser.add_argument("--kb-records", required=True, help="Knowledge-base records JSONL")
    parser.add_argument("--entities", required=True, help="Knowledge-base entities JSON")
    parser.add_argument("--output", required=True, help="Plan JSONL output path")
    parser.add_argument("--summary-output", required=False)
    args = parser.parse_args(argv)

    mastery_records = read_jsonl(Path(args.mastery))
    kb_records = read_jsonl(Path(args.kb_records))
    entity_bank = read_json(Path(args.entities))
    plan = build_post_mastery_plan(mastery_records, kb_records, entity_bank)

    output_path = Path(args.output)
    write_jsonl(output_path, plan)
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )
    write_json(
        summary_path,
        {
            "seed_count": len(mastery_records),
            "plan_count": len(plan),
            "unique_scene_domains": len(
                {
                    record.get("knowledge", {})
                    .get("diversity", {})
                    .get("primary_scene", {})
                    .get("domain")
                    for record in plan
                }
                - {None, ""}
            ),
            "unique_plan_signatures": len(
                {
                    record.get("knowledge", {})
                    .get("diversity", {})
                    .get("plan_signature")
                    for record in plan
                }
                - {None, ""}
            ),
            "output": str(output_path),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "summary": str(summary_path),
                "count": len(plan),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
