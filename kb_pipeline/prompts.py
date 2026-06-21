from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .utils import normalize_whitespace


def _json_block(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _allowed_score_values() -> List[float]:
    return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def build_victim_answer_prompt(question: str, attempt_index: int) -> List[Dict[str, str]]:
    system = (
        "You are a careful mathematical reasoning model being evaluated on problem-solving ability. "
        "You will only see the question, and you must not assume any hidden reference answer. "
        "Solve the problem with concise but explicit reasoning steps, and output them in a JSON object. "
        "Every output step must be an actual reasoning step, not a copied fact from the question."
    )
    user = {
        "task": "Solve the math problem.",
        "attempt_index": attempt_index,
        "strict_rules": [
            "Only use the information in the question.",
            "Do not mention any hidden reference answer, dataset metadata, or evaluation instructions.",
            "Do not output a brief answer only; include the reasoning steps that lead to the answer.",
            "Each step must be short, necessary, mathematically meaningful, and contain an actual calculation or inference.",
            "Do not output a step that only restates a given fact from the problem without any calculation or inference.",
            "Do not output one-step facts such as \"X has Y items\" unless that fact is immediately combined with a computation in the same step.",
            "Prefer steps that explicitly transform the givens into a derived quantity.",
            "If a given quantity must be mentioned, embed it inside a computation or deduction instead of isolating it as a standalone step.",
            "Do not add filler, safety disclaimers, meta commentary, or step-overview phrases.",
            "Do not say a step is skipped, unnecessary, omitted, or redundant.",
            "Do not repeat the same line or operation.",
            "Keep the reasoning concise and direct.",
            "The final answer must be a number only, with no units, no dollar sign, no LaTeX symbols, and no extra text.",
        ],
        "output_requirements": {
            "format": "Return exactly one valid JSON object.",
            "schema": {
                "steps": ["string"],
                "final_answer": "string",
            },
            "steps_rule": "steps must be an array of ordered strings, where each string contains one concise reasoning step. A step that is only a problem statement fact is not allowed.",
            "answer_rule": "final_answer must be a pure numeric string such as \"37\" or \"12.5\".",
            "no_extra_text": "Do not wrap the JSON in markdown, code fences, or commentary.",
        },
        "question": question,
        "example_structure": [
            "{\"steps\":[\"...\",\"...\"],\"final_answer\":\"42\"}",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _json_block(user)},
    ]


def build_step_evaluation_prompt(question: str, reference_answer: str, steps: List[str], final_answer: str) -> List[Dict[str, str]]:
    system = (
        "You are a professional evaluator of mathematical problem-solving steps. "
        "You must score each provided step independently and conservatively. "
        "The input is already segmented by the answer generator, so you must not split, merge, reorder, or rewrite any step. "
        "Score strictly according to the rubric, and return only a valid JSON object."
    )
    user = {
        "task": "Evaluate all solution steps of the following mathematical solution.",
        "scoring_standard": {
            "allowed_values": _allowed_score_values(),
            "dimensions": {
                "correctness": {
                    "1.0": "Fully correct. The calculation, transformation, or conclusion is mathematically accurate and consistent with the problem.",
                    "0.8": "Almost fully correct. Only a very minor imprecision exists, but the step is still mathematically valid.",
                    "0.6": "Mostly correct but contains one noticeable error, imprecision, or weak inference.",
                    "0.4": "Contains multiple errors or a serious flaw, but part of the step is still related to the solution.",
                    "0.2": "Contains a major mathematical error and is largely unusable for the solution.",
                    "0.0": "Completely incorrect, or the element is not a mathematical statement that can be judged for correctness.",
                },
                "logical": {
                    "1.0": "Necessary and clearly connected to the previous and next steps. It advances the solution in a natural way.",
                    "0.8": "Reasonable and related, with only a slight indirectness or mild gap in flow.",
                    "0.6": "Noticeable logical jump. The step is still relevant, but the connection to surrounding steps is weak.",
                    "0.4": "Weak connection or partially redundant. The step contributes only limited progress.",
                    "0.2": "Largely disconnected, contradictory, or very hard to justify from the surrounding steps.",
                    "0.0": "Irrelevant to the solution process.",
                },
                "standardization": {
                    "1.0": "Fully standard, clear, and unambiguous mathematical language or notation.",
                    "0.8": "Mostly standard, with only a minor notation or wording issue.",
                    "0.6": "Several non-standard expressions or a somewhat awkward presentation, but still understandable.",
                    "0.4": "Difficult to interpret without extra effort, or contains confusing notation.",
                    "0.2": "Severely non-standard or poorly written, making the step hard to trust.",
                    "0.0": "Unintelligible or effectively impossible to interpret.",
                },
                "completeness": {
                    "1.0": "Fully justified. The step contains enough information to support its own conclusion.",
                    "0.8": "Missing only a very small detail, but the intended reasoning is clear.",
                    "0.6": "Missing several supporting details, though the step is still understandable.",
                    "0.4": "Missing important justification or intermediate reasoning needed to support the claim.",
                    "0.2": "Severely incomplete. The step leaves out most of the needed information.",
                    "0.0": "Provides no meaningful information toward the solution.",
                },
                "efficiency": {
                    "1.0": "Fully necessary, concise, and non-redundant. The step is worth keeping.",
                    "0.8": "Mostly necessary, with only minor redundancy or slight extra wording.",
                    "0.6": "Partially redundant or somewhat unnecessary, but still provides some value.",
                    "0.4": "Mostly redundant, weakly useful, or too verbose for its contribution.",
                    "0.2": "Highly redundant or mostly unnecessary for solving the problem.",
                    "0.0": "Completely unnecessary or purely filler.",
                },
            },
        },
        "step_rules": [
            "Evaluate every step independently.",
            "Some elements may not be actual mathematical steps (for example introductory phrases or meta-comments). You must still score every single element.",
            "Do not skip any element, even if it seems redundant, non-mathematical, or incorrectly segmented.",
            "You may use neighboring steps as context when evaluating logical consistency and necessity.",
            "Do NOT adjust the score of a step based on whether the overall solution is ultimately correct or incorrect.",
            "Score each dimension independently.",
            "Use the score descriptions as strict anchors. Do not freely interpolate outside the rubric.",
            "Be conservative: if a step is only partially justified or only loosely connected, prefer the lower score.",
            "Use only the allowed values.",
            "Do not reward verbosity or penalize brevity by itself; judge whether the step is necessary and clear.",
            "Do not assume hidden calculations or unstated reasoning that are not actually present in the text of the step.",
        ],
        "question": question,
        "reference_answer": reference_answer,
        "solution_steps": steps,
        "final_answer": final_answer,
        "output_schema": {
            "step_count": "integer",
            "correctness": ["one score per step"],
            "logical": ["one score per step"],
            "standardization": ["one score per step"],
            "completeness": ["one score per step"],
            "efficiency": ["one score per step"],
            "final_answer_correct": "boolean",
            "overall_reason": "string",
        },
        "output_rules": [
            "Return only a valid JSON object.",
            "The number of scores in each list must exactly equal the number of input steps.",
            "The i-th score in every list must correspond to the i-th input step.",
            "All score arrays must be compact and presented on a single line, with no line breaks inside the arrays.",
            "All score values must be from the allowed set.",
            "Do not output markdown, prose, or extra commentary.",
            "Do not output any extra keys outside the schema.",
            "If a step is clearly non-mathematical or filler, still score it explicitly using the rubric rather than skipping it.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _json_block(user)},
    ]


def build_generation_prompt(plan_card: Dict[str, Any], target: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    bucket = str(target.get("bucket") or "medium")
    difficulty_level = str(target.get("difficulty_level") or target.get("target_difficulty") or bucket)
    prompt_profile = _generation_prompt_profile(difficulty_level)
    mode_profile = _generation_mode_profile(str(target.get("mode") or plan_card.get("mode") or "same_structure"))
    system = prompt_profile["role_description"]
    step_range = target.get("step_count_range") or prompt_profile.get("step_count_range", [2, 4])
    source_question = normalize_whitespace(plan_card.get("question") or "")
    source_answer = normalize_whitespace(plan_card.get("answer") or "")
    source_scene = normalize_whitespace(plan_card.get("scene_text") or plan_card.get("plan_source_scene_text") or "")
    source_concepts = plan_card.get("concepts") or plan_card.get("plan_source_concepts") or {}
    source_knowledge = plan_card.get("knowledge") or plan_card.get("plan_source_knowledge") or {}
    feedback_text = ""
    if feedback:
        feedback_text = json.dumps(feedback, ensure_ascii=False, separators=(",", ":"))

    user_lines = [
        prompt_profile["task"],
        "",
        f"Reference question: {source_question}",
        f"Reference answer: {source_answer}",
        f"Reference scene/template: {source_scene or plan_card.get('scene_template') or plan_card.get('scenario_template') or ''}",
        f"Reference concepts: {json.dumps(source_concepts, ensure_ascii=False, separators=(',', ':')) if source_concepts else '{}'}",
        f"Reference knowledge: {json.dumps(source_knowledge, ensure_ascii=False, separators=(',', ':')) if source_knowledge else '{}'}",
        f"Target difficulty: {difficulty_level}",
        f"Target bucket: {bucket}",
        f"Target step count range: {step_range[0]}-{step_range[1]}",
        f"Target mode: {target.get('mode') or plan_card.get('mode') or 'same_structure'}",
        "",
        "Rules:",
    ]
    for item in prompt_profile["constraints"] + mode_profile["diversity_rules"] + mode_profile["mode_rules"] + [
        "Keep the mathematical topic aligned with the seed knowledge.",
        "Change the surface form, narrative context, and entities so the result is not a copy.",
        "The problem must have a unique and well-defined answer.",
        "The answer must be directly derivable from the stated conditions.",
        f"The solution should usually have {step_range[0]} to {step_range[1]} reasoning lines.",
        "Use concise wording and avoid redundant exposition.",
        "Do not output placeholder fields, empty strings, or null values.",
        "Do not output any field outside the required JSON schema.",
        "If feedback is provided, fix the issue directly.",
    ]:
        user_lines.append(f"- {item}")
    if feedback_text:
        user_lines.extend(["", f"Feedback: {feedback_text}"])
    user_lines.extend(
        [
            "",
            "Output requirements:",
            "- Return only one valid JSON object.",
            '- Use exactly these keys: "question", "solution", "answer".',
            '- "question" must be a self-contained math word problem.',
            '- "solution" must be a concise step-by-step solution, one reasoning step per line.',
            '- Every line in "solution" must be an actual reasoning step, not a fact copied from the question.',
            '- "answer" must be a pure numeric string with no units, no dollar sign, no commas, and no extra text.',
            "Do not wrap the JSON in markdown, code fences, or commentary.",
            'Example: {"question":"...","solution":"step 1\\nstep 2","answer":"42"}',
        ]
    )
    user = "\n".join(user_lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_evaluation_prompt(candidate: Dict[str, Any], rubric: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are a strict evaluator for generated math problems. "
        "Judge the candidate against the rubric and return only JSON. "
        "Be conservative and penalize ambiguity, incorrectness, redundancy, and weak answer quality."
    )
    user = {
        "task": "Evaluate the candidate.",
        "rubric": rubric,
        "candidate": candidate,
        "output_schema": {
            "verdict": "pass|fail",
            "scores": {
                "correctness": "1-5",
                "difficulty_match": "1-5",
                "brevity": "1-5",
                "non_redundancy": "1-5",
                "answer_quality": "1-5",
            },
            "issues": ["string"],
            "short_reason": "string",
        },
        "output_rules": [
            "Return only a valid JSON object.",
            "Use the rubric to judge mathematical correctness and quality.",
            "Keep the reasoning implicit; do not output chain-of-thought.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _json_block(user)},
    ]


def build_repair_prompt(candidate: Dict[str, Any], issues: List[str], target: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You repair noisy synthetic math problems. "
        "Your goal is to keep the intended difficulty and mathematical topic while fixing ambiguity, contamination, incorrectness, or redundancy. "
        "Return only JSON."
    )
    user = "\n".join(
        [
            "Repair the candidate problem.",
            f"Issues: {json.dumps(issues, ensure_ascii=False, separators=(',', ':'))}",
            f"Target: {json.dumps(target, ensure_ascii=False, separators=(',', ':'))}",
            f"Candidate: {json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))}",
            "",
            "Repair rules:",
            "- Preserve the target difficulty as much as possible.",
            "- Keep the same core mathematical concept unless a minimal structural change is required.",
            "- Remove contamination, copied phrasing, duplicated structure, and ambiguity.",
            "- Ensure the answer is unique and directly derivable.",
            "- Keep the repaired question concise and solvable.",
            "",
            "Output requirements:",
            "- Return only one valid JSON object.",
            '- Use exactly these keys: "question", "solution", "answer", "repair_notes".',
            '- "solution" must be a concise step-by-step solution, one reasoning step per line.',
            '- "answer" must be a pure numeric string with no units or symbols.',
            "Do not wrap the JSON in markdown, code fences, or commentary.",
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_quality_prompt(candidate: Dict[str, Any], source: Dict[str, Any], target: Dict[str, Any], hint: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    system = (
        "You are a rigorous verifier for synthetic math problems. "
        "Check whether the candidate is suitable for training data. "
        "Evaluate difficulty, correctness, solvability, uniqueness, conciseness, and whether the solution is brief and effective. "
        "Return only JSON."
    )
    user = {
        "task": "Judge whether the candidate passes quality control.",
        "source": source,
        "target": target,
        "candidate": candidate,
        "hint": hint or {},
        "quality_checks": [
            "Is the question unambiguous and self-contained?",
            "Does it have a unique and well-defined answer?",
            "Are the steps and final answer correct?",
            "Does the difficulty match the target bucket?",
            "Is the wording concise and non-redundant?",
            "Does it avoid trivial duplication of the source or sibling items?",
            "Is the answer directly derivable from the stated conditions?",
        ],
        "failure_categories": [
            "Ambiguity",
            "MissingCondition",
            "ParameterOmitted",
            "MathFormatError",
            "LogicalInconsistency",
            "Unsolvable",
            "RedundantContent",
            "DuplicateProblem",
        ],
        "output_schema": {
            "verdict": "pass|fail",
            "scores": {
                "difficulty_match": "1-5",
                "correctness": "1-5",
                "answer_uniqueness": "1-5",
                "step_validity": "1-5",
                "brevity": "1-5",
            },
            "issues": ["string"],
            "repair_suggestions": ["string"],
            "short_reason": "string",
        },
        "output_rules": [
            "If the candidate passes all checks, return verdict pass.",
            "If any important issue exists, return verdict fail and include a specific category.",
            "Repair suggestions must be concrete and actionable.",
            "Keep the response short and machine-readable.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _json_block(user)},
    ]


def build_retry_generation_prompt(plan_card: Dict[str, Any], target: Dict[str, Any], feedback: Dict[str, Any]) -> List[Dict[str, str]]:
    return build_generation_prompt(plan_card, target, feedback=feedback)


def _generation_prompt_profile(bucket: str) -> Dict[str, Any]:
    bucket = str(bucket or "medium")
    profiles: Dict[str, Dict[str, Any]] = {
        "easy": {
            "role_description": "You are a mathematics expert. Create a problem that is clearly easier than the reference problem. The result must be simple, direct, and fully determined from the stated conditions. Do not copy the reference wording, entities, or number pattern.",
            "task": "Generate an easier math problem with a very short solution and final answer.",
            "constraints": [
                "Target 1 to 2 reasoning steps.",
                "Use only one main operation when possible.",
                "Avoid hidden conditions, chained inferences, and nested subproblems.",
                "Keep the question short and easy to parse.",
            ],
            "diversity_rules": [
                "Change the scenario or domain when possible.",
                "Do not preserve the same storyline, object types, or numerical pattern.",
                "Avoid producing multiple sibling samples with the same surface structure.",
            ],
        },
        "medium": {
            "role_description": "You are a mathematics expert. Create a new math problem with difficulty comparable to the reference problem, but with a clearly different surface form, story setup, and entity set. The output must be compact, self-contained, and unambiguous.",
            "task": "Generate an equal-difficulty math problem with a concise solution and final answer.",
            "constraints": [
                "Target 2 to 4 reasoning steps.",
                "Maintain similar reasoning depth and similar cognitive load.",
                "Preserve the mathematical topic, but change the representation style.",
                "Keep the problem compact and unambiguous.",
            ],
            "diversity_rules": [
                "Do not reuse the same context, scenario, or entity types.",
                "Avoid preserving the same numerical relationship pattern.",
                "Ensure sibling diversity when multiple samples are generated from the same seed.",
            ],
        },
        "hard": {
            "role_description": "You are a mathematics expert. Create a new math problem that is harder than the reference problem. The result must require deeper reasoning, extra constraints, and a unique solution. Keep the wording compact but precise.",
            "task": "Generate a harder math problem with a concise but complete step-by-step solution and final answer.",
            "constraints": [
                "Target 4 to 6 reasoning steps.",
                "Add at least one additional reasoning step compared to the reference.",
                "Introduce extra constraints or intermediate variables.",
                "Prefer inferable hidden conditions, but never make the answer ambiguous.",
                "The solution should require at least one non-trivial intermediate transformation.",
            ],
            "diversity_rules": [
                "Significantly change the scenario, context, and numeric structure.",
                "Avoid template reuse across siblings.",
                "Do not create superficial variants that differ only by numbers.",
            ],
        },
        "very_hard": {
            "role_description": "You are a mathematics expert. Create a new math problem that is substantially harder than the reference problem. It must remain solvable, have a single unambiguous answer, and use a compact but layered reasoning chain. You must obey the requested JSON schema exactly.",
            "task": "Generate a very hard math problem with explicit reasoning steps and final answer.",
            "constraints": [
                "Target 6 to 10 reasoning steps.",
                "Introduce layered constraints or combined operations.",
                "Require careful intermediate reasoning rather than direct arithmetic.",
                "Use hidden conditions only if they are still inferable and do not make the task ambiguous.",
                "Ensure the solution cannot be answered by a single arithmetic expression alone.",
            ],
            "diversity_rules": [
                "Fully change context, story, and numeric scale.",
                "Avoid isomorphic transformation of the reference problem.",
                "Ensure siblings are not near duplicates.",
            ],
        },
    }
    aliases = {
        "easy": "easy",
        "slightly easy": "easy",
        "equal": "medium",
        "slightly hard": "hard",
        "hard": "very_hard",
        "medium": "medium",
        "very_hard": "very_hard",
    }
    normalized = aliases.get(bucket.strip().lower(), bucket.strip().lower())
    return profiles.get(normalized, profiles["medium"])


def _generation_mode_profile(mode: str) -> Dict[str, Any]:
    mode = str(mode or "same_structure").strip().lower()
    profiles: Dict[str, Dict[str, Any]] = {
        "same_structure": {
            "mode_rules": [
                "Keep the underlying operation sequence and reasoning skeleton close to the selected plan source.",
                "Change the names, quantities, and surface story, but preserve the high-level algebraic structure.",
                "Do not copy the full wording of the selected plan source.",
            ],
            "diversity_rules": [
                "Use different entities or context labels while keeping the same structural pattern.",
                "Vary numbers enough to avoid duplicate-looking siblings.",
            ],
        },
        "scene_swap": {
            "mode_rules": [
                "Keep the core structure and operation sequence, but replace the scene/context with a clearly different real-world setting.",
                "The new scene must not be a trivial paraphrase of the selected plan source.",
                "Keep the quantities compatible with the new scene.",
            ],
            "diversity_rules": [
                "Switch the domain or scenario frame across siblings when possible.",
                "Avoid repeating the same event structure in the story text.",
            ],
        },
        "surface_swap": {
            "mode_rules": [
                "Preserve the core mathematical structure, but rewrite the surface wording and phrasing substantially.",
                "Use a noticeably different narrative style while keeping the same reasoning backbone.",
                "The changed surface form must still be natural and solvable.",
            ],
            "diversity_rules": [
                "Use alternate phrasing, different sentence order, and different entity descriptions.",
                "Avoid near-paraphrase variants across siblings.",
            ],
        },
        "entity_swap": {
            "mode_rules": [
                "Preserve the structure and reasoning backbone, but replace key entities, actors, and objects with different ones.",
                "The mathematical relationships should remain compatible after entity replacement.",
                "Do not change into a different problem type.",
            ],
            "diversity_rules": [
                "Vary persons, objects, and units across siblings.",
                "Avoid reusing the same named entities from the source or neighboring outputs.",
            ],
        },
    }
    return profiles.get(mode, profiles["same_structure"])
