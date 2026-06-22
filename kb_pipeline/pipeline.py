from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .assessment import (
    answer_questions,
    build_mastery_records,
    evaluate_answers,
    project_victim_answer_raw_record,
    project_victim_answer_record,
)
from .build import build_knowledge_base
from .client import VLLMClient
from .distribute import distribute_mastery_records
from .post_mastery_generate import generate_post_mastery_questions
from .post_mastery_plan import build_post_mastery_plan
from .validation import validate_generated_questions
from .utils import normalize_whitespace, read_json, read_jsonl, write_json, write_jsonl


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log(message: str) -> None:
    print(message, flush=True)


class VLLMManager:
    def __init__(self, runtime_dir: Path, start_timeout_sec: int = 300, start_poll_sec: int = 5) -> None:
        self.runtime_dir = runtime_dir
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.runtime_dir / "vllm.pid"
        self.log_file = self.runtime_dir / "vllm.log"
        self.current_model: Optional[str] = None
        self.owned: bool = False
        self.start_timeout_sec = max(30, int(start_timeout_sec))
        self.start_poll_sec = max(1, int(start_poll_sec))

    def _run(self, command: Sequence[str], env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        return subprocess.run(command, check=True, env=env, cwd=str(_project_root()))

    def probe(self) -> Optional[str]:
        script = _project_root() / "run" / "probe_vllm.py"
        try:
            output = subprocess.check_output(["python", str(script)], cwd=str(_project_root()))
            model = normalize_whitespace(output.decode("utf-8"))
            return model or None
        except Exception:
            return None

    def stop(self, force: bool = False) -> None:
        if not force and not self.owned:
            return
        script = _project_root() / "run" / "stop_vllm.sh"
        if self.pid_file.exists():
            self._run(["bash", str(script), "--pid-file", str(self.pid_file)])
        else:
            self._run(["bash", str(script)])
        for _ in range(30):
            time.sleep(1)
            if self.probe() is None:
                break
        self.current_model = None
        self.owned = False

    def start(self, model: str) -> None:
        if self.current_model == model:
            _log(f"[vLLM] already using model: {model}")
            return
        running = self.probe()
        if running == model:
            self.current_model = model
            self.owned = False
            _log(f"[vLLM] reusing existing model: {model}")
            return
        if running and running != model:
            _log(f"[vLLM] stopping mismatched model: {running}")
            self.stop(force=True)
            for _ in range(30):
                time.sleep(1)
                if self.probe() is None:
                    _log("[vLLM] old model stopped, ready to launch new model")
                    break
        script = _project_root() / "run" / "start_vllm.sh"
        env = os.environ.copy()
        env["VLLM_MODEL"] = model
        env["VLLM_PID_FILE"] = str(self.pid_file)
        env["VLLM_LOG_FILE"] = str(self.log_file)
        _log(f"[vLLM] starting model: {model}")
        self._run(["bash", str(script), "--background", "--pid-file", str(self.pid_file), "--log-file", str(self.log_file)], env=env)
        total_polls = max(1, self.start_timeout_sec // self.start_poll_sec)
        for poll_idx in range(total_polls):
            time.sleep(self.start_poll_sec)
            running = self.probe()
            if running == model:
                self.current_model = model
                self.owned = True
                _log(f"[vLLM] ready: {model}")
                return
            waited = (poll_idx + 1) * self.start_poll_sec
            _log(
                f"[vLLM] waiting for ready state: {model} "
                f"({waited}s/{self.start_timeout_sec}s)"
            )
        log_tail = ""
        try:
            lines = self.log_file.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            log_tail = "\n".join(lines[-30:])
        except OSError:
            pass
        detail = f"\nLast vLLM log lines:\n{log_tail}" if log_tail else ""
        raise RuntimeError(
            f"vLLM did not become ready for model {model}. "
            f"Log: {self.log_file}{detail}"
        )


def _load_source_target_maps(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[Any, Dict[str, Any]]]:
    source_lookup: Dict[Any, Dict[str, Any]] = {}
    target_lookup: Dict[Any, Dict[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        source_lookup[task_id] = record
        target_lookup[task_id] = {
            "bucket": record.get("difficulty_bucket") or record.get("knowledge", {}).get("difficulty_bucket", "medium"),
            "step_count_range": {
                "easy": [1, 2],
                "medium": [2, 4],
                "hard": [4, 6],
                "very_hard": [6, 10],
            }.get(record.get("difficulty_bucket") or record.get("knowledge", {}).get("difficulty_bucket", "medium"), [2, 4]),
            "reference_step_count": int(record.get("knowledge", {}).get("step_count", 0) or 0),
        }
    return {"source": source_lookup, "target": target_lookup}


def _parse_float_map(value: Optional[str]) -> Dict[str, float]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    outputs: Dict[str, float] = {}
    for key, raw in parsed.items():
        try:
            outputs[str(key)] = float(raw)
        except Exception:
            continue
    return outputs


def _parse_json_float_map_env(*names: str) -> Dict[str, float]:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            parsed = json.loads(value)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        outputs: Dict[str, float] = {}
        for key, raw in parsed.items():
            try:
                outputs[str(key)] = float(raw)
            except Exception:
                continue
        if outputs:
            return outputs
    return {}


def _parse_float_env(*names: str, default: float) -> float:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _parse_int_env(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return int(default)


def _parse_bool_env(*names: str, default: bool) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is None or value == "":
            continue
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def run_pipeline(
    input_path: str,
    output_dir: str,
    dataset_name: Optional[str] = None,
    sample_limit: Optional[int] = None,
    n_answers: int = 10,
    victim_model: Optional[str] = None,
    victim_temperature: Optional[float] = None,
    victim_top_p: Optional[float] = None,
    step_model: Optional[str] = None,
    qc_model: Optional[str] = None,
    gen_model: Optional[str] = None,
    repair_model: Optional[str] = None,
    qc_votes: Optional[int] = None,
    qc_max_rounds: Optional[int] = None,
    gen_temperature: Optional[float] = None,
    gen_top_p: Optional[float] = None,
    gen_temperature_map: Optional[Dict[str, float]] = None,
    gen_top_p_map: Optional[Dict[str, float]] = None,
    synthesis_target_multiplier: Optional[int] = None,
    synthesis_min_per_seed: Optional[int] = None,
    synthesis_max_per_seed: Optional[int] = None,
    synthesis_balance_lambda: Optional[float] = None,
    run_validation: Optional[bool] = None,
    vllm_start_timeout_sec: int = 300,
    vllm_start_poll_sec: int = 5,
) -> Dict[str, Any]:
    root = _project_root()
    input_path = str(input_path)
    output_dir_path = Path(output_dir)
    dataset_name = dataset_name or Path(input_path).stem

    victim_model = victim_model or os.environ.get("VICTIM_MODEL") or os.environ.get("VLLM_VICTIM_MODEL") or os.environ.get("VLLM_MODEL") or "/root/brjverl/models/Meta-Llama-3-8B-Instruct"
    victim_temperature = victim_temperature if victim_temperature is not None else _parse_float_env("VICTIM_TEMPERATURE", "VLLM_VICTIM_TEMPERATURE", default=0.3)
    victim_top_p = victim_top_p if victim_top_p is not None else _parse_float_env("VICTIM_TOP_P", "VLLM_VICTIM_TOP_P", default=0.95)
    gen_temperature = gen_temperature if gen_temperature is not None else _parse_float_env("GEN_TEMPERATURE", "VLLM_GEN_TEMPERATURE", default=0.5)
    gen_top_p = gen_top_p if gen_top_p is not None else _parse_float_env("GEN_TOP_P", "VLLM_GEN_TOP_P", default=0.5)
    if not gen_temperature_map:
        gen_temperature_map = _parse_json_float_map_env("GEN_TEMPERATURE_MAP", "VLLM_GEN_TEMPERATURE_MAP")
    if not gen_top_p_map:
        gen_top_p_map = _parse_json_float_map_env("GEN_TOP_P_MAP", "VLLM_GEN_TOP_P_MAP")
    synthesis_target_multiplier = synthesis_target_multiplier if synthesis_target_multiplier is not None else _parse_int_env("SYNTHESIS_TARGET_MULTIPLIER", default=26)
    synthesis_min_per_seed = synthesis_min_per_seed if synthesis_min_per_seed is not None else _parse_int_env("SYNTHESIS_MIN_PER_SEED", default=10)
    synthesis_max_per_seed = synthesis_max_per_seed if synthesis_max_per_seed is not None else _parse_int_env("SYNTHESIS_MAX_PER_SEED", default=50)
    synthesis_balance_lambda = synthesis_balance_lambda if synthesis_balance_lambda is not None else _parse_float_env("SYNTHESIS_BALANCE_LAMBDA", default=0.3)
    run_validation = run_validation if run_validation is not None else _parse_bool_env("RUN_VALIDATION", default=True)
    step_model = step_model or os.environ.get("STEP_MODEL") or os.environ.get("JUDGE_MODEL") or os.environ.get("VLLM_JUDGE_MODEL") or os.environ.get("GEN_MODEL") or os.environ.get("VLLM_GEN_MODEL") or os.environ.get("VLLM_MODEL") or "/root/brjverl/models/Qwen3.6-27B"
    qc_model = qc_model or os.environ.get("QC_MODEL") or os.environ.get("QUALITY_MODEL") or step_model
    gen_model = gen_model or os.environ.get("GEN_MODEL") or os.environ.get("VLLM_GEN_MODEL") or os.environ.get("VLLM_MODEL") or "/root/brjverl/models/Qwen3.6-27B"
    repair_model = repair_model or os.environ.get("REPAIR_MODEL") or os.environ.get("VLLM_REPAIR_MODEL") or gen_model

    runtime = VLLMManager(
        output_dir_path / "runtime" / dataset_name,
        start_timeout_sec=vllm_start_timeout_sec,
        start_poll_sec=vllm_start_poll_sec,
    )
    try:
        _log(f"[pipeline] dataset={dataset_name}")
        _log(f"[pipeline] input_path={input_path}")
        _log(f"[pipeline] output_dir={output_dir}")
        _log(f"[pipeline] victim_model={victim_model}")
        _log(f"[pipeline] victim_temperature={victim_temperature}")
        _log(f"[pipeline] victim_top_p={victim_top_p}")
        _log(f"[pipeline] gen_temperature={gen_temperature}")
        _log(f"[pipeline] gen_top_p={gen_top_p}")
        _log(f"[pipeline] gen_temperature_map={gen_temperature_map}")
        _log(f"[pipeline] gen_top_p_map={gen_top_p_map}")
        _log(f"[pipeline] synthesis_target_multiplier={synthesis_target_multiplier}")
        _log(f"[pipeline] synthesis_min_per_seed={synthesis_min_per_seed}")
        _log(f"[pipeline] synthesis_max_per_seed={synthesis_max_per_seed}")
        _log(f"[pipeline] synthesis_balance_lambda={synthesis_balance_lambda}")
        _log(f"[pipeline] run_validation={run_validation}")
        _log(f"[pipeline] step_model={step_model}")
        _log(f"[pipeline] gen_model={gen_model}")
        _log("[pipeline] building knowledge base...")
        kb_outputs = build_knowledge_base(input_path=input_path, output_dir=str(output_dir_path), dataset_name=dataset_name, sample_limit=sample_limit)
        records = read_jsonl(kb_outputs["records"])
        _log(f"[pipeline] knowledge base ready: {kb_outputs['kb_dir']}")

        analysis_dir = output_dir_path / "analysis" / dataset_name
        planning_dir = output_dir_path / "planning" / dataset_name
        pipeline_dir = output_dir_path / "pipeline" / dataset_name
        analysis_dir.mkdir(parents=True, exist_ok=True)
        planning_dir.mkdir(parents=True, exist_ok=True)
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        map_data = _load_source_target_maps(records)
        write_json(pipeline_dir / "source_map.json", map_data["source"])
        write_json(pipeline_dir / "target_map.json", map_data["target"])

        _log("[pipeline] answering seed questions...")
        runtime.start(victim_model)
        victim_client = VLLMClient(model=victim_model)
        victim_answers = answer_questions(
            records,
            n_answers=n_answers,
            client=victim_client,
            temperature=victim_temperature,
            top_p=victim_top_p,
        )
        victim_answer_path = analysis_dir / "victim_answers.jsonl"
        victim_answer_raw_path = analysis_dir / "victim_answers.raw.jsonl"
        write_jsonl(victim_answer_path, [project_victim_answer_record(record) for record in victim_answers])
        write_jsonl(victim_answer_raw_path, [project_victim_answer_raw_record(record) for record in victim_answers])
        _log(f"[pipeline] victim answers saved: {len(victim_answers)}")

        _log("[pipeline] evaluating answers and computing mastery...")
        runtime.start(step_model)
        step_client = VLLMClient(model=step_model)
        step_evaluation_path = analysis_dir / "step_evaluations.jsonl"
        step_checkpoint_path = analysis_dir / "step_evaluations.jsonl.partial"
        step_reports = evaluate_answers(
            victim_answers,
            client=step_client,
            checkpoint_path=step_checkpoint_path,
        )
        write_jsonl(step_evaluation_path, step_reports)
        step_checkpoint_path.unlink(missing_ok=True)
        mastery_records = build_mastery_records(step_reports, map_data["source"])
        mastery_records = distribute_mastery_records(
            mastery_records,
            map_data["source"],
            target_multiplier=synthesis_target_multiplier,
            n_min=synthesis_min_per_seed,
            n_max=synthesis_max_per_seed,
            lambda_balance=synthesis_balance_lambda,
        )
        write_jsonl(analysis_dir / "mastery_records.jsonl", mastery_records)
        write_json(analysis_dir / "mastery.json", mastery_records)
        _log(f"[pipeline] mastery computed for {len(mastery_records)} seeds")

        _log("[pipeline] building compact post-mastery synthesis plan...")
        entity_bank = read_json(kb_outputs["entities"])
        synthesis_plan = build_post_mastery_plan(
            mastery_records,
            records,
            entity_bank,
        )
        synthesis_plan_path = planning_dir / "synthesis_plan.jsonl"
        synthesis_plan_summary_path = planning_dir / "synthesis_plan.summary.json"
        write_jsonl(synthesis_plan_path, synthesis_plan)
        write_json(
            synthesis_plan_summary_path,
            {
                "seed_count": len(mastery_records),
                "plan_count": len(synthesis_plan),
                "unique_scene_domains": len(
                    {
                        record.get("knowledge", {})
                        .get("diversity", {})
                        .get("primary_scene", {})
                        .get("domain")
                        for record in synthesis_plan
                    }
                    - {None, ""}
                ),
                "unique_plan_signatures": len(
                    {
                        record.get("knowledge", {})
                        .get("diversity", {})
                        .get("plan_signature")
                        for record in synthesis_plan
                    }
                    - {None, ""}
                ),
                "fields": ["source_task_id", "plan_id", "knowledge"],
            },
        )
        _log(f"[pipeline] synthesis plan ready: {len(synthesis_plan)} items")

        _log("[pipeline] generating synthetic questions...")
        runtime.start(gen_model)
        generated_path = pipeline_dir / "generated.jsonl"
        generated_raw_path = pipeline_dir / "generated.raw.jsonl"
        generated_failed_path = pipeline_dir / "generated.failed.jsonl"
        generated_summary_path = pipeline_dir / "generated.summary.json"
        generated, generated_raw, generated_failed = generate_post_mastery_questions(
            synthesis_plan,
            mastery_records,
            model=gen_model,
            temperature_map=gen_temperature_map,
            top_p_map=gen_top_p_map,
            output_path=generated_path,
            raw_output_path=generated_raw_path,
            failed_output_path=generated_failed_path,
        )
        write_jsonl(generated_path, generated)
        write_jsonl(generated_raw_path, generated_raw)
        write_jsonl(generated_failed_path, generated_failed)
        write_json(
            generated_summary_path,
            {
                "planned": len(synthesis_plan),
                "generated": len(generated),
                "failed": len(generated_failed),
                "rounds_completed": (
                    max(
                        [
                            int(item.get("round") or 0)
                            for item in [*generated_raw, *generated_failed]
                        ],
                        default=-1,
                    )
                    + 1
                ),
                "output_fields": [
                    "source_task_id",
                    "plan_id",
                    "difficulty",
                    "question",
                    "steps",
                    "answer",
                ],
                "round_output_dir": str(
                    pipeline_dir / "generated.rounds"
                ),
            },
        )
        _log(
            f"[pipeline] generation complete: "
            f"{len(generated)} success, {len(generated_failed)} failed"
        )

        validated_path: Optional[Path] = None
        validation_reports_path: Optional[Path] = None
        validation_failed_path: Optional[Path] = None
        repair_history_path: Optional[Path] = None
        if run_validation:
            _log("[pipeline] validating generated questions...")
            runtime.start(qc_model)
            validated_path = pipeline_dir / "validated.jsonl"
            validation_reports_path = pipeline_dir / "validation_reports.jsonl"
            validation_failed_path = pipeline_dir / "validation.failed.jsonl"
            repair_history_path = pipeline_dir / "repair_history.jsonl"
            validated, validation_reports, validation_failed = validate_generated_questions(
                generated,
                synthesis_plan,
                mastery_records,
                model=qc_model,
                blind_votes=qc_votes,
                max_rounds=qc_max_rounds,
                validated_path=validated_path,
                reports_path=validation_reports_path,
                failed_path=validation_failed_path,
                repair_history_path=repair_history_path,
            )
            write_jsonl(validated_path, validated)
            write_jsonl(validation_reports_path, validation_reports)
            write_jsonl(validation_failed_path, validation_failed)
            write_json(
                pipeline_dir / "validated.summary.json",
                {
                    "input": len(generated),
                    "validated": len(validated),
                    "failed": len(validation_failed),
                    "validation_model": qc_model,
                    "round_output_dir": str(pipeline_dir / "validation.rounds"),
                },
            )
            _log(
                f"[pipeline] validation complete: "
                f"{len(validated)} passed, {len(validation_failed)} failed"
            )
        else:
            _log("[pipeline] validation skipped")

        outputs = {
            "kb": {k: str(v) for k, v in kb_outputs.items()},
            "analysis_dir": str(analysis_dir),
            "planning_dir": str(planning_dir),
            "pipeline_dir": str(pipeline_dir),
            "victim_answers": str(analysis_dir / "victim_answers.jsonl"),
            "step_evaluations": str(analysis_dir / "step_evaluations.jsonl"),
            "mastery_records": str(analysis_dir / "mastery_records.jsonl"),
            "mastery": str(analysis_dir / "mastery.json"),
            "synthesis_plan": str(synthesis_plan_path),
            "generated": str(generated_path),
            "generated_raw": str(generated_raw_path),
            "generated_failed": str(generated_failed_path),
            "validated": str(validated_path) if validated_path else None,
            "validation_reports": str(validation_reports_path) if validation_reports_path else None,
            "validation_failed": str(validation_failed_path) if validation_failed_path else None,
            "repair_history": str(repair_history_path) if repair_history_path else None,
        }
        _log("[pipeline] done")
        return outputs
    finally:
        runtime.stop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full gradual data synthesis pipeline.")
    parser.add_argument("--input", required=False, help="Input dataset JSONL path")
    parser.add_argument("--output-dir", required=False, help="Output directory")
    parser.add_argument("--dataset-name", required=False, help="Dataset name")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--n-answers", type=int, default=10)
    parser.add_argument("--victim-model", required=False)
    parser.add_argument("--victim-temperature", type=float, required=False)
    parser.add_argument("--victim-top-p", type=float, required=False)
    parser.add_argument("--step-model", required=False)
    parser.add_argument("--qc-model", required=False)
    parser.add_argument("--gen-model", required=False)
    parser.add_argument("--repair-model", required=False)
    parser.add_argument("--qc-votes", type=int, default=None, help="Initial blind-solve vote count")
    parser.add_argument("--qc-max-rounds", type=int, default=None, help="Maximum repair/revalidation rounds")
    parser.add_argument("--gen-temperature", type=float, required=False)
    parser.add_argument("--gen-top-p", type=float, required=False)
    parser.add_argument("--gen-temperature-map", required=False, help="JSON map of bucket to temperature")
    parser.add_argument("--gen-top-p-map", required=False, help="JSON map of bucket to top_p")
    parser.add_argument("--synthesis-target-multiplier", type=int, required=False)
    parser.add_argument("--synthesis-min-per-seed", type=int, required=False)
    parser.add_argument("--synthesis-max-per-seed", type=int, required=False)
    parser.add_argument("--synthesis-balance-lambda", type=float, required=False)
    parser.add_argument("--skip-validation", action="store_true", help="Stop after generation")
    parser.add_argument("--vllm-start-timeout-sec", type=int, default=300, help="Seconds to wait for vLLM startup before failing")
    parser.add_argument("--vllm-start-poll-sec", type=int, default=5, help="Polling interval in seconds while waiting for vLLM startup")
    args = parser.parse_args(argv)

    root = _project_root()
    input_path = Path(
        args.input
        or os.environ.get("INPUT_PATH")
        or os.environ.get("DATASET_INPUT_PATH")
        or root / "data" / "gsm8k.jsonl"
    )
    output_dir = Path(
        args.output_dir
        or os.environ.get("OUTPUT_DIR")
        or os.environ.get("PIPELINE_OUTPUT_DIR")
        or root / "outputs"
    )
    dataset_name = args.dataset_name or os.environ.get("DATASET_NAME") or input_path.stem

    outputs = run_pipeline(
        input_path=str(input_path),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        sample_limit=args.sample_limit,
        n_answers=args.n_answers,
        victim_model=args.victim_model,
        victim_temperature=args.victim_temperature,
        victim_top_p=args.victim_top_p,
        step_model=args.step_model,
        qc_model=args.qc_model,
        gen_model=args.gen_model,
        repair_model=args.repair_model,
        qc_votes=args.qc_votes,
        qc_max_rounds=args.qc_max_rounds,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        gen_temperature_map=_parse_float_map(args.gen_temperature_map) or None,
        gen_top_p_map=_parse_float_map(args.gen_top_p_map) or None,
        synthesis_target_multiplier=args.synthesis_target_multiplier,
        synthesis_min_per_seed=args.synthesis_min_per_seed,
        synthesis_max_per_seed=args.synthesis_max_per_seed,
        synthesis_balance_lambda=args.synthesis_balance_lambda,
        run_validation=False if args.skip_validation else None,
        vllm_start_timeout_sec=args.vllm_start_timeout_sec,
        vllm_start_poll_sec=args.vllm_start_poll_sec,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
