# data_gradual_new

Independent gradual math-data synthesis pipeline. Accepted mastery and quantity/difficulty distribution logic is preserved. The downstream planning, generation, blind validation, and targeted repair stages are implemented in this directory.

Chinese documentation: [README.zh.md](./README.zh.md)

## Current flow

1. Format the source dataset and build the KB.
2. Let the victim model answer each seed question `N` times using only the question.
3. Compare numeric answers and score victim-provided reasoning steps.
4. Compute mastery and assign five-level relative difficulty plus synthesis count.
5. Build a diversity-oriented synthesis plan: knowledge, scene, problem pattern, relative difficulty, and diversity are decided here.
6. Generate question, steps, and answer asynchronously from the plan. This stage only checks parseable fields and lightweight plan alignment; it does not run global similarity filtering.
7. Run deterministic prechecks.
8. Produce two independent blind solutions; add a tie-break vote when needed.
9. Audit correctness, solvability, uniqueness, steps, and relative difficulty.
10. Apply targeted repair and revalidate in the next batch round.
11. Export passed records to `validated.jsonl`.
12. Optionally export validated records to training-format JSONL.

For the recommended restart-friendly workflow, use the numbered stage scripts
instead of the monolithic pipeline. See [run/README_stages.md](./run/README_stages.md)
for the full stage reference.

## Run By Stage

```bash
cd /root/brjverl/data_gradual_new
export STAGE_VLLM_MODE=external
```

The stage launchers automatically load `config/pipeline.env`, activate or use
the configured `PIPELINE_PYTHON`, and write outputs under `OUTPUT_DIR`.

The default stage mode is external vLLM: start the needed model yourself, then
run the corresponding stage. The script only checks `/v1/models` and will not
start, switch, or stop vLLM unless you explicitly set `STAGE_VLLM_MODE=managed`.

### Stage Commands

```bash
bash run/01_build_kb.sh gsm8k
bash run/02_answer_seed.sh gsm8k
bash run/03_score_seed.sh gsm8k
bash run/04_build_synthesis_plan.sh gsm8k
bash run/05_generate_questions.sh gsm8k
bash run/06_validate_generated.sh gsm8k
bash run/07_export_training_data.sh gsm8k
```

Stage model requirements:

| Stage | vLLM requirement | Main output |
| --- | --- | --- |
| `01_build_kb.sh` | None | `outputs/kb/<dataset>/records.jsonl` |
| `02_answer_seed.sh` | `VICTIM_MODEL` served | `outputs/analysis/<dataset>/victim_answers.raw.jsonl` |
| `03_score_seed.sh` | `STEP_MODEL` served | `outputs/analysis/<dataset>/mastery_records.jsonl` |
| `04_build_synthesis_plan.sh` | None | `outputs/planning/<dataset>/synthesis_plan.jsonl` |
| `05_generate_questions.sh` | `GEN_MODEL` served | `outputs/pipeline/<dataset>/generated.jsonl` |
| `06_validate_generated.sh` | `QC_MODEL` served | `outputs/pipeline/<dataset>/validated.jsonl` |
| `07_export_training_data.sh` | None | `outputs/pipeline/<dataset>/train.jsonl` |

For example, if stage 2 uses Llama and stages 3/5/6 use Qwen, start or switch
the external vLLM server before each model-dependent stage.

### Stage Responsibilities

- `04_build_synthesis_plan.sh` owns diversity and similarity prevention. It selects the knowledge focus, scene, variation mode, number strategy, problem pattern, target difficulty, and synthesis count.
- `05_generate_questions.sh` only realizes each plan into `question`, `steps`, and numeric `answer`. Its checks are limited to JSON/field parsing and lightweight plan alignment, such as whether the generated question reflects the assigned scene or inspiration keywords.
- `06_validate_generated.sh` owns mathematical correctness, solvability, uniqueness, exact relative difficulty, repair, regeneration, and eventual replan after repeated failures.
- Generate-stage failures caused by invalid JSON, missing fields, or plan mismatch stay in the generation failed queue and retry the same plan. Validation-stage stubborn failures can regenerate from a repaired plan through `replan_failed_plan`.

### Resume And Rerun

Resume is enabled by default:

```bash
export STAGE_RESUME=1
export ANSWER_CHECKPOINT_EVERY=50
export GEN_CHECKPOINT_EVERY=50
```

Recovery behavior:

| Stage | Recovery behavior |
| --- | --- |
| `01_build_kb.sh` | Skips if KB records and entities already exist. |
| `02_answer_seed.sh` | Resumes from `victim_answers.raw.jsonl`; periodically saves answers. |
| `03_score_seed.sh` | Resumes from `step_evaluations.jsonl.partial`; appends completed score records. |
| `04_build_synthesis_plan.sh` | Skips if plan and summary already exist. |
| `05_generate_questions.sh` | Resumes from `generated.jsonl`; skips successful `plan_id`s; periodically checkpoints. |
| `06_validate_generated.sh` | Saves canonical files after each validation round; skips if validated output exists. |
| `07_export_training_data.sh` | Skips if train output and summary already exist. |

Force a stage to rebuild from scratch:

```bash
STAGE_FORCE=1 bash run/05_generate_questions.sh gsm8k
```

The old convenience commands are still available as wrappers:

```bash
bash run/run_full_pipeline.sh gsm8k
bash run/run_generate_questions.sh gsm8k
```

`run_full_pipeline.sh` delegates to the numbered stage sequence. By default it
uses managed vLLM mode so one command can start/switch the stage-specific model
and stop vLLM when the sequence exits. If you already started vLLM manually, run:

```bash
STAGE_VLLM_MODE=external bash run/run_full_pipeline.sh gsm8k
```

In external mode, you must switch vLLM yourself before each stage that requires
a different served model.

## Validation configuration

Edit `config/pipeline.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONDA_SH` | `/root/miniconda3/etc/profile.d/conda.sh` | Conda initialization script; leave empty for auto-detection |
| `PIPELINE_CONDA_ENV` | `brj` | Conda environment used by the main pipeline |
| `PIPELINE_PYTHON` | empty | Optional absolute pipeline Python path; skips Conda activation |
| `VLLM_CONDA_ENV` | `qwen` | Conda environment used by the vLLM server |
| `VLLM_PYTHON` | empty | Optional absolute vLLM Python path; skips Conda activation |
| `RUN_VALIDATION` | `1` | Enable validation in the full pipeline |
| `QC_CONCURRENCY` | `256` | Concurrent verifier requests |
| `QC_BLIND_VOTES` | `2` | Initial independent blind solutions |
| `QC_TIEBREAK_VOTES` | `1` | Additional vote when initial solutions disagree |
| `QC_MAX_ROUNDS` | `3` | Repair/revalidation rounds; `-1` means unlimited |
| `QC_MAX_TOKENS` | `900` | Maximum verifier output tokens |
| `QC_ENABLE_THINKING` | `0` | Disable Qwen thinking mode |
| `QC_FORCE_JSON` | `0` | Optional JSON response format |
| `QC_ROUND_RETRY_DELAY` | `1` | Delay between validation rounds |

### Moving to another machine

Normally, only these values need to change:

```bash
CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
PIPELINE_CONDA_ENV=math_pipeline
PIPELINE_PYTHON=
VLLM_CONDA_ENV=vllm_runtime
VLLM_PYTHON=
```

Alternatively, leave both environment names empty and set
`PIPELINE_PYTHON`/`VLLM_PYTHON` to absolute interpreter paths. Model, input,
and output paths must also match the new machine.

## Multi-GPU vLLM/NCCL profiles

The default profile preserves the current deployed server behavior:

```bash
VLLM_BASE_URL=http://127.0.0.1:8911/v1
VLLM_API_PORT=8911
VLLM_ENFORCE_EAGER=0
VLLM_ENABLE_AUTO_TOOL_CHOICE=1
VLLM_TOOL_CALL_PARSER=hermes
VLLM_CUDA_VISIBLE_DEVICES=0,1
VLLM_NCCL_P2P_DISABLE=1
VLLM_NCCL_IB_DISABLE=1
VLLM_NCCL_DEBUG=INFO
VLLM_NCCL_SOCKET_IFNAME=lo
VLLM_NCCL_BLOCKING_WAIT=1
```

Each environment setting accepts a normal exported value, `unset` to remove
the variable, or `inherit` to preserve the parent value. Machines where native
GPU P2P works should merge `config/vllm.p2p-enabled.example.env` into
`pipeline.env`.

Inspect the resolved environment and command without stopping or launching
vLLM:

```bash
bash run/start_vllm.sh --dry-run
```

The default runtime lets the pipeline manage vLLM on one fixed API port:

```bash
VLLM_RUNTIME_MODE=managed
VLLM_BASE_URL=http://127.0.0.1:8911/v1
VLLM_API_PORT=8911
VLLM_START_TIMEOUT=600
VLLM_START_POLL_SEC=5
```

In `managed` mode, the pipeline starts the victim model, stops it after the
answering stage, and starts the evaluator/generator model on the same port
`8911`. Every client request continues to use the single configured
`VLLM_BASE_URL`.

All datasets share `outputs/runtime/vllm/` for the managed PID, model marker,
and server log. This prevents a dataset-name change from misclassifying the
same managed service as an external process. Do not run two full pipelines
concurrently against the same managed port.

Foreground logging is configurable:

```bash
VLLM_LOG_FILE=/root/brjverl/data_gradual_new/outputs/runtime/vllm.log
VLLM_FOREGROUND_LOG=1
VLLM_LOG_APPEND=0
```

Start a configured model while keeping vLLM attached to the terminal:

```bash
bash run/start_vllm.sh \
  --model /root/brjverl/models/Meta-Llama-3-8B-Instruct
```

The terminal still displays output, the configured log file receives the same
output, and `Ctrl+C` stops the server.

Manual external mode remains available through `VLLM_RUNTIME_MODE=external`.
`VLLM_API_PORT` is used by the managed launcher.
Do not export `VLLM_PORT` on vLLM 0.8.x because vLLM also uses that variable
for internal worker communication.

## Validation design

Blind solvers receive only the generated question. They never see the candidate steps or answer. The auditor later receives the candidate, blind consensus, target difficulty, and seed question/solution as a relative-difficulty reference.

Repair actions:

- `repair_solution`: keep the question exactly unchanged and replace steps/answer.
- `repair_question`: minimally fix ambiguity, missing conditions, uniqueness, or difficulty.
- `regenerate_question`: create a fresh problem from the plan.
- request errors: keep the candidate and retry validation next round.
- repeated validation failures: replan the diversity assignment, then regenerate.

Every repaired candidate must pass a fresh blind solve and audit round.

## Outputs

Generated candidates:

```text
outputs/pipeline/<dataset>/generated.jsonl
```

Validated records:

```text
outputs/pipeline/<dataset>/validated.jsonl
```

Both use the compact schema:

```json
{
  "source_task_id": 0,
  "plan_id": "0_0",
  "difficulty": "Hard",
  "question": "...",
  "steps": ["...", "..."],
  "answer": "540"
}
```

Detailed validation artifacts:

```text
validation_reports.jsonl
validation.failed.jsonl
repair_history.jsonl
validated.summary.json
validation.rounds/
```

Legacy `quality.py` and `noise.py` are retained for historical comparison. The new flow uses `kb_pipeline/validation.py`.
