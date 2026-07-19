# data_gradual_new

Independent gradual math-data synthesis pipeline. Accepted mastery and quantity/difficulty distribution logic is preserved. The downstream planning, generation, blind validation, and targeted repair stages are implemented in this directory.

Chinese documentation: [README.zh.md](./README.zh.md)

## Current flow

1. Inspect and prepare the source dataset. Raw GSM8K-style data is formatted into the project schema, and missing `question_type` values are filled by the classification model.
2. Build the KB from the prepared records.
3. Let the victim model answer each seed question `N` times using only the question.
4. Compare numeric answers and score victim-provided reasoning steps.
5. Compute mastery and assign five-level relative difficulty plus synthesis count.
6. Build a diversity-oriented synthesis plan: knowledge, scene, problem pattern, relative difficulty, and diversity are decided here.
7. Generate question, steps, and answer asynchronously from the plan. This stage only checks parseable fields and lightweight plan alignment; it does not run global similarity filtering.
8. Run deterministic prechecks.
9. Produce two independent blind solutions; add a tie-break vote when needed.
10. Audit correctness, solvability, uniqueness, steps, and relative difficulty.
11. Apply targeted repair and revalidate in the next batch round.
12. Export passed records to `validated.jsonl`.
13. Optionally refine only the validated `steps` into dependency-aware training targets.
14. Export refined or validated records to training-format JSONL.

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
bash run/00_prepare_data.sh gsm8k
bash run/01_build_kb.sh gsm8k
bash run/02_answer_seed.sh gsm8k
bash run/03_score_seed.sh gsm8k
bash run/04_build_synthesis_plan.sh gsm8k
bash run/05_generate_questions.sh gsm8k
bash run/06_validate_generated.sh gsm8k
bash run/07_refine_solution_steps.sh gsm8k
bash run/08_export_training_data.sh gsm8k
```

Stage model requirements:

| Stage | vLLM requirement | Main output |
| --- | --- | --- |
| `00_prepare_data.sh` | `CLASSIFY_MODEL` served only when `question_type` is missing | `outputs/prepared/<dataset>/<dataset>.prepared.jsonl` |
| `01_build_kb.sh` | None | `outputs/kb/<dataset>/records.jsonl` |
| `02_answer_seed.sh` | `VICTIM_MODEL` served | `outputs/analysis/<dataset>/victim_answers.raw.jsonl` |
| `03_score_seed.sh` | `STEP_MODEL` served | `outputs/analysis/<dataset>/mastery_records.jsonl` |
| `04_build_synthesis_plan.sh` | None | `outputs/planning/<dataset>/synthesis_plan.jsonl` |
| `05_generate_questions.sh` | `GEN_MODEL` served | `outputs/pipeline/<dataset>/generated.jsonl` |
| `06_validate_generated.sh` | `QC_MODEL` served | `outputs/pipeline/<dataset>/validated.jsonl` |
| `07_refine_solution_steps.sh` | `REFINE_MODEL`/`REPAIR_MODEL` served | `outputs/pipeline/<dataset>/refined.jsonl` |
| `08_export_training_data.sh` | None | `outputs/pipeline/<dataset>/train.jsonl` |

For example, if stage 2 uses Llama and stages 3/5/6 use Qwen, start or switch
the external vLLM server before each model-dependent stage.

### Test Data Preparation Only

Run the preparation stage directly when you want to test raw-data formatting and
question classification without running the rest of the pipeline:

```bash
cd /root/brjverl/data_gradual_new
STAGE_FORCE=1 SAMPLE_LIMIT=5 \
RAW_INPUT_PATH=/root/brjverl/datas/gsm8k_2.jsonl \
PREPARED_INPUT_PATH=/tmp/gsm8k_2.prepared.test.jsonl \
bash run/00_prepare_data.sh gsm8k_2
```

Inspect the result:

```bash
head -n 5 /tmp/gsm8k_2.prepared.test.jsonl
```

The stage first inspects the input schema. If records already contain
`task_id`, `question`, `answer`, `solution_steps`, and `proficiency_score`, it
skips formatting. If all records already contain non-empty `question_type`, it
skips classification and does not check or start vLLM. To test formatting only:

```bash
STAGE_FORCE=1 PREPARE_CLASSIFY=0 SAMPLE_LIMIT=5 \
RAW_INPUT_PATH=/root/brjverl/datas/gsm8k_2.jsonl \
PREPARED_INPUT_PATH=/tmp/gsm8k_2.formatted.test.jsonl \
bash run/00_prepare_data.sh gsm8k_2
```

### Stage Responsibilities

- `04_build_synthesis_plan.sh` owns diversity and similarity prevention. It selects the knowledge focus, scene, variation mode, number strategy, problem pattern, target difficulty, and synthesis count.
- `05_generate_questions.sh` only realizes each plan into `question`, `steps`, and numeric `answer`. Its checks are limited to JSON/field parsing and lightweight plan alignment, such as whether the generated question reflects the assigned scene or inspiration keywords.
- `06_validate_generated.sh` owns mathematical correctness, solvability, uniqueness, exact relative difficulty, repair, regeneration, and eventual replan after repeated failures.
- `07_refine_solution_steps.sh` runs after validation and changes only `steps`; it keeps `question`, `answer`, `difficulty`, IDs, and the validated solution path unchanged.
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
| `00_prepare_data.sh` | Skips if prepared input exists; set `STAGE_FORCE=1` to rebuild. |
| `01_build_kb.sh` | Skips if KB records and entities already exist. |
| `02_answer_seed.sh` | Resumes from `victim_answers.raw.jsonl`; periodically saves answers. |
| `03_score_seed.sh` | Resumes from `step_evaluations.jsonl.partial`; appends completed score records. |
| `04_build_synthesis_plan.sh` | Skips if plan and summary already exist. |
| `05_generate_questions.sh` | Resumes from `generated.jsonl`; skips successful `plan_id`s; periodically checkpoints. |
| `06_validate_generated.sh` | Saves canonical files after each validation round; skips if validated output exists. |
| `07_refine_solution_steps.sh` | Resumes from `refined.jsonl`; skips already refined records; clears `refine.failed.jsonl` each round. |
| `08_export_training_data.sh` | Skips if train output and summary already exist. |

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

## Independent Model Evaluation

Model accuracy evaluation is intentionally separate from the synthesis pipeline.
It lives under `evaluation/` and is launched manually. The evaluator first
prepares the validation set for answer comparison, for example extracting
`#### 72` from raw GSM8K answers, then asks the served model to answer each
question with `evaluation/prompt/generate.json`. It writes predictions plus a
JSON/Markdown report with sample accuracy and pass@k.

Edit `evaluation/eval.env` for routine changes such as model path, input path,
output directory, temperature, top-p, concurrency, and answers per question.
`evaluation/eval.example.env` documents the available keys.
Set `EVAL_MAX_RETRIES=-1` to retry model-answer requests indefinitely.

```bash
cd /root/brjverl/data_gradual_new
bash evaluation/run_model_eval.sh gsm8k_2
```

For multiple answers per question, increase `EVAL_N_ANSWERS`; the report will
include `pass@1 ... pass@k`.

```bash
EVAL_N_ANSWERS=5 EVAL_TEMPERATURE=0.7 EVAL_TOP_P=0.95 bash evaluation/run_model_eval.sh gsm8k_2
```

For data preparation, `CLASSIFY_MAX_RETRIES=-1` makes question classification
retry indefinitely. If a model returns text outside the configured categories,
the classifier asks it to choose again from the allowed category list.

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
| `PLAN_USE_FULL_SCENE_DOMAINS` | `0` | Use the full scene pool when `1`; default `0` favors GSM8K-like everyday scenes |
| `QC_MAX_QUESTION_CHARS` | `700` | Reject overly long questions before blind validation |
| `QC_MAX_SOLUTION_CHARS` | `900` | Reject overly verbose generated solutions |
| `QC_MAX_STEP_COUNT` | `10` | Reject generated solutions with too many steps |
| `QC_TEMPLATE_CALCULATE_MAX_STEPS` | `1` | Reject formulaic outputs that start many steps with `Calculate` |
| `QC_BLOCK_TRAINING_UNFRIENDLY_SCENES` | `1` | Reject technical warehouse/lab/software/engineering-style scenes by default |
| `QC_WARN_OVERUSED_FINAL_ANSWERS` | `1` | Add warnings for very common final answers such as `10`, `20`, `60`, or `120` |
| `QC_TRAINING_STYLE_HARD_FAIL` | `0` | Treat training-style issues as warnings by default, avoiding expensive repair loops |
| `QC_SEVERE_MAX_QUESTION_CHARS` | `1200` | Still hard-fail extremely long questions |
| `QC_SEVERE_MAX_SOLUTION_CHARS` | `1800` | Still hard-fail extremely verbose solutions |
| `QC_SEVERE_MAX_STEP_COUNT` | `16` | Still hard-fail extremely long step lists |
| `QC_DIFFICULTY_TOLERANCE` | `1` | Allow adjacent difficulty estimates to pass validation |
| `QC_REQUIRE_EXACT_DIFFICULTY` | `0` | Set to `1` to require exact target difficulty matching |
| `RUN_STEP_REFINEMENT` | `1` | Enable the step-refinement stage in the full pipeline |
| `REFINE_CONCURRENCY` | `128` | Concurrent step-refinement requests |
| `REFINE_MAX_ROUNDS` | `-1` | Retry rounds for failed step refinements; `-1` means unlimited |
| `REFINE_MAX_TOKENS` | `900` | Maximum output tokens for step refinement |
| `REFINE_CHECKPOINT_EVERY` | `50` | Save refined output every N completed records |
| `RUN_DATA_PREPARE` | `1` | Enable data preparation in the full pipeline; set `0` to bypass |
| `DATA_FORMAT_TEMPLATE` | `gsm8k` | Raw-data format adapter, such as `gsm8k` or `passthrough` |
| `PREPARE_CLASSIFY` | `1` | Fill missing `question_type` values during data preparation |
| `CLASSIFY_MODEL` | `VLLM_MODEL` | Model used for question classification |
| `CLASSIFY_CONCURRENCY` | `16` | Concurrent classification requests |

### Training-Quality Controls

The synthesis planner now defaults to a GSM8K-friendly scene pool: school,
shopping, chores, sports, food, money, time, distance, community events, and
other everyday contexts. Technical domains such as warehouses, computer labs,
solar projects, water stations, airports, and recycling centers are excluded
unless `PLAN_USE_FULL_SCENE_DOMAINS=1`.

Generation prompts also ask for compact GSM8K-style problems and concise
solutions. Validation precheck records candidates that are too long, too
formulaic, or obviously training-unfriendly as warnings by default, so
mathematically correct samples are not forced into expensive repair loops.
Set `QC_TRAINING_STYLE_HARD_FAIL=1` if you prefer strict filtering.

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

## Step Refinement And Training Export

After validation, the optional step-refinement stage rewrites only the `steps`
field. It preserves all other fields and keeps the validated answer and solution
path unchanged:

```bash
bash run/07_refine_solution_steps.sh gsm8k
```

Outputs:

```text
outputs/pipeline/<dataset>/refined.jsonl
outputs/pipeline/<dataset>/refine.failed.jsonl
outputs/pipeline/<dataset>/refine.raw.jsonl
outputs/pipeline/<dataset>/refine.summary.json
outputs/pipeline/<dataset>/refine.rounds/
```

`refine.rounds/` stores per-round `input`, `success`, `raw`, `failed`, and
`summary` files, mirroring the generate/validation debug style.

The final export stage converts `refined.jsonl` into the legacy
supervised-fine-tuning JSONL format. If `refined.jsonl` does not exist, it falls
back to `validated.jsonl`:

```bash
bash run/08_export_training_data.sh gsm8k
```

`run/07_export_training_data.sh` remains as a compatibility wrapper for export.

Default output:

```text
outputs/pipeline/<dataset>/train.jsonl
```

Each line contains exactly:

```json
{"instruction":"...","input":"question","output":"Step 1: ...\nStep 2: ...\nThe answer is $\\boxed{XXX}$."}
```

`output` is built by joining the validated `steps` with one step per line and
appending the final answer on its own line with the strict template
`The answer is $\\boxed{XXX}$.`. If a validated step lacks an ordinal label, the
exporter prefixes `Step N:`. Simple mechanical steps such as `Calculate X: ...`
are lightly rewritten at export time into goal-oriented wording so the training
target exposes the purpose of each intermediate value.

## Ablations

Ablation code lives under `ablations/` and does not change the normal stage
commands. Run the main pipeline through Stage 03 first, then launch isolated
ablation variants:

```bash
bash ablations/run_ablation.sh gsm8k answer_accuracy_only
bash ablations/run_ablation.sh gsm8k hard_all
bash ablations/run_ablation.sh gsm8k equal_all
bash ablations/run_ablation.sh gsm8k easy_all
bash ablations/run_ablation.sh gsm8k uniform_count
```

Outputs are written to:

```text
outputs/ablations/<dataset>/<variant>/
```

Variants:

- `answer_accuracy_only`: removes step scoring from the mastery signal and
  recomputes allocation from final-answer accuracy only.
- `hard_all`: keeps the computed per-seed counts but forces every target
  difficulty to `Hard`.
- `equal_all`: keeps the computed per-seed counts but forces every target
  difficulty to `Equal`.
- `easy_all`: keeps the computed per-seed counts but forces every target
  difficulty to `Easy`.
- `uniform_count`: keeps the computed difficulty but gives every seed the same
  target count. Set `ABLATION_UNIFORM_COUNT=...` to choose the count manually;
  otherwise the rounded mean of original counts is used.

The runner builds the ablation mastery file, then reuses Stage 04 and Stage 05
with overridden output paths. Add `--run-validation`, `--run-refine`, and
`--export` when you want later stages too:

```bash
bash ablations/run_ablation.sh gsm8k hard_all --run-validation --run-refine --export
```

For validation ablations, skipping Stage 07 is already supported because Stage
08 falls back from `refined.jsonl` to `validated.jsonl`:

```bash
bash run/06_validate_generated.sh gsm8k
bash run/08_export_training_data.sh gsm8k
```

Skipping Stage 06 requires an explicit export input because Stage 08 normally
expects validated or refined records:

```bash
EXPORT_INPUT_PATH=/root/brjverl/data_gradual_new/outputs/pipeline/gsm8k/generated.jsonl \
  bash run/08_export_training_data.sh gsm8k
```
