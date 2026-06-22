# data_gradual_new

Independent gradual math-data synthesis pipeline. Accepted mastery and quantity/difficulty distribution logic is preserved. The downstream planning, generation, blind validation, and targeted repair stages are implemented in this directory.

Chinese documentation: [README.zh.md](./README.zh.md)

## Current flow

1. Format the source dataset and build the KB.
2. Let the victim model answer each seed question `N` times using only the question.
3. Compare numeric answers and score victim-provided reasoning steps.
4. Compute mastery and assign five-level relative difficulty plus synthesis count.
5. Build a diversity-oriented synthesis plan.
6. Generate question, steps, and answer asynchronously.
7. Run deterministic prechecks.
8. Produce two independent blind solutions; add a tie-break vote when needed.
9. Audit correctness, solvability, uniqueness, steps, and relative difficulty.
10. Apply targeted repair and revalidate in the next batch round.
11. Export passed records to `validated.jsonl`.

Training-format export is not connected yet.

## Run

```bash
cd /root/brjverl/data_gradual_new
bash run/run_full_pipeline.sh gsm8k
```

The launcher automatically loads and activates the configured pipeline
environment.

The full pipeline validates by default. To stop after generation:

```bash
bash run/run_full_pipeline.sh gsm8k --skip-validation
```

Standalone downstream stages:

```bash
bash run/run_build_synthesis_plan.sh gsm8k
bash run/run_generate_questions.sh gsm8k
bash run/run_validate_generated.sh gsm8k
```

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

## Validation design

Blind solvers receive only the generated question. They never see the candidate steps or answer. The auditor later receives the candidate, blind consensus, target difficulty, and seed question/solution as a relative-difficulty reference.

Repair actions:

- `repair_solution`: keep the question exactly unchanged and replace steps/answer.
- `repair_question`: minimally fix ambiguity, missing conditions, uniqueness, or difficulty.
- `regenerate_question`: create a fresh problem from the plan.
- request errors: keep the candidate and retry validation next round.

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
