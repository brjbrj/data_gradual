# data_gradual_new

这是一个独立的渐进式数学数据合成项目。项目保留了原始的 mastery 计算、合成数量分配和相对难度分配逻辑，并在当前目录中实现了后续的计划构建、题目生成、盲解验证、定向修复和训练数据导出流程。

英文文档：[README.md](./README.md)

## 当前完整流程

1. 格式化原始数学题数据，并构建知识库 KB。
2. 使用被测模型仅根据题目内容回答每道种子题 `N` 次。
3. 保存被测模型输出的推理步骤和最终答案。
4. 与标准答案做数值比对，并对被测模型的推理步骤进行评分。
5. 根据正确率和步骤质量计算每道种子题的 mastery。
6. 根据 mastery 为每道种子题分配合成数量和五级相对难度。
7. 根据 KB、mastery 和多样性策略构建合成计划。
8. 异步并发生成新题、步骤和答案。
9. 对生成题做程序预检查。
10. 验证模型对题目进行独立盲解，必要时追加 tie-break vote。
11. 审计候选答案、步骤、可解性、唯一性和相对难度。
12. 根据错误类型定向修复，并进入下一轮盲解和审计。
13. 输出通过验证的简洁数据，并可导出训练格式 JSONL。

## 推荐运行方式：分步执行

现在推荐使用编号阶段脚本，而不是一次性跑完整 pipeline。分步方式更适合外部手动启动 vLLM，也更容易从中断位置恢复。

```bash
cd /jizhicfs/hymiezhao/lpc/repos/brj/data_gradual_new
export STAGE_VLLM_MODE=external
```

阶段脚本会自动读取 `config/pipeline.env`，并使用其中配置的 `PIPELINE_PYTHON` 或 pipeline conda 环境。默认情况下，阶段脚本不会启动、切换或关闭 vLLM；你需要在外部提前启动好当前阶段所需模型。

### 阶段命令

```bash
bash run/01_build_kb.sh gsm8k
bash run/02_answer_seed.sh gsm8k
bash run/03_score_seed.sh gsm8k
bash run/04_build_synthesis_plan.sh gsm8k
bash run/05_generate_questions.sh gsm8k
bash run/06_validate_generated.sh gsm8k
bash run/07_export_training_data.sh gsm8k
```

### 每阶段模型要求和主要输出

| 阶段 | vLLM 要求 | 主要输出 |
| --- | --- | --- |
| `01_build_kb.sh` | 不需要 vLLM | `outputs/kb/<dataset>/records.jsonl` |
| `02_answer_seed.sh` | 需要外部服务当前提供 `VICTIM_MODEL` | `outputs/analysis/<dataset>/victim_answers.raw.jsonl` |
| `03_score_seed.sh` | 需要外部服务当前提供 `STEP_MODEL` | `outputs/analysis/<dataset>/mastery_records.jsonl` |
| `04_build_synthesis_plan.sh` | 不需要 vLLM | `outputs/planning/<dataset>/synthesis_plan.jsonl` |
| `05_generate_questions.sh` | 需要外部服务当前提供 `GEN_MODEL` | `outputs/pipeline/<dataset>/generated.jsonl` |
| `06_validate_generated.sh` | 需要外部服务当前提供 `QC_MODEL` | `outputs/pipeline/<dataset>/validated.jsonl` |
| `07_export_training_data.sh` | 不需要 vLLM | `outputs/pipeline/<dataset>/train.jsonl` |

例如：如果第 2 阶段使用 Llama，第 3、5、6 阶段使用 Qwen，那么你需要在进入相应阶段前，手动启动或切换外部 vLLM 服务到对应模型。

### 外部 vLLM 检查方式

阶段脚本会访问：

```text
${VLLM_BASE_URL}/models
```

并携带：

```text
Authorization: Bearer ${VLLM_API_KEY}
```

模型匹配支持以下几种形式：

```text
/path/to/Model
/path/to/Model/
Model
```

也就是说完整路径、末尾斜杠、仅 basename 都可以识别为同一个模型。

## 恢复机制

默认开启恢复：

```bash
export STAGE_RESUME=1
```

如果要强制某个阶段从头重跑：

```bash
STAGE_FORCE=1 bash run/05_generate_questions.sh gsm8k
```

各阶段恢复行为如下：

| 阶段 | 恢复行为 |
| --- | --- |
| `01_build_kb.sh` | 如果 KB records 和 entities 已存在，则跳过。 |
| `02_answer_seed.sh` | 从 `victim_answers.raw.jsonl` 恢复；每 `ANSWER_CHECKPOINT_EVERY` 条答案保存一次。 |
| `03_score_seed.sh` | 从 `step_evaluations.jsonl.partial` 恢复；评分完成一条就追加保存。 |
| `04_build_synthesis_plan.sh` | 如果 plan 和 summary 已存在，则跳过。 |
| `05_generate_questions.sh` | 从 `generated.jsonl` 恢复；跳过已经成功的 `plan_id`；每 `GEN_CHECKPOINT_EVERY` 条完成项保存一次。 |
| `06_validate_generated.sh` | 每轮验证后保存 canonical 文件和 `validation.rounds`；如果 validated 输出已存在，则跳过。 |
| `07_export_training_data.sh` | 如果 train 输出和 summary 已存在，则跳过。 |

常用恢复参数：

```bash
ANSWER_RESUME=1
ANSWER_CHECKPOINT_EVERY=50
SCORE_RESUME=1
GEN_RESUME=1
GEN_CHECKPOINT_EVERY=50
```

如果你明确希望生成阶段从头再来：

```bash
GEN_RESUME=0 bash run/05_generate_questions.sh gsm8k --no-resume
```

## 可选托管 vLLM 模式

默认推荐外部启动 vLLM。如果你明确希望某个阶段脚本自己启动 vLLM，可以使用：

```bash
STAGE_VLLM_MODE=managed bash run/05_generate_questions.sh gsm8k
```

如果希望阶段结束时自动关闭这个 managed vLLM：

```bash
STAGE_VLLM_MODE=managed STAGE_VLLM_STOP_ON_EXIT=1 bash run/05_generate_questions.sh gsm8k
```

## 旧命令兼容

旧命令仍然保留为 wrapper，但现在会转发到新的编号阶段脚本：

```bash
bash run/run_full_pipeline.sh gsm8k
bash run/run_generate_questions.sh gsm8k
bash run/run_validate_generated.sh gsm8k
```

## 主要配置文件

配置文件：

```text
config/pipeline.env
```

常见路径和环境配置：

```bash
CONDA_SH=/jizhicfs/hymiezhao/miniconda3/etc/profile.d/conda.sh
PIPELINE_CONDA_ENV=
PIPELINE_PYTHON=/jizhicfs/hymiezhao/miniconda3/envs/brj/bin/python
VLLM_CONDA_ENV=
VLLM_PYTHON=/jizhicfs/hymiezhao/miniconda3/envs/brjqwen/bin/python
VLLM_BASE_URL=http://127.0.0.1:8911/v1
VLLM_API_KEY=EMPTY
```

模型配置：

```bash
VICTIM_MODEL=/path/to/Meta-Llama-3-8B-Instruct
STEP_MODEL=/path/to/Qwen3.6-27B/
GEN_MODEL=/path/to/Qwen3.6-27B/
QC_MODEL=/path/to/Qwen3.6-27B/
REPAIR_MODEL=/path/to/Qwen3.6-27B/
```

生成配置示例：

```bash
GEN_CONCURRENCY=256
GEN_CONCURRENCY_CAP=256
GEN_MAX_RETRIES=3
GEN_MAX_TOKENS=900
GEN_FORCE_JSON=1
GEN_ENABLE_THINKING=0
GEN_ROUND_RETRY_DELAY=1
GEN_RESUME=1
GEN_CHECKPOINT_EVERY=50
```

验证配置示例：

```bash
RUN_VALIDATION=1
QC_CONCURRENCY=256
QC_BLIND_VOTES=2
QC_TIEBREAK_VOTES=1
QC_MAX_ROUNDS=-1
QC_MAX_TOKENS=900
QC_ENABLE_THINKING=0
QC_FORCE_JSON=0
QC_ROUND_RETRY_DELAY=1
```

## vLLM / NCCL 配置说明

`start_vllm.sh --dry-run` 可以查看最终环境变量和启动命令：

```bash
bash run/start_vllm.sh --dry-run
```

如果外部手动启动 vLLM，则阶段脚本只依赖这些接口配置：

```bash
VLLM_BASE_URL=http://127.0.0.1:8911/v1
VLLM_API_KEY=EMPTY
```

如果使用项目脚本启动 vLLM，则可通过 `pipeline.env` 配置：

```bash
VLLM_CUDA_VISIBLE_DEVICES=0,1,2,3
VLLM_NCCL_DEBUG=INFO
VLLM_NCCL_BLOCKING_WAIT=1
VLLM_NCCL_P2P_DISABLE=0
VLLM_NCCL_IB_DISABLE=0
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_USE_FLASH_ATTN=0
VLLM_USE_FLASHINFER=0
FLASHINFER_DISABLE_JIT=1
```

## 合成计划输出

合成计划文件：

```text
outputs/planning/<dataset>/synthesis_plan.jsonl
```

每条记录主要包含：

```json
{
  "source_task_id": 0,
  "plan_id": "0_0",
  "knowledge": {
    "math": {},
    "diversity": {},
    "kb_inspiration": {}
  }
}
```

- `math`：目标数学能力、操作序列、相对难度等。
- `diversity`：主场景、备选场景、叙事风格、数值策略等。
- `kb_inspiration`：知识库提供的可选灵感，不要求照搬原题。

职责边界：

- `04_build_synthesis_plan.sh` 负责多样性和相似性预防，包括知识点选择、场景构建、问题模式、难度预设、数值策略和生成数量。
- `05_generate_questions.sh` 只负责根据 plan 生成题目、步骤和数值答案，并做轻量结构检查：JSON 是否可解析、字段是否齐全、答案是否为数值、题面是否体现 plan 的场景或关键词。不再做全局相似度扫描。
- `06_validate_generated.sh` 负责后置校验，包括数学正确性、可解性、唯一答案、难度匹配、多轮修复、重新生成；多次失败后再回退到 replan，重新调整 plan 后继续生成。
- generate 阶段的失败队列只处理格式错误、字段缺失、未按 plan 输出等问题；相似性控制应在 plan/replan 阶段完成。

## 生成输出

生成题标准文件：

```text
outputs/pipeline/<dataset>/generated.jsonl
```

格式：

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

过程文件：

```text
generated.raw.jsonl
generated.failed.jsonl
generated.summary.json
generated.rounds/
```

## 验证输出

通过验证的文件：

```text
outputs/pipeline/<dataset>/validated.jsonl
```

详细过程文件：

```text
validation_reports.jsonl
validation.failed.jsonl
repair_history.jsonl
validated.summary.json
validation.rounds/
```

## 更多阶段说明

详见：

```text
run/README_stages.md
```
