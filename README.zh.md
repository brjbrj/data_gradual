# data_gradual_new

这是一个独立的渐进式数学数据合成项目。项目保留了原始的 mastery 计算、合成数量分配和相对难度分配逻辑，并在当前目录中实现了后续的计划构建、题目生成、盲解验证、定向修复和训练数据导出流程。

英文文档：[README.md](./README.md)

## 当前完整流程

1. 先浏览并准备源数据。GSM8K 等原始格式会转成项目标准字段，缺失的 `question_type` 会由分类模型补齐。
2. 根据准备好的标准数据构建知识库 KB。
3. 使用被测模型仅根据题目内容回答每道种子题 `N` 次。
4. 保存被测模型输出的推理步骤和最终答案。
5. 与标准答案做数值比对，并对被测模型的推理步骤进行评分。
6. 根据正确率和步骤质量计算每道种子题的 mastery。
7. 根据 mastery 为每道种子题分配合成数量和五级相对难度。
8. 根据 KB、mastery 和多样性策略构建合成计划。
9. 异步并发生成新题、步骤和答案。
10. 对生成题做程序预检查。
11. 验证模型对题目进行独立盲解，必要时追加 tie-break vote。
12. 审计候选答案、步骤、可解性、唯一性和相对难度。
13. 根据错误类型定向修复，并进入下一轮盲解和审计。
14. 输出通过验证的简洁数据。
15. 可选地只改写 `steps`，把正确但机械的步骤改成更适合训练的依赖关系推理链。
16. 导出训练格式 JSONL。

## 推荐运行方式：分步执行

现在推荐使用编号阶段脚本，而不是一次性跑完整 pipeline。分步方式更适合外部手动启动 vLLM，也更容易从中断位置恢复。

```bash
cd /jizhicfs/hymiezhao/lpc/repos/brj/data_gradual_new
export STAGE_VLLM_MODE=external
```

阶段脚本会自动读取 `config/pipeline.env`，并使用其中配置的 `PIPELINE_PYTHON` 或 pipeline conda 环境。默认情况下，阶段脚本会跟随 `config/pipeline.env` 中的 `VLLM_RUNTIME_MODE`；如果想强制外部手动管理 vLLM，可设置 `STAGE_VLLM_MODE=external`，如果想强制脚本托管启动/切换 vLLM，可设置 `STAGE_VLLM_MODE=managed`。

### 阶段命令

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

### 每阶段模型要求和主要输出

| 阶段 | vLLM 要求 | 主要输出 |
| --- | --- | --- |
| `00_prepare_data.sh` | 只有缺少 `question_type` 时才需要 `CLASSIFY_MODEL` | `outputs/prepared/<dataset>/<dataset>.prepared.jsonl` |
| `01_build_kb.sh` | 不需要 vLLM | `outputs/kb/<dataset>/records.jsonl` |
| `02_answer_seed.sh` | 需要外部服务当前提供 `VICTIM_MODEL` | `outputs/analysis/<dataset>/victim_answers.raw.jsonl` |
| `03_score_seed.sh` | 需要外部服务当前提供 `STEP_MODEL` | `outputs/analysis/<dataset>/mastery_records.jsonl` |
| `04_build_synthesis_plan.sh` | 不需要 vLLM | `outputs/planning/<dataset>/synthesis_plan.jsonl` |
| `05_generate_questions.sh` | 需要外部服务当前提供 `GEN_MODEL` | `outputs/pipeline/<dataset>/generated.jsonl` |
| `06_validate_generated.sh` | 需要外部服务当前提供 `QC_MODEL` | `outputs/pipeline/<dataset>/validated.jsonl` |
| `07_refine_solution_steps.sh` | 需要外部服务当前提供 `REFINE_MODEL`/`REPAIR_MODEL` | `outputs/pipeline/<dataset>/refined.jsonl` |
| `08_export_training_data.sh` | 不需要 vLLM | `outputs/pipeline/<dataset>/train.jsonl` |

例如：如果第 2 阶段使用 Llama，第 3、5、6 阶段使用 Qwen，那么你需要在进入相应阶段前，手动启动或切换外部 vLLM 服务到对应模型。

### 单独测试前置处理

如果只想测试原始数据格式化和题型分类，不运行后续 pipeline，可以直接运行：

```bash
cd /root/brjverl/data_gradual_new
STAGE_FORCE=1 SAMPLE_LIMIT=5 \
RAW_INPUT_PATH=/root/brjverl/datas/gsm8k_2.jsonl \
PREPARED_INPUT_PATH=/tmp/gsm8k_2.prepared.test.jsonl \
bash run/00_prepare_data.sh gsm8k_2
```

查看输出：

```bash
head -n 5 /tmp/gsm8k_2.prepared.test.jsonl
```

该阶段会先浏览输入数据 schema。如果记录已经包含 `task_id`、`question`、`answer`、`solution_steps` 和 `proficiency_score`，会自然跳过 format；如果所有记录都已经有非空 `question_type`，会自然跳过分类，并且不会检查或启动 vLLM。若只想测试格式化、不调用分类模型：

```bash
STAGE_FORCE=1 PREPARE_CLASSIFY=0 SAMPLE_LIMIT=5 \
RAW_INPUT_PATH=/root/brjverl/datas/gsm8k_2.jsonl \
PREPARED_INPUT_PATH=/tmp/gsm8k_2.formatted.test.jsonl \
bash run/00_prepare_data.sh gsm8k_2
```

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
| `00_prepare_data.sh` | 如果 prepared input 已存在则跳过；设置 `STAGE_FORCE=1` 可重建。 |
| `01_build_kb.sh` | 如果 KB records 和 entities 已存在，则跳过。 |
| `02_answer_seed.sh` | 从 `victim_answers.raw.jsonl` 恢复；每 `ANSWER_CHECKPOINT_EVERY` 条答案保存一次。 |
| `03_score_seed.sh` | 从 `step_evaluations.jsonl.partial` 恢复；评分完成一条就追加保存。 |
| `04_build_synthesis_plan.sh` | 如果 plan 和 summary 已存在，则跳过。 |
| `05_generate_questions.sh` | 从 `generated.jsonl` 恢复；跳过已经成功的 `plan_id`；每 `GEN_CHECKPOINT_EVERY` 条完成项保存一次。 |
| `06_validate_generated.sh` | 每轮验证后保存 canonical 文件和 `validation.rounds`；如果 validated 输出已存在，则跳过。 |
| `07_refine_solution_steps.sh` | 从 `refined.jsonl` 恢复；跳过已经改写成功的记录；每轮会清空并重写 `refine.failed.jsonl`。 |
| `08_export_training_data.sh` | 如果 train 输出和 summary 已存在，则跳过。 |

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

单独运行某个阶段时，managed 模式会先检查已有 vLLM：如果模型匹配，就直接复用并保留；如果服务不可用或模型不匹配，就先关闭当前 vLLM，再启动该阶段需要的模型。凡是由这个单阶段脚本启动或切换出来的 vLLM，阶段结束会默认自动关闭。若你想保留模型给下一条手动 stage 命令复用，可以显式关闭这个清理行为：

```bash
STAGE_VLLM_MODE=managed STAGE_VLLM_STOP_ON_EXIT=0 bash run/05_generate_questions.sh gsm8k
```

## 旧命令兼容

旧命令仍然保留为 wrapper，但现在会转发到新的编号阶段脚本：

```bash
bash run/run_full_pipeline.sh gsm8k
bash run/run_generate_questions.sh gsm8k
bash run/run_validate_generated.sh gsm8k
```

`run_full_pipeline.sh` 仍然兼容，而且默认使用 managed vLLM 模式：一条命令会按阶段自动启动或切换所需模型，并在全流程退出时关闭由脚本启动的 vLLM。

如果你已经在外部手动启动 vLLM，希望脚本只检查现有服务，可以显式使用：

```bash
STAGE_VLLM_MODE=external bash run/run_full_pipeline.sh gsm8k
```

外部模式下，如果不同阶段使用不同模型，需要你在对应阶段前手动切换 vLLM。

## 独立模型评测

模型准确率评测不放入主合成流程，代码独立放在 `evaluation/` 目录下，需要手动执行。评测脚本会先对验证集做评测专用预处理，例如从 GSM8K 原始答案中提取 `#### 72` 作为标准答案；然后严格使用 `evaluation/prompt/generate.json` 让模型回答；最后输出 predictions、JSON 报告和 Markdown 报告，包含 sample accuracy 和 pass@k。

日常修改评测参数时，优先编辑 `evaluation/eval.env`，例如模型路径、输入路径、输出目录、温度、top_p、并发数和每题回答次数。`evaluation/eval.example.env` 记录了可用配置项。
设置 `EVAL_MAX_RETRIES=-1` 时，评测生成回答请求会无限重试。

```bash
cd /root/brjverl/data_gradual_new
bash evaluation/run_model_eval.sh gsm8k_2
```

如果每题要生成多次回答，把 `EVAL_N_ANSWERS` 调大即可，报告会给出 `pass@1 ... pass@k`：

```bash
EVAL_N_ANSWERS=5 EVAL_TEMPERATURE=0.7 EVAL_TOP_P=0.95 bash evaluation/run_model_eval.sh gsm8k_2
```

前置数据处理里的分类也支持 `CLASSIFY_MAX_RETRIES=-1` 无限重试。如果模型输出不在预设类别中，分类器会带着允许类别列表要求模型重新分类。

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

前置处理配置示例：

```bash
RUN_DATA_PREPARE=1
DATA_FORMAT_TEMPLATE=gsm8k
PREPARE_CLASSIFY=1
CLASSIFY_MODEL=/path/to/Qwen3.6-27B/
CLASSIFY_BASE_URL=http://127.0.0.1:8911/v1
CLASSIFY_API_KEY=EMPTY
CLASSIFY_CONCURRENCY=16
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

训练质量控制参数：

```bash
PLAN_USE_FULL_SCENE_DOMAINS=0
QC_MAX_QUESTION_CHARS=700
QC_MAX_SOLUTION_CHARS=900
QC_MAX_STEP_COUNT=10
QC_TEMPLATE_CALCULATE_MAX_STEPS=1
QC_BLOCK_TRAINING_UNFRIENDLY_SCENES=1
QC_WARN_OVERUSED_FINAL_ANSWERS=1
QC_TRAINING_STYLE_HARD_FAIL=0
QC_SEVERE_MAX_QUESTION_CHARS=1200
QC_SEVERE_MAX_SOLUTION_CHARS=1800
QC_SEVERE_MAX_STEP_COUNT=16
QC_DIFFICULTY_TOLERANCE=1
QC_REQUIRE_EXACT_DIFFICULTY=0
RUN_STEP_REFINEMENT=1
REFINE_CONCURRENCY=128
REFINE_MAX_ROUNDS=-1
REFINE_MAX_TOKENS=900
REFINE_CHECKPOINT_EVERY=50
```

含义：

- `PLAN_USE_FULL_SCENE_DOMAINS=0` 时，plan 阶段默认使用更接近 GSM8K 的日常场景池，如学校、购物、家务、运动、食物、金钱、时间、距离和社区活动。
- `PLAN_USE_FULL_SCENE_DOMAINS=1` 时，恢复完整场景池，允许仓储、实验室、软件、太阳能、水站、机场等技术/工程化场景。
- `QC_MAX_QUESTION_CHARS`、`QC_MAX_SOLUTION_CHARS`、`QC_MAX_STEP_COUNT` 用于拦截过长、过啰嗦、步骤过多的样本。
- `QC_TEMPLATE_CALCULATE_MAX_STEPS` 用于拦截大量步骤都以 `Calculate...` 开头的模板化解答。
- `QC_BLOCK_TRAINING_UNFRIENDLY_SCENES=1` 时，校验预检查会拒绝明显不利于 GSM8K 风格训练的技术化场景。
- `QC_TRAINING_STYLE_HARD_FAIL=0` 时，训练风格问题只作为 warning 记录，不进入修复循环；这能显著提高每轮 pass 率。
- `QC_SEVERE_MAX_QUESTION_CHARS`、`QC_SEVERE_MAX_SOLUTION_CHARS`、`QC_SEVERE_MAX_STEP_COUNT` 仍会拦截极端冗长样本，避免完全失控的输出进入训练集。
- `QC_DIFFICULTY_TOLERANCE=1` 表示允许相邻难度档位通过，避免审计模型对难度边界判断过严导致反复修复。
- `QC_REQUIRE_EXACT_DIFFICULTY=1` 时才恢复严格难度匹配。
- `RUN_STEP_REFINEMENT=1` 时，全流程会在验证后运行步骤改写阶段。
- `REFINE_*` 参数控制步骤改写阶段的并发、最大轮数、token 和 checkpoint；`REFINE_MAX_ROUNDS=-1` 表示无限重试。
- 被拒绝的样本不会直接进入训练数据，而是进入已有的重新生成、修复或 replan 流程。

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

### 同一机器并行跑两个 managed 实验

可以并行，但不能只改端口。两个实验的 vLLM 端口、输出目录、PID 文件、日志文件和 GPU 分配都应分开。现在支持通过 `PIPELINE_CONFIG_FILE` 加载 overlay 配置文件：先读取 `config/pipeline.env`，再读取指定的覆盖配置。

两个终端可以这样启动：

```bash
PIPELINE_CONFIG_FILE=config/parallel_exp_a.example.env bash run/run_stage_sequence.sh gsm8k
PIPELINE_CONFIG_FILE=config/parallel_exp_b.example.env bash run/run_stage_sequence.sh gsm8k
```

每个 overlay 至少需要区分这些项：

```bash
VLLM_BASE_URL=http://127.0.0.1:8911/v1
VLLM_API_PORT=8911
OUTPUT_DIR=/path/to/outputs_exp_a
VLLM_PID_FILE=/path/to/outputs_exp_a/runtime/vllm/vllm.pid
VLLM_LOG_FILE=/path/to/outputs_exp_a/runtime/vllm/vllm.log
VLLM_CUDA_VISIBLE_DEVICES=0,1
```

第二个实验使用另一个端口、另一个输出/运行目录和另一组 GPU。这样脚本切换或关闭 vLLM 时会按当前配置的 PID 文件和端口处理，不会误关另一组实验。

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
- `07_refine_solution_steps.sh` 只负责在验证通过的基础上改写 `steps`，不允许修改题目、答案、难度、ID 或数学解法路径。
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

## 步骤改写输出

步骤改写阶段输入 `validated.jsonl`，输出：

```text
outputs/pipeline/<dataset>/refined.jsonl
outputs/pipeline/<dataset>/refine.failed.jsonl
outputs/pipeline/<dataset>/refine.raw.jsonl
outputs/pipeline/<dataset>/refine.summary.json
outputs/pipeline/<dataset>/refine.rounds/
```

`refine.rounds/` 会保存每一轮的 `input`、`success`、`raw`、`failed` 和 `summary` 文件，便于像 generate/validation 一样按轮次排查。

该阶段只替换 `steps` 字段，其余字段原样保留。若手动中断，重新运行会从 `refined.jsonl` 继续。

## 训练数据导出

最后阶段会优先将 `refined.jsonl` 转换为旧版监督训练 JSONL 格式；如果没有 `refined.jsonl`，则回退到 `validated.jsonl`：

```bash
bash run/08_export_training_data.sh gsm8k
```

`run/07_export_training_data.sh` 仍保留为兼容 wrapper，会转发到 `08_export_training_data.sh`。

默认输出位置：

```text
outputs/pipeline/<dataset>/train.jsonl
```

每一行严格包含：

```json
{"instruction":"...","input":"题目","output":"Step 1: ...\nStep 2: ...\nThe answer is $\\boxed{XXX}$."}
```

其中 `output` 会将 validated 记录中的 `steps` 按“一步一行”拼接，并把最终答案单独放在最后一行：`The answer is $\\boxed{XXX}$.`。
如果某个步骤没有编号，导出器会自动补充 `Step N:`。对于 `Calculate X: ...` 这类机械步骤，导出器会做轻量改写，补充“本步要得到什么中间量、为什么需要它”，让训练目标更像连续推理链条。

## 消融实验

消融实验代码放在 `ablations/`，不会改变主流程已有命令。先将主流程跑到 Stage 03，得到 KB、种子回答和原始 mastery 结果，然后单独运行：

```bash
bash ablations/run_ablation.sh gsm8k answer_accuracy_only
bash ablations/run_ablation.sh gsm8k hard_all
bash ablations/run_ablation.sh gsm8k equal_all
bash ablations/run_ablation.sh gsm8k easy_all
bash ablations/run_ablation.sh gsm8k uniform_count
```

输出目录：

```text
outputs/ablations/<dataset>/<variant>/
```

各变体含义：

- `answer_accuracy_only`：移除步骤评价信号，只根据最终答案准确率重新计算 mastery 和合成数量/难度分配。
- `hard_all`：合成数量仍使用原计算结果，但所有相对难度强制为 `Hard`。
- `equal_all`：合成数量仍使用原计算结果，但所有相对难度强制为 `Equal`。
- `easy_all`：合成数量仍使用原计算结果，但所有相对难度强制为 `Easy`。
- `uniform_count`：难度仍使用原计算结果，但每个种子题的合成数量相同。可通过 `ABLATION_UNIFORM_COUNT=...` 手动指定；不指定时使用原始 `target_count` 的四舍五入均值。

runner 会先生成消融版 `mastery_records.jsonl`，再复用 Stage 04 和 Stage 05，并把所有输出路径改到消融目录。需要继续跑后置环节时加参数：

```bash
bash ablations/run_ablation.sh gsm8k hard_all --run-validation --run-refine --export
```

关于后置校验消融：不跑 Stage 07 是支持的，因为 Stage 08 会在没有 `refined.jsonl` 时自动回退到 `validated.jsonl`：

```bash
bash run/06_validate_generated.sh gsm8k
bash run/08_export_training_data.sh gsm8k
```

如果要连 Stage 06 也跳过，需要显式指定导出输入，因为 Stage 08 默认不会直接吃 `generated.jsonl`：

```bash
EXPORT_INPUT_PATH=/root/brjverl/data_gradual_new/outputs/pipeline/gsm8k/generated.jsonl \
  bash run/08_export_training_data.sh gsm8k
```

详见：

```text
run/README_stages.md
```
