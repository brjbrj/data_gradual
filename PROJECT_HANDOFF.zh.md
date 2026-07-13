# data_gradual_new 项目交接文档

## 1. 项目目标

本项目用于构建“渐进式数学推理数据合成”流水线：基于原始 GSM8K 等数学题数据，结合被测模型的掌握度评估，自动规划并合成不同难度的新题，经过严格数学验证、修复、步骤改写后，导出可用于监督微调的训练数据。

核心目标是生成能够提升模型数学推理能力的高质量训练数据，而不是单纯扩大数据量。

## 2. 当前目录结构

```text
data_gradual_new/
├── config/
│   ├── pipeline.env
│   └── pipeline.example.env
├── data/
│   └── *.jsonl
├── kb_pipeline/
│   ├── assessment.py
│   ├── post_mastery_plan.py
│   ├── post_mastery_generate.py
│   ├── validation.py
│   ├── step_refine.py
│   ├── export.py
│   └── utils.py
├── run/
│   ├── 01_build_kb.sh
│   ├── 02_answer_seed.sh
│   ├── 03_score_seed.sh
│   ├── 04_build_synthesis_plan.sh
│   ├── 05_generate_questions.sh
│   ├── 06_validate_generated.sh
│   ├── 07_refine_solution_steps.sh
│   ├── 08_export_training_data.sh
│   ├── run_full_pipeline.sh
│   ├── stage_common.sh
│   └── start_vllm.sh / stop_vllm.sh
├── outputs/
├── README.md
├── README.zh.md
└── run/README_stages.md
```

## 3. 当前已经完成的功能

- 分阶段流水线脚本，支持单独运行和全流程运行。
- 支持 managed / external 两种 vLLM 模式。
- 支持不同阶段使用不同模型。
- 支持断点恢复、checkpoint、失败队列。
- 支持 Ctrl+C 后尽量清理 managed vLLM。
- 基于 mastery 的合成数量和难度分配。
- plan 阶段负责场景、知识点、难度、数值策略和多样性控制。
- generate 阶段只负责按 plan 生成候选题、步骤和答案。
- validate 阶段负责数学正确性、可解性、唯一答案、难度、修复、回退生成和 replan。
- 新增 `07_refine_solution_steps.sh`：只改写 `steps`，不改题目、答案、ID、难度和解题路径。
- refine 阶段新增 `refine.rounds/` 分轮日志，保存每轮 input/success/raw/failed/summary。
- export 阶段导出 legacy SFT 格式，最终答案严格为 `The answer is $\boxed{XXX}$.`。
- 中英文 README 和阶段文档已更新。

## 4. 正在开发完善的功能

- 合成数据训练效果优化。
- 步骤表达质量优化：让步骤不仅是机械计算，而是包含推理依赖、题意来源、中间量意义。
- validate 与 refine 的职责边界持续调整：validate 偏数学正确性，refine 偏训练友好的步骤表达。
- 生成数据分布对齐 GSM8K：减少工程化、促销化、过长题面和模板化步骤。

## 5. 关键技术栈

- Python 3.12
- Bash stage scripts
- OpenAI-compatible API client
- vLLM
- Qwen / Llama 本地模型
- JSONL 数据流
- GitHub 仓库：`brjbrj/data_gradual`

## 6. 重要文件说明

- `config/pipeline.env`：当前机器实际运行配置。
- `config/pipeline.example.env`：示例配置。
- `run/stage_common.sh`：阶段脚本公共环境、路径、vLLM 检查逻辑。
- `run/run_stage_sequence.sh`：全流程阶段顺序。
- `run/05_generate_questions.sh`：生成候选题。
- `run/06_validate_generated.sh`：后置数学验证、修复、回退生成。
- `run/07_refine_solution_steps.sh`：步骤改写阶段，只修改 `steps`。
- `run/08_export_training_data.sh`：导出最终训练数据。
- `kb_pipeline/post_mastery_plan.py`：合成计划构建。
- `kb_pipeline/post_mastery_generate.py`：候选题生成。
- `kb_pipeline/validation.py`：盲解、审计、修复、replan。
- `kb_pipeline/step_refine.py`：步骤表达改写。
- `kb_pipeline/export.py`：训练格式导出。
- `outputs/pipeline/<dataset>/refine.rounds/`：refine 每轮输入、成功、原始输出、失败和摘要日志。

## 7. 已知问题

- 生成题目若过多依赖复杂商业规则、小数、返现、促销等，训练收益可能不如旧版数据。
- 机械步骤如 `Calculate ...` 对训练不友好，需要 refine 阶段改写。
- validate 阶段若规则过严，会导致 pass 率下降、耗时显著增加。
- 盲解请求量大，`QC_BLIND_VOTES=2` 时 3w 样本约产生 6w+ 请求，耗时较长。
- 远程 `pipeline.env` 曾出现过异常追加的 `nSTAGE...` 字符，需要后续有空清理。
- 当前远程环境和本地 Windows 环境不同，远程路径不要用本地配置覆盖。

## 8. 下一步要做什么

- 用新 `07_refine_solution_steps.sh` 对已有 `validated.jsonl` 跑步骤改写。
- 再运行 `08_export_training_data.sh` 生成新的 `train.jsonl`。
- 抽样检查 `refined.jsonl` 中是否只修改了 `steps`。
- 若 refine 有样例反复失败，优先查看 `outputs/pipeline/<dataset>/refine.rounds/round_XXX.failed.jsonl` 和对应 `raw.jsonl`。
- 用 refined 训练数据和旧版数据做训练效果对比。
- 如果 refine 太慢，可调：

```bash
REFINE_CONCURRENCY=256
REFINE_MAX_TOKENS=700
```

- 如果 validate 太慢，可调：

```bash
QC_CONCURRENCY=512
QC_MAX_TOKENS=512
```

## 9. 不能改动 / 需要注意的约束

- `07_refine_solution_steps.sh` 只能修改 `steps`。
- refine 阶段不能修改：
  - `question`
  - `answer`
  - `difficulty`
  - `source_task_id`
  - `plan_id`
  - 数学解法路径
- validate 阶段主要负责数学正确性，不应把表达风格问题过度硬判，避免拖慢。
- generate 阶段不要重新加入重型全局相似度校验；相似性和多样性应由 plan/replan 控制。
- 不要覆盖远程服务器的真实 `pipeline.env` 路径配置。
- 不要把 GitHub token、SSH 密码等敏感信息写入文件或文档。
- 不要破坏旧命令兼容性：`run/07_export_training_data.sh` 当前是兼容 wrapper。

## 10. 项目整体运行逻辑和方案描述

完整流程：

```text
01_build_kb
  -> 02_answer_seed
  -> 03_score_seed
  -> 04_build_synthesis_plan
  -> 05_generate_questions
  -> 06_validate_generated
  -> 07_refine_solution_steps
  -> 08_export_training_data
```

各阶段目标：

- `01_build_kb`：格式化原始数据并构建知识库。
- `02_answer_seed`：让被测模型回答种子题。
- `03_score_seed`：评估被测模型步骤质量和正确率，计算 mastery。
- `04_build_synthesis_plan`：根据 mastery 规划合成数量、难度、场景、知识点和数值策略。
- `05_generate_questions`：根据 plan 生成候选题、候选步骤和答案。
- `06_validate_generated`：通过预检查、盲解、审计和修复，保证题目可解、唯一、答案正确、步骤数学正确。
- `07_refine_solution_steps`：在验证通过基础上，只将步骤改写为更适合训练的逻辑推理链。
- `08_export_training_data`：导出最终 SFT 训练 JSONL。

关键数据流：

```text
generated.jsonl
  -> validated.jsonl
  -> refined.jsonl
  -> train.jsonl
```

推荐运行：

```bash
cd /root/brjverl/data_gradual_new
bash run/run_full_pipeline.sh gsm8k
```

或分步运行：

```bash
bash run/06_validate_generated.sh gsm8k
bash run/07_refine_solution_steps.sh gsm8k
bash run/08_export_training_data.sh gsm8k
```
