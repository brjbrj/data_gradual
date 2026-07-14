# data_gradual_new 项目交接文档

## 1. 项目目标

构建一套“渐进式数学推理数据合成”流水线：基于 GSM8K 等种子题，评估被测模型的掌握度，按薄弱知识点和目标难度合成新题，经过数学验证、修复、步骤表达改写后，导出可用于监督微调的训练数据。

核心目标不是单纯扩大数据量，而是生成能提升模型数学推理能力的高质量训练样本。

## 2. 当前目录结构

```text
data_gradual_new/
├── config/
│   ├── pipeline.env              # 机器私有配置，不应随意覆盖
│   └── pipeline.example.env
├── data/
│   └── *.jsonl
├── kb_pipeline/
│   ├── assessment.py             # seed answer / step scoring / mastery
│   ├── post_mastery_plan.py      # 合成计划
│   ├── post_mastery_generate.py  # 按 plan 生成候选题
│   ├── validation.py             # 后置校验、修复、replan
│   ├── step_refine.py            # 只改写 steps
│   ├── export.py                 # 导出训练数据
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
│   ├── run_stage_sequence.sh
│   ├── stage_common.sh
│   └── start_vllm.sh / stop_vllm.sh
├── outputs/
├── README.md
├── README.zh.md
└── run/README_stages.md
```

## 3. 当前已经完成的功能

- 支持全流程运行和分阶段运行。
- 支持 `managed` / `external` 两种 vLLM 模式。
- managed 模式下支持自动启动、切换和退出时清理 vLLM。
- managed 模式已优化：如果下一阶段使用同一个模型，会复用当前 vLLM，不再重复 stop/start。
- vLLM readiness probe 已强制绕过 HTTP 代理，避免 `127.0.0.1` 请求被 Squid 劫持。
- 内部调用 `start_vllm.sh` / `stop_vllm.sh` 已改为 `bash xxx.sh`，不再依赖脚本执行权限。
- 各阶段支持 checkpoint / resume / failed 队列。
- `04_build_synthesis_plan` 负责知识点、场景、难度、数量和多样性规划。
- `05_generate_questions` 只负责按 plan 生成可解析候选题，不再承担重度相似度/数学正确性验证。
- `06_validate_generated` 负责可解性、唯一答案、答案正确、步骤正确、难度检查、repair、regenerate、replan。
- `07_refine_solution_steps` 只改写 `steps`，保持题目、答案、ID、难度和数学路径不变。
- refine 阶段已有多轮重试、历史 raw 恢复、分轮日志 `refine.rounds/`。
- `08_export_training_data` 导出 legacy SFT 格式，答案统一为 `The answer is $\boxed{XXX}$.`。
- GitHub 仓库已包含最近代码修复：`brjbrj/data_gradual`。

## 4. 正在开发完善的功能

- 合成数据训练收益优化，目标是超过旧版合成数据。
- 控制题面/步骤分布，减少工程化、促销化、过长题面、模板化步骤。
- refine 步骤表达质量继续优化：步骤要体现题意来源、中间量意义和逻辑依赖，不只是机械计算。
- validate/refine 职责继续保持清晰：validate 管数学正确性，refine 管训练友好的表达。
- 数据规模消融建议：`5x / 10x / 15x / 20x`，不建议一开始盲目上很大倍数。

## 5. 关键技术栈

- Python 3.12
- Bash stage scripts
- OpenAI-compatible API
- vLLM 0.19.0
- 本地 Qwen / Llama 模型
- JSONL 数据流
- GitHub API / GitHub 仓库同步

## 6. 重要文件说明

- `config/pipeline.env`：当前机器真实运行配置，远程服务器版本和 GitHub 版本不同，不能乱覆盖。
- `config/pipeline.example.env`：示例配置，可同步。
- `run/stage_common.sh`：阶段通用路径、环境、vLLM 检查、同模型复用逻辑。
- `run/start_vllm.sh`：vLLM 启动逻辑，负责 Python/env 检查、dry-run、日志、pid/model/python 文件。
- `run/stop_vllm.sh`：清理 vLLM 主进程和 worker。
- `run/run_full_pipeline.sh` / `run/run_stage_sequence.sh`：全流程入口。
- `kb_pipeline/assessment.py`：seed 回答、步骤评分、mastery 计算。
- `kb_pipeline/post_mastery_plan.py`：根据 mastery 分配目标难度、数量、场景和知识点。
- `kb_pipeline/post_mastery_generate.py`：按计划生成候选题。
- `kb_pipeline/validation.py`：盲解、审计、修复、重生成、replan。
- `kb_pipeline/step_refine.py`：只改写 steps，支持恢复历史 raw 输出和分轮日志。
- `kb_pipeline/export.py`：导出最终训练 JSONL。

## 7. 已知问题

- vLLM 日志中可能出现大量 flashinfer/GDN warmup warning；若最终有 `Application startup complete` 和 `/v1/models 200 OK`，通常不算启动失败。
- 如果 `FLASHINFER_DISABLE_JIT=1` 且缺少 JIT cache，可能刷 `MissingJITCacheError`。不同机器配置不同，不能直接照搬。
- 远程服务器 `/root/brjverl/data_gradual_new/config/pipeline.env` 是远程专属配置，不应同步成 GitHub 里的 `/jizhicfs` 版本。
- GitHub 中的 `config/pipeline.env` 曾按另一台 `/jizhicfs/...` 机器调整，不能直接覆盖远程 `/root/...` 服务器。
- `pipeline.env` 曾出现异常 `nSTAGE...` 污染行；若出现要删除。
- 代理环境会导致本地 `127.0.0.1:8911/v1/models` 被 Squid 劫持；代码已绕过代理，但 shell/curl 仍需注意 `NO_PROXY`。
- validate 盲解请求量大，`QC_BLIND_VOTES=2` 时耗时明显。
- refine 若验收规则过严会反复重试；当前已放宽合理表达并支持从失败 raw 中恢复。

## 8. 下一步要做什么

- 在远程继续跑实验前，先确认：

```bash
cd /root/brjverl/data_gradual_new
bash run/start_vllm.sh --dry-run
```

- 全流程运行：

```bash
bash run/run_full_pipeline.sh gsm8k
```

- 若只想继续后半段：

```bash
bash run/06_validate_generated.sh gsm8k
bash run/07_refine_solution_steps.sh gsm8k
bash run/08_export_training_data.sh gsm8k
```

- 如果 validate/refine 中断，优先看：

```text
outputs/pipeline/<dataset>/validation.failed.jsonl
outputs/pipeline/<dataset>/validation.rounds/
outputs/pipeline/<dataset>/refine.failed.jsonl
outputs/pipeline/<dataset>/refine.rounds/
```

- 若修改 `.py`、`.sh`、README、example 配置，记得同步 GitHub。
- 若只修改远程 `config/pipeline.env`，通常不要推 Git，除非用户明确要求更新仓库中的那版配置。

## 9. 不能改动 / 需要注意的约束

- 不要把 token、密码、SSH 信息写入仓库或交接文档。
- 不要覆盖远程真实 `config/pipeline.env`，除非明确是在远程机器上按其环境修复。
- 不要把远程 `/root/...` 配置和 GitHub `/jizhicfs/...` 配置混用。
- `07_refine_solution_steps` 只能修改 `steps` 字段，不能修改：
  - `question`
  - `answer`
  - `difficulty`
  - `source_task_id`
  - `plan_id`
  - 数学解法路径
- generate 阶段不要重新加入重度全局相似度过滤；多样性应主要由 plan/replan 控制。
- validate 阶段不要把表达风格问题硬判失败，避免 pass 率和速度大幅下降。
- `run/07_export_training_data.sh` 是兼容 wrapper，不要删除。
- 同步远程时避免 `scp -r run remote/run` 造成 `run/run/` 嵌套；应同步目录内容到目标目录。

## 10. 项目整体运行逻辑和方案描述

完整阶段：

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

阶段职责：

- `01_build_kb`：格式化原始题目，构建 KB、实体、场景、模板和知识信息。
- `02_answer_seed`：用 victim model 回答种子题。
- `03_score_seed`：用 step/QC 模型评价步骤质量和正确性，计算 mastery。
- `04_build_synthesis_plan`：按 mastery 分配目标数量、难度、知识点、场景和数值策略。
- `05_generate_questions`：按 plan 生成 `question / steps / answer` 候选。
- `06_validate_generated`：预检、盲解、审计、修复、重生成、必要时 replan。
- `07_refine_solution_steps`：在数学已验证基础上，只把 steps 改写成更适合训练的逻辑推理链。
- `08_export_training_data`：导出最终 SFT 训练数据。

关键数据流：

```text
generated.jsonl
  -> validated.jsonl
  -> refined.jsonl
  -> train.jsonl
```

vLLM 逻辑：

- `STAGE_VLLM_MODE=managed`：阶段脚本自动启动/切换 vLLM。
- 同模型跨阶段会复用，不再重复关闭和启动。
- 不同模型阶段会切换。
- `STAGE_VLLM_MODE=external`：用户自己提前启动 vLLM，脚本只探测服务。

