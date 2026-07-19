# data_gradual_new 项目交接文档

## 1. 项目目标

构建一套面向数学推理能力提升的“渐进式合成数据”流水线。项目以 GSM8K 等种子题为基础，先评估被测模型在原始题上的作答和步骤掌握情况，再根据薄弱知识点、目标难度和场景规划合成新题，经过后置校验、修复、步骤表达优化，最终导出可用于 SFT 的训练数据。

核心目标不是单纯扩大数据量，而是得到正确、可解、唯一答案、步骤清晰且适合训练的数学推理样本，用于提升被测模型性能。

## 2. 当前目录结构

```text
data_gradual_new/
├── config/
│   ├── pipeline.env                  # 当前机器真实配置，机器私有
│   ├── pipeline.example.env          # 示例配置
│   └── vllm.p2p-enabled.example.env
├── data/
│   └── *.jsonl                       # 输入数据，如 gsm8k/gsm8k_train
├── kb_pipeline/
│   ├── assessment.py                 # answer seed / score seed / mastery
│   ├── post_mastery_plan.py          # 合成计划
│   ├── post_mastery_generate.py      # 按 plan 生成候选题
│   ├── validation.py                 # validate/repair/regenerate/replan
│   ├── step_refine.py                # 只优化 steps 表达
│   ├── export.py                     # 导出训练 JSONL
│   ├── client.py                     # OpenAI-compatible vLLM 客户端
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
│   ├── start_vllm.sh
│   └── stop_vllm.sh
├── outputs/
├── README.md
├── README.zh.md
├── run/README_stages.md
└── PROJECT_HANDOFF.zh.md
```

## 3. 当前已经完成的功能

- 支持全流程运行：`bash run/run_full_pipeline.sh <dataset>`。
- 支持分阶段运行：`01` 到 `08` 每个阶段都有独立 bash 脚本。
- 支持 `managed` / `external` 两种 vLLM 模式。
- `managed` 模式可自动启动、切换、复用和退出清理 vLLM。
- 同一模型跨阶段会复用 vLLM，避免重复 stop/start。
- vLLM readiness probe 已绕过 HTTP 代理，避免本地 `127.0.0.1` 被 Squid 劫持。
- `start_vllm.sh` / `stop_vllm.sh` 通过 `bash xxx.sh` 调用，不依赖执行权限。
- `start_vllm.sh` 支持 `--dry-run`，可检查最终启动命令。
- 支持 answer/generate/validate/refine/export 的 checkpoint、failed 文件和中断恢复。
- `02_answer_seed` 中 steps 是被测模型同一次 JSON 输出中的步骤，不是后续重新生成。
- `03_score_seed` 评分模型只评价 answer 阶段解析出的步骤。
- `04_build_synthesis_plan` 负责知识点、场景、难度、数量、多样性和分布规划。
- `05_generate_questions` 负责按 plan 生成可解析候选题，不承担重度正确性验证。
- `06_validate_generated` 负责 precheck、blind solve、strict audit、repair、regenerate、replan。
- validate 已修复本地 precheck 对符号方程的误判，例如不再把 `x + 2x + 0 + 1 = 7` 错切为 `0 + 1 = 7`。
- validate 已增加请求错误防卡死逻辑，`blind_request_error` 不再长期消耗 tiebreak。
- validate 已增加 `QC_REPLAN_AFTER_ARITHMETIC_ERRORS`，重复算术错误会更快回退重生成题目。
- validate 会拦截步骤中的自我修稿污染，例如 `Let's re-read...`、`adjust the numbers...`、`rewrite the question...`。
- `07_refine_solution_steps` 只改写 `steps`，不改题目、答案、难度和 ID。
- refine 阶段有多轮日志 `refine.rounds/`，并可从 raw/failed 中恢复。
- `08_export_training_data` 导出 legacy SFT 格式，最终答案统一为 `The answer is $\boxed{XXX}$.`。

## 4. 正在开发完善的功能

- validate 阶段的恢复机制仍需加强：当前能保存每轮日志和失败队列，但中断后不一定能精确从上一轮 pending 队列继续。
- 合成数据训练收益仍在优化，目标是超过旧版合成数据。
- 继续控制题面/步骤风格，减少过长题面、机械步骤、工程化场景、模板化表达。
- refine 步骤质量继续优化，目标是让步骤体现“由题意得出中间量，再承接下一步”的逻辑链。
- 合成规模和混合比例需要继续做消融。当前经验是不要盲目追求 5x 以上数据量，优先看质量和混合权重。

## 5. 关键技术栈

- Python 3.12
- Bash stage scripts
- vLLM 0.19.0
- OpenAI-compatible Chat Completions API
- 本地 Qwen / Llama 模型
- JSONL 数据流
- GitHub API / GitHub 仓库同步
- 远程服务器 SSH/scp 同步

## 6. 重要文件说明

- `config/pipeline.env`：真实运行配置，强机器相关，不要随意跨机器覆盖。
- `config/pipeline.example.env`：示例配置，可以随代码更新。
- `run/stage_common.sh`：路径初始化、阶段参数、vLLM 探测、模型匹配、同模型复用。
- `run/start_vllm.sh`：vLLM 启动入口，处理 Python 环境、NCCL、API key、日志、pid/model 记录。
- `run/stop_vllm.sh`：清理 vLLM 主进程和 worker，处理 Ctrl+C 后残留。
- `run/run_stage_sequence.sh`：全流程阶段编排。
- `kb_pipeline/assessment.py`：被测模型作答、步骤评分、mastery 记录构建。
- `kb_pipeline/prompts.py`：answer/score 等提示词。
- `kb_pipeline/post_mastery_plan.py`：根据 mastery 构建合成计划。
- `kb_pipeline/post_mastery_generate.py`：按计划生成题目、步骤、答案。
- `kb_pipeline/validation.py`：后置校验、修复、重生成、replan，是当前最关键的质量控制文件。
- `kb_pipeline/step_refine.py`：步骤表达优化，只能改 steps。
- `kb_pipeline/export.py`：导出最终训练数据。
- `outputs/pipeline/<dataset>/generated.jsonl`：生成候选。
- `outputs/pipeline/<dataset>/validated.jsonl`：校验通过样本。
- `outputs/pipeline/<dataset>/refined.jsonl`：步骤优化后样本。
- `outputs/pipeline/<dataset>/train.jsonl`：最终 SFT 数据。

## 7. 已知问题

- validate 中断恢复还不完整：如果全流程重跑，可能会重新验证一部分已跑内容。需要进一步实现“从 `validation.failed.jsonl` 的 `next_candidate` 和已有 `validated.jsonl` 精确续跑”。
- 不同机器 vLLM 启动配置差异很大。`/root/...` 远程服务器和 `/jizhicfs/...` 机器不能共用同一份 `pipeline.env`。
- `/jizhicfs/...` 机器上 Qwen3.6 曾确认最稳手动命令是 2 卡、TP=2、`--enforce-eager`、只禁用 IB、unset 其他 NCCL/FlashInfer 变量。
- 如果 `FLASHINFER_DISABLE_JIT=1` 泄漏到 vLLM，可能出现大量 `MissingJITCacheError` 和 500。
- `VLLM_API_KEY=` 为空时，vLLM 可无鉴权启动，但 Python OpenAI SDK 仍需要非空占位 key；代码已在客户端侧 fallback 为 `EMPTY`。
- 如果 `VLLM_API_KEY=EMPTY`，vLLM 会要求请求带 Authorization；如果要完全贴近无鉴权手动命令，应显式写 `VLLM_API_KEY=` 并使用最新 `start_vllm.sh`。
- validate 的 `blind_solve` 请求慢时，日志会出现 heartbeat 长时间 0/2；若最终 error 多，要看 `validation.failed.jsonl` 中 `blind_request_meta`。
- 合成规模不是越大越好。7473 原始样本下，5x 约 3.7w 合成样本，可能压过原始分布。建议优先试 3x 或 4x。

## 8. 下一步要做什么

- 优先补 validate 的真正 resume：
  - 启动时读取已有 `validated.jsonl` 作为 accepted。
  - 若 `validation.failed.jsonl` 非空，优先用其中 `next_candidate` 作为 pending。
  - 只有 `STAGE_FORCE=1` 才从 `generated.jsonl` 全量重验。
- 继续跑数据规模消融：
  - `origin only`
  - `origin + synthetic 1x`
  - `origin + synthetic 2x`
  - `origin + synthetic 3x`
  - `origin + synthetic 4x/5x`
- 当前更稳的合成分配建议：
  - `SYNTHESIS_TARGET_MULTIPLIER=3`
  - `SYNTHESIS_MIN_PER_SEED=0`
  - `SYNTHESIS_MAX_PER_SEED=20`
- 若坚持 5x，建议：
  - `SYNTHESIS_TARGET_MULTIPLIER=5`
  - `SYNTHESIS_MIN_PER_SEED=0`
  - `SYNTHESIS_MAX_PER_SEED=25`
- 在任何机器运行前先执行：

```bash
bash run/start_vllm.sh --dry-run
```

- 远程 `/root/brjverl/data_gradual_new` 已同步最近代码修复，但远程 `config/pipeline.env` 未覆盖。

## 9. 不能改动 / 需要注意的约束

- 不要把 GitHub token、SSH 密码、服务器密码写入仓库或文档。
- 不要随意覆盖远程服务器的 `config/pipeline.env`。
- 不要把 `/root/...` 远程服务器配置和 `/jizhicfs/...` 机器配置混用。
- `07_refine_solution_steps` 只能修改 `steps`，不能修改：
  - `question`
  - `answer`
  - `difficulty`
  - `source_task_id`
  - `plan_id`
  - 数学解法路径
- generate 阶段不要重新加入重度全局相似度过滤；多样性应主要由 plan/replan 控制。
- validate 阶段不要把普通风格问题默认硬失败；风格问题主要作为 warning，避免 pass 率过低。
- 如果步骤中出现模型自我修稿、试错、改题过程，必须拦截，不能进入训练数据。
- 保留 `run/07_export_training_data.sh` 兼容 wrapper。
- 远程同步时不要用会造成嵌套的 scp 方式，例如避免把 `run` 复制成 `run/run`。

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

阶段说明：

- `01_build_kb`：读取原始数据，抽取题目、答案、场景、实体、知识点。
- `02_answer_seed`：被测模型回答原始题，并直接输出 `steps` 和 `final_answer`。
- `03_score_seed`：评分模型按被测模型原始 steps 打分，构建 mastery。
- `04_build_synthesis_plan`：根据 mastery 和分布策略规划每个 seed 的合成目标。
- `05_generate_questions`：根据 plan 生成候选题，不做重度数学校验。
- `06_validate_generated`：对候选做 deterministic precheck、blind solve、strict audit、repair、regenerate、replan。
- `07_refine_solution_steps`：在数学已经通过的基础上，把 steps 改成更适合训练的逻辑表达。
- `08_export_training_data`：拼接步骤并导出训练格式，答案使用 `The answer is $\boxed{XXX}$.`。

核心数据流：

```text
generated.jsonl
  -> validated.jsonl
  -> refined.jsonl
  -> train.jsonl
```

vLLM 模式：

- `STAGE_VLLM_MODE=managed`：阶段脚本自动启动/切换/复用 vLLM。
- `STAGE_VLLM_MODE=external`：用户自己启动 vLLM，脚本只探测服务。
- 如果连续阶段使用同一模型，managed 模式会复用当前 vLLM。
- 如果模型不同，managed 模式会切换服务。

常用命令：

```bash
bash run/run_full_pipeline.sh gsm8k
bash run/06_validate_generated.sh gsm8k
bash run/07_refine_solution_steps.sh gsm8k
bash run/08_export_training_data.sh gsm8k
```
