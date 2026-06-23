# data_gradual_new

这是独立于 `/root/brjverl/data_gradual` 的渐进式数学数据合成项目。掌握度计算及合成数量、难度分配部分保持已验收逻辑；后续计划、生成、验证和修正均在新目录中实现。

## 当前完整流程

1. 格式化原始问题集并构建知识库。
2. 被测模型只看到问题内容，对每道种子题回答 `N` 次，默认 `N=10`。
3. 保存被测模型输出的推理步骤和纯数值答案。
4. 与标准答案比对，计算正确率。
5. 对被测模型已经划分好的步骤做五维评分。
6. 根据正确率和步骤评分计算每道题的掌握度。
7. 根据掌握度分配每道种子题的合成数量和五级相对难度。
8. 根据知识库和多样性策略生成合成计划。
9. 异步并发生成问题、步骤和答案；失败项按批次轮转。
10. 对生成题做程序预检。
11. 验证模型对题目进行两次独立盲解；意见不一致时追加第三票。
12. 审计候选答案、步骤、可解性、唯一性和相对难度。
13. 根据错误类型定向修正，并在下一轮重新盲解和审计。
14. 输出全部通过验证的简洁数据。

训练数据格式转换仍未接入，等待验证和修正效果验收后再实现。

## 环境与模型

主程序：

```bash
conda activate brj
```

vLLM：

```bash
conda activate qwen
```

默认模型：

```text
被测模型：/root/brjverl/models/Meta-Llama-3-8B-Instruct
步骤评分模型：/root/brjverl/models/Qwen3.6-27B
题目生成模型：/root/brjverl/models/Qwen3.6-27B
验证和修正模型：/root/brjverl/models/Qwen3.6-27B
```

主程序运行在 `brj` 环境，vLLM 服务由脚本在 `qwen` 环境中启动。完整流程会根据模型路径判断是否需要关闭并切换 vLLM。

## 主要配置

配置文件：

```text
/root/brjverl/data_gradual_new/config/pipeline.env
```

### 生成配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `GEN_CONCURRENCY` | `256` | 合成请求并发数 |
| `GEN_CONCURRENCY_CAP` | `256` | 合成并发硬上限 |
| `GEN_MAX_RETRIES` | `3` | 失败批次最大重试轮数；`-1` 表示无限轮转 |
| `GEN_MAX_TOKENS` | `640` | 合成最大输出 token |
| `GEN_ENABLE_THINKING` | `0` | 关闭 Qwen thinking 模式 |
| `GEN_FORCE_JSON` | `0` | 是否使用 JSON response format |
| `GEN_SIMILARITY_THRESHOLD` | `0.88` | 生成题表面相似度阈值 |
| `GEN_ROUND_RETRY_DELAY` | `1` | 失败批次之间等待秒数 |

不同难度的采样参数：

```bash
GEN_TEMPERATURE_MAP={"Easy":0.3,"Slightly Easy":0.4,"Equal":0.5,"Slightly Hard":0.6,"Hard":0.7}
GEN_TOP_P_MAP={"Easy":0.3,"Slightly Easy":0.4,"Equal":0.5,"Slightly Hard":0.6,"Hard":0.7}
```

### 验证与修正配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `RUN_VALIDATION` | `1` | 完整流程是否自动执行验证和修正 |
| `QC_MODEL` | `Qwen3.6-27B` | 盲解、审计和修正模型 |
| `QC_CONCURRENCY` | `256` | 验证模型请求并发数 |
| `QC_BLIND_VOTES` | `2` | 每题初始独立盲解票数 |
| `QC_TIEBREAK_VOTES` | `1` | 初始答案不一致时追加票数 |
| `QC_MAX_ROUNDS` | `3` | 最大修正重验轮数；`-1` 表示无限轮转 |
| `QC_MAX_TOKENS` | `900` | 盲解、审计和修正最大输出 token |
| `QC_ENABLE_THINKING` | `0` | 关闭验证模型 thinking 模式 |
| `QC_FORCE_JSON` | `0` | 是否强制 JSON response format |
| `QC_ROUND_RETRY_DELAY` | `1` | 验证轮次之间等待秒数 |

`QC_MAX_ROUNDS=3` 表示初始验证轮加最多 3 个修正重验轮。

## 一键运行

```bash
cd /root/brjverl/data_gradual_new
bash run/run_full_pipeline.sh gsm8k
```

默认会执行到：

```text
outputs/pipeline/gsm8k/validated.jsonl
```

如果本次只想运行到生成阶段：

```bash
bash run/run_full_pipeline.sh gsm8k --skip-validation
```

## 分步执行后半段

生成计划：

```bash
bash run/run_build_synthesis_plan.sh gsm8k
```

生成题目：

```bash
bash run/run_generate_questions.sh gsm8k
```

验证和修正：

```bash
bash run/run_validate_generated.sh gsm8k
```

分步运行时需要确保 Qwen vLLM 已在配置端口运行。

## 合成计划

文件：

```text
outputs/planning/<dataset>/synthesis_plan.jsonl
```

每条记录只有三个顶层字段：

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

- `math`：目标数学能力和相对难度。
- `diversity`：主场景、备用场景、结构变化、叙事方式和数值策略。
- `kb_inspiration`：知识库提供的可选灵感，不是必须照搬的模板。

允许相同数学模板放到不同场景中；不允许机械复制知识库原题措辞、数字和实体。

## 生成输出

标准文件：

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

生成失败按批次轮转。每个 plan 在一轮中只请求一次，失败项写入 `generated.failed.jsonl`，下一轮只处理失败项。相似度失败会重新规划场景；网络、JSON 和字段错误复用原 plan。

过程文件：

```text
generated.raw.jsonl
generated.failed.jsonl
generated.summary.json
generated.rounds/
```

## 验证方案

### 1. 程序预检

不调用模型，检查：

- 必要字段和纯数值答案。
- 多子问题。
- 重复步骤。
- 可解析算式的计算正确性。
- 最后步骤数值与候选答案是否明显冲突。

### 2. 2+1 独立盲解

验证模型只看到合成问题，不会看到候选步骤和答案。

- 两份盲解答案一致时形成初步共识。
- 两份答案不一致或无效时追加第三票。
- 验证可解性、答案唯一性和数值答案。

### 3. 步骤与难度审计

审计模型看到：

- 合成题目、候选步骤和候选答案。
- 独立盲解共识和代表性解题步骤。
- 种子题及其标准步骤，仅用于判断相对难度。
- 目标五级难度。

审计内容：

- 问题是否完整、可解且答案唯一。
- 候选答案是否正确。
- 每个步骤是否正确。
- 难度是否相对种子题达到预设等级。

### 4. 定向修正

- `repair_solution`：问题完全不变，只重新生成步骤和答案。
- `repair_question`：只修补必要条件、歧义或难度，随后重新求解。
- `regenerate_question`：严重结构错误时根据 plan 重新生成整道题。
- 验证请求异常：不修改题目，下一轮重新验证。

修正结果不会直接通过，必须进入下一轮重新盲解和审计。如果连续修正得到相同问题和答案，下一次强制重新生成整道题。

## 验证输出

通过验证的标准文件：

```text
outputs/pipeline/<dataset>/validated.jsonl
```

格式仍然只有六个字段：

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

详细过程单独保存：

```text
validation_reports.jsonl
validation.failed.jsonl
repair_history.jsonl
validated.summary.json
validation.rounds/
```

## 环境配置与机器迁移

启动脚本会自动读取 `config/pipeline.env`，无需提前手动执行
`conda activate`。主流程环境和 vLLM 环境可以分别配置：

```bash
CONDA_SH=/root/miniconda3/etc/profile.d/conda.sh
PIPELINE_CONDA_ENV=brj
PIPELINE_PYTHON=
VLLM_CONDA_ENV=qwen
VLLM_PYTHON=
```

- `PIPELINE_CONDA_ENV`：主流程使用的 Conda 环境名。
- `VLLM_CONDA_ENV`：vLLM 服务使用的 Conda 环境名。
- `CONDA_SH`：新机器上的 Conda 初始化脚本路径；留空时自动检测。
- `PIPELINE_PYTHON`：可选的主流程 Python 绝对路径，设置后跳过 Conda 激活。
- `VLLM_PYTHON`：可选的 vLLM Python 绝对路径，设置后跳过 Conda 激活。

例如迁移到另一台机器：

```bash
CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
PIPELINE_CONDA_ENV=math_pipeline
VLLM_CONDA_ENV=vllm_runtime
```

如果不想依赖环境名，也可以直接指定解释器：

```bash
PIPELINE_CONDA_ENV=
PIPELINE_PYTHON=/opt/miniconda3/envs/math_pipeline/bin/python
VLLM_CONDA_ENV=
VLLM_PYTHON=/opt/miniconda3/envs/vllm_runtime/bin/python
```

模型、输入和输出路径也需要修改为新机器上的实际路径。

## 双卡 vLLM/NCCL 配置

当前部署服务器的默认配置保持不变：

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

配置值支持三种形式：

- 普通值，如 `1`、`lo`：启动前执行 `export`。
- `unset`：启动前明确删除该环境变量。
- `inherit`：保留启动脚本父进程中的原值。

如果另一台机器原生 GPU P2P 可以工作，且强制
`NCCL_P2P_DISABLE=1` 会卡住，可将
`config/vllm.p2p-enabled.example.env` 中的配置合并到
`pipeline.env`。该模板会启用 eager，并取消强制 P2P、loopback
网卡和 blocking-wait 配置。

验证配置但不停止或启动 vLLM：

```bash
bash run/start_vllm.sh --dry-run
```

默认使用流水线自动管理模式：

```bash
VLLM_RUNTIME_MODE=managed
VLLM_BASE_URL=http://127.0.0.1:8911/v1
VLLM_API_PORT=8911
VLLM_START_TIMEOUT=600
VLLM_START_POLL_SEC=5
```

在 `managed` 模式下，流水线自动启动被测模型，回答完成后关闭该 vLLM，
再在同一个 `8911` 端口启动评测/生成模型。所有客户端请求始终发送到
`VLLM_BASE_URL`，不会为不同阶段配置多个外部 API 端口。

所有数据集共用 `outputs/runtime/vllm/` 中的 PID、模型标记和 vLLM 日志，
避免切换数据集名称后把同一个自动启动服务误判为外部进程。请勿同时启动两条
完整流水线争用同一个端口。

日志也可以在配置文件中统一设置：

```bash
VLLM_LOG_FILE=/root/brjverl/data_gradual_new/outputs/runtime/vllm.log
VLLM_FOREGROUND_LOG=1
VLLM_LOG_APPEND=0
```

- `VLLM_FOREGROUND_LOG=1`：终端实时显示日志，同时写入文件。
- `VLLM_LOG_APPEND=0`：每次启动覆盖旧日志；设为 `1` 时追加。

在部署服务器启动被测模型：

```bash
bash run/start_vllm.sh \
  --model /root/brjverl/models/Meta-Llama-3-8B-Instruct
```

切换到评测/生成模型：

```bash
bash run/start_vllm.sh \
  --model /root/brjverl/models/Qwen3.6-27B
```

这两个命令都会读取 `config/pipeline.env` 中的 Python 环境、GPU、NCCL、
端口和日志设置，并在前台运行；按 `Ctrl+C` 停止。

如果需要改为手动启动 vLLM，可选用：

```bash
VLLM_RUNTIME_MODE=external
VLLM_EXTERNAL_WAIT_TIMEOUT=-1
```

`managed` 是默认模式。vLLM 0.8.x 中不要导出 `VLLM_PORT`，因为该变量也会
被 vLLM 用于内部 Worker 通信。

- `validation_reports.jsonl`：每轮预检、盲解、审计和最终决策。
- `validation.failed.jsonl`：当前最终未通过的题目。
- `repair_history.jsonl`：每次修正前后内容、错误原因和原始响应。
- `validation.rounds/`：每轮报告、修正、失败项和统计。

## 暂未接入

训练格式转换和实际训练仍未接入完整流程。旧 `quality.py`、`noise.py` 等文件只保留用于历史对照，新验证流程使用 `kb_pipeline/validation.py`。
