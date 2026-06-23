source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_BLOCKING_WAIT=1
unset VLLM_PORT

python -m vllm.entrypoints.openai.api_server \
  --model /root/brjverl/models/Meta-Llama-3-8B-Instruct \
  --port 8911 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes