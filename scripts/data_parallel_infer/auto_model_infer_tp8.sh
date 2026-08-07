#!/bin/bash
# Run distributed inference with one TP=8 instance per node by default.

set -euo pipefail

export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-1}"

cpu_offload_gb="${CPU_OFFLOAD_GB:-0}"
swap_space="${SWAP_SPACE:-0}"
vllm_dtype="${VLLM_DTYPE:-bfloat16}"
export EXTRA_ENGINE_ARGS="--cpu-offload-gb ${cpu_offload_gb} --swap-space ${swap_space} --dtype ${vllm_dtype} ${EXTRA_ENGINE_ARGS:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/auto_model_infer_common.sh"

main "$@"
