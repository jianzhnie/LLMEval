#!/bin/bash
# Run distributed inference with two TP=4 instances per node by default.

set -euo pipefail

export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export INSTANCES_PER_NODE="${INSTANCES_PER_NODE:-2}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/auto_model_infer_common.sh"

main "$@"
