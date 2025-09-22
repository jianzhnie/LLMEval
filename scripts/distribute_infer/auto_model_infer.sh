#!/bin/bash
set -euo pipefail


# =======================================================
#                  脚本配置
# =======================================================

# SSH 选项（更快更稳：跳过指纹交互、设置超时、连接复用）
readonly SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPersist=60s"
# 如需自定义 ssh 用户：export SSH_USER=root；默认使用当前用户
readonly SSH_USER="${SSH_USER:-}"

# === 模型与资源参数 ===
readonly MODEL_PATH="/home/jianzhnie/llmtuner/hfhub/mindspeed/models/mindspore/hf_sft_packing_0703_step6476"
readonly NUM_GPUS=8
readonly MEMORY_UTILIZATION=0.9
readonly MAX_MODEL_LEN=65536
readonly N_SAMPLES=8            # 每个样本重复采样次数
readonly SERVED_MODEL_NAME="PCL-Reasoner"
# Ascend 设备可见性（根据 NUM_GPUS 自动生成 0..N-1）
readonly ASCEND_VISIBLE=$(seq -s, 0 $((NUM_GPUS-1)))

# === 项目与脚本路径（确保远程节点与本机一致）===
readonly PROJECT_DIR="/home/jianzhnie/llmtuner/llm/LLMEval"
readonly INFER_SCRIPT="${PROJECT_DIR}/llmeval/vllm/online_server.py"
readonly SET_ENV_SCRIPT="${PROJECT_DIR}/set_env.sh"

# === 输出与日志目录（在远程节点上创建与写入）===
readonly OUTPUT_ROOT="/home/jianzhnie/llmtuner/llm/LLMEval/output"
readonly OUTPUT_DIR="${OUTPUT_ROOT}/$SERVED_MODEL_NAME"
readonly LOG_DIR="${OUTPUT_ROOT}/logs-rl"
mkdir -p ${OUTPUT_DIR}
# 仅清理本地残留目录（避免干扰本地状态文件）；远程清理会在后续专门执行
rm -rf ${LOG_DIR}
mkdir -p ${LOG_DIR}
readonly API_SERVER_LOG_PREFIX="api_server_"
readonly TASK_LOG_PREFIX="task_"
readonly MAX_WAIT_TIME=900

# === 数据集相关（数据已在远程各节点同步，自动发现文件）===
readonly DATASET_DIR="${PROJECT_DIR}/data_process/model_infer"  # 远程节点上的数据目录
# 自动发现的数据文件通配（可通过环境变量覆盖），支持 part1 和 part_001 等两种命名
readonly DATASET_GLOB="${DATASET_GLOB:-top_100K_final_verified_samples_shard*}"
# 并发度限制（节点任务提交时本地 wait 的节流）
readonly MAX_JOBS=8

# rsync 选项（如有需要可复用；当前逻辑不再主动下发数据）
readonly RSYNC_OPTS="-avz --checksum --partial --inplace --no-whole-file --exclude='.*'"

# 可选：推理客户端额外参数（按需启用）
readonly SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-empty}"
readonly MAX_WORKERS="${MAX_WORKERS:-8}"

# =======================================================
#                  工具函数
# =======================================================

# 打印帮助信息并退出
usage() {
    echo "Usage: $0 [NODE_LIST_FILE]"
    echo
    echo "跨多节点自动部署 vLLM 并分布式推理。"
    echo
    echo "Arguments:"
    echo "  NODE_LIST_FILE    包含节点 IP 或主机名的文件路径 (默认: ./node_list_all.txt)"
    echo "                    文件内每行一个节点，支持 # 注释与空行。"
    exit 1
}

# 统一的 SSH 执行封装（自动拼接 user@host）
ssh_run() {
    local node="$1"
    shift
    local userhost="${SSH_USER:+${SSH_USER}@}${node}"
    ssh ${SSH_OPTS} "${userhost}" "$@"
}

# 将本地文件通过 rsync 同步到远程（当前未使用，保留以备扩展）
rsync_to_node() {
    local src_path="$1"   # 本地绝对或相对路径
    local node="$2"
    local dst_path="$3"   # 远程绝对路径
    local userhost="${SSH_USER:+${SSH_USER}@}${node}"
    rsync ${RSYNC_OPTS} "${src_path}" "${userhost}:${dst_path}"
}

# =======================================================
#                  核心函数区
# =======================================================

# 检查节点与端口列表是否对齐
check_node_port_alignment() {
  if [[ ${#NODES[@]} -ne ${#PORTS[@]} ]]; then
        echo "❌ 错误: NODES (${#NODES[@]}) 与 PORTS (${#PORTS[@]}) 数量不一致。" >&2
        exit 1
  fi
    echo "✅ 节点和端口配置检查通过。"
}

# 在第一个节点上发现数据集文件（结果用于所有节点任务分配）
discover_remote_dataset_files() {
    if [[ ${#NODES[@]} -eq 0 ]]; then
        echo "❌ 错误: 无节点可用于发现数据文件。" >&2
        exit 1
    fi
    local head_node="${NODES[0]}"
    echo "🔎 正在 ${head_node} 上发现数据文件：${DATASET_DIR}/${DATASET_GLOB}"
    # 仅列名（basename），并进行自然数值排序（支持 part_2 < part_10 的正确顺序）
    local out
    if ! out=$(ssh_run "$head_node" "sh -lc 'ls -1 ${DATASET_DIR}/${DATASET_GLOB} 2>/dev/null | xargs -n1 basename | LC_ALL=C sort -V'"); then
        echo "❌ 错误: 无法在 ${head_node} 上列出数据文件，请检查路径与权限。" >&2
        exit 1
    fi
    mapfile -t FILES < <(printf "%s\n" "$out" || true)
    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "❌ 错误: 未发现任何匹配的数据文件（glob: ${DATASET_GLOB}）。" >&2
        exit 1
    fi
    echo "✅ 发现数据集文件数量：${#FILES[@]}"
}

# 检查并创建远程输出与数据目录（并清理远程日志目录，确保本次运行干净）
check_and_prepare_remote_dirs() {
    echo "⚙️ 正在检查并创建远程目录，并清理旧日志..."
    for node in "${NODES[@]}"; do
        ssh_run "$node" "mkdir -p '${OUTPUT_DIR}' '${DATASET_DIR}' && rm -rf '${LOG_DIR}' && mkdir -p '${LOG_DIR}'" || {
            echo "❌ 错误：无法在节点 ${node} 上准备目录或验证路径。请检查SSH连接和权限。" >&2
            exit 1
        }
    done
    echo "✅ 所有远程目录已就绪，日志已清理。"
}


# 停止所有部署在远程节点的模型服务
stop_services() {
    echo "🛑 脚本退出，正在尝试停止所有远程模型服务..."
    local search_pattern="vllm.entrypoints.openai.api_server"
    for node in "${NODES[@]}"; do
        echo "   -> 正在停止 ${node} 上的 vLLM 进程..."
        ssh_run "$node" "pkill -f '${search_pattern}' || true" &
    done
    wait || true
    echo "✅ 所有远程模型服务停止完成。"
}

# 部署单个 vLLM 模型服务到某节点
deploy_model_service() {
    local node=$1
    local port=$2
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}.log"

    echo "🚀 在节点 ${node} 上部署模型，端口 ${port}. (TP=${NUM_GPUS}, mem_util=${MEMORY_UTILIZATION})"

    ssh_run "$node" "cd '${PROJECT_DIR}' && \
        source '${SET_ENV_SCRIPT}' && \
        export ASCEND_RT_VISIBLE_DEVICES='${ASCEND_VISIBLE}' && \
        nohup python -m vllm.entrypoints.openai.api_server \
            --model '${MODEL_PATH}' \
            --trust-remote-code \
            --served-model-name '${SERVED_MODEL_NAME}' \
            --tensor-parallel-size ${NUM_GPUS} \
            --gpu-memory-utilization ${MEMORY_UTILIZATION} \
            --max-model-len ${MAX_MODEL_LEN} \
            --enforce-eager \
            --port ${port} > '${log_file}' 2>&1 &" &
}

# 轮询检查所有模型服务是否启动成功（通过远程日志关键字）
wait_for_services() {
    echo "⏳ 正在等待所有模型服务启动并就绪... 最长等待 ${MAX_WAIT_TIME} 秒。"
    local total_wait_time=0
    local interval=10
    local total_services=${#NODES[@]}
    local status_dir="${LOG_DIR}/status"

    rm -rf "${status_dir}" || true
    mkdir -p "${status_dir}"
    while [ "$total_wait_time" -lt "$MAX_WAIT_TIME" ]; do
        local running_pids=()

        for ((i = 0; i < total_services; i++)); do
            local node=${NODES[i]}
            local port=${PORTS[i]}
            local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}.log"
            local status_file="${status_dir}/status_${node//./_}.ok"

            if [ -f "$status_file" ]; then
                continue
            fi

            (
                if ssh_run "$node" "grep -q 'Application startup complete.' '${log_file}'"; then
                    echo "✅ 服务 ${node}:${port} 已就绪 (日志确认)."
                    touch "$status_file"
                fi
            ) &
            running_pids+=($!)
        done

        if [ ${#running_pids[@]} -gt 0 ]; then
            wait "${running_pids[@]}"
        fi

        local ready_count
        ready_count=$(ls -1 "${status_dir}" 2>/dev/null | wc -l | tr -d ' ')

        if [ "$ready_count" -eq "$total_services" ]; then
            echo "✅ All ${total_services} services ready."
            return 0
        fi

        echo "    -> ${ready_count}/${total_services} ready. Waiting..."
        sleep "$interval"
        total_wait_time=$((total_wait_time + interval))
    done

    echo "❌ Timeout: services not ready in ${MAX_WAIT_TIME}s. Check remote logs."
    exit 1
}

# 将数据(文件列表)按轮询分配到 N 个实例（N=节点数）
assign_data_to_instances() {
    local total_instances="$1"
    for ((i = 0; i < total_instances; i++)); do
        eval "INSTANCE_ASSIGNMENTS_$i=()"
    done

    for idx in "${!FILES[@]}"; do
        local file=${FILES[idx]}
        local instance_idx=$((idx % total_instances))
        eval "INSTANCE_ASSIGNMENTS_${instance_idx}+=(\"\$file\")"
        echo "Assign ${file} -> instance ${instance_idx}"
    done
}

# 在某节点上批量提交推理任务（每个文件一个后台任务）
run_task_batch() {
    local node=$1
    local model_name=$2
    local base_url=$3
    shift 3
    local files=("$@")

    echo "👉 Starting tasks on node: $node with model $model_name"

    for file in "${files[@]}"; do
        local input_file="${DATASET_DIR}/${file}"
        local base_name=$(basename "$file" .jsonl)
        local output_file="${OUTPUT_DIR}/infer_${model_name//\//_}_${base_name}_bz${N_SAMPLES}.jsonl"
        local log_file="${LOG_DIR}/${TASK_LOG_PREFIX}${node//./_}_${base_name}.log"

        echo "👉 Running inference on ${file} using ${model_name} at ${base_url}"

        ssh_run "$node" "cd '${PROJECT_DIR}' && \
        source '${SET_ENV_SCRIPT}' && \
            nohup python '${INFER_SCRIPT}' \
            --input_file '${input_file}' \
            --output_file '${output_file}' \
            --input_key "question" \
            --base_url '${base_url}' \
            --model_name '${model_name}' \
            --n_samples ${N_SAMPLES} \
            --system_prompt_type '${SYSTEM_PROMPT_TYPE}' \
            --max_workers ${MAX_WORKERS} > '${log_file}' 2>&1 &" &
    done
}

# 根据节点数量分发并提交任务
distribute_and_launch_jobs() {
    local total_instances=${#NODES[@]}
    assign_data_to_instances "$total_instances"

    local in_flight=0
    for ((i = 0; i < total_instances; i++)); do
        local node=${NODES[i]}
        local port=${PORTS[i]}
        local base_url="http://127.0.0.1:${port}/v1"
        local model_name="${SERVED_MODEL_NAME}"

        IFS=$'\n' read -r -d '' -a ASSIGNED < <(eval "printf '%s\0' \"\${INSTANCE_ASSIGNMENTS_${i}[@]}\"")
        # 若该节点本轮未分配到任何文件，则跳过
        if [[ ${#ASSIGNED[@]} -eq 0 ]]; then
            continue
        fi

        run_task_batch "$node" "$model_name" "$base_url" "${ASSIGNED[@]:-}"

        in_flight=$((in_flight + 1))
        if [ $in_flight -ge $MAX_JOBS ]; then
            wait
            in_flight=0
        fi
    done

    wait
    echo "✅ All inference jobs launched."
}

# =======================================================
#                  主流程
# =======================================================
main() {

    # ========== 读取节点列表并配置端口 ==========
    if [ "$#" -gt 1 ]; then
        echo "❌ 错误: 参数过多。"
        usage
    fi

    local NODE_LIST_FILE="${1:-"./node_list_all.txt"}"

    if [ ! -f "$NODE_LIST_FILE" ]; then
        echo "❌ 错误: 节点列表文件 '${NODE_LIST_FILE}' 不存在！" >&2
        usage
    fi

    echo "✅ 从文件 '${NODE_LIST_FILE}' 加载节点列表。"
    mapfile -t NODES < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

    if [ ${#NODES[@]} -eq 0 ]; then
        echo "❌ 错误: 节点列表 '${NODE_LIST_FILE}' 为空。"
        exit 1
    fi

    PORTS=()
    local start_port=6000
    for ((i=0; i<${#NODES[@]}; i++)); do
        PORTS+=($((start_port + i * 10)))
    done
    echo "✅ 自动生成端口列表：${PORTS[@]}"

    # 设置清理陷阱，确保退出时关闭所有远程 vLLM
    trap stop_services EXIT

    # 自动发现数据集文件（在第一个节点上扫描）
    discover_remote_dataset_files

    # 步骤1: 检查节点与端口对齐
    check_node_port_alignment

    # 步骤2: 创建远程目录并清理日志
    # check_and_prepare_remote_dirs

    # 步骤4: 在所有节点上并行部署模型服务
    echo "🔄 正在并行部署所有模型服务..."
    for ((i = 0; i < ${#NODES[@]}; i++)); do
        local node=${NODES[i]}
        local port=${PORTS[i]}
        deploy_model_service "$node" "$port"
    done

    # 步骤5: 轮询检查所有服务是否就绪
    wait_for_services

    # 步骤6: 分配数据集并并行启动推理任务
    distribute_and_launch_jobs
}

# 运行主函数
main "$@"
