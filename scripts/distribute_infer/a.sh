#!/bin/bash
# =======================================================
# 分布式 vLLM 模型推理部署脚本（高并发优化版）
# =======================================================
#
# 功能描述：
#   1. 跨多节点自动部署 vLLM 模型服务（单节点多卡张量并行）
#   2. 基于健康检查与端口探活的稳健式启动与监控
#   3. 动态批处理与并行度参数优化，支持高并发推理
#   4. 数据文件轮询分配与任务并行执行
#   5. 优雅清理（退出信号捕获）与失败回滚
#
# 使用方法：
#   ./auto_model_infer.sh [NODE_LIST_FILE]
#
# 作者：LLM Eval Team
# 版本：3.0
# 更新日期：2025
# =======================================================

set -euo pipefail

# =======================================================
#                  全局配置区域
# =======================================================

# SSH 连接配置
# 优化选项：跳过主机密钥检查、设置连接超时、启用连接复用
readonly SSH_OPTS="-o StrictHostKeyChecking=no \
                   -o UserKnownHostsFile=/dev/null \
                   -o ConnectTimeout=5 \
                   -o ServerAliveInterval=30 \
                   -o ServerAliveCountMax=3 \
                   -o ControlMaster=auto \
                   -o ControlPersist=60s"

# SSH 用户配置（可通过环境变量覆盖）
readonly SSH_USER="${SSH_USER:-$(whoami)}"
# =======================================================
#                  模型与资源配置
# =======================================================

# 模型路径配置
readonly MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/mindspeed/models/mindspore/hf_sft_packing_0703_step6476}"

# GPU/ASCEND 资源配置
readonly NUM_GPUS=${NUM_GPUS:-8}                     # 张量并行大小（单实例用光整机多卡）
readonly MEMORY_UTILIZATION=${MEMORY_UTILIZATION:-0.9}
readonly MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

# vLLM 高并发关键参数（按需调整；需结合显存与上下文长度）
# - MAX_NUM_SEQS: 同时并发处理的序列数（越大越能吞吐，受显存影响较大）
# - MAX_NUM_BATCHED_TOKENS: 动态批次内总 token 上限（控制显存与吞吐权衡）
# 注：两者不宜同时设过大，推荐根据模型大小按 1-2 次试跑观测 GPU 利用率后调整
readonly MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
readonly MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}

# 其他推理参数
readonly N_SAMPLES=${N_SAMPLES:-8}                   # 每条样本的重复采样次数
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PCL-Reasoner}"

# Ascend 设备可见性（如使用 GPU 可忽略；若为 CUDA 可替换为 CUDA_VISIBLE_DEVICES）
readonly ASCEND_VISIBLE="$(seq -s, 0 $((NUM_GPUS-1)))"

# =======================================================
#                  vLLM API Server 运行参数
# =======================================================

# 关闭请求逐条日志，减少 IO 抖动
readonly DISABLE_LOG_REQUESTS=${DISABLE_LOG_REQUESTS:-1}

# 禁用 OpenAI 兼容层的请求体保存（如版本支持）
readonly DISABLE_STATE_DUMP=${DISABLE_STATE_DUMP:-1}

# Uvicorn/Server 设置（注意：vLLM 引擎内并行为主，过多服务进程可能适得其反）
# 如果 vLLM 支持 --num-servers 或 --workers，可以在此开启；默认 1
readonly API_WORKERS=${API_WORKERS:-1}

# 额外引擎参数（按需追加，例如 "--dtype bfloat16 --enforce-eager"）
readonly EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:-}"

# =======================================================
#                  路径配置
# =======================================================

# 项目路径配置
readonly PROJECT_DIR="${PROJECT_DIR:-/home/jianzhnie/llmtuner/llm/LLMEval}"
readonly INFER_SCRIPT="${INFER_SCRIPT:-${PROJECT_DIR}/llmeval/vllm/online_server.py}"
readonly SET_ENV_SCRIPT="${SET_ENV_SCRIPT:-${PROJECT_DIR}/set_env.sh}"

# 输出与日志路径配置
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jianzhnie/llmtuner/llm/LLMEval/output}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${SERVED_MODEL_NAME}}"
readonly LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs-rl}"

# 日志文件前缀配置
readonly API_SERVER_LOG_PREFIX="api_server_"
readonly TASK_LOG_PREFIX="task_"

# 服务等待最大时长（秒）
readonly MAX_WAIT_TIME=${MAX_WAIT_TIME:-900}

# 健康检查设置
readonly HEALTH_PATH="${HEALTH_PATH:-/health}"        # OpenAI 兼容服务通常暴露 /health
readonly HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-3}          # 单次健康检查超时（秒）

# =======================================================
#                  数据集配置
# =======================================================

# 数据集路径配置（假定各节点路径一致或挂同一 NAS）
readonly DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data_process/model_infer}"

# 数据集文件匹配模式（可覆盖）
readonly DATASET_GLOB="${DATASET_GLOB:-top_100K_final_verified_samples_shard*}"

# 并发控制配置
readonly MAX_JOBS=${MAX_JOBS:-128}                    # 总体一次性拉起的最大任务数量（进程数）

# =======================================================
#                  推理客户端参数
# =======================================================

readonly SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-amthinking}"
readonly MAX_WORKERS=${MAX_WORKERS:-32}               # 客户端每进程内部的线程/协程并发

# =======================================================
#                  全局变量声明
# =======================================================

# 节点和端口数组（在 main 函数中初始化）
declare -a NODES
declare -a PORTS
declare -a FILES

# =======================================================
#                  工具函数区域
# =======================================================

# 打印使用帮助信息
# 参数：无
# 返回值：无（直接退出）
usage() {
    cat << EOF
用法: $0 [NODE_LIST_FILE]

跨多节点自动部署 vLLM 并执行分布式推理任务（高并发优化版）。

参数:
  NODE_LIST_FILE         节点列表文件 (默认: ./node_list_all.txt)；每行一个节点，支持 # 注释与空行

可用环境变量（可覆盖默认值）:
  SSH_USER               远程 SSH 用户名（默认：当前用户）
  MODEL_PATH             模型文件路径
  NUM_GPUS               GPU/ASCEND 数量（默认：8）
  MEMORY_UTILIZATION     显存利用率（默认：0.9）
  MAX_MODEL_LEN          最大上下文长度（默认：65536）
  MAX_NUM_SEQS           vLLM 动态批并发序列数（默认：1024）
  MAX_NUM_BATCHED_TOKENS vLLM 动态批 token 上限（默认：32768）
  N_SAMPLES              每个样本采样次数（默认：8）
  SERVED_MODEL_NAME      服务模型名称（默认：PCL-Reasoner）
  MAX_WAIT_TIME          服务启动最大等待时间（默认：900秒）
  DATASET_GLOB           数据集文件匹配模式
  SYSTEM_PROMPT_TYPE     系统提示类型（默认：amthinking）
  MAX_WORKERS            推理客户端内部并发（默认：32）
  DISABLE_LOG_REQUESTS   是否关闭请求日志（默认：1）
  API_WORKERS            API 进程数（如版本支持；默认：1）
  EXTRA_ENGINE_ARGS      附加引擎参数字符串（默认：空）

示例:
  $0
  SSH_USER=root NUM_GPUS=4 MAX_NUM_SEQS=2048 $0 ./nodes.txt
EOF
    exit 1
}

# 统一的 SSH 执行封装
# 参数：
#   $1: 节点地址
#   $@: 要执行的命令
# 返回值：SSH 命令的退出码
ssh_run() {
    local node="$1"
    shift
    local userhost="${SSH_USER:+${SSH_USER}@}${node}"
    ssh ${SSH_OPTS} "${userhost}" "$@"
}

# 通过 rsync 同步文件到远程节点
# 参数：
#   $1: 本地源路径
#   $2: 目标节点
#   $3: 远程目标路径
# 返回值：rsync 命令的退出码
rsync_to_node() {
    local src_path="$1"
    local node="$2"
    local dst_path="$3"
    local userhost="${SSH_USER:+${SSH_USER}@}${node}"
    local RSYNC_OPTS="-avz --checksum --partial --inplace --no-whole-file --exclude='.*'"

    if ! rsync ${RSYNC_OPTS} "${src_path}" "${userhost}:${dst_path}"; then
        echo "❌ rsync 同步失败: ${src_path} -> ${userhost}:${dst_path}" >&2
        return 1
    fi
}

# 验证配置参数
# 参数：无
# 返回值：无（验证失败时退出）
validate_config() {
    echo "正在验证配置参数..."

    # 验证必要的路径是否存在
    if [[ ! -f "${INFER_SCRIPT}" ]]; then
        echo "❌ 错误: 推理脚本不存在: ${INFER_SCRIPT}" >&2
        exit 1
    fi

    if [[ ! -f "${SET_ENV_SCRIPT}" ]]; then
        echo "❌ 错误: 环境设置脚本不存在: ${SET_ENV_SCRIPT}" >&2
        exit 1
    fi

    if [[ ${NUM_GPUS} -lt 1 || ${NUM_GPUS} -gt 8 ]]; then
        echo "❌ 错误: NUM_GPUS 需在 1-8 之间: ${NUM_GPUS}" >&2
        exit 1
    fi

    if [[ $(echo "${MEMORY_UTILIZATION} < 0.1 || ${MEMORY_UTILIZATION} > 1.0" | bc -l) -eq 1 ]]; then
        echo "❌ 错误: 内存利用率需在 0.1-1.0 之间: ${MEMORY_UTILIZATION}" >&2
        exit 1
    fi

    if [[ ${N_SAMPLES} -lt 1 || ${N_SAMPLES} -gt 100 ]]; then
        echo "❌ 错误: 采样次数需在 1-100 之间: ${N_SAMPLES}" >&2
        exit 1
    fi

    if [[ ${MAX_NUM_SEQS} -lt 1 || ${MAX_NUM_SEQS} -gt 16384 ]]; then
        echo "❌ 错误: MAX_NUM_SEQS 范围异常: ${MAX_NUM_SEQS}" >&2
        exit 1
    fi
    if [[ ${MAX_NUM_BATCHED_TOKENS} -lt 512 || ${MAX_NUM_BATCHED_TOKENS} -gt 1048576 ]]; then
        echo "❌ 错误: MAX_NUM_BATCHED_TOKENS 范围异常: ${MAX_NUM_BATCHED_TOKENS}" >&2
        exit 1
    fi

    echo "✅ 配置参数验证通过"
}

# =======================================================
#                  核心功能函数区域
# =======================================================

# 检查节点与端口列表数量是否一致
# 参数：无
# 返回值：无（检查失败时退出）
check_node_port_alignment() {
    if [[ ${#NODES[@]} -ne ${#PORTS[@]} ]]; then
        echo "❌ 错误: 节点数量 (${#NODES[@]}) 与端口数量 (${#PORTS[@]}) 不一致" >&2
        exit 1
    fi
    echo "✅ 节点和端口配置检查通过"
}

# 端口探活（远程是否可用）
check_remote_port_free() {
    local node="$1"
    local port="$2"
    local used
    used=$(ssh_run "$node" "ss -ltn '( sport = :$port )' 2>/dev/null | tail -n +2 | wc -l" 2>/dev/null || echo 0)
    if [[ "${used:-0}" -eq 0 ]]; then
        used=$(ssh_run "$node" "netstat -ltn 2>/dev/null | awk '{print \$4}' | grep -E '[:.]${port}\$' | wc -l" 2>/dev/null || echo 0)
    fi
    if [[ "${used:-0}" -eq 0 ]]; then
        used=$(ssh_run "$node" "lsof -iTCP:${port} -sTCP:LISTEN -nP 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    fi
    if [[ "${used:-0}" -gt 0 ]]; then
        echo "⚠️  节点 ${node} 端口 ${port} 已被占用，尝试清理旧 vLLM 进程..."
        ssh_run "$node" "pkill -f 'vllm.entrypoints.openai.api_server.*--port ${port}' || true" >/dev/null 2>&1 || true
        sleep 1
    fi
}

# 在第一个节点上发现数据集文件
# 参数：无
# 返回值：无（发现失败时退出）
discover_remote_dataset_files() {
    if [[ ${#NODES[@]} -eq 0 ]]; then
        echo "❌ 错误: 无可用节点进行数据文件发现" >&2
        exit 1
    fi

    local head_node="${NODES[0]}"
    echo "🔎 正在节点 ${head_node} 上发现数据文件: ${DATASET_DIR}/${DATASET_GLOB}"

    # 执行文件发现命令，支持自然数值排序
    local out
    if ! out=$(ssh_run "$head_node" "sh -lc 'ls -1 ${DATASET_DIR}/${DATASET_GLOB} 2>/dev/null | xargs -n1 basename | LC_ALL=C sort -V'"); then
        echo "❌ 错误: 无法在节点 ${head_node} 上列出数据文件，请检查路径与权限" >&2
        exit 1
    fi

    # 将结果存储到全局数组
    mapfile -t FILES < <(printf "%s\n" "$out" || true)

    if [[ ${#FILES[@]} -eq 0 ]]; then
        echo "❌ 错误: 未发现任何匹配的数据文件 (模式: ${DATASET_GLOB})" >&2
        exit 1
    fi

    echo "✅ 发现数据集文件数量: ${#FILES[@]}"
    echo "文件列表: ${FILES[*]}"
}

# 检查并创建远程目录，清理旧日志
# 参数：无
# 返回值：无（操作失败时退出）
check_and_prepare_remote_dirs() {
    echo "⚙️ 正在检查并创建远程目录，清理旧日志..."

    for node in "${NODES[@]}"; do
        echo "   -> 处理节点: ${node}"
        if ! ssh_run "$node" "mkdir -p '${OUTPUT_DIR}' '${DATASET_DIR}' '${LOG_DIR}' && rm -rf '${LOG_DIR}/status' && mkdir -p '${LOG_DIR}/status' && rm -f '${LOG_DIR}/${API_SERVER_LOG_PREFIX}'*.log '${LOG_DIR}/${TASK_LOG_PREFIX}'*.log 2>/dev/null || true"; then
            echo "❌ 错误: 无法在节点 ${node} 上准备目录，请检查SSH连接和权限" >&2
            exit 1
        fi
    done

    echo "✅ 所有远程目录已就绪，旧日志已清理"
}

# 停止所有远程节点上的模型服务
# 参数：无
# 返回值：无
stop_services() {
    echo "🛑 脚本退出，正在停止所有远程模型服务..."

    local search_pattern="vllm.entrypoints.openai.api_server"
    local stop_pids=()

    for node in "${NODES[@]}"; do
        echo "   -> 正在停止节点 ${node} 上的 vLLM 进程..."
        (
            ssh_run "$node" "pkill -f '${search_pattern}' || true"
            echo "   ✅ 节点 ${node} 服务已停止"
        ) &
        stop_pids+=($!)
    done

    # 等待所有停止操作完成
    wait "${stop_pids[@]}" || true
    echo "✅ 所有远程模型服务停止完成"
}

# 在指定节点部署 vLLM 模型服务
# 参数：
#   $1: 节点地址
#   $2: 服务端口
# 返回值：无
deploy_model_service() {
    local node="$1"
    local port="$2"
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}.log"

    echo "🚀 在节点 ${node} 上部署模型服务，端口 ${port} (TP=${NUM_GPUS}, mem_util=${MEMORY_UTILIZATION})"

    # 构建 vLLM 启动命令
    # 关键参数：
    #   --max-num-seqs              并发序列数上限
    #   --max-num-batched-tokens    动态批内 token 上限
    #   --disable-log-requests      关闭请求日志（减小 I/O）
    #   --tensor-parallel-size      使用多卡并行
    #   --gpu-memory-utilization    控制显存水位（避免 OOM）
    #   --max-model-len             控制上下文长度
    # 提示：如需开启混合精度/强制 eager，可在 EXTRA_ENGINE_ARGS 中追加
    local vllm_cmd="cd '${PROJECT_DIR}' && \
        source '${SET_ENV_SCRIPT}' && \
        export ASCEND_RT_VISIBLE_DEVICES='${ASCEND_VISIBLE}' && \
        nohup python -m vllm.entrypoints.openai.api_server \
            --model '${MODEL_PATH}' \
            --trust-remote-code \
            --served-model-name '${SERVED_MODEL_NAME}' \
            --tensor-parallel-size ${NUM_GPUS} \
            --gpu-memory-utilization ${MEMORY_UTILIZATION} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-num-seqs ${MAX_NUM_SEQS} \
            --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
            --port ${port} \
            > '${log_file}' 2>&1 &"

    # 端口探活与服务启动
    check_remote_port_free "$node" "$port"
    # 在后台启动服务
    ssh_run "$node" "$vllm_cmd" &
}

# 健康检查（HTTP 探活 + 日志回退）
check_service_ready() {
    local node="$1"
    local port="$2"
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}.log"
    local base_url="http://127.0.0.1:${port}"

    # 先尝试 HTTP 健康检查
    if ssh_run "$node" "curl -s --max-time ${HEALTH_TIMEOUT} ${base_url}${HEALTH_PATH} 2>/dev/null | grep -qi 'ok\|healthy\|ready'"; then
        echo "✅ 服务 ${node}:${port} 健康检查通过"
        return 0
    fi
    # 兼容部分版本：尝试 /v1/models
    if ssh_run "$node" "curl -s --max-time ${HEALTH_TIMEOUT} ${base_url}/v1/models 2>/dev/null | grep -qi '${SERVED_MODEL_NAME}\|data'"; then
        echo "✅ 服务 ${node}:${port} /v1/models 检查通过"
        return 0
    fi

    # 回退到日志关键字
    if ssh_run "$node" "grep -q 'Application startup complete' '${log_file}'"; then
        echo "✅ 服务 ${node}:${port} 日志检查通过"
        return 0
    fi

    return 1
}

# 轮询检查所有模型服务是否启动成功
# 参数：无
# 返回值：无（超时或失败时退出）
wait_for_services() {
    echo "⏳ 正在等待所有模型服务启动并就绪... 最长等待 ${MAX_WAIT_TIME} 秒"

    local total_wait_time=0
    local interval=5
    local total_services=${#NODES[@]}
    local status_dir="${LOG_DIR}/status"

    # 清理并创建状态目录
    rm -rf "${status_dir}" || true
    mkdir -p "${status_dir}"

    while [[ $total_wait_time -lt $MAX_WAIT_TIME ]]; do
        local running_pids=()

        # 并行检查所有服务状态
        for ((i = 0; i < total_services; i++)); do
            local node="${NODES[i]}"
            local port="${PORTS[i]}"
            local status_file="${status_dir}/status_${node//./_}.ok"

            # 跳过已就绪的服务
            if [[ -f "$status_file" ]]; then
                continue
            fi

            # 后台检查服务状态
            (
                if check_service_ready "$node" "$port"; then
                    touch "$status_file"
                fi
            ) &
            running_pids+=($!)
        done

        # 等待所有检查完成
        if [[ ${#running_pids[@]} -gt 0 ]]; then
            wait "${running_pids[@]}" || true
        fi

        # 统计就绪服务数量
        local ready_count
        ready_count=$(ls -1 "${status_dir}" 2>/dev/null | wc -l | tr -d ' ')

        if [[ $ready_count -eq $total_services ]]; then
            echo "✅ 所有 ${total_services} 个服务已就绪"
            return 0
        fi

        echo "   -> ${ready_count}/${total_services} 服务就绪，继续等待..."
        sleep "$interval"
        total_wait_time=$((total_wait_time + interval))
    done

    echo "❌ 超时: 服务在 ${MAX_WAIT_TIME} 秒内未完全就绪，请检查远程日志" >&2
    exit 1
}

# 将数据文件按轮询方式分配到各个实例
# 参数：
#   $1: 总实例数量
# 返回值：无
assign_data_to_instances() {
    local total_instances="$1"

    echo "📊 正在分配全部 ${#FILES[@]} 个数据文件到 ${total_instances} 个实例..."

    # 初始化实例分配数组
    for ((i = 0; i < total_instances; i++)); do
        eval "INSTANCE_ASSIGNMENTS_$i=()"
    done

    # 轮询分配文件
    for idx in "${!FILES[@]}"; do
        local file="${FILES[idx]}"
        local instance_idx=$((idx % total_instances))
        eval "INSTANCE_ASSIGNMENTS_${instance_idx}+=(\"\$file\")"
        echo "   分配文件: ${file} -> 实例 ${instance_idx}"
    done

    for ((i = 0; i < total_instances; i++)); do
        eval "local count=\${#INSTANCE_ASSIGNMENTS_${i}[@]}"
        echo "   -> 实例 ${i} 分配 ${count} 个文件"
    done

    echo "✅ 数据文件分配完成"
}

# 在指定节点上批量提交推理任务
# 参数：
#   $1: 节点地址
#   $2: 模型名称
#   $3: 基础URL
#   $@: 文件列表
# 返回值：无
run_task_batch() {
    local node="$1"
    local model_name="$2"
    local base_url="$3"
    shift 3
    local files=("$@")

    echo "👉 在节点 ${node} 上启动推理任务，模型: ${model_name}"

    local tasks_started=0
    for file in "${files[@]}"; do
        local input_file="${DATASET_DIR}/${file}"
        local base_name
        base_name=$(basename "$file" .jsonl)
        local output_file="${OUTPUT_DIR}/infer_${model_name//\//_}_${base_name}_bz${N_SAMPLES}.jsonl"
        local log_file="${LOG_DIR}/${TASK_LOG_PREFIX}${node//./_}_${base_name}.log"

        echo "   -> 处理文件: ${file} (输出: ${output_file})"

        # 构建推理客户端命令（注意：在线客户端脚本内部应支持异步/批并发与重试）
        local infer_cmd="cd '${PROJECT_DIR}' && \
            source '${SET_ENV_SCRIPT}' && \
            nohup python '${INFER_SCRIPT}' \
                --input_file '${input_file}' \
                --output_file '${output_file}' \
                --base_url '${base_url}' \
                --model_name '${model_name}' \
                --n_samples ${N_SAMPLES} \
                --max_workers ${MAX_WORKERS} > '${log_file}' 2>&1 &"

        # 在后台启动推理任务
        ssh_run "$node" "$infer_cmd" &
        tasks_started=$((tasks_started + 1))

        # 简单的全局节流，避免一次性拉起过多任务导致瞬时拥塞
        # 如需更精细的节流策略，可替换为远程 semaphore 或基于队列的派发
        if [[ $tasks_started -ge $MAX_JOBS ]]; then
            wait
            tasks_started=0
        fi
    done

    wait || true
}

# 分发并启动所有推理任务
# 参数：无
# 返回值：无
distribute_and_launch_jobs() {
    local total_instances=${#NODES[@]}

    echo "🚀 开始分发并启动推理任务..."

    # 分配数据文件
    assign_data_to_instances "$total_instances"

    # 为每个节点启动对应的推理任务（并行）
    local pids=()
    local submitted=0
    for ((i = 0; i < total_instances; i++)); do
        local node="${NODES[i]}"
        local port="${PORTS[i]}"
        local base_url="http://127.0.0.1:${port}/v1"
        local model_name="${SERVED_MODEL_NAME}"

        # 获取分配给当前实例的文件列表
        IFS=$'\n' read -r -d '' -a ASSIGNED < <(eval "printf '%s\0' \"\${INSTANCE_ASSIGNMENTS_${i}[@]}\"")

        # 跳过没有分配文件的节点
        if [[ ${#ASSIGNED[@]} -eq 0 ]]; then
            echo "   -> 节点 ${node} 未分配到文件，跳过"
            continue
        fi

        echo "   -> 节点 ${node} 分配到 ${#ASSIGNED[@]} 个文件"

        # 并行提交每个节点的任务批次（本地后台，远端内部再并行）
        (
            run_task_batch "$node" "$model_name" "$base_url" "${ASSIGNED[@]:-}"
        ) &
        pids+=($!)
        submitted=$((submitted + 1))

        # 简单本地节流：限制同时存在的提交批次数量，避免本地进程过多
        if [[ $submitted -ge $MAX_JOBS ]]; then
            wait "${pids[@]}" || true
            pids=()
            submitted=0
        fi
    done

    # 等待所有节点的任务提交完成（不等待远端具体推理完成）
    if [[ ${#pids[@]} -gt 0 ]]; then
        wait "${pids[@]}" || true
    fi

    echo "✅ 所有推理任务已启动"
}

# 监控远端推理任务直至完成（基于进程存活）
# 返回值：无（阻塞直到所有节点上不再存在 INFER_SCRIPT 进程）
wait_for_remote_jobs() {
    echo "⏳ 等待所有远端推理任务完成..."
    local interval=10

    while true; do
        local running_total=0
        local pids=()
        for node in "${NODES[@]}"; do
            (
                # 统计匹配推理客户端脚本的存活进程数
                # 用 basename 兼容符号链接/不同路径
                cnt=$(ssh_run "$node" "pgrep -fal 'python .*${INFER_SCRIPT##*/}' | wc -l" 2>/dev/null || echo 0)
                echo "${node}:${cnt}"
            ) &
            pids+=($!)
        done
        wait "${pids[@]}" || true

        # 汇总
        while read -r line; do
            [[ -z "$line" ]] && continue
            c=${line##*:}
            running_total=$((running_total + c))
        done < <(
            for node in "${NODES[@]}"; do
                # 再次获取，避免 subshell输出竞争；轻微重复成本可接受
                ssh_run "$node" "pgrep -fal 'python .*${INFER_SCRIPT##*/}' | wc -l" 2>/dev/null || echo 0
            done
        )

        if [[ ${running_total} -eq 0 ]]; then
            echo "✅ 所有远端推理任务已完成"
            break
        fi
        echo "   -> 仍有 ${running_total} 个远端推理进程在运行，${interval}s 后重试..."
        sleep "${interval}"
    done
}



# =======================================================
#                  主程序入口
# =======================================================

# 主函数：协调整个部署和推理流程
# 参数：
#   $@: 命令行参数
# 返回值：无
main() {
    echo "🎯 开始执行分布式 vLLM 模型推理部署"
    echo "================================================"

    # 参数解析
    if [[ $# -gt 1 ]]; then
        echo "❌ 错误: 参数过多" >&2
        usage
    fi

    local NODE_LIST_FILE="${1:-./node_list_all.txt}"

    # 验证节点列表文件
    if [[ ! -f "$NODE_LIST_FILE" ]]; then
        echo "❌ 错误: 节点列表文件 '${NODE_LIST_FILE}' 不存在" >&2
        usage
    fi

    echo "✅ 从文件 '${NODE_LIST_FILE}' 加载节点列表"

    # 读取节点列表（过滤空行和注释）
    mapfile -t NODES < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

    if [[ ${#NODES[@]} -eq 0 ]]; then
        echo "❌ 错误: 节点列表 '${NODE_LIST_FILE}' 为空" >&2
        exit 1
    fi

    echo "📋 发现 ${#NODES[@]} 个节点: ${NODES[*]}"

    # 自动生成端口列表（节点间避免冲突，间隔 10 端口）
    PORTS=()
    local start_port=6000
    for ((i=0; i<${#NODES[@]}; i++)); do
        PORTS+=($((start_port + i * 10)))
    done
    echo "✅ 自动生成端口列表: ${PORTS[*]}"

    # 验证配置参数
    validate_config

    # 设置退出时的清理陷阱
    trap stop_services EXIT

    # 执行主要流程
    echo "🔄 开始执行部署流程..."

    # 步骤1: 发现数据集文件
    discover_remote_dataset_files

    # 步骤2: 检查节点与端口配置
    check_node_port_alignment

    # 步骤3: 准备远程目录
    check_and_prepare_remote_dirs

    # 步骤4: 并行部署模型服务
    echo "🚀 正在并行部署所有模型服务..."
    for ((i = 0; i < ${#NODES[@]}; i++)); do
        local node="${NODES[i]}"
        local port="${PORTS[i]}"
        deploy_model_service "$node" "$port"
    done

    # 步骤5: 等待服务就绪（HTTP 健康检查 + 日志回退）
    wait_for_services

    # 步骤6: 分发并启动推理任务（远端并行 + 本地节流）
    distribute_and_launch_jobs

    # 步骤7: 等待远端推理任务完成后再关闭服务
    wait_for_remote_jobs
    stop_services

    echo "🎉 分布式推理部署完成！"
    echo "📊 部署统计:"
    echo "   - 节点数量: ${#NODES[@]}"
    echo "   - 数据文件: ${#FILES[@]}"
    echo "   - 服务端口: ${PORTS[*]}"
    echo "   - 输出目录: ${OUTPUT_DIR}"
    echo "   - 日志目录: ${LOG_DIR}"
    echo "================================================"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
