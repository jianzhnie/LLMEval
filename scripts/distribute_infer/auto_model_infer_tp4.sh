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
# 核心特性：
#   - 自动发现与分配数据文件
#   - 多层次并行（节点间并行 + 节点内多卡并行 + 单卡动态批处理）
#   - 健康检查机制（HTTP探活 + 日志检查）
#   - 进程级任务监控
#   - 失败节点自动跳过
#   - 资源限制与任务节流
#
# 执行流程：
#   1. 参数校验与环境初始化
#   2. 读取节点列表并生成端口配置
#   3. 发现数据集文件并进行分配
#   4. 并行部署 vLLM 服务实例
#   5. 等待服务就绪（健康检查）
#   6. 分发并启动推理任务
#   7. 监控任务执行直至完成
#   8. 优雅关闭服务并清理资源
#
# 可配置项：
#   - GPU/NPU 资源配置（卡数、显存比例等）
#   - 推理批处理参数（并发序列数、批次大小等）
#   - 网络超时与重试设置
#   - 并发度与节流控制
#   - 日志与输出路径
#
# 使用建议：
#   1. 根据硬件配置调整资源参数
#   2. 结合数据规模设置并发度
#   3. 配置合适的超时与重试策略
#   4. 规划好日志与输出管理
#
# 配置建议：
#   1. NUM_GPUS: 根据实际显卡数量设置
#   2. MAX_NUM_SEQS: 结合显存大小调整
#   3. MAX_JOBS: 依据系统资源调整并发数
#   4. HEALTH_TIMEOUT: 根据网络情况调整检查超时
#
# 使用方法：
#   ./auto_model_infer.sh [NODE_LIST_FILE]
#
# 环境要求：
#   - bash 4.0+
#   - ssh 免密配置
#   - python 3.9+
#   - vLLM
#   - CUDA/NPU 驱动
#
# 作者：LLM Eval Team
# 版本：3.0
# 更新日期：2025
# =======================================================

set -euo pipefail

# =======================================================
#                  调试模式配置
# =======================================================
# 启用调试模式（设置 DEBUG=1 开启）
if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x  # 打印执行的每条命令
    export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'  # 增强调试输出
fi

# =======================================================
#                  全局配置区域
# =======================================================

# SSH 连接配置
# ----------------------
# SSH 优化选项配置:
# - StrictHostKeyChecking=no: 关闭主机密钥检查，避免首次连接询问
# - UserKnownHostsFile=/dev/null: 不记录主机密钥，减少维护负担
# - LogLevel=ERROR: 仅记录错误日志，减少日志噪声
# - ConnectTimeout=5: 连接超时设置，快速失败
# - ServerAliveInterval=30: 每30秒发送保活包
# - ServerAliveCountMax=3: 最多允许3次保活失败
# - ControlMaster=auto: 启用连接复用，提高性能
# - ControlPersist=60s: 保持连接60秒，减少重连开销
readonly SSH_OPTS="-o StrictHostKeyChecking=no \
                   -o UserKnownHostsFile=/dev/null \
                   -o LogLevel=ERROR \
                   -o ConnectTimeout=5 \
                   -o ServerAliveInterval=30 \
                   -o ServerAliveCountMax=3 \
                   -o ControlMaster=auto \
                   -o ControlPersist=60s"

# SSH 用户配置: 优先使用环境变量，否则使用当前用户
readonly SSH_USER="${SSH_USER:-$(whoami)}"
# =======================================================
#                  模型与资源配置
# =======================================================

# 模型路径配置
readonly MODEL_PATH="${MODEL_PATH:-/home/jianzhnie/llmtuner/hfhub/mindspeed/models/mindspore/hf_sft_packing_0703_step6476}"

# GPU/ASCEND 资源配置
readonly NUM_GPUS=${NUM_GPUS:-4}                     # 张量并行大小（每实例4卡）
readonly INSTANCES_PER_NODE=${INSTANCES_PER_NODE:-2} # 每节点部署实例数（2实例）
readonly MEMORY_UTILIZATION=${MEMORY_UTILIZATION:-0.9}
readonly MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

# vLLM 高并发关键参数（按需调整；需结合显存与上下文长度）
# - MAX_NUM_SEQS: 同时并发处理的序列数（越大越能吞吐，受显存影响较大）
# - MAX_NUM_BATCHED_TOKENS: 动态批次内总 token 上限（控制显存与吞吐权衡）
# 注：两者不宜同时设过大，推荐根据模型大小按 1-2 次试跑观测 GPU 利用率后调整
# 提示：如需开启混合精度/强制 eager，可在 EXTRA_ENGINE_ARGS 中追加
readonly MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
readonly MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}

# 其他推理参数
readonly N_SAMPLES=${N_SAMPLES:-8}                   # 每条样本的重复采样次数
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PCL-Reasoner}"

# 计算每个实例的设备可见性
get_device_visibility() {
    local instance_id=$1  # 0-based
    local start_idx=$((instance_id * NUM_GPUS))
    # Check if we have enough devices available
    local end_idx=$((start_idx + NUM_GPUS - 1))

    # Just in case we're using ASCEND devices, we should check the actual available devices
    # This is a more robust approach than just assuming sequential device IDs
    seq -s, $start_idx $end_idx
}

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
readonly INPUT_KEY="${INPUT_KEY:-question}"           # 输入字段键名
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
        log_error "rsync 同步失败: ${src_path} -> ${userhost}:${dst_path}"
        return 1
    fi
}

# 日志函数
log_info() {
    local msg="$*"
    # 根据消息内容选择合适的emoji
    case "$msg" in
        *"开始执行"*|*"启动"*) local emoji="🚀 " ;;
        *"完成"*|*"成功"*|*"通过"*) local emoji="✅ " ;;
        *"发现"*|*"检查"*) local emoji="🔍 " ;;
        *"配置"*|*"设置"*) local emoji="⚙️ " ;;
        *"等待"*) local emoji="⏳ " ;;
        *"清理"*) local emoji="🧹 " ;;
        *"分配"*|*"部署"*) local emoji="📦 " ;;
        *"节点"*|*"服务"*) local emoji="💻 " ;;
        *"端口"*) local emoji="🔌 " ;;
        *"文件"*) local emoji="📄 " ;;
        *"统计"*) local emoji="📊 " ;;
        *) local emoji="ℹ️ " ;;
    esac
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: ${emoji}$msg"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: ⚠️ $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ❌ $*" >&2
}

# 错误处理函数
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg"

    # 清理资源
    stop_services

    exit "$exit_code"
}

# 文件锁管理
LOCK_FILE="/tmp/vllm_deploy.lock"

acquire_lock() {
    if [ -e "$LOCK_FILE" ]; then
        local pid
        pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            handle_error 1 "另一个部署进程 (PID: $pid) 正在运行"
        fi
        rm -f "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
}

release_lock() {
    rm -f "$LOCK_FILE"
}

# 权限检查函数
check_permissions() {
    local dir=$1
    if [[ ! -w "$dir" ]]; then
        handle_error 1 "目录 $dir 没有写入权限"
    fi
}

# 节点连通性检查
validate_node() {
    local node=$1
    if ! ssh -q "$node" exit 2>/dev/null; then
        handle_error 1 "无法连接到节点 $node"
    fi
}

# 清理函数
cleanup_and_exit() {
    local exit_code=$?

    log_info "开始清理资源..."

    # 停止所有服务
    stop_services

    # 释放文件锁
    release_lock

    # 如果是调试模式，关闭它
    [[ -n "$DEBUG" ]] && set +x

    log_info "清理完成，退出代码: $exit_code"
    exit "$exit_code"
}

# 验证配置参数
# 参数：无
# 返回值：无（验证失败时通过 handle_error 退出）
validate_config() {
    log_info "开始验证配置参数..."

    # 验证必要文件存在性
    local required_files=(
        "$INFER_SCRIPT"
        "$SET_ENV_SCRIPT"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            handle_error 1 "必需文件不存在: $file"
        fi
        if [[ ! -r "$file" ]]; then
            handle_error 1 "文件没有读取权限: $file"
        fi
    done

    # 验证目录权限
    local required_dirs=(
        "${OUTPUT_DIR}"
        "${LOG_DIR}"
        "${DATASET_DIR}"
    )

    for dir in "${required_dirs[@]}"; do
        check_permissions "$dir"
    done

    # 验证数值参数范围
    local param_checks=(
        "NUM_GPUS:1:8:GPU数量"
        "N_SAMPLES:1:100:采样次数"
        "MAX_NUM_SEQS:1:16384:并发序列数"
        "MAX_NUM_BATCHED_TOKENS:512:1048576:批处理Token数"
    )

    for check in "${param_checks[@]}"; do
        IFS=':' read -r param min max desc <<< "$check"
        local value
        value=$(eval echo "\$$param")
        if [[ $value -lt $min || $value -gt $max ]]; then
            handle_error 1 "$desc ($param) 需在 $min-$max 之间，当前值: $value"
        fi
    done

    # 验证浮点数参数
    if [[ $(echo "${MEMORY_UTILIZATION} < 0.1 || ${MEMORY_UTILIZATION} > 1.0" | bc -l) -eq 1 ]]; then
        handle_error 1 "显存利用率需在 0.1-1.0 之间，当前值: ${MEMORY_UTILIZATION}"
    fi

    log_info "✅ 配置参数验证通过"
}

# =======================================================
#                  核心功能函数区域
# =======================================================

# 检查节点与端口列表数量是否一致
# 参数：无
# 返回值：无（检查失败时退出）
check_node_port_alignment() {
    local expected_ports=$(( ${#NODES[@]} * INSTANCES_PER_NODE ))
    if [[ ${#PORTS[@]} -ne $expected_ports ]]; then
        log_error "错误: 节点数量 (${#NODES[@]}) * 实例数 (${INSTANCES_PER_NODE}) 与端口数量 (${#PORTS[@]}) 不一致"
        exit 1
    fi
    log_info "✅ 节点和端口配置检查通过"
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
        log_warn "节点 ${node} 端口 ${port} 已被占用，尝试清理旧 vLLM 进程..."
        ssh_run "$node" "pkill -f 'vllm.entrypoints.openai.api_server.*--port ${port}' || true" >/dev/null 2>&1 || true
        sleep 1
    fi
}

# 在第一个节点上发现数据集文件
# 参数：无
# 返回值：无（发现失败时退出）
discover_remote_dataset_files() {
    if [[ ${#NODES[@]} -eq 0 ]]; then
        log_error "错误: 无可用节点进行数据文件发现"
        exit 1
    fi

    local head_node="${NODES[0]}"
    log_info "正在节点 ${head_node} 上发现数据文件: ${DATASET_DIR}/${DATASET_GLOB}"

    # 执行文件发现命令，支持自然数值排序
    local out
    if ! out=$(ssh_run "$head_node" "sh -lc 'ls -1 ${DATASET_DIR}/${DATASET_GLOB} 2>/dev/null | xargs -n1 basename | LC_ALL=C sort -V'"); then
        log_error "无法在节点 ${head_node} 上列出数据文件，请检查路径与权限"
        exit 1
    fi

    # 将结果存储到全局数组
    mapfile -t FILES < <(printf "%s\n" "$out" || true)

    if [[ ${#FILES[@]} -eq 0 ]]; then
        log_error "未发现任何匹配的数据文件 (模式: ${DATASET_GLOB})"
        exit 1
    fi

    log_info "发现数据集文件数量: ${#FILES[@]}"
    log_info "文件列表: ${FILES[*]}"
}

# 检查并创建远程目录，清理旧日志
# 参数：无
# 返回值：无（操作失败时退出）
check_and_prepare_remote_dirs() {
    log_info "正在检查并创建远程目录，清理旧日志..."

    for node in "${NODES[@]}"; do
        log_info "处理节点: ${node}"
        if ! ssh_run "$node" "mkdir -p '${OUTPUT_DIR}' '${DATASET_DIR}' '${LOG_DIR}' && rm -rf '${LOG_DIR}/status' && mkdir -p '${LOG_DIR}/status' && rm -f '${LOG_DIR}/${API_SERVER_LOG_PREFIX}'*.log '${LOG_DIR}/${TASK_LOG_PREFIX}'*.log 2>/dev/null || true"; then
            log_error "无法在节点 ${node} 上准备目录，请检查SSH连接和权限"
            exit 1
        fi
    done

    log_info "✅ 所有远程目录已就绪，旧日志已清理"
}

# 停止所有远程节点上的模型服务
# 参数：无
# 返回值：无
stop_services() {
    log_info "🛑 脚本退出，正在停止所有远程模型服务..."

    local search_pattern="vllm.entrypoints.openai.api_server"
    local stop_pids=()

    for node in "${NODES[@]}"; do
        log_info "   -> 正在停止节点 ${node} 上的 vLLM 进程..."
        (
            ssh_run "$node" "pkill -f '${search_pattern}' || true"
            log_info "   ✅ 节点 ${node} 服务已停止"
        ) &
        stop_pids+=($!)
    done

    # 等待所有停止操作完成
    wait "${stop_pids[@]}" || true
    log_info "✅ 所有远程模型服务停止完成"
}

# 在指定节点部署 vLLM 模型服务
# 功能: 在远程节点上启动 vLLM 模型服务实例
# 参数:
#   $1: 节点地址 - 远程服务器的域名或IP
#   $2: 服务端口 - 服务监听的端口号
# 返回值:
#   0: 部署命令发送成功
#   1: 节点验证或命令发送失败
# 注意事项:
#   - 会自动清理已占用端口的旧进程
#   - 服务启动为异步操作，需要后续健康检查确认
#   - 日志会重定向到指定文件
#   - 使用 nohup 确保服务在 SSH 断开后继续运行
deploy_model_service() {
    local node="$1"
    local port="$2"
    local instance_id="$3"
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}_${instance_id}.log"
    local devices=$(get_device_visibility "$instance_id")

    log_info "🚀 在节点 ${node} 上部署模型服务实例 ${instance_id}，端口 ${port} (TP=${NUM_GPUS}, GPUs=${devices}, mem_util=${MEMORY_UTILIZATION})"    # 1. 节点连通性验证
    if ! validate_node "$node"; then
        log_error "节点 ${node} 连通性验证失败"
        return 1
    fi

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
        export ASCEND_RT_VISIBLE_DEVICES='${devices}' && \
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
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}_${3:-0}.log"
    local base_url="http://127.0.0.1:${port}"

    # 检查日志文件是否存在
    if ! ssh_run "$node" "[[ -f '${log_file}' ]]"; then
        log_warn "节点 ${node} 的日志文件尚未创建: ${log_file}"
        return 1
    fi

    # 检查进程是否存在
    if ! ssh_run "$node" "pgrep -f 'vllm.entrypoints.openai.api_server.*--port ${port}' > /dev/null"; then
        log_warn "节点 ${node} 上的服务进程未运行"
        return 1
    fi

    # 尝试 HTTP 健康检查
    local http_status
    http_status=$(ssh_run "$node" "curl -s -o /dev/null -w '%{http_code}' --max-time ${HEALTH_TIMEOUT} \
        ${base_url}${HEALTH_PATH} 2>/dev/null || echo 0")

    if [[ $http_status -eq 200 ]]; then
        log_info "✅ 服务 ${node}:${port} 健康检查通过"
        return 0
    else
        log_warn "节点 ${node} 健康检查接口返回状态码: ${http_status}"
    fi

    # 兼容性检查：尝试 /v1/models
    local models_status
    models_status=$(ssh_run "$node" "curl -s -o /dev/null -w '%{http_code}' --max-time ${HEALTH_TIMEOUT} \
        ${base_url}/v1/models 2>/dev/null || echo 0")

    if [[ $models_status -eq 200 ]]; then
        log_info "✅ 服务 ${node}:${port} /v1/models 检查通过"
        return 0
    else
        log_warn "节点 ${node} /v1/models 接口返回状态码: ${models_status}"
    fi

    # 检查日志关键字和错误
    if ssh_run "$node" "grep -q 'Application startup complete' '${log_file}'"; then
        # 检查最近的错误日志
        local error_logs
        error_logs=$(ssh_run "$node" "grep -i '[ERROR]\|error\|exception\|failed' '${log_file}' | tail -n 5")
        if [[ -n "$error_logs" ]]; then
            log_warn "节点 ${node} 日志中发现错误:"
            echo "$error_logs" | while read -r line; do
                log_warn "错误日志: $line"
            done
            return 1
        else
            log_info "✅ 服务 ${node}:${port} 日志检查通过"
            return 0
        fi
    else
        log_warn "节点 ${node} 服务启动未完成，日志中未找到启动完成标志"
    fi
    return 1
}

# 轮询检查所有模型服务是否启动成功
# 参数：无
# 返回值：就绪节点的索引数组
wait_for_services() {
    log_info "⏳ 等待服务启动..."
    log_info "总等待时间: ${MAX_WAIT_TIME} 秒"

    local total_wait_time=0
    local interval=10
    local total_nodes=${#NODES[@]}
    local status_dir="${LOG_DIR}/status"
    local -a ready_indices=()

    # 新增：用于跟踪完成和失败的实例
    local -A completed_instances=()  # 跟踪已完成的实例
    local -A failed_instances=()     # 跟踪已失败的实例

    # 清理并创建状态目录
    rm -rf "${status_dir}" || true
    mkdir -p "${status_dir}"

    while [[ $total_wait_time -lt $MAX_WAIT_TIME ]]; do
        local running_pids=()
        local checked_instances=0
        local total_instances=$((total_nodes * INSTANCES_PER_NODE))

        # 并行检查所有节点的所有实例状态
        for ((i = 0; i < total_nodes; i++)); do
            local node="${NODES[i]}"
            for ((j = 0; j < INSTANCES_PER_NODE; j++)); do
                local port_idx=$((i * INSTANCES_PER_NODE + j))
                local port="${PORTS[port_idx]}"
                local status_file="${status_dir}/status_${node//./_}_${j}.ok"
                local fail_file="${status_dir}/status_${node//./_}_${j}.fail"

                # 跳过已就绪的服务实例
                if [[ -f "$status_file" ]]; then
                    checked_instances=$((checked_instances + 1))
                    continue
                fi

                # 跳过已标记为失败的实例
                if [[ -f "$fail_file" ]]; then
                    checked_instances=$((checked_instances + 1))
                    continue
                fi

                # 后台检查服务状态
                (
                    if check_service_ready "$node" "$port" "$j"; then
                        touch "$status_file"
                        echo "[OK] 实例就绪: 节点 ${node} 实例 ${j} (端口 ${port})"
                    elif [[ $total_wait_time -gt $((MAX_WAIT_TIME / 2)) ]]; then
                        # 如果等待时间已过半，标记为失败
                        touch "$fail_file"
                        echo "[ERROR] 实例失败: 节点 ${node} 实例 ${j} (端口 ${port}) - 超时或启动失败"
                    fi
                ) &
                running_pids+=($!)
            done
        done

        # 等待所有检查完成
        if [[ ${#running_pids[@]} -gt 0 ]]; then
            wait "${running_pids[@]}" || true
        fi

        # 收集就绪节点索引
        ready_indices=()
        local ready_instances=0
        local failed_instance_count=0

        # 清空跟踪数组
        completed_instances=()
        failed_instances=()

        for ((i = 0; i < total_nodes; i++)); do
            local node="${NODES[i]}"
            local node_ready_count=0
            local node_failed_count=0
            local node_instance_info=()

            # 检查该节点的所有实例状态
            for ((j = 0; j < INSTANCES_PER_NODE; j++)); do
                local port_idx=$((i * INSTANCES_PER_NODE + j))
                local port="${PORTS[port_idx]}"
                local status_file="${status_dir}/status_${node//./_}_${j}.ok"
                local fail_file="${status_dir}/status_${node//./_}_${j}.fail"

                if [[ -f "$status_file" ]]; then
                    completed_instances["${node}:${j}"]="${port}"
                    node_ready_count=$((node_ready_count + 1))
                    ready_instances=$((ready_instances + 1))
                    node_instance_info+=("[OK]实例${j}(端口:${port})")
                elif [[ -f "$fail_file" ]]; then
                    failed_instances["${node}:${j}"]="${port}"
                    node_failed_count=$((node_failed_count + 1))
                    failed_instance_count=$((failed_instance_count + 1))
                    node_instance_info+=("[ERROR]实例${j}(端口:${port})")
                fi
            done

            # 如果节点的所有实例都就绪，则标记节点为就绪
            if [[ $node_ready_count -eq $INSTANCES_PER_NODE ]]; then
                ready_indices+=($i)
            fi

            # 如果有实例信息要报告，显示节点状态
            if [[ ${#node_instance_info[@]} -gt 0 ]]; then
                echo "   节点 ${node}: ${node_instance_info[*]}"
            fi
        done

        # 检查是否所有节点都就绪
        if [[ ${#ready_indices[@]} -eq $total_nodes ]]; then
            echo "所有 ${total_nodes} 个节点的 ${total_instances} 个服务实例已就绪"
            echo "${ready_indices[@]}"
            return 0
        fi

        # 显示进度
        local pending_instances=$((total_instances - ready_instances - failed_instance_count))
        echo "   -> 就绪: ${ready_instances}, 失败: ${failed_instance_count}, 等待: ${pending_instances}/${total_instances} 服务，已等待: ${total_wait_time}s"

        # 如果等待时间过长，提前退出
        local time_threshold=$((MAX_WAIT_TIME * 8 / 10))  # 80% of MAX_WAIT_TIME
        if [[ $total_wait_time -gt $time_threshold ]] && [[ $pending_instances -gt 0 ]]; then
            echo "接近最大等待时间，部分实例仍未就绪"
        fi

        sleep "$interval"
        total_wait_time=$((total_wait_time + interval))
    done

    # 超时后收集最终状态
    echo "等待超时，收集最终部署状态..."

    # 显示已完成的实例
    if [[ ${#completed_instances[@]} -gt 0 ]]; then
        echo "已完成部署的实例 (${#completed_instances[@]} 个):"
        for instance in "${!completed_instances[@]}"; do
            echo "   - ${instance} (端口: ${completed_instances[$instance]})"
        done
    fi

    # 显示失败的实例
    if [[ ${#failed_instances[@]} -gt 0 ]]; then
        echo "部署失败的实例 (${#failed_instances[@]} 个):"
        for instance in "${!failed_instances[@]}"; do
            echo "   - ${instance} (端口: ${failed_instances[$instance]})"
        done
    fi

    if [[ ${#ready_indices[@]} -gt 0 ]]; then
        echo "超时但有 ${#ready_indices[@]} 个节点已就绪，将继续使用可用节点"
        echo "${ready_indices[@]}"
        return 0
    fi

    echo "错误: 没有任何节点成功启动，请检查远程日志" >&2
    exit 1
}

# 将数据文件按轮询方式分配到各个实例
# 参数：
#   $1: 总实例数量
# 返回值：无
assign_data_to_instances() {
    local total_instances="$1"

    log_info "正在分配全部 ${#FILES[@]} 个数据文件到 ${total_instances} 个实例..."

    # 初始化实例分配数组
    for ((i = 0; i < total_instances; i++)); do
        eval "INSTANCE_ASSIGNMENTS_$i=()"
    done

    # 轮询分配文件
    for idx in "${!FILES[@]}"; do
        local file="${FILES[idx]}"
        local instance_idx=$((idx % total_instances))
        eval "INSTANCE_ASSIGNMENTS_${instance_idx}+=(\"\$file\")"
        log_info "分配文件: ${file} -> 实例 ${instance_idx}"
    done

    for ((i = 0; i < total_instances; i++)); do
        eval "local count=\${#INSTANCE_ASSIGNMENTS_${i}[@]}"
        log_info "实例 ${i} 分配 ${count} 个文件"
    done

    log_info "数据文件分配完成"
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

    log_info "👉 在节点 ${node} 上启动推理任务，模型: ${model_name}"
    # 创建临时状态目录

    local tasks_started=0
    for file in "${files[@]}"; do
        local input_file="${DATASET_DIR}/${file}"
        local base_name=$(basename "$file" .jsonl)
        local output_file="${OUTPUT_DIR}/infer_${model_name//\//_}_${base_name}_bz${N_SAMPLES}.jsonl"
        local log_file="${LOG_DIR}/${TASK_LOG_PREFIX}${node//./_}_${base_name}.log"

        log_info "  -> 处理文件: ${file} (输出: ${output_file})"
        # 构建推理命令
        local infer_cmd="cd '${PROJECT_DIR}' && \
            source '${SET_ENV_SCRIPT}' && \
            nohup python '${INFER_SCRIPT}' \
                --input_file '${input_file}' \
                --output_file '${output_file}' \
                --input_key '${INPUT_KEY}' \
                --base_url '${base_url}' \
                --model_name '${model_name}' \
                --n_samples ${N_SAMPLES} \
                --system_prompt_type '${SYSTEM_PROMPT_TYPE}' \
                --max_workers ${MAX_WORKERS} \
                > '${log_file}' 2>&1 &"

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
    local total_nodes=${#NODES[@]}
    local total_instances=$((total_nodes * INSTANCES_PER_NODE))

    log_info "开始分发并启动推理任务..."

    # 分配数据文件到所有实例
    assign_data_to_instances "$total_instances"

    # 为每个节点的每个实例启动对应的推理任务（并行）
    local pids=()
    local submitted=0
    for ((i = 0; i < total_nodes; i++)); do
        local node="${NODES[i]}"
        for ((j = 0; j < INSTANCES_PER_NODE; j++)); do
            local port_idx=$((i * INSTANCES_PER_NODE + j))
            local port="${PORTS[port_idx]}"
            local base_url="http://127.0.0.1:${port}/v1"
            local model_name="${SERVED_MODEL_NAME}"

            # 获取分配给当前实例的文件列表
            local instance_idx=$((i * INSTANCES_PER_NODE + j))
            IFS=$'\n' read -r -d '' -a ASSIGNED < <(eval "printf '%s\0' \"\${INSTANCE_ASSIGNMENTS_${instance_idx}[@]}\"")

            # 跳过没有分配文件的实例
            if [[ ${#ASSIGNED[@]} -eq 0 ]]; then
                log_info "节点 ${node} 未分配到文件，跳过"
                continue
            fi

            log_info "节点 ${node} 实例 ${j} 分配到 ${#ASSIGNED[@]} 个文件"

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
        done  # Close the inner for loop
    done  # Close the outer for loop

    # 等待所有节点的任务提交完成（不等待远端具体推理完成）
    if [[ ${#pids[@]} -gt 0 ]]; then
        wait "${pids[@]}" || true
    fi

    echo "✅ 所有推理任务已启动"
}

# 监控远端推理任务直至完成（基于进程存活）
# 返回值：无（阻塞直到所有节点上不再存在 INFER_SCRIPT 进程）
wait_for_remote_jobs() {
    log_info "等待所有远端推理任务完成..."
    local interval=10

    while true; do
        local running_total=0
        local pids=()
        for node in "${NODES[@]}"; do
            (
                # 统计匹配推理客户端脚本的存活进程数
                # 用 basename 兼容符号链接/不同路径
                cnt=$(ssh_run "$node" "pgrep -fal 'python .*${INFER_SCRIPT##*/}' | wc -l" 2>/dev/null || echo 0)
                log_info "${node}:${cnt}"
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
            log_info "所有远端推理任务已完成"
            break
        fi
        log_info "仍有 ${running_total} 个远端推理进程在运行，${interval}s 后重试..."
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
    log_info "[START] 开始执行分布式 vLLM 模型推理部署"
    echo "================================================"

    # 获取文件锁，确保只有一个实例在运行
    acquire_lock

    # 设置退出时的清理陷阱
    trap 'cleanup_and_exit' EXIT TERM INT

    # 参数解析
    if [[ $# -gt 1 ]]; then
        echo "❌ 错误: 参数过多" >&2
        usage
    fi

    local NODE_LIST_FILE="${1:-./node_list_all.txt}"

    # 验证节点列表文件
    if [[ ! -f "$NODE_LIST_FILE" ]]; then
        log_error "节点列表文件 '${NODE_LIST_FILE}' 不存在"
        usage
    fi

    log_info "从文件 '${NODE_LIST_FILE}' 加载节点列表"

    # 读取节点列表（过滤空行和注释）
    mapfile -t NODES < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

    if [[ ${#NODES[@]} -eq 0 ]]; then
        log_error "节点列表 '${NODE_LIST_FILE}' 为空"
        exit 1
    fi

    log_info "发现 ${#NODES[@]} 个节点: ${NODES[*]}"

    # 自动生成端口列表（节点间及节点内实例避免冲突，间隔 10 端口）
    PORTS=()
    local start_port=6000
    for ((i=0; i<${#NODES[@]}; i++)); do
        for ((j=0; j<INSTANCES_PER_NODE; j++)); do
            # Increase port spacing to avoid conflicts
            PORTS+=($((start_port + i * 100 + j * 20)))  # 每节点间隔100，实例间隔20
        done
    done
    log_info "自动生成端口列表: ${PORTS[*]}"

    # 验证配置参数
    validate_config

    # 设置退出时的清理陷阱
    trap stop_services EXIT

    # 执行主要流程
    log_info "开始执行部署流程..."

    # 步骤1: 发现数据集文件
    discover_remote_dataset_files

    # 步骤2: 检查节点与端口配置
    check_node_port_alignment

    # 步骤3: 准备远程目录
    check_and_prepare_remote_dirs

    # 步骤4: 并行部署模型服务
    log_info "正在并行部署所有模型服务..."
    for ((i = 0; i < ${#NODES[@]}; i++)); do
        local node="${NODES[i]}"
        for ((j = 0; j < INSTANCES_PER_NODE; j++)); do
            local port_idx=$((i * INSTANCES_PER_NODE + j))
            local port="${PORTS[port_idx]}"
            deploy_model_service "$node" "$port" "$j"
        done
    done

    # 步骤5: 等待服务就绪并获取可用节点（HTTP 健康检查 + 日志回退）
    local -a ready_indices
    mapfile -t ready_indices < <(wait_for_services)

    if [[ ${#ready_indices[@]} -eq 0 ]]; then
        echo "❌ 错误: 没有可用的服务节点" >&2
        exit 1
    fi

    # 构建可用节点和失败节点信息
    local -a available_nodes=()
    local -a available_ports=()
    local -a failed_nodes=()
    declare -A failed_instances  # 关联数组存储每个节点的失败实例信息

    # 使用关联数组标记就绪的节点
    declare -A ready_node_map
    for idx in "${ready_indices[@]}"; do
        ready_node_map["${NODES[idx]}"]=1
        available_nodes+=("${NODES[idx]}")
        # 添加该节点的所有实例端口
        for ((j = 0; j < INSTANCES_PER_NODE; j++)); do
            available_ports+=("${PORTS[$((idx * INSTANCES_PER_NODE + j))]}")
        done
    done

    # 基于ready_node_map快速识别失败的节点和实例
    for ((i = 0; i < ${#NODES[@]}; i++)); do
        local node="${NODES[i]}"
        if [[ -z "${ready_node_map[$node]}" ]]; then
            failed_nodes+=("$node")
            local failed_instance_info=""
            # 收集该节点所有实例的端口信息
            for ((j = 0; j < INSTANCES_PER_NODE; j++)); do
                local port_idx=$((i * INSTANCES_PER_NODE + j))
                if [[ -n "$failed_instance_info" ]]; then
                    failed_instance_info+=", "
                fi
                failed_instance_info+="实例${j}(端口:${PORTS[port_idx]})"
            done
            failed_instances["$node"]="$failed_instance_info"
        fi
    done

    # 输出部署失败的节点和实例信息
    if [[ ${#failed_nodes[@]} -gt 0 ]]; then
        echo "⚠️ 以下节点存在部署失败的实例:"
        for node in "${failed_nodes[@]}"; do
            echo "   - 节点: ${node}"
            echo "     失败实例: ${failed_instances[$node]}"
        done
        echo "❗ 请检查这些节点的日志文件:"
        echo "   ${LOG_DIR}/${API_SERVER_LOG_PREFIX}<节点名>_<实例ID>.log"
    fi

    # 更新全局节点和端口数组
    NODES=("${available_nodes[@]}")
    PORTS=("${available_ports[@]}")

    log_info "将使用 ${#NODES[@]} 个可用节点进行推理"

    # 步骤6: 使用可用节点分发并启动推理任务
    distribute_and_launch_jobs

    # 步骤7: 等待远端推理任务完成后再关闭服务
    wait_for_remote_jobs
    stop_services

    log_info "分布式推理部署完成！"
    log_info "部署统计:"
    log_info "   - 节点数量: ${#NODES[@]}"
    log_info "   - 数据文件: ${#FILES[@]}"
    log_info "   - 服务端口: ${PORTS[*]}"
    log_info "   - 输出目录: ${OUTPUT_DIR}"
    log_info "   - 日志目录: ${LOG_DIR}"
    log_info "================================================"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
