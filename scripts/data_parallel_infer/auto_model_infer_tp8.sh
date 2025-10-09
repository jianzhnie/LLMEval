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
#   - 自动发现与分配数据文件，支持自然数值排序
#   - 多层次并行（节点间并行 + 节点内多卡并行 + 单卡动态批处理）
#   - 混合健康检查机制（HTTP探活 + 日志检查 + 进程检查）
#   - 基于 PID/文件名 的任务监控
#   - 失败节点自动跳过，只使用可用节点进行推理
#   - 资源限制与任务节流
#
# 执行流程：
#   1. 参数校验与环境初始化
#   2. 读取节点列表并生成端口配置
#   3. 发现数据集文件并进行分配
#   4. 并行部署 vLLM 服务实例
#   5. 等待服务就绪（健康检查）并筛选可用节点
#   6. 分发并启动推理任务（数据文件轮询分配到可用节点）
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

# 设置脚本健壮性标志：
# -e: 任何命令失败立即退出
# -u: 使用未设置的变量视为错误
# -o pipefail: 管道中任何命令失败都退出
set -euo pipefail

# =======================================================
#                  调试模式配置
# =======================================================
# 启用调试模式（设置 DEBUG=1 开启）
if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x  # 打印执行的每条命令
    # 增强调试输出，显示文件名、行号和函数名
    export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
fi

# =======================================================
#                  全局常量与配置区域
# =======================================================

# ----------------------
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
readonly NUM_GPUS=${NUM_GPUS:-8}                     # 张量并行大小（单实例用光整机多卡）
readonly MEMORY_UTILIZATION=${MEMORY_UTILIZATION:-0.9} # 显存利用率 (0.0 - 1.0)
readonly MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}        # 最大上下文长度

# vLLM 高并发关键参数（按需调整；需结合显存与上下文长度）
# - MAX_NUM_SEQS: 同时并发处理的序列数（越大越能吞吐，受显存影响较大）
# - MAX_NUM_BATCHED_TOKENS: 动态批次内总 token 上限（控制显存与吞吐权衡）
# 注：两者不宜同时设过大，推荐根据模型大小按 1-2 次试跑观测 GPU 利用率后调整
readonly MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}            # 同时并发处理的序列数
readonly MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768} # 动态批次内总 token 上限

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
readonly INPUT_KEY="${INPUT_KEY:-question}"           # 输入字段键名
readonly SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-amthinking}"
readonly MAX_WORKERS=${MAX_WORKERS:-32}               # 客户端每进程内部的线程/协程并发

# =======================================================
#                  全局变量声明
# =======================================================

# 节点和端口数组（在 main 函数中初始化）
declare -a NODES       # 存储节点地址
declare -a PORTS       # 存储对应的服务端口
declare -a FILES       # 存储发现的数据文件列表（文件名）

# =======================================================
#                  工具函数区域
# =======================================================

# 打印使用帮助信息
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
# Args:
#   $1: node (string) - 节点地址
#   $@: command (string array) - 要执行的命令
# Returns:
#   SSH 命令的退出码
ssh_run() {
    local node="$1"
    shift
    local userhost="${SSH_USER:+${SSH_USER}@}${node}"
    # 使用 $@ 确保命令中的空格和引号被正确传递
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
        log_error "❌ rsync 同步失败: ${src_path} -> ${userhost}:${dst_path}" >&2
        return 1
    fi
}


# 日志函数 (带有 Emoji 提示)
# Args:
#   $@: msg (string) - 日志消息内容
# Returns:
#   None (输出到 stdout/stderr)
log_info() {
    local msg="$*"
    local emoji="ℹ️ "
    # 根据消息内容选择合适的emoji
    case "$msg" in
        *"开始执行"*|*"启动"*) emoji="🚀 " ;;
        *"完成"*|*"成功"*|*"通过"*) emoji="✅ " ;;
        *"失败"*|*"错误"*|*"异常"*) emoji="❌ " ;;
        *"发现"*|*"检查"*) emoji="🔍 " ;;
        *"配置"*|*"设置"*) emoji="⚙️ " ;;
        *"等待"*) emoji="⏳ " ;;
        *"清理"*) emoji="🧹 " ;;
        *"分配"*|*"部署"*) emoji="📦 " ;;
        *"节点"*|*"服务"*) emoji="💻 " ;;
        *"端口"*) emoji="🔌 " ;;
        *"文件"*) emoji="📄 " ;;
        *"统计"*) emoji="📊 " ;;
    esac
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: ${emoji}$msg"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: ⚠️ $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: ❌ $*" >&2
}

# 错误处理函数，并在退出前清理资源
# Args:
#   $1: exit_code (int) - 退出码
#   $2: error_msg (string) - 错误消息
# Returns:
#   None (直接退出脚本)
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg"

    # 调用清理函数
    cleanup_and_exit "$exit_code"
}

# 文件锁管理 (使用 PID)
LOCK_FILE="/tmp/vllm_deploy.lock"

acquire_lock() {
    if [ -e "$LOCK_FILE" ]; then
        local pid
        pid=$(cat "$LOCK_FILE")
        # 检查 PID 是否仍在运行
        if kill -0 "$pid" 2>/dev/null; then
            handle_error 1 "另一个部署进程 (PID: $pid) 正在运行"
        fi
        # 如果 PID 不存在，删除旧锁
        rm -f "$LOCK_FILE"
    fi
    echo $$ > "$LOCK_FILE"
}

release_lock() {
    rm -f "$LOCK_FILE"
}

# 权限检查函数
# Args:
#   $1: dir (string) - 目录路径
# Returns:
#   0: 成功，1: 失败 (通过 handle_error 退出)
check_permissions() {
    local dir="$1"
    if [[ ! -w "$dir" ]]; then
        handle_error 1 "本地目录 $dir 没有写入权限"
    fi
}

# 节点连通性检查
# Args:
#   $1: node (string) - 节点地址
# Returns:
#   0: 成功，1: 失败
validate_node() {
    local node="$1"
    # 使用 -q (quiet) 避免输出，通过退出码判断连通性
    if ssh -q "${SSH_USER:+${SSH_USER}@}${node}" exit 2>/dev/null; then
        return 0
    else
        log_warn "无法连接到节点 $node"
        return 1
    fi
}

# 优雅清理所有资源并退出
# Args:
#   $1: exit_code (int, optional) - 退出码，默认为最后一次命令的退出码
# Returns:
#   None (退出脚本)
cleanup_and_exit() {
    # 如果没有传递退出码，使用上一个命令的退出码
    local exit_code="${1:-$?}"

    log_info "开始清理资源..."

    # 停止所有服务
    stop_services

    # 释放文件锁
    release_lock

    # 如果是调试模式，关闭它
    [[ "${DEBUG:-0}" == "1" ]] && set +x

    log_info "清理完成，退出代码: $exit_code"
    exit "$exit_code"
}


# 验证配置参数
# Args:
#   None
# Returns:
#   None (验证失败时通过 handle_error 退出)
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

    # 验证本地目录权限
    local required_dirs=(
        "$OUTPUT_DIR"
        "$LOG_DIR"
        "$DATASET_DIR"
    )

    # 提前创建输出目录，确保权限检查通过
    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" || true

    for dir in "${required_dirs[@]}"; do
        check_permissions "$dir"
    done

    # 验证数值参数范围
    # 参数名: 最小值: 最大值: 描述
    local param_checks=(
        "NUM_GPUS:1:8:GPU数量"
        "N_SAMPLES:1:100:采样次数"
        "MAX_NUM_SEQS:1:16384:并发序列数"
        "MAX_NUM_BATCHED_TOKENS:512:1048576:批处理Token数"
    )

    for check in "${param_checks[@]}"; do
        # Bash 技巧: 使用 IFS 拆分字符串
        IFS=':' read -r param min max desc <<< "$check"
        local value
        # Bash 技巧: 使用 eval 获取变量值 (仅限配置区，风险可控)
        value=$(eval echo "\$$param")
        # Bash 技巧: 使用 [[ ... ]] 和算术扩展进行数值比较
        if [[ $value -lt $min || $value -gt $max ]]; then
            handle_error 1 "$desc ($param) 需在 $min-$max 之间，当前值: $value"
        fi
    done

    # 验证浮点数参数 (使用 bc 进行浮点比较)
    if [[ $(echo "${MEMORY_UTILIZATION} < 0.1 || ${MEMORY_UTILIZATION} > 1.0" | bc -l) -eq 1 ]]; then
        handle_error 1 "显存利用率需在 0.1-1.0 之间，当前值: ${MEMORY_UTILIZATION}"
    fi

    log_info "✅ 配置参数验证通过"
}

# =======================================================
#                  核心功能函数区域
# =======================================================

# 停止所有远程节点上的模型服务
# Args:
#   None
# Returns:
#   None
stop_services() {
    log_info "🛑 脚本退出，正在停止所有远程模型服务..."

    local search_pattern="vllm.entrypoints.openai.api_server"
    local pids=()

    # 遍历当前已知的节点列表 (可能已被 main 函数更新为 available_nodes)
    for node in "${NODES[@]}"; do
        log_info "---> 正在停止节点 ${node} 上的 vLLM 进程..."
        (
            # 使用 pkill 优雅地发送 SIGTERM，并忽略错误（如果进程已停止）
            ssh_run "$node" "pkill -f '${search_pattern}' || true"
            # 等待一段时间确保进程完全停止
            sleep 3
            # 再次检查是否还有相关进程在运行
            local remaining_processes
            remaining_processes=$(ssh_run "$node" "pgrep -f '${search_pattern}' | wc -l" 2>/dev/null || echo "0")
            if [[ "${remaining_processes:-0}" -gt 0 ]]; then
                log_warn "节点 ${node} 上仍有 ${remaining_processes} 个 vLLM 进程在运行，尝试强制终止..."
                ssh_run "$node" "pkill -9 -f '${search_pattern}' || true"
            fi
            log_info "节点 ${node} 服务已停止"
        ) &
        pids+=($!)
    done

    # 等待所有停止操作完成
    wait "${pids[@]}" || true
    log_info "✅ 所有远程模型服务停止完成"
}

# 端口探活（远程是否可用）
# Args:
#   $1: node (string) - 节点地址
#   $2: port (int) - 服务端口
# Returns:
#   None (如果端口被占用，尝试清理)
check_remote_port_free() {
    local node="$1"
    local port="$2"
    local used=0

    # 尝试通过 ss, netstat 或 lsof 检查端口占用情况
    # 注意：这些命令在不同系统上可能不同，尝试多个以提高兼容性
    used=$(ssh_run "$node" "ss -ltn '( sport = :$port )' 2>/dev/null | tail -n +2 | wc -l" 2>/dev/null || echo 0)
    if [[ "${used:-0}" -eq 0 ]]; then
        used=$(ssh_run "$node" "netstat -ltn 2>/dev/null | awk '{print \$4}' | grep -E '[:.]${port}\$' | wc -l" 2>/dev/null || echo 0)
    fi
    if [[ "${used:-0}" -eq 0 ]]; then
        used=$(ssh_run "$node" "lsof -iTCP:${port} -sTCP:LISTEN -nP 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    fi
    if [[ "${used:-0}" -gt 0 ]]; then
        log_warn "节点 ${node} 端口 ${port} 已被占用，尝试清理旧 vLLM 进程..."
        # 尝试通过匹配端口的 vLLM 进程杀掉旧服务
        ssh_run "$node" "pkill -f 'vllm.entrypoints.openai.api_server.*--port ${port}' || true" >/dev/null 2>&1 || true
        sleep 1
    fi
}


# 检查节点与端口列表数量是否一致
# 参数：无
# 返回值：无（检查失败时退出）
check_node_port_alignment() {
    if [[ ${#NODES[@]} -ne ${#PORTS[@]} ]]; then
        log_error "节点数量 (${#NODES[@]}) 与端口数量 (${#PORTS[@]}) 不一致"
        exit 1
    fi
    log_info "节点和端口配置检查通过"
}


# 在第一个节点上发现数据集文件
# Args:
#   None
# Returns:
#   None (文件列表存储到全局 FILES 数组)
discover_remote_dataset_files() {
    if [[ ${#NODES[@]} -eq 0 ]]; then
        log_error "错误: 无可用节点进行数据文件发现"
        exit 1
    fi

    local head_node="${NODES[0]}"
    local search_path="${DATASET_DIR}/${DATASET_GLOB}"
    log_info "🔍 正在节点 ${head_node} 上发现数据文件: ${search_path}"

    # Bash 技巧: 使用 xargs -n1 basename | sort -V 实现按自然数值排序的文件名列表
    local find_cmd="sh -lc 'find ${DATASET_DIR} -maxdepth 1 -name \"${DATASET_GLOB}\" 2>/dev/null | xargs -n1 basename | LC_ALL=C sort -V'"

    local out
    if ! out=$(ssh_run "$head_node" "$find_cmd"); then
        log_error "❌ 无法在节点 ${head_node} 上列出数据文件，请检查路径与权限"
        exit 1
    fi

    # 将结果存储到全局数组 FILES
    # Bash 技巧: mapfile -t < <(...) 避免创建 subshell 导致变量无法修改
    mapfile -t FILES < <(printf "%s\n" "$out" || true)

    if [[ ${#FILES[@]} -eq 0 ]]; then
        log_error "❌ 未发现任何匹配的数据文件 (模式: ${DATASET_GLOB})，请检查 ${DATASET_DIR} 和 ${DATASET_GLOB} 配置"
        exit 1
    fi

    log_info "✅ 发现数据集文件数量: ${#FILES[@]}"
    # 仅输出前5个文件示例
    log_info "文件列表 (前5个): ${FILES[*]:0:5}..."
}

# 检查并创建远程目录，清理旧日志
# Args:
#   None
# Returns:
#   None
check_and_prepare_remote_dirs() {
    log_info "⚙️ 正在检查并创建远程目录，清理旧日志..."

    for node in "${NODES[@]}"; do
        log_info "处理节点: ${node}"
        # 创建目录，清理旧的状态/日志文件
        local prep_cmd="mkdir -p '${OUTPUT_DIR}' '${DATASET_DIR}' '${LOG_DIR}' && \
            rm -rf '${LOG_DIR}/status' && mkdir -p '${LOG_DIR}/status' && \
            rm -f '${LOG_DIR}/${API_SERVER_LOG_PREFIX}'*.log '${LOG_DIR}/${TASK_LOG_PREFIX}'*.log 2>/dev/null || true"

        if ! ssh_run "$node" "$prep_cmd"; then
            log_error "❌ 无法在节点 ${node} 上准备目录，请检查SSH连接和权限"
                exit 1 # 在 subshell 中退出
        fi
    done

    log_info "✅ 所有远程目录已就绪，旧日志已清理"
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
    # Bash 技巧: 使用 ${variable//pattern/replacement} 替换所有 . 为 _ 以避免文件名问题
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}.log"

    log_info "🚀 正在节点 ${node} 上部署模型服务 (端口: ${port}, TP: ${NUM_GPUS}, 内存: ${MEMORY_UTILIZATION})"

    # 1. 节点连通性验证
    if ! validate_node "$node"; then
        return 1
    fi

    # 2. 检查并清理旧端口占用
    check_remote_port_free "$node" "$port"

    # 3. 构建 vLLM 启动命令
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

    # 4. 在后台启动服务
    ssh_run "$node" "$vllm_cmd" &
    log_info "✅ 节点 ${node} 启动命令发送成功"
}

# 健康检查（HTTP 探活 + 日志回退）
# Args:
#   $1: node (string) - 节点地址
#   $2: port (int) - 服务端口
# Returns:
#   0: 健康检查通过，1: 检查失败
check_service_ready() {
    local node="$1"
    local port="$2"
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}.log"
    local base_url="http://127.0.0.1:${port}"
    local http_status models_status


    # 检查日志文件是否存在
    if ! ssh_run "$node" "[[ -f '${log_file}' ]]"; then
        log_warn "节点 ${node} 的日志文件尚未创建: ${log_file}"
        return 1
    fi

    # 1. 检查服务进程是否存在
    if ! ssh_run "$node" "pgrep -f 'vllm.entrypoints.openai.api_server.*--port ${port}' > /dev/null"; then
        log_warn "节点 ${node} 上的服务进程未运行或已退出"
        return 1
    fi

    # 2. 尝试 HTTP 健康检查 (/health)
    http_status=$(ssh_run "$node" "curl -s -o /dev/null -w '%{http_code}' --max-time ${HEALTH_TIMEOUT} \
        ${base_url}${HEALTH_PATH} 2>/dev/null || echo 0")

    if [[ $http_status -eq 200 ]]; then
        log_info "✅ 服务 ${node}:${port} 健康检查 (${HEALTH_PATH}) 通过"
        return 0
    fi

    # 3. 兼容性检查：尝试 /v1/models (vLLM OpenAI 兼容层标准)
    models_status=$(ssh_run "$node" "curl -s -o /dev/null -w '%{http_code}' --max-time ${HEALTH_TIMEOUT} \
        ${base_url}/v1/models 2>/dev/null || echo 0")

    if [[ $models_status -eq 200 ]]; then
        log_info "✅ 服务 ${node}:${port} /v1/models 接口检查通过"
        return 0
    fi

    # 4. 日志回退检查：查找启动完成标志
    if ssh_run "$node" "grep -q 'Application startup complete' '${log_file}' 2>/dev/null"; then
        log_info "✅ 服务 ${node}:${port} 日志启动完成标志通过 (HTTP状态码: ${http_status}/${models_status})"
        return 0
    fi
    log_warn "节点 ${node} 的 vllm 服务启动未完成 (HTTP状态码: ${http_status}/${models_status})，日志中未找到启动完成标志"
    return 1
}

# 轮询检查所有模型服务是否启动成功
# Args:
#   None
# Returns:
#   None (不返回任何值，改为在函数外部检查状态文件)
wait_for_services() {
    log_info "⏳ 正在等待所有模型服务启动并就绪... 最长等待 ${MAX_WAIT_TIME} 秒"

    local total_wait_time=0
    local interval=10
    local total_services=${#NODES[@]}
    local status_dir="${LOG_DIR}/status"

    # 确保状态目录干净
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

        # 等待所有节点的检查完成
        if [[ ${#running_pids[@]} -gt 0 ]]; then
            wait "${running_pids[@]}" || true
        fi

        # 统计已就绪服务数量
        local ready_count
        ready_count=$(ls -1 "${status_dir}" 2>/dev/null | wc -l | tr -d ' ')

        if [[ $ready_count -eq $total_services ]]; then
            log_info "✅ 所有 ${total_services} 个服务已就绪"
            return 0
        fi

        log_info "---> ${ready_count}/${total_services} 服务就绪，继续等待... (已等待 ${total_wait_time}s)"
        sleep "$interval"
        total_wait_time=$((total_wait_time + interval))
    done

    log_warn "⏰ 超时: 服务在 ${MAX_WAIT_TIME} 秒内未完全就绪，将继续使用已就绪的服务"
}


# 将数据文件按轮询方式分配到各个实例
# Args:
#   $1: total_instances (int) - 总实例数量
# Returns:
#   None (分配结果存储在全局变量 INSTANCE_ASSIGNMENTS_X 中)
assign_data_to_instances() {
    local total_instances="$1"

    log_info "📊 正在分配全部 ${#FILES[@]} 个数据文件到 ${total_instances} 个实例..."

    # 销毁并初始化实例分配数组
    for ((i = 0; i < total_instances; i++)); do
        # 动态声明/清空数组变量
        eval "INSTANCE_ASSIGNMENTS_$i=()"
    done

    # 轮询分配文件
    for idx in "${!FILES[@]}"; do
        local file="${FILES[idx]}"
        local instance_idx=$((idx % total_instances))
        # 动态赋值数组元素
        eval "INSTANCE_ASSIGNMENTS_${instance_idx}+=(\"\$file\")"
        log_info "分配文件: ${file} -> 实例 ${instance_idx}"
    done

    # 打印分配结果统计
    for ((i = 0; i < total_instances; i++)); do
        local count
        eval "count=\${#INSTANCE_ASSIGNMENTS_${i}[@]}"
        log_info "实例 ${i} 分配 ${count} 个文件"
    done

    log_info "✅ 数据文件分配完成"
}

# 在指定节点上批量提交推理任务，包含重试和资源控制机制
# Args:
#   $1: node (string) - 节点地址
#   $2: model_name (string) - 模型名称
#   $3: base_url (string) - 服务 URL (如 http://127.0.0.1:port/v1)
#   $@: files (string array) - 分配给该节点的全部文件列表
# Returns:
#   None (任务在远程后台启动，不等待完成)
run_task_batch() {
    local node="$1"
    local model_name="$2"
    local base_url="$3"
    shift 3
    local files=("$@")

    log_info "👉 在节点 ${node} 上启动 ${#files[@]} 个推理任务..."

    # 构建所有文件的推理命令并一次性发送
    local commands=()
    for file in "${files[@]}"; do
        local input_file="${DATASET_DIR}/${file}"
        # 移除文件扩展名
        local base_name="${file%.*}"
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

        commands+=("$infer_cmd")
    done

    # 将所有命令组合成一个命令字符串并执行
    if [[ ${#commands[@]} -gt 0 ]]; then
        local combined_cmd=$(printf "%s; " "${commands[@]}")
        ssh_run "$node" "$combined_cmd" >/dev/null 2>&1 &
    fi

    log_info "✅ 节点 ${node} 上的 ${#files[@]} 个推理任务已提交"
}



# 分发并启动所有推理任务
# Args:
#   None
# Returns:
#   None
distribute_and_launch_jobs() {
    local total_instances=${#NODES[@]}

    log_info "开始分发并启动推理任务..."

    # 1. 分配数据文件到可用实例
    assign_data_to_instances "$total_instances"

    # 2. 为每个节点启动对应的推理任务（并行）
    local pids=()
    for ((i = 0; i < total_instances; i++)); do
        local node="${NODES[i]}"
        local port="${PORTS[i]}"
        # 注意: vLLM OpenAI 兼容层 API 通常在 /v1 路径下
        local base_url="http://127.0.0.1:${port}/v1"
        local model_name="${SERVED_MODEL_NAME}"

        # 获取分配给当前实例的文件列表
        local instance_files_var="INSTANCE_ASSIGNMENTS_$i"
        local -n instance_files_ref="$instance_files_var"

        # 检查文件是否分配 (如果 assign_data_to_instances 中有节点没有分配到文件，这里跳过)
        if [[ ${#instance_files_ref[@]} -eq 0 ]]; then
            log_info "节点 ${node} 未分配到文件，跳过"
            continue
        fi

        # 获取分配给当前实例的文件列表
        log_info "节点 ${node} 分配到 ${#instance_files_ref[@]} 个文件"
        # 在本地后台启动任务提交批次
        (
            run_task_batch "$node" "$model_name" "$base_url" "${instance_files_ref[@]}"
        ) &
        pids+=($!)
    done

    # 3. 等待所有节点的任务提交完成（不等待远端具体推理完成）
    if [[ ${#pids[@]} -gt 0 ]]; then
        wait "${pids[@]}" || true
    fi
    log_info "✅ 所有推理任务已启动，进入远端任务监控阶段"

    # 4. 等待所有远程推理任务完成
    wait_for_inference_completion
}
# 等待所有推理任务完成
# Args:
#   None
# Returns:
#   None
wait_for_inference_completion() {
    log_info "⏳ 等待所有推理任务完成..."

    local total_nodes=${#NODES[@]}
    local completed_nodes=0

    while [[ $completed_nodes -lt $total_nodes ]]; do
        completed_nodes=0

        for ((i = 0; i < total_nodes; i++)); do
            local node="${NODES[i]}"

            # 检查节点上是否还有运行中的推理任务
            local running_tasks
            running_tasks=$(ssh_run "$node" "pgrep -f '${INFER_SCRIPT}' | wc -l" 2>/dev/null || echo "0")

            if [[ "${running_tasks:-0}" -eq 0 ]]; then
                completed_nodes=$((completed_nodes + 1))
                log_info "✅ 节点 ${node} 上的推理任务已完成"
            else
                log_info "⏳ 节点 ${node} 上仍有 ${running_tasks} 个推理任务在运行"
            fi
        done

        if [[ $completed_nodes -lt $total_nodes ]]; then
            log_info "等待 10 秒后再次检查任务状态..."
            sleep 10
        fi
    done

    log_info "✅ 所有节点上的推理任务已完成"
}


# =======================================================
#                  主程序入口
# =======================================================

# 主函数：协调整个部署和推理流程
# Args:
#   $@: 命令行参数 (可选: NODE_LIST_FILE)
# Returns:
#   None
main() {
    log_info "[START] 开始执行分布式 vLLM 模型推理部署"
    echo "================================================"

    # 设置退出时的清理陷阱 (最先设置，确保任何失败都能调用清理)
    trap 'cleanup_and_exit' EXIT TERM INT

    # 获取文件锁
    acquire_lock

    # 参数解析
    if [[ $# -gt 1 ]]; then
        log_error "参数错误"
        usage
    fi

    local NODE_LIST_FILE="${1:-./node_list_all.txt}"

    # 验证节点列表文件
    if [[ ! -f "$NODE_LIST_FILE" ]]; then
        handle_error 1 "节点列表文件 '${NODE_LIST_FILE}' 不存在"
    fi

    log_info "从文件 '${NODE_LIST_FILE}' 加载节点列表"

    # 读取节点列表（过滤空行和注释），存入全局 NODES
    mapfile -t NODES < <(grep -v -e '^\s*$' -e '^\s*#' "$NODE_LIST_FILE")

    if [[ ${#NODES[@]} -eq 0 ]]; then
        handle_error 1 "节点列表 '${NODE_LIST_FILE}' 为空"
    fi

    log_info "发现 ${#NODES[@]} 个节点: ${NODES[*]}"

    # 自动生成端口列表（节点间间隔 10 端口），存入全局 PORTS
    PORTS=()
    local start_port=6000
    for ((i=0; i<${#NODES[@]}; i++)); do
        PORTS+=($((start_port + i * 10)))
    done
    log_info "自动生成端口列表: ${PORTS[*]}"

    # 验证配置参数
    validate_config

    # --- 执行主要流程 ---
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
        local port="${PORTS[i]}"
        # 在本地后台部署，加速并发
        deploy_model_service "$node" "$port" &
    done

    # 等待所有部署命令发送完成 (即使失败，deploy_model_service 也会返回)
    wait || true

    # 步骤5: 等待服务就绪并获取可用节点（HTTP 健康检查 + 日志回退）
    wait_for_services

    # 不再使用 wait_for_services 的返回值，而是主动检查所有节点状态
    log_info "正在检查各节点服务状态..."

    # 初始化可用节点和失败节点列表
    local -a available_nodes=()
    local -a available_ports=()
    local -a failed_nodes=()
    local -a failed_ports=()

    # 检查每个节点的状态
    for ((i = 0; i < ${#NODES[@]}; i++)); do
        local node="${NODES[i]}"
        local port="${PORTS[i]}"
        # 获取节点的 API 服务状态文件
        local status_file="${LOG_DIR}/status/status_${node//./_}.ok"

        if [[ -f "$status_file" ]]; then
            log_info "✅ 节点 ${node} (端口: ${port}) 服务就绪"
            available_nodes+=("${node}")
            available_ports+=("${port}")
        else
            log_warn "❌ 节点 ${node} (端口: ${port}) 服务未就绪"
            failed_nodes+=("${node}")
            failed_ports+=("${port}")
        fi
    done

    # 输出部署结果统计
    log_info "📊 服务部署结果统计:"
    log_info "   - 成功节点数量: ${#available_nodes[@]}/${#NODES[@]}"

    if [[ ${#failed_nodes[@]} -gt 0 ]]; then
        log_warn "以下节点未能成功部署:"
        for ((i = 0; i < ${#failed_nodes[@]}; i++)); do
            log_warn "   - ${failed_nodes[i]} (端口: ${failed_ports[i]})"
        done
        log_warn "请检查这些节点的日志文件: ${LOG_DIR}/${API_SERVER_LOG_PREFIX}<节点名>.log"
    fi

    # 更新全局 NODES 和 PORTS 数组为可用节点
    NODES=("${available_nodes[@]}")
    PORTS=("${available_ports[@]}")

    # 检查是否有可用节点
    if [[ ${#NODES[@]} -eq 0 ]]; then
        handle_error 1 "❌ 没有任何节点成功启动服务，无法继续执行推理任务"
    fi

    log_info "将使用 ${#NODES[@]} 个可用节点进行推理"

    # 步骤6: 使用可用节点分发并启动推理任务
    distribute_and_launch_jobs

    # 步骤7: 优雅关闭服务（由 EXIT 陷阱调用 stop_services）
    log_info "✅ 分布式推理部署和任务执行完成，正在退出并清理资源..."

    log_info "📊 部署统计:"
    log_info "   - 节点总数: ${#NODES[@]}"
    log_info "   - 可用节点: ${#available_nodes[@]}"
    log_info "   - 数据文件: ${#FILES[@]}"
    log_info "   - 输出目录: ${OUTPUT_DIR}"
    log_info "   - 日志目录: ${LOG_DIR}"
    echo "================================================"
}


# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
