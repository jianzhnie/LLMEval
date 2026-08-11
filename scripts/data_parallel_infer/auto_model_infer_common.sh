#!/bin/bash
# =======================================================
# 分布式 vLLM 模型推理共享实现（高并发优化版）
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
#   1. TENSOR_PARALLEL_SIZE: 根据实际显卡数量设置
#   2. MAX_NUM_SEQS: 结合显存大小调整
#   3. HEALTH_TIMEOUT: 根据网络情况调整检查超时
#
# 使用方法：
#   由 auto_model_infer.sh 调用
#
# 环境要求：
#   - bash 4.3+（使用了 local -n nameref）
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
# 为避免行内注释破坏多行字符串，将注释前移
# - BatchMode=yes: 禁止交互提示，便于自动化
readonly -a SSH_OPTS=(
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
    -o BatchMode=yes
    -o ConnectTimeout=5
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    -o ControlMaster=auto
    -o ControlPersist=60s
)

# SSH 用户配置: 优先使用环境变量，否则使用当前用户
readonly SSH_USER="${SSH_USER:-$(whoami)}"
# =======================================================
#                  模型与资源配置
# =======================================================

# 模型路径配置
readonly MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B}"

# GPU/ASCEND 资源配置
readonly TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}    # 张量并行大小，支持 1/2/4/8
readonly INSTANCES_PER_NODE=${INSTANCES_PER_NODE:-2}        # 每节点部署实例数（灵活配置）
readonly MEMORY_UTILIZATION=${MEMORY_UTILIZATION:-0.9}      # 显存利用率 (0.0 - 1.0)
readonly MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}              # 最大上下文长度
# 针对 Ascend 场景中 npu-smi 返回值与可用设备数量不一致的问题，允许引入修正因子
readonly DEVICE_COUNT_MULTIPLIER=${DEVICE_COUNT_MULTIPLIER:-2}

# vLLM 高并发关键参数（按需调整；需结合显存与上下文长度）
# - MAX_NUM_SEQS: 同时并发处理的序列数（越大越能吞吐，受显存影响较大）
# - MAX_NUM_BATCHED_TOKENS: 动态批次内总 token 上限（控制显存与吞吐权衡）
# 注：两者不宜同时设过大，推荐根据模型大小按 1-2 次试跑观测 GPU 利用率后调整
readonly MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}                         # 同时并发处理的序列数
readonly MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}    # 动态批次内最大 token 数

# 其他推理参数
readonly N_SAMPLES=${N_SAMPLES:-8}                   # 每条样本的重复采样次数
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PCL-Reasoner}"
readonly RUN_ID="${RUN_ID:-$(date '+%Y%m%d%H%M%S')-$$}"

# 计算每个实例的设备可见性
get_device_visibility() {
    local instance_id=$1  # 0-based
    local start_idx=$((instance_id * TENSOR_PARALLEL_SIZE))
    # Check if we have enough devices available
    local end_idx=$((start_idx + TENSOR_PARALLEL_SIZE - 1))

    # Just in case we're using ASCEND devices, we should check the actual available devices
    # This is a more robust approach than just assuming sequential device IDs
    seq -s, $start_idx $end_idx
}

get_remote_device_count() {
    local node=$1
    # 尝试连接并执行命令。SSH_OPTS 已禁用交互式 host-key 检查；额外的
    # ssh-keyscan 会让不可达节点在 set -e 下提前终止整个部署。
    local output
    if ! output=$(ssh_run "$node" "npu-smi info 2>/dev/null" 2>/dev/null); then
        # 如果ssh命令失败（例如连接超时），则直接判定为不可用
        # 诊断信息输出到 stderr，避免污染命令替换捕获的 stdout 返回值
        echo "🔴 节点 $node: 连接失败或命令执行失败" >&2
        echo "0"
        return 0
    fi

    # 检查输出中是否包含"No running processes found in NPU"
    # 我们可以通过统计"No running processes found"的行数来判断所有卡是否都空闲
    local device_count
    device_count=$(printf '%s\n' "$output" | awk '/No running processes found in NPU/ { count++ } END { print count + 0 }')

    # 检查是否有错误信息
    local error_lines
    error_lines=$(printf '%s\n' "$output" | awk '/Error/ { count++ } END { print count + 0 }')

    if [ "$error_lines" -gt 0 ]; then
        echo "❌ 节点 $node: NPU命令执行出错" >&2
        echo "0"
        return 0
    fi

    # 通过echo返回实际的设备数量
    echo "$device_count"
}

# 验证节点的设备数量是否满足实例配置需求
verify_node_device_capacity() {
    local node=$1

    # 根据参数决定NPU卡数量
    local required_devices=$((INSTANCES_PER_NODE * TENSOR_PARALLEL_SIZE))

    # 从远程节点获取实际的设备数量
    local device_count_raw
    device_count_raw=$(get_remote_device_count "$node")

    # get_remote_device_count 可能返回空或非数字，统一兜底为 0
    if ! [[ "$device_count_raw" =~ ^[0-9]+$ ]]; then
        device_count_raw=0
    fi

    # 根据硬件/驱动输出情况应用修正系数（Ascend 910B 常见为 2）
    local device_count=$((device_count_raw * DEVICE_COUNT_MULTIPLIER))

    # 确保所有NPU都空闲（device_count 由算术展开得出，恒为数字）
    if [[ "$device_count" -lt "$required_devices" ]]; then
        handle_error 1 "节点 ${node} 可用设备数量 (${device_count}) 少于运行 ${INSTANCES_PER_NODE} 个实例所需的 ${required_devices} 张设备 (TP=${TENSOR_PARALLEL_SIZE})"
    fi
    log_info "✅ 节点 ${node} 可用设备数 ${device_count} 满足 ${INSTANCES_PER_NODE} 实例 * TP=${TENSOR_PARALLEL_SIZE} 的需求"
}

# =======================================================
#                  vLLM API Server 运行参数
# =======================================================

# 关闭请求逐条日志，减少 IO 抖动
readonly DISABLE_LOG_REQUESTS=${DISABLE_LOG_REQUESTS:-True}

# 额外引擎参数（按需追加，例如 "--dtype bfloat16 --enforce-eager"）
readonly EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:-}"

# =======================================================
#                  路径配置
# =======================================================

# 项目路径配置
COMMON_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly COMMON_SCRIPT_DIR
DEFAULT_PROJECT_DIR="$(cd -- "${COMMON_SCRIPT_DIR}/../.." && pwd)"
readonly DEFAULT_PROJECT_DIR
readonly PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
readonly INFER_SCRIPT="${INFER_SCRIPT:-${PROJECT_DIR}/llmeval/inference/online.py}"
readonly SET_ENV_SCRIPT="${SET_ENV_SCRIPT:-${PROJECT_DIR}/set_env.sh}"

# 输出与日志路径配置
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output}"
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

# =======================================================
#                  推理客户端参数
# =======================================================
readonly INPUT_KEY="${INPUT_KEY:-question}"           # 输入字段键名
readonly SYSTEM_PROMPT_TYPE="${SYSTEM_PROMPT_TYPE:-amthinking}"
readonly MAX_WORKERS=${MAX_WORKERS:-128}               # 客户端每进程内部的线程/协程并发

# =======================================================
#                  全局变量声明
# =======================================================

# 节点和端口数组（在 main 函数中初始化）
declare -a NODES                    # 存储节点地址
declare -a PORTS                    # 存储对应的服务端口
declare -a FILES                    # 存储发现的数据文件列表（文件名）
declare -a READY_INSTANCE_NODES     # 存储已就绪实例所属节点（按实例展开）
declare -a READY_INSTANCE_PORTS     # 存储已就绪实例端口
declare -a READY_INSTANCE_IDS       # 存储已就绪实例在节点内的 index

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
  TENSOR_PARALLEL_SIZE   张量并行卡数，支持 {1,2,4,8}（当前：${TENSOR_PARALLEL_SIZE}）
  INSTANCES_PER_NODE     每节点实例数（当前：${INSTANCES_PER_NODE}）
  MEMORY_UTILIZATION     显存利用率（默认：0.9）
  MAX_MODEL_LEN          最大上下文长度（默认：65536）
  MAX_NUM_SEQS           vLLM 动态批并发序列数（默认：1024）
  MAX_NUM_BATCHED_TOKENS vLLM 动态批 token 上限（默认：32768）
  N_SAMPLES              每个样本采样次数（默认：8）
  SERVED_MODEL_NAME      服务模型名称（默认：PCL-Reasoner）
  MAX_WAIT_TIME          服务启动最大等待时间（默认：900秒）
  DATASET_GLOB           数据集文件匹配模式
  SYSTEM_PROMPT_TYPE     系统提示类型（默认：amthinking）
  MAX_WORKERS            推理客户端内部并发（默认：128）
  DISABLE_LOG_REQUESTS   是否关闭请求日志（默认：True）
  EXTRA_ENGINE_ARGS      附加引擎参数字符串（默认：空）
  DEVICE_COUNT_MULTIPLIER npu-smi 统计修正因子（默认：2）
  DEBUG                  启用调试模式（默认：0）

示例:
  $0
  SSH_USER=root TENSOR_PARALLEL_SIZE=4 MAX_NUM_SEQS=2048 $0 ./nodes.txt
  DEBUG=1 $0
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
    # Commands are intentionally assembled locally and passed as one remote command.
    # shellcheck disable=SC2029
    ssh "${SSH_OPTS[@]}" "${userhost}" "$@"
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

wait_for_pids() {
    local pending=("$@")
    local next=()
    local pid remaining state

    while [[ ${#pending[@]} -gt 0 ]]; do
        next=()
        for pid in "${pending[@]}"; do
            state=$(ps -p "$pid" -o stat= 2>/dev/null || true)
            if [[ -n "$state" && "$state" != Z* ]]; then
                next+=("$pid")
                continue
            fi
            if ! wait "$pid"; then
                # Stop every monitor still running as soon as any peer fails.
                for remaining in "${pending[@]}"; do
                    [[ "$remaining" == "$pid" ]] || kill "$remaining" 2>/dev/null || true
                done
                for remaining in "${pending[@]}"; do
                    [[ "$remaining" == "$pid" ]] || wait "$remaining" 2>/dev/null || true
                done
                return 1
            fi
        done
        if [[ ${#next[@]} -eq 0 ]]; then
            break
        fi
        pending=("${next[@]}")
        sleep 1
    done
    return 0
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
# 使用 mkdir 实现原子加锁：mkdir 对已存在路径返回非零，是原子操作。
# 锁目录内的 pid 文件记录持锁进程，供占用锁诊断（kill -0 验证）使用。
LOCK_DIR="$LOG_DIR/vllm_deploy.lock"
LOCK_PID_FILE="${LOCK_DIR}/pid"

acquire_lock() {
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        local pid
        pid=$(cat "$LOCK_PID_FILE" 2>/dev/null || true)
        if [[ -z "$pid" ]]; then
            handle_error 1 "部署锁正在初始化或已损坏: ${LOCK_DIR}；请稍后重试，确认无部署进程后再手动清理"
        fi
        # kill -0 may return EPERM for another user's live process. ps provides
        # a second existence check before classifying the lock as stale.
        if kill -0 "$pid" 2>/dev/null || ps -p "$pid" -o pid= >/dev/null 2>&1; then
            handle_error 1 "另一个部署进程 (PID: $pid) 正在运行"
        fi
        # 不能在检查后自动删除：另一个进程可能已在两步之间取得同一路径的
        # 新锁。保留现场并要求显式清理，避免误删活跃部署的锁。
        handle_error 1 "检测到无有效持有者的部署锁: ${LOCK_DIR}，请确认没有部署进程运行后手动删除该目录"
    fi
    printf '%s\n' "$$" > "$LOCK_PID_FILE"
}

release_lock() {
    # 仅当锁内的 PID 等于当前进程 PID 时才删除锁，
    # 避免未持锁的进程（如发现他人正在运行而报错退出时）误删持锁进程的锁，
    # 导致第三个进程可以并发进入
    if [[ -d "$LOCK_DIR" ]]; then
        local lock_pid
        lock_pid=$(cat "$LOCK_PID_FILE" 2>/dev/null || true)
        if [[ -n "$lock_pid" && "$lock_pid" == "$$" ]]; then
            rm -f "$LOCK_PID_FILE"
            if ! rmdir "$LOCK_DIR" 2>/dev/null; then
                log_warn "锁目录包含非预期内容，未自动删除: ${LOCK_DIR}"
            fi
        else
            log_warn "锁不属于当前进程 (锁内 PID: '${lock_pid:-空}', 当前 PID: $$)，跳过删除"
        fi
    fi
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
    # 复用 ssh_run（带 BatchMode/ConnectTimeout），避免密钥认证失败时
    # 交互式密码提示导致脚本永久挂起；通过退出码判断连通性
    if ssh_run "$node" exit 2>/dev/null; then
        log_info "✅ 节点 ${node} 连通性检查通过"
        return 0
    else
        log_warn "⚠️ 无法连接到节点 $node"
        return 1
    fi
}

# 检查节点上的 PROJECT_DIR 和 DATASET_DIR 是否存在
# Args:
#   $1: node (string) - 节点地址
# Returns:
#   0: 成功，1: 失败
validate_node_directories() {
    local node="$1"

    log_info "🔍 检查节点 ${node} 上的必要目录是否存在"

    # 检查 PROJECT_DIR 是否存在
    if ! ssh_run "$node" "[[ -d '${PROJECT_DIR}' ]]"; then
        log_error "❌ 节点 ${node} 上 PROJECT_DIR 不存在: ${PROJECT_DIR}"
        return 1
    fi

    # 检查 DATASET_DIR 是否存在
    if ! ssh_run "$node" "[[ -d '${DATASET_DIR}' ]]"; then
        log_error "❌ 节点 ${node} 上 DATASET_DIR 不存在: ${DATASET_DIR}"
        return 1
    fi

    log_info "✅ 节点 ${node} 上的 PROJECT_DIR 和 DATASET_DIR 检查通过"
    return 0
}


# 优雅清理所有资源并退出
# Args:
#   $1: exit_code (int, optional) - 退出码，默认为最后一次命令的退出码
# Returns:
#   None (退出脚本)
cleanup_and_exit() {
    # 如果没有传递退出码，使用上一个命令的退出码
    local exit_code="${1:-$?}"
    # cleanup_and_exit 自己会 exit；先移除 trap，避免 exit 再次触发同一
    # 清理流程，造成重复的远程 SSH 停服往返。
    trap - EXIT TERM INT

    log_info "开始清理资源..."

    # 停止所有服务, but preserve the original failure code and always release
    # the lock even when a remote cleanup command fails.
    local cleanup_failed=0
    if ! stop_services; then
        cleanup_failed=1
    fi

    # 释放文件锁
    release_lock

    # 如果是调试模式，关闭它
    [[ "${DEBUG:-0}" == "1" ]] && set +x

    if [[ "$cleanup_failed" -eq 1 && "$exit_code" -eq 0 ]]; then
        exit_code=1
    fi
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
    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

    for dir in "${required_dirs[@]}"; do
        check_permissions "$dir"
    done

    # 验证数值参数范围
    # 参数名: 最小值: 最大值: 描述
    local param_checks=(
        "TENSOR_PARALLEL_SIZE:1:8:GPU数量"
        "INSTANCES_PER_NODE:1:8:每节点实例数"
        "N_SAMPLES:1:100:采样次数"
        "MAX_NUM_SEQS:1:16384:并发序列数"
        "MAX_NUM_BATCHED_TOKENS:512:1048576:批处理Token数"
    )

    for check in "${param_checks[@]}"; do
        # Bash 技巧: 使用 IFS 拆分字符串
        IFS=':' read -r param min max desc <<< "$check"
        local -n value_ref="$param"
        local value="$value_ref"
        if ! [[ "$value" =~ ^[0-9]+$ ]]; then
            handle_error 1 "$desc ($param) 必须是整数，当前值: $value"
        fi
        if [[ $value -lt $min || $value -gt $max ]]; then
            handle_error 1 "$desc ($param) 需在 $min-$max 之间，当前值: $value"
        fi
    done

    # 张量并行度仅支持 1/2/4/8
    local -a supported_tp_sizes=(1 2 4 8)
    local tp_valid=0
    for tp_size in "${supported_tp_sizes[@]}"; do
        if [[ "$TENSOR_PARALLEL_SIZE" -eq "$tp_size" ]]; then
            tp_valid=1
            break
        fi
    done
    if [[ $tp_valid -ne 1 ]]; then
        handle_error 1 "该脚本仅支持 TENSOR_PARALLEL_SIZE ∈ {${supported_tp_sizes[*]}}，当前为 ${TENSOR_PARALLEL_SIZE}"
    fi

    # 验证 TENSOR_PARALLEL_SIZE * INSTANCES_PER_NODE = 8
    local total_devices=$((TENSOR_PARALLEL_SIZE * INSTANCES_PER_NODE))
    if [[ $total_devices -ne 8 ]]; then
        handle_error 1 "TENSOR_PARALLEL_SIZE (${TENSOR_PARALLEL_SIZE}) * INSTANCES_PER_NODE (${INSTANCES_PER_NODE}) 必须等于 8，当前乘积为: ${total_devices}"
    fi

    # 验证浮点数参数，不依赖额外的 bc 命令。
    if ! awk -v value="${MEMORY_UTILIZATION}" \
        'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value >= 0.1 && value <= 1.0) }'; then
        handle_error 1 "显存利用率需在 0.1-1.0 之间，当前值: ${MEMORY_UTILIZATION}"
    fi

    log_info "✅ 配置参数验证通过"
}

# =======================================================
#                  核心功能函数区域
# =======================================================

# 由端口列表构建匹配本任务 vLLM 进程的 ERE 模式（供 pkill/pgrep -f 使用）
# 带端口边界 ([[:space:]]|$)，避免 "--port 6000" 误匹配 "--port 60001" 等前缀端口
# Args:
#   $@: ports (int...) - 本任务使用的服务端口列表
# Returns:
#   打印匹配模式字符串
build_vllm_kill_pattern() {
    local port_regex
    port_regex=$(IFS='|'; echo "$*")
    echo "vllm serve.*--port (${port_regex})([[:space:]]|\$)"
}

# 获取指定节点上本任务使用的端口列表（按全局 NODES/PORTS 下标对齐规则）
# Args:
#   $1: node (string) - 节点地址
# Returns:
#   打印该节点的端口列表（空格分隔），未找到时输出为空
get_node_ports() {
    local node="$1"
    local -a node_ports=()
    if [[ ${#PORTS[@]} -gt 0 ]]; then
        for ((i = 0; i < ${#NODES[@]}; i++)); do
            if [[ "${NODES[i]}" == "$node" ]]; then
                for ((instance_idx = 0; instance_idx < INSTANCES_PER_NODE; instance_idx++)); do
                    local port_idx=$((i * INSTANCES_PER_NODE + instance_idx))
                    node_ports+=("${PORTS[port_idx]}")
                done
                break
            fi
        done
    fi
    if [[ ${#node_ports[@]} -gt 0 ]]; then
        echo "${node_ports[@]}"
    else
        echo ""
    fi
}

# 停止指定节点上的 vLLM 服务
# Args:
#   $1: node (string) - 节点地址
#   $2: port (int, optional) - 服务端口（可选，用于精确停止特定端口的服务）
# Returns:
#   0: 成功，1: 失败
stop_service_on_node() {
    local node="$1"
    local port="${2:-}"

    log_info "🛑 正在停止节点 ${node} 上的 vLLM 服务..."

    # 仅匹配本任务使用的端口，避免裸 'vllm serve' 模式误杀节点上其他任务/用户的 vLLM 进程
    local -a ports=()
    if [[ -n "$port" ]]; then
        ports=("$port")
    else
        # 未指定端口时，回退到该节点在全局端口列表中的所有端口
        read -r -a ports <<< "$(get_node_ports "$node")" || true
    fi

    if [[ ${#ports[@]} -eq 0 ]]; then
        log_warn "⚠️ 无法确定节点 ${node} 上本任务使用的端口，跳过清理以避免误杀其他 vLLM 进程"
        return 0
    fi

    local search_pattern
    search_pattern=$(build_vllm_kill_pattern "${ports[@]}")

    # 优雅关闭：先发送 SIGTERM
    if ! ssh_run "$node" "pkill -f '${search_pattern}' || true"; then
        log_error "❌ 节点 ${node} 上的 vLLM 进程停止命令发送失败"
        return 1
    fi

    # 等待进程优雅退出
    sleep 2

    # 检查进程是否已停止
    local remaining
    if ! remaining=$(ssh_run "$node" "pgrep -f '${search_pattern}' | wc -l" 2>/dev/null); then
        log_error "❌ 节点 ${node} 进程状态查询失败，停止流程"
        return 1
    fi

    if [[ "${remaining:-0}" -gt 0 ]]; then
        log_warn "⚠️ 节点 ${node} 上仍有 ${remaining} 个 vLLM 进程，尝试强制终止..."
        ssh_run "$node" "pkill -9 -f '${search_pattern}' || true"
        sleep 1
    fi

    log_info "✅ 节点 ${node} 上的 vLLM 服务已停止"
    return 0
}



# 停止所有远程节点上的模型服务
# Args:
#   None
# Returns:
#   None
stop_services() {
    log_info "🛑 脚本退出，正在停止所有远程模型服务..."

    local pids=()

    # 遍历当前已知的节点列表 (可能已被 main 函数更新为 available_nodes),
    # 每个节点在并行子 shell 中调用 stop_service_on_node,由该函数统一负责
    # 端口定位(get_node_ports)与优雅/强制终止。
    for ((i = 0; i < ${#NODES[@]}; i++)); do
        local node="${NODES[i]}"
        (
            stop_service_on_node "$node"
        ) &
        pids+=($!)
    done

    # 等待所有停止操作完成
    if [[ ${#pids[@]} -gt 0 ]]; then
        log_info "⏳ 等待所有节点服务停止..."
        if ! wait_for_pids "${pids[@]}"; then
            log_error "部分节点服务清理失败"
            return 1
        fi
    fi
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
        # 使用带端口边界的共享模式，避免 6000 误匹配 60001。
        local search_pattern
        search_pattern=$(build_vllm_kill_pattern "$port")
        ssh_run "$node" "pkill -f '${search_pattern}' || true" >/dev/null 2>&1 || true
        sleep 1
    fi
}


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

    # out 为空时直接判失败：printf "%s\n" "" 也会输出一个空行，
    # 使 mapfile 得到含空元素的单元素数组，绕过下方"未发现文件"检查
    if [[ -z "$out" ]]; then
        log_error "❌ 未发现任何匹配的数据文件 (模式: ${DATASET_GLOB})，请检查 ${DATASET_DIR} 和 ${DATASET_GLOB} 配置"
        exit 1
    fi

    # 将结果存储到全局数组 FILES（过滤可能的空行）
    # Bash 技巧: mapfile -t < <(...) 避免创建 subshell 导致变量无法修改
    mapfile -t FILES < <(printf "%s\n" "$out" | grep -v -e '^[[:space:]]*$' || true)

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

        # 首先验证节点上的 PROJECT_DIR 和 DATASET_DIR 是否存在
        if ! validate_node_directories "$node"; then
            exit 1
        fi

        # 确保单节点资源满足 INSTANCES_PER_NODE * TENSOR_PARALLEL_SIZE 的部署要求
        verify_node_device_capacity "$node"

        # 创建目录，清理旧的状态/日志文件
        local prep_cmd="mkdir -p '${OUTPUT_DIR}' '${DATASET_DIR}' '${LOG_DIR}' && \
            rm -rf '${LOG_DIR}/status' && mkdir -p '${LOG_DIR}/status' && \
            (rm -f '${LOG_DIR}/${API_SERVER_LOG_PREFIX}'*.log '${LOG_DIR}/${TASK_LOG_PREFIX}'*.log 2>/dev/null || true)"

        if ! ssh_run "$node" "$prep_cmd"; then
            log_error "❌ 无法在节点 ${node} 上准备目录，请检查SSH连接和权限"
                exit 1 # 直接终止整个部署流程
        fi
    done

    log_info "✅ 所有远程目录已就绪，旧日志已清理"
}

# 在指定节点部署 vLLM 模型服务
# 功能: 在远程节点上启动 vLLM 模型服务实例
# 参数:
#   $1: 节点地址 - 远程服务器的域名或IP
#   $2: 服务端口 - 服务监听的端口号
#   $3: 实例ID - 实例编号 (0-based)
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
    local devices
    devices=$(get_device_visibility "$instance_id")

    log_info "🚀 在节点 ${node} 上部署模型服务实例 ${instance_id}，端口 ${port} (TP=${TENSOR_PARALLEL_SIZE}, GPUs=${devices}, mem_util=${MEMORY_UTILIZATION})"    # 1. 节点连通性验证

    # 1. 节点连通性验证
    if ! validate_node "$node"; then
        return 1
    fi

    # 2. 检查并清理旧端口占用
    log_info "🔍 检查节点 ${node} 端口 ${port} 占用情况"
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
    # DISABLE_LOG_REQUESTS 为 True/1/true 时追加 --disable-log-requests
    local log_requests_flag=""
    case "${DISABLE_LOG_REQUESTS}" in
        1 | [Tt][Rr][Uu][Ee]) log_requests_flag="--disable-log-requests" ;;
    esac
    local vllm_cmd="cd '${PROJECT_DIR}' && \
        source '${SET_ENV_SCRIPT}' && \
        export ASCEND_RT_VISIBLE_DEVICES='${devices}' && \
        nohup vllm serve \
            --model '${MODEL_PATH}' \
            --trust-remote-code \
            --enforce-eager \
            --served-model-name '${SERVED_MODEL_NAME}' \
            --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
            --gpu-memory-utilization ${MEMORY_UTILIZATION} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-num-seqs ${MAX_NUM_SEQS} \
            --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
            ${log_requests_flag} \
            ${EXTRA_ENGINE_ARGS} \
            --port ${port} \
            > '${log_file}' 2>&1 &"

    # 4. 在后台启动服务
    log_info "🔄 执行部署命令到节点 ${node}, 实例 ${instance_id}, 端口 ${port}"
    ssh_run "$node" "$vllm_cmd" &
    log_info "✅ 节点 ${node} vllm 模型部署启动命令发送成功"
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
    local instance_id="$3"
    local log_file="${LOG_DIR}/${API_SERVER_LOG_PREFIX}${node//./_}_${instance_id}.log"
    local base_url="http://127.0.0.1:${port}"
    local http_status models_status

    log_info "🔍 检查节点 ${node}  (端口: ${port}) 上 vllm 模型部署状态"
    # 检查日志文件是否存在
    if ! ssh_run "$node" "[[ -f '${log_file}' ]]"; then
        log_warn "⚠️ 节点 ${node}/port${port}/instance-${instance_id} 的日志文件尚未创建: ${log_file}"
        return 1
    fi

    # 1. 检查服务进程是否存在（使用带端口边界的共享模式，避免 6000 误匹配 60001）
    local search_pattern
    search_pattern=$(build_vllm_kill_pattern "$port")
    if ! ssh_run "$node" "pgrep -f '${search_pattern}' > /dev/null"; then
        log_warn "⚠️ 节点 ${node}/port${port}/instance-${instance_id} 上的服务进程未运行或已退出"
        return 1
    fi

    # 2. 尝试 HTTP 健康检查 (/health)
    http_status=$(ssh_run "$node" "curl -s -o /dev/null -w '%{http_code}' --max-time ${HEALTH_TIMEOUT} \
        ${base_url}${HEALTH_PATH} 2>/dev/null || echo 0")

    if [[ $http_status -eq 200 ]]; then
        log_info "✅ 服务 ${node}/port${port}/instance-${instance_id} 健康检查 (${HEALTH_PATH}) 通过"
        return 0
    fi

    # 3. 兼容性检查：尝试 /v1/models (vLLM OpenAI 兼容层标准)
    models_status=$(ssh_run "$node" "curl -s -o /dev/null -w '%{http_code}' --max-time ${HEALTH_TIMEOUT} \
        ${base_url}/v1/models 2>/dev/null || echo 0")

    if [[ $models_status -eq 200 ]]; then
        log_info "✅ 服务 ${node}/port${port}/instance-${instance_id} /v1/models 接口检查通过"
        return 0
    fi

    # 4. 日志回退检查：查找启动完成标志
    if ssh_run "$node" "grep -q 'Application startup complete' '${log_file}' 2>/dev/null"; then
        log_info "✅ 服务 ${node}/port${port}/instance-${instance_id}  日志检测到 [Application startup complete] 标志, vllm 启动完成"
            return 0
        fi
    log_warn "⚠️ 节点 ${node}/port${port}/instance-${instance_id} 的 vllm 服务启动未完成 (HTTP状态码: ${http_status}/${models_status})，日志中未找到启动完成标志"
    return 1
}

# 轮询检查所有模型服务是否启动成功
# 参数：无
# 返回值：就绪节点的索引数组
wait_for_services() {
    log_info "⏳ 正在等待所有模型服务启动并就绪... 最长等待 ${MAX_WAIT_TIME} 秒"

    local total_wait_time=0
    local interval=10
    local total_nodes=${#NODES[@]}
    local total_instances=$((total_nodes * INSTANCES_PER_NODE))
    local total_services=$total_instances
    local status_dir="${LOG_DIR}/status"

    # 确保状态目录干净
    rm -rf "${status_dir}" || true
    mkdir -p "${status_dir}"

    while [[ $total_wait_time -lt $MAX_WAIT_TIME ]]; do
        local running_pids=()

        # 并行检查所有节点的所有实例状态
        for ((i = 0; i < total_nodes; i++)); do
            local node="${NODES[i]}"
            for ((instance_idx = 0; instance_idx < INSTANCES_PER_NODE; instance_idx++)); do
                local port_idx=$((i * INSTANCES_PER_NODE + instance_idx))
                local port="${PORTS[port_idx]}"
                local status_file="${status_dir}/status_${node//./_}_${instance_idx}.ok"

                # 跳过已就绪的服务实例
                if [[ -f "$status_file" ]]; then
                    continue
                fi

                # 后台检查服务状态
                (
                    if check_service_ready "$node" "$port" "$instance_idx" ; then
                        touch "$status_file"
                    fi
                ) &
                running_pids+=($!)
            done
        done

        # 等待所有检查完成
        if [[ ${#running_pids[@]} -gt 0 ]]; then
            wait "${running_pids[@]}" || true
        fi

        # 统计已就绪服务数量
        local ready_count
        ready_count=$(find "${status_dir}" -maxdepth 1 -type f -name '*.ok' | wc -l | tr -d ' ')

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

    # 销毁并初始化实例分配数组（nameref 替代 eval，避免文件名注入）
    for ((i = 0; i < total_instances; i++)); do
        local -n _assign_init="INSTANCE_ASSIGNMENTS_$i"
        _assign_init=()
    done

    # 轮询分配文件
    for idx in "${!FILES[@]}"; do
        local file="${FILES[idx]}"
        local instance_idx=$((idx % total_instances))
        local -n _assign_files="INSTANCE_ASSIGNMENTS_${instance_idx}"
        _assign_files+=("$file")
        log_info "分配文件: ${file} -> 实例 ${instance_idx}"
    done

    # 打印分配结果统计
    for ((i = 0; i < total_instances; i++)); do
        local -n _assign_cnt="INSTANCE_ASSIGNMENTS_$i"
        local count="${#_assign_cnt[@]}"
        log_info "实例 ${i} 分配 ${count} 个文件"
    done

    log_info "✅ 数据文件分配完成"
}

# 在指定节点上批量提交推理任务，包含重试和资源控制机制
# Args:
#   $1: node (string) - 节点地址
#   $2: port (int) - 服务端口
#   $3: model_name (string) - 模型名称
#   $4: base_url (string) - 服务 URL (如 http://127.0.0.1:port/v1)
#   $@: files (string array) - 分配给该节点的全部文件列表
# Returns:
#   None (提交后会调用 wait_for_batch_completion_and_cleanup 等待本节点任务完成)
run_task_batch_parallel() {
    local node="$1"
    local port="$2"
    local model_name="$3"
    local base_url="$4"
    local instance_idx="$5"
    shift 5
    local files=("$@")
    local node_tag="${node//./_}"
    local status_dir="${LOG_DIR}/task_status_${RUN_ID}_${node_tag}_${instance_idx}"

    log_info "👉 在节点 ${node}, instance ${instance_idx} 上启动 ${#files[@]} 个推理任务..."

    # 检查是否有文件需要处理
    if [[ ${#files[@]} -eq 0 ]]; then
        log_warn "节点 ${node} 没有分配到任何文件，跳过任务启动"
        return 0
    fi

    # 构建所有文件的推理命令并一次性发送
    if ! ssh_run "$node" "rm -rf '${status_dir}' && mkdir -p '${status_dir}'"; then
        log_error "节点 ${node} 无法创建任务状态目录: ${status_dir}"
        return 1
    fi

    local commands=()
    local task_index=0
    for file in "${files[@]}"; do
        local input_file="${DATASET_DIR}/${file}"
        # 移除文件扩展名
        local base_name="${file%.*}"
        local output_file="${OUTPUT_DIR}/infer_${model_name//\//_}_${base_name}_bz${N_SAMPLES}.jsonl"
        local log_file="${LOG_DIR}/${TASK_LOG_PREFIX}${node//./_}_${instance_idx}_${base_name}.log"
        local status_file="${status_dir}/${task_index}.status"

        log_info "  -> 准备处理文件: ${file} (输出: ${output_file})"

        # 检查输入文件是否存在
        if ! ssh_run "$node" "test -f '${input_file}'" >/dev/null 2>&1; then
            log_error "❌ 输入文件 ${input_file} 在节点 ${node} 上不存在"
            return 1
        fi

        # 构建推理命令
        local job_cmd="if cd '${PROJECT_DIR}' && \
            source '${SET_ENV_SCRIPT}' && \
            python '${INFER_SCRIPT}' \
                --input_file '${input_file}' \
                --output_file '${output_file}' \
                --input_key '${INPUT_KEY}' \
                --base_url '${base_url}' \
                --model_name '${model_name}' \
                --n_samples ${N_SAMPLES} \
                --system_prompt_type '${SYSTEM_PROMPT_TYPE}' \
                --max_workers ${MAX_WORKERS} \
                > '${log_file}' 2>&1; then \
            rc=0; \
        else \
            rc=\$?; \
        fi; \
            printf '%s\n' \"\$rc\" > '${status_file}.tmp'; \
            mv '${status_file}.tmp' '${status_file}'; \
            exit \"\$rc\""

        local quoted_job_cmd
        printf -v quoted_job_cmd '%q' "$job_cmd"
        local infer_cmd="nohup bash -lc ${quoted_job_cmd} >/dev/null 2>&1 &"

        commands+=("$infer_cmd")
        task_index=$((task_index + 1))
    done

    # 将所有命令组合成一个命令字符串并执行
    if [[ ${#commands[@]} -gt 0 ]]; then
        local combined_cmd
        combined_cmd=$(printf "%s " "${commands[@]}")
        log_info "🚀 节点 ${node}: ${instance_idx}) .. 提交  ${#commands[@]} 个 OpenAI API Server 推理任务..."
        # 提交失败（SSH 异常）时明确报错退出，而不是静默继续：
        # 否则后续等待循环会因节点上没有任务而误判“任务已完成”
        if ! ssh_run "$node" "$combined_cmd" >/dev/null 2>&1; then
            log_error "❌ 节点 ${node} 推理任务提交失败（SSH 异常），请检查节点连通性后重试"
            return 1
        fi
        # 添加一个小延迟以确保任务正确启动
        sleep 2
    else
        log_error "节点 ${node} 分配到的输入文件均不可用，无法启动推理任务"
        return 1
    fi
    # 等待任务完成
    wait_for_batch_completion_and_cleanup \
        "$node" "$port" "${#commands[@]}" "$status_dir"
}

# 监控单节点推理任务并在完成后清理服务
wait_for_batch_completion_and_cleanup() {
    local node="$1"
    local port="$2"
    local expected_count="$3"
    local status_dir="$4"
    local max_wait_time=864000  # 最大等待时间（秒）(10天)
    local wait_interval=30      # 检查间隔（秒）
    local total_wait_time=0

    log_info "⏳ 等待节点 ${node} 上的 ${expected_count} 个任务完成..."

    while [[ $total_wait_time -lt $max_wait_time ]]; do
        local completed_count
        if ! completed_count=$(ssh_run "$node" \
            "test -d '${status_dir}' && find '${status_dir}' -maxdepth 1 -type f -name '*.status' | wc -l" \
            2>/dev/null); then
            log_warn "⚠️ 节点 ${node} 状态检查失败（SSH 异常，可能是网络抖动），跳过本轮检查，继续等待..."
            sleep "$wait_interval"
            total_wait_time=$((total_wait_time + wait_interval))
            continue
        fi

        if ! [[ "$completed_count" =~ ^[0-9]+$ ]]; then
            log_warn "节点 ${node} 返回无效任务状态计数: ${completed_count}"
            sleep "$wait_interval"
            total_wait_time=$((total_wait_time + wait_interval))
            continue
        fi

        if [[ $completed_count -eq $expected_count ]]; then
            local failed_count
            if ! failed_count=$(ssh_run "$node" \
                "awk '\$1 != 0 { failures++ } END { print failures + 0 }' '${status_dir}'/*.status" \
                2>/dev/null); then
                log_error "节点 ${node} 无法读取任务退出状态"
                stop_service_on_node "$node" "$port"
                return 1
            fi

            log_info "节点 ${node} 上的 ${expected_count} 个推理任务已结束，正在清理资源..."
            stop_service_on_node "$node" "$port"
            if [[ ! "$failed_count" =~ ^[0-9]+$ || "$failed_count" -ne 0 ]]; then
                log_error "节点 ${node} 有 ${failed_count} 个推理任务执行失败"
                return 1
            fi
            log_info "✅ 节点 ${node} 上的推理任务全部成功"
            return 0
        fi

        log_info "⏳ 节点 ${node} 已完成 ${completed_count}/${expected_count} 个任务，已等待 ${total_wait_time} 秒"
        sleep "$wait_interval"
        total_wait_time=$((total_wait_time + wait_interval))
    done

    log_warn "⏰ 等待超时，节点 ${node} 上的任务可能仍在运行，已等待 ${total_wait_time} 秒"
    # 即使超时，仍然尝试停止服务
    log_warn "正在强制停止节点 ${node} 上的 vLLM 服务..."
    stop_service_on_node "$node" "$port"
    return 1
}

# 分发并启动所有推理任务
# Args:
#   None
# Returns:
#   None
distribute_and_launch_jobs() {
    local total_instances=${#READY_INSTANCE_PORTS[@]}

    if [[ $total_instances -eq 0 ]]; then
        handle_error 1 "没有可用实例可供执行推理任务"
    fi

    log_info "开始分发并启动推理任务 (实例总数: ${total_instances})..."

    # 1. 分配数据文件到可用实例
    assign_data_to_instances "$total_instances"

    # 2. 为每个节点启动对应的推理任务（并行）
    local pids=()
    for ((i = 0; i < total_instances; i++)); do
        local node="${READY_INSTANCE_NODES[i]}"
        local port="${READY_INSTANCE_PORTS[i]}"
        local instance_idx="${READY_INSTANCE_IDS[i]}"
        # 注意: vLLM OpenAI 兼容层 API 通常在 /v1 路径下
        local base_url="http://127.0.0.1:${port}/v1"
        local model_name="${SERVED_MODEL_NAME}"

        # 获取分配给当前实例的文件列表
        local instance_files_var="INSTANCE_ASSIGNMENTS_$i"
        local -n instance_files_ref="$instance_files_var"

            # 检查文件是否分配 (如果 assign_data_to_instances 中有节点没有分配到文件，这里跳过)
        if [[ ${#instance_files_ref[@]} -eq 0 ]]; then
            log_info "节点 ${node} 实例 ${instance_idx} (端口 ${port}) 未分配到文件，跳过"
            continue
        fi
        # 获取分配给当前实例的文件列表
        log_info "节点 ${node} 实例 ${instance_idx} (端口 ${port}) 分配到 ${#instance_files_ref[@]} 个文件"
        # 在后台启动任务提交批次
        (
            run_task_batch_parallel "$node" "$port" "$model_name" "$base_url" "$instance_idx" "${instance_files_ref[@]}"
        ) &
        pids+=($!)
    done

    # 3. 等待所有节点的任务批次完成（run_task_batch_parallel 内部会等待
    #    远端推理完成并清理对应实例的服务）；任一节点失败则整体失败
    if [[ ${#pids[@]} -gt 0 ]]; then
        if ! wait_for_pids "${pids[@]}"; then
            handle_error 1 "部分节点的推理任务执行失败，请检查各节点任务日志: ${LOG_DIR}"
        fi
    fi
    log_info "✅ 所有推理任务已启动，进入远端任务监控阶段, 请查看推理结果: ${OUTPUT_DIR}"
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
    log_info " 开始执行分布式 vLLM 模型推理部署"
    echo "================================================"

    # 设置退出时的清理陷阱 (最先设置，确保任何失败都能调用清理)
    trap 'cleanup_and_exit' EXIT TERM INT

    # 验证配置参数
    validate_config

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

    # 自动生成端口列表（节点间间隔 100 端口、实例间间隔 20 端口），存入全局 PORTS
    PORTS=()
    local start_port=6000
    for ((i=0; i<${#NODES[@]}; i++)); do
        for ((instance_idx=0; instance_idx<INSTANCES_PER_NODE; instance_idx++)); do
            # Increase port spacing to avoid conflicts
            PORTS+=($((start_port + i * 100 + instance_idx * 20)))  # 每节点间隔100，实例间隔20
        done
    done
    log_info "自动生成端口列表: ${PORTS[*]}"

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
        for ((instance_idx = 0; instance_idx < INSTANCES_PER_NODE; instance_idx++)); do
            local port_idx=$((i * INSTANCES_PER_NODE + instance_idx))
            local port="${PORTS[port_idx]}"
            # 在本地后台部署，加速并发
            deploy_model_service "$node" "$port" "$instance_idx" &
        done
    done

    # 等待所有部署命令发送完成 (即使失败，deploy_model_service 也会返回)
    wait || true

    # 步骤5: 等待服务就绪并获取可用节点（HTTP 健康检查 + 日志回退）
    wait_for_services

    # 不再使用 wait_for_services 的返回值，而是主动检查所有节点状态
    log_info "正在检查各节点服务状态..."

    # 初始化可用节点和失败节点列表
    READY_INSTANCE_NODES=()
    READY_INSTANCE_PORTS=()
    READY_INSTANCE_IDS=()
    local -a failed_nodes=()
    local -a failed_ports=()
    local -a failed_instance_ids=()

    # 检查每个节点的状态
    local configured_node_count=${#NODES[@]}
    local total_services_expected=$(( configured_node_count * INSTANCES_PER_NODE ))
    for ((i = 0; i < ${#NODES[@]}; i++)); do
        local node="${NODES[i]}"
        for ((instance_idx = 0; instance_idx < INSTANCES_PER_NODE; instance_idx++)); do
            local port_idx=$((i * INSTANCES_PER_NODE + instance_idx))
            local port="${PORTS[port_idx]}"
            # 获取节点实例的 API 服务状态文件
            local status_file="${LOG_DIR}/status/status_${node//./_}_${instance_idx}.ok"

            if [[ -f "$status_file" ]]; then
                log_info "✅ 节点 ${node} 实例 ${instance_idx} (端口: ${port}) 服务就绪"
                READY_INSTANCE_NODES+=("${node}")
                READY_INSTANCE_PORTS+=("${port}")
                READY_INSTANCE_IDS+=("${instance_idx}")
            else
                log_warn "❌ 节点 ${node} 实例 ${instance_idx} (端口: ${port}) 服务未就绪"
                failed_nodes+=("${node}")
                failed_ports+=("${port}")
                failed_instance_ids+=("${instance_idx}")
            fi
        done
    done

    # 输出部署结果统计
    log_info "📊 服务部署结果统计:"
    local ready_instance_count=${#READY_INSTANCE_PORTS[@]}
    local ready_node_count=0
    if [[ $ready_instance_count -gt 0 ]]; then
        mapfile -t ready_nodes < <(printf "%s\n" "${READY_INSTANCE_NODES[@]}" | sort -u)
        ready_node_count=${#ready_nodes[@]}
    else
        # 如果没有可用节点，确保数组为空
        ready_nodes=()
    fi
    log_info "   - 成功实例数量: ${ready_instance_count}/${total_services_expected}"
    log_info "   - 可用节点数量: ${ready_node_count}/${configured_node_count}"

    if [[ ${#failed_nodes[@]} -gt 0 ]]; then
        log_warn "以下实例未能成功部署:"
        for ((i = 0; i < ${#failed_nodes[@]}; i++)); do
            log_warn "   - ${failed_nodes[i]} instance ${failed_instance_ids[i]} (端口: ${failed_ports[i]})"
        done
        log_warn "请检查这些节点的日志文件: ${LOG_DIR}/${API_SERVER_LOG_PREFIX}<节点名>.log"
    fi

    # 检查是否有可用节点
    if [[ ${#READY_INSTANCE_PORTS[@]} -eq 0 ]]; then
        handle_error 1 "❌ 没有任何节点成功启动服务，无法继续执行推理任务"
    fi

    local actual_total_instances=${#READY_INSTANCE_PORTS[@]}
    log_info "将使用 ${actual_total_instances} 个可用实例 (覆盖 ${ready_node_count} 个节点) 进行推理"

    # 步骤6: 使用可用节点分发并启动推理任务（内部等待远端推理完成并清理对应实例服务）
    distribute_and_launch_jobs

    # 步骤7: 优雅关闭服务（由 EXIT 陷阱调用 stop_services）
    log_info "✅ 分布式推理部署和任务执行完成，正在退出并清理资源..."

    log_info "📊 部署统计:"
    log_info "   - 成功实例: ${ready_instance_count}/${total_services_expected}"
    log_info "   - 覆盖节点: ${ready_node_count}/${configured_node_count}"
    log_info "   - 数据文件: ${#FILES[@]}"
    log_info "   - 输出目录: ${OUTPUT_DIR}"
    log_info "   - 日志目录: ${LOG_DIR}"
    echo "================================================"
}
