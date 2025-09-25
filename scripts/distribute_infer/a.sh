# 轮询检查所有模型服务是否启动成功
# 参数：无
# 返回值：就绪节点的索引数组
wait_for_services() {
    echo "⏳ 正在等待所有模型服务启动并就绪... 最长等待 ${MAX_WAIT_TIME} 秒"

    local total_wait_time=0
    local interval=5
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
                        echo "✅ 实例就绪: 节点 ${node} 实例 ${j} (端口 ${port})"
                    elif [[ $total_wait_time -gt $((MAX_WAIT_TIME / 2)) ]]; then
                        # 如果等待时间已过半，标记为失败
                        touch "$fail_file"
                        echo "❌ 实例失败: 节点 ${node} 实例 ${j} (端口 ${port}) - 超时或启动失败"
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
                    node_instance_info+=("✅实例${j}(端口:${port})")
                elif [[ -f "$fail_file" ]]; then
                    failed_instances["${node}:${j}"]="${port}"
                    node_failed_count=$((node_failed_count + 1))
                    failed_instance_count=$((failed_instance_count + 1))
                    node_instance_info+=("❌实例${j}(端口:${port})")
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
            echo "✅ 所有 ${total_nodes} 个节点的 ${total_instances} 个服务实例已就绪"
            echo "${ready_indices[@]}"
            return 0
        fi

        # 显示进度
        local pending_instances=$((total_instances - ready_instances - failed_instance_count))
        echo "   -> 就绪: ${ready_instances}, 失败: ${failed_instance_count}, 等待: ${pending_instances}/${total_instances} 服务，已等待: ${total_wait_time}s"

        # 如果等待时间过长，提前退出
        if [[ $total_wait_time -gt $((MAX_WAIT_TIME * 0.8)) ]] && [[ $pending_instances -gt 0 ]]; then
            echo "⚠️  接近最大等待时间，部分实例仍未就绪"
        fi

        sleep "$interval"
        total_wait_time=$((total_wait_time + interval))
    done

    # 超时后收集最终状态
    echo "⏰ 等待超时，收集最终部署状态..."

    # 显示已完成的实例
    if [[ ${#completed_instances[@]} -gt 0 ]]; then
        echo "✅ 已完成部署的实例 (${#completed_instances[@]} 个):"
        for instance in "${!completed_instances[@]}"; do
            echo "   - ${instance} (端口: ${completed_instances[$instance]})"
        done
    fi

    # 显示失败的实例
    if [[ ${#failed_instances[@]} -gt 0 ]]; then
        echo "❌ 部署失败的实例 (${#failed_instances[@]} 个):"
        for instance in "${!failed_instances[@]}"; do
            echo "   - ${instance} (端口: ${failed_instances[$instance]})"
        done
    fi

    if [[ ${#ready_indices[@]} -gt 0 ]]; then
        echo "⚠️ 超时但有 ${#ready_indices[@]} 个节点已就绪，将继续使用可用节点"
        echo "${ready_indices[@]}"
        return 0
    fi

    echo "❌ 错误: 没有任何节点成功启动，请检查远程日志" >&2
    exit 1
}
