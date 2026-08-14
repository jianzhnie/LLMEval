"""Static regressions for the distributed inference shell runner."""

import shlex
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data_parallel_infer"
    / "auto_model_infer_common.sh"
)


def test_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_remote_tasks_publish_and_check_exit_statuses() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "task_status_${RUN_ID}_${node_tag}_${instance_idx}" in source
    assert "${task_index}.status" in source
    assert "awk '\\$1 != 0" in source
    assert "if cd '${PROJECT_DIR}'" in source
    assert "rc=\\$?" in source
    assert "pgrep -f '${INFER_SCRIPT}'" not in source


def test_remote_setup_and_cleanup_are_failure_safe() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'ssh-keyscan -H "$node"' not in source
    assert 'search_pattern=$(build_vllm_kill_pattern "$port")' in source
    assert "trap - EXIT TERM INT" in source
    assert "2>/dev/null || true)" in source


def test_zero_match_counts_do_not_trip_errexit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "grep -c" not in source
    assert "awk '/Error/" in source


def test_stale_lock_is_not_automatically_deleted() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'rm -rf "$LOCK_DIR"' not in source
    assert "请确认没有部署进程运行后手动删除该目录" in source


def test_task_timeout_and_missing_inputs_fail_the_run() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'log_error "节点 ${node} 分配到的输入文件均不可用' in source
    assert (
        'log_error "❌ 输入文件 ${input_file} 在节点 ${node} 上不存在"\n'
        "            return 1"
    ) in source
    assert 'stop_service_on_node "$node" "$port"\n    return 1' in source


def test_failed_monitor_terminates_remaining_monitors() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'ps -p "$pid" -o stat=' in source
    assert 'kill "$remaining"' in source
    assert 'wait "$remaining"' in source


def test_wait_for_pids_handles_success_and_failure() -> None:
    command = f"""
        source {shlex.quote(str(SCRIPT))}
        (sleep 10) & long_pid=$!
        (exit 7) & failed_pid=$!
        if wait_for_pids "$long_pid" "$failed_pid"; then exit 1; fi
        if kill -0 "$long_pid" 2>/dev/null; then exit 2; fi
        (sleep 0.1) & first_pid=$!
        (sleep 0.2) & second_pid=$!
        wait_for_pids "$first_pid" "$second_pid"
    """
    subprocess.run(["bash", "-c", command], check=True, timeout=5)


def test_partial_service_startup_is_reported_as_degraded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'readonly FAIL_ON_DEGRADED="${FAIL_ON_DEGRADED:-0}"' in source
    assert 'run_status="DEGRADED"' in source
    assert 'log_warn "运行状态: ${run_status}' in source
    assert (
        'if [[ "$FAIL_ON_DEGRADED" == "1" && "$run_status" == "DEGRADED" ]]' in source
    )


def test_deployment_command_statuses_are_collected() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'deployment_pids+=("$!")' in source
    assert 'if wait "${deployment_pids[$i]}"' in source
    assert 'ssh_run "$node" "$vllm_cmd" &' not in source


def test_service_status_directory_cleanup_is_required() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'rm -rf "${status_dir}" || true' not in source
    assert 'if ! rm -rf "${status_dir}" || ! mkdir -p "${status_dir}"; then' in source
    assert 'handle_error 1 "❌ 无法重建服务状态目录: ${status_dir}"' in source
