"""Static regressions for the distributed inference shell runner."""

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


def test_remote_process_checks_do_not_match_their_wrapper() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "[p]ython" in source
    assert "pgrep -f '${INFER_SCRIPT}'" not in source


def test_remote_setup_and_cleanup_are_failure_safe() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'ssh-keyscan -H "$node"' not in source
    assert 'search_pattern=$(build_vllm_kill_pattern "$port")' in source
    assert "trap - EXIT TERM INT" in source


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
    assert 'stop_service_on_node "$node" "$port"\n    return 1' in source


def test_failed_monitor_terminates_remaining_monitors() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'kill "$remaining"' in source
    assert 'wait "$remaining"' in source
