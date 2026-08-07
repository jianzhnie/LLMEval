---
Bug 与优化排查报告(不改代码,仅报告)

基于 4 个审查 agent + 我的独立验证。当前基线:428 测试全绿(你 IDE 的 config 改动加了新测试)。注:审查过程中 HEAD 从 f06dd41 移到 0583c38(你的并发提交),我验证的是当前 HEAD 状态。

🔴 高严重度

BUG-1 — shell wait_for_batch_completion_and_cleanup 永远检测不到任务结束,脚本挂 ~10 天
auto_model_infer_common.sh:1157:ssh_run "$node" "pgrep -f '${INFER_SCRIPT}' | wc -l"。远端 bash -c 包装进程的 cmdline 本身就含脚本路径,pgrep -f 恒≥1 → current_running_tasks -le 0 永不触发 → 所有推理结束后 vLLM 服务空转,直到 864000s(10天)上限才强制停止。修复:用 pgrep -f "[o]nline.py" 括号技巧或匹配 python argv。

BUG-2 — ssh-keyscan 在 set -e 下让首个不可达节点中止整个部署
auto_model_infer_common.sh:152:ssh-keyscan -H "$node" 无 || true,脚本有 set -euo pipefail(line 69)。节点不可达 → keyscan 退出码 1 → 脚本在进入"连接失败→跳过"优雅 fallback 前就 abort(触发 EXIT trap 清理)。与文档声称的"失败节点自动跳过"不符。

🟠 中严重度

BUG-3 — unsafe_execute 用 exec(code, {}),__name__ == "builtins" 而非 "__main__"
execute.py:422 + _worker 不传 exec_globals。任何用 if __name__ == "__main__": 门控测试的 harness 断言永远不执行,unsafe_execute 返回 ("passed","") → 假阳性正确分。内置 HumanEval/MBPP harness 顶层调用 check() 所以不受影响,但 extract_code 的 _STOP_MARKERS 又专门在 if __name__ 截断——埋了雷。修复:exec_globals 注入 {"__name__": "__main__"}。

BUG-4 — mc generate 模式 None gold 被计为错误而非跳过
mc_score.py:745:str(item.get(label_key, "")) → str(None) = "NONE"(truthy)→ 不 skipped,进分母拉低 acc。loglikelihood 模式(gold→-1→skipped)和 math 模式(label is None→skipped)都跳过。不一致:2 题含 1 个 null-gold 会报 0.5 而非 1.0。

BUG-5 — _sample_weight 对 per_sample 模式的下池超时/崩溃记录权重为 0,结构化计数看不见
mc_score.py:501-510 + _to_scorer_result:_error_record 产物无 sample_total,per_sample 模式 _sample_weight 返回 0 → ScorerResult.timeout_count/failed_count 恒 0。而 write_cache 的 summary 报 1。同一事件两个输出不一致(metric 本身不受影响)。

🟡 低严重度 / 优化建议

- BUG-6 check_remote_port_free 的 pkill -f 'vllm serve.*--port ${port}' 无端口边界,共享节点上 --port 6000 会误杀 --port 60001(脚本自己的 build_vllm_kill_pattern 已加边界,这处漏了)。
- BUG-7 MCLoglikelihoodClient.get_choices_logprobs 的宽 except Exception 吞掉非重试错误(401/编程错误)→ 全变成 -inf failed 行,root cause 只在 warning。与 retry.py 明确"re-raise 非 APIError"的意图相悖。
- BUG-8 code_score._score_items 串行路径无 per-item 异常隔离,并行路径有 → 同样数据 max_workers=1 时一个异常 abort 整个评分,max_workers>1 时被吸收。不一致。
- BUG-9 time_limit 一次性 ITIMER_REAL 可被 try/except BaseException 中和(SIGALRM 只在 Unix)。handle_error/cleanup_and_exit 在错误路径清理跑两遍(幂等但多一轮 ssh 往返)。
- 已知设计限制(非 bug):resume 的按行计数 completed_counts 假设已完成行是连续前缀,fail_fast=False/online 模式下失败样本会留空洞 → resume 重复尾部样本、丢失失败序号(你此前确认"sample_index 功能已删除"是有意设计)。
- 优化:pass@1 日志 correct/total 是 micro 计数但 pass@1 是 macro 平均,数字对不上(展示误导,metric 正确);is_chat_template_applied 的 Human:/Assistant: 正则对对话式 prompt 误判;non-ASCII 在 continuation 模式走 failed(影响小)。
