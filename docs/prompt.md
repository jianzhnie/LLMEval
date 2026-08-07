
---
Review 报告

基线:485 passed 全绿。工作树干净,所有改动已提交。以下按严重度排序,分为 正确性 bug 和 优化项 两组。

🔴 正确性 Bug(建议优先修)

BUG-1 — MC loglikelihood resume 永不完成,每次重复请求 ⭐ 已验证

inference/common.py:270 + inference/mc.py:772-774
- 根因:MC loglikelihood 结果行把 -inf logprob 持久化为 None(注释明说 "persisted as null"),而 _is_completed_record 要求 all(isinstance(value, int|float)) → None 使检查失败,该行不算 completed。
- 后果:prepare_sample_requests 重新展开全部 n_samples → 每次 resume 都重新请求已完成的样本 → 输出文件无限增长、acc 分母膨胀。
- 修复:_is_completed_record 的 logprobs 检查应接受 None(即 isinstance(v, (int, float)) or v is None)。

BUG-2 — 推理空响应不落盘,resume 永远重试 ⭐ 已验证

inference/online.py:373-377
- 根因:get_content 把 content=None(reasoning 模型 max_tokens 耗尽在 thinking 上)归一为 "",_build_result 对空响应返回 None 且不写行(仅 failed++)。
- 后果:该 item 永不落盘 → 每次 resume 重新发同一请求 → 永久重试,failed 计数随运行增长。
- 对比:context-length 路径(process_item:402-412)会写永久失败行。此路径应同样处理。

BUG-3 — 并行 worker 崩溃被误标为 timeout(2 处)⭐ 已验证

math_score.py:647 / code_score.py:544
- 根因:pebble 迭代器抛 ProcessExpired(非 TimeoutError),被 except Exception 吞掉 → 索引缺失 → 兜底标为 timeout。
- 后果:同一 item 在 max_workers=1(串行→failed)和 max_workers>1(并行→timeout)下状态不同,分母统计不一致。崩溃≠超时,应区分。

BUG-4 — prep_cmd 尾部 || true 吞掉整个 mkdir 链失败

scripts/.../auto_model_infer_common.sh:859
- 根因:mkdir && rm && mkdir && rm -f ... || true 中 || true 覆盖整条链,远端 mkdir 失败被静默吞掉。
- 后果:目录没建好但脚本以为成功 → vLLM 写日志失败 → 服务起不来,报错点离根因很远。

BUG-5 — 单节点不可达中止整个部署(与"失败节点自动跳过"矛盾)

auto_model_infer_common.sh:850,201,813
- validate_node_directories 失败 → exit 1;verify_node_device_capacity 设备不足 → handle_error 退出;discover_remote_dataset_files 只探测 NODES[0],首节点不通即整体退出。
- 后果:一个死节点(ssh 超时 / 缺 npu-smi / 被其他任务占用)让所有健康节点一起作废。

BUG-6 — wait_for_pids 杀兄弟子 shell:清理漏执行 / 推理被中断

auto_model_infer_common.sh:370
- 清理中一个节点 ssh 失败 → 健康节点的 stop_service_on_node 子 shell 被 SIGTERM → vLLM 漏停(泄漏)。
- 任务中某节点失败 → 兄弟 10 天监控循环被杀 → cleanup 杀掉所有 vLLM 而远端 python 还在跑 → 结果丢失。

BUG-7 — 单死节点挂起整个运行最长 10 天 + 成功运行被翻成失败

auto_model_infer_common.sh:1195,524
- 推理中途节点宕机 → pgrep ssh 持续失败 → 每轮 sleep 600 continue → 864000 秒后失败。其它节点结果被阻塞。
- 清理时某节点 ssh 失败会把成功运行的 exit_code 从 0 改成 1,调度侧误判失败。

BUG-8 — 端口复用无跨运行协调:并发部署互杀 vLLM

auto_model_infer_common.sh:1333,774
- 两个部署覆盖同一节点同一端口(都是 6000)→ 后启动者的 check_remote_port_free pkill 掉先启动者的 vLLM(端口边界正则本身正确,但端口分配无互斥)。

🟠 中低严重度正确性

- BUG-9 retry.py:132 — context-length 若被网关包成 429(RateLimitError),is_context_length_error 分支被跳过 → 重试后 ClientError → 不落永久失败行 → resume 重试。
- BUG-10 online.py:443 — 空 prompt 的 skip 项被同时写进 *_failed.jsonl,audit 误标数据集问题为推理失败。
- BUG-11 execute.py:446 — __name__="__main__" 让残留的 if __name__ == "__main__": guard 在 harness 前执行,sys.exit() → "failed: SystemExit" 假阴性。
- BUG-12 check_remote_port_free:762 — ss/netstat/lsof 全缺或 ssh 失败时端口被当空闲 → 占用端口上启动 → bind 失败。
- BUG-13 DEVICE_COUNT_MULTIPLIER ×2:197 — 4 卡节点报告 4×2=8 通过校验 → 实例引用不存在的 4-7 卡 → vLLM 崩溃被静默跳过。

🟢 优化项(去冗余/删空壳)

OPT-1 — score_math_result 与 compute_scores 重复聚合 ⭐ 已验证

math_score.py:738-763 重新扫描 eval_dataset 算 observations/timeout/failed/skipped/failure_counts/wrong_answer,而 compute_scores 已算好 stats + failure_counts + eligible_accuracy。~30 行重复,可让 score_math_result 直接消费 compute_scores 的返回值。

OPT-2 — MCScoreResult.correct/.correct_norm/.correct_bytes 死字段 ⭐ 已验证

mc_score.py:131-135 字段在 build_result 写入但无任何消费者(_to_scorer_result/write_cache 只读 acc/acc_norm/acc_bytes/exact_match/total/per_item)。删除字段 + 写入。

OPT-3 — set_log_level 死函数

utils/log.py:92 — 生产零调用,仅测试用。其 level 归一逻辑已在 init_logger 内重复。

OPT-4 — write_cache.sample_count 与 _sample_weight 已漂移

mc_score.py:904 vs :519 — 同一记录可能在不同输出中权重不同(已漂移),应合并为一个共享 helper。

OPT-5 — load_data 三处复制粘贴

online.py:310 / offline.py:209 / mc.py:540 — ~35 行相同 resume/expansion 管线,可抽共享 helper。

OPT-6 — _process_concurrently 两处复制

online.py:424 / mc.py:604 — 并发 worker 循环几乎相同,可抽结果处理器参数化 helper。

OPT-7 — OpenAI client 构建三处复制

online.py:91 / mc.py:235 / mc.py:835 — api_key 解析 + 客户端实例化,可抽 build_openai_client。

▎ ⚠️ 注意:OPT-5/6/7 与我上轮你明确说"保持不变"的 4 项(OpenAI client×3、pool 循环×3、CLI×3、长函数)在 _process_concurrently/load_data/OpenAI client 上有重合。我已在上轮标记为不改。本报告仅重提验证过的新证据,不默认执行。







审查基于当前 HEAD=83dc2ae，工作区干净。当前验证结果：

  - 485 passed, 34 warnings
  - Ruff 通过
  - Shell 语法检查通过
  - 暂未修改任何文件

  没有发现新的 P0 阻断问题，但仍有以下问题值得修复。

  P1 正确性与健壮性

  1. Continuation 模式未严格校验 choice.index

     llmeval/inference/mc.py:349 在服务端缺失或返回非整数 index 时，会回退到响应顺序。这
     样 malformed response 可能被静默接受，导致 choice 与 continuation 错位。

     建议：要求每个 choice 都有非负、唯一且不越界的整数 index；缺失时直接标记 malformed
     response。

  2. Resume / generation schema 仍接受非法空样本

     llmeval/tasks/postprocess.py:220 会把 [None] 归一化为 []，[""] 也不会直接报错。对于
     resume 文件，这可能把损坏或未完成记录混入正常协议。

     建议：
      - resume loader 严格拒绝 None、空字符串和空 generation；
      - 明确区分“推理失败输出”和“非法 resume 行”；
      - 对 NaN、Infinity 等非有限 logprob 也直接拒绝；
      - 评估是否继续保留 response_key != "gen" 时的 gen fallback。

  3. 数学 worker 异常分类不准确

     llmeval/tasks/math_eval/math_score.py:459 将所有 worker 异常标记为
     verification_failed，其中可能包含序列化错误、字段错误或 scorer 基础设施异常。

     同时，llmeval/tasks/math_eval/math_score.py:592 对 pool 迭代器异常直接继续，最终在
     llmeval/tasks/math_eval/math_score.py:647 统一标成 timeout。

     建议新增 worker_failed / infrastructure_failed 分类，并让无法定位样本的 pool 异常默
     认 fail-fast，避免把基础设施故障伪装成验证超时。

  4. 代码 benchmark 准备脚本可能生成冲突的 task ID

     scripts/data_process/prepare_code_benchmarks.py:55 和 scripts/data_process/
     prepare_code_benchmarks.py:89 对缺失 task_id 使用行号兜底；如果源数据显式包含
     task_id: null，则会生成字符串 "None"，多条记录可能聚合为同一题。

     建议：
      - 正式数据强制要求非空字符串 task_id；
      - 删除 "None" 等隐式兜底；
      - 在准备阶段检查唯一性；
      - 数学数据的 _make_doc_id() 也采用同样策略。

  5. MC 结果写入失败没有进入统一处理

     llmeval/inference/mc.py:628 只捕获 future.result() 的异常；_write_result() 在后续
     else 分支执行，文件写入失败会直接中止收集循环，且不会进入 failed audit。

     建议统一让 worker/调度层负责写入失败分类，或者明确规定写入异常为不可恢复错误并在最
     外层只记录一次。

  6. 数据准备脚本不是原子写入

     MC、数学和代码准备脚本都直接使用 "w" 写目标文件，例如 scripts/data_process/
     prepare_mc_benchmarks.py:96。进程中断后可能留下部分文件；如果部分文件恰好拥有唯一
     doc_id，下一次运行可能误判为有效缓存并跳过。

     建议统一使用临时文件写入后 os.replace()，并在输出中记录行数或数据集版本。

  P2 结构与可维护性

  1. Resume 当前只能保证 append-only 语义

     llmeval/inference/common.py:171 使用每个 doc_id 的完成数量，而不是样本身份。若中间
     结果被删除或输出顺序被打乱，无法检测缺口。

     在删除 sample_index 的设计下，建议明确文档约束：resume 文件只能追加，禁止手工删除或
     重排；否则需要增加稳定的 run manifest 或迁移协议。

  2. 直接 scorer API 的参数校验不一致

     MC 的 llmeval/tasks/mc_eval/mc_score.py:374 在 max_workers=0 时会静默走串行路径；代
     码 scorer 也有类似行为，而配置层却要求 worker 数必须大于零。

     建议在三个 scorer 的公共入口统一校验 max_workers、timeout、exec_timeout 和
     k_values。

  3. 代码执行安全 guard 使用全局可变快照

     llmeval/tasks/code_eval/execute.py:57 的 reliability_guard() /
     reliability_restore() 依赖全局字典。生产路径通过子进程隔离，风险较低，但直接并发调
     用 unsafe_execute() 时可能互相覆盖恢复状态。

     建议将 guard 状态改为调用级对象，或增加明确的进程内锁，并在文档中声明并发约束。

  4. 兼容包装和旧持久化入口仍然较多

     仍有 write_cache()、score_generate()、score_loglikelihood() 等兼容入口。它们目前有
     测试或公开导出，不能直接删除。

     建议先：
      - 标记弃用；
      - 将测试迁移到结构化 scorer；
      - 保留一个兼容周期后再删除；
      - 同步缩短过长 docstring，避免重复描述相同协议。

  5. 公共持久化模块仍可进一步收敛

     postprocess.py 同时承担文本过滤、样本归一化、worker 限制和持久化。虽然当前功能正
     确，但职责已经偏多。后续可拆成：
      - sample_schema.py
      - text_filters.py
      - persistence.py

     这属于结构优化，不建议与 P1 修复混在同一批提交中。

  建议执行顺序

  1. 先修复 P1-1 至 P1-6，并补充对应回归测试；
  2. 再统一 scorer 参数校验和 resume 协议文档；
  3. 最后处理兼容 API、模块拆分和全局 guard 等结构优化。
