# LLMEval 代码审查与优化建议

> 审查基线:`a9c12f6..HEAD`(8 个 commit,29 个文件,+2456/−2536),外加工作区进行中的并发重构。
> 建议按优先级排序;**P0 为阻断性问题,必须先解决**;P1 为稳定文件的可执行清理;P2 为跨文件重构(改动面大,建议单独立项)。
> 执行状态:⬜ 未执行 / 🟡 部分执行 / ✅ 已执行。

---

## P0 — 阻断性问题(并发重构未完成,树处于 broken 状态)

当前工作区:`verifier.py`、`sample_index.py`、`verifier_prompts.py` 已删除,但多个模块/测试仍引用它们 → **`import llmeval.utils.config` 已失败,101 个测试挂**。以下必须随重构一起修复。

- **P0-1** ⬜ `llmeval/utils/config.py:31` — `from llmeval.utils.verifier_prompts import VERIFY_PROMPT_FACTORY`。`verifier_prompts.py` 已删除,`VERIFY_PROMPT_FACTORY` 是全项目唯一引用点。删除该 import,否则 config 无法加载。
- **P0-2** ⬜ `llmeval/utils/config.py:574-640` — `VerifierInferArguments` 整类。`verifier.py` 已删除,该类无生产消费者(仅 `__all__` 自引用),约 66 行死代码。连同其字段(`verifier_prompt_type`/`keep_origin_data`/`verifier_prompt`)与验证逻辑一并删除。
- **P0-3** ⬜ `llmeval/inference/__init__.py:10`、`tests/test_verifier_infer_helpers.py:47` — 引用已删除的 `llmeval.inference.verifier`。删除/更新这些引用。
- **P0-4** ⬜ `llmeval/inference/mc.py` — 未跟上重构:仍 import/调用已删的 `sample_seed_for_item`、旧 `expand_data_with_resume` 签名。需迁移到新 API(`get_request_seed`/`base_seed`),否则 `import llmeval.inference.mc` 报错。
- **P0-5** ⬜ `llmeval/tasks/math_eval/math_score.py:32`、`mc_eval/mc_score.py:57`、`code_eval/code_score.py:30` — 均 `from llmeval.tasks.sample_index import ...`,而 `sample_index.py` 已被删除(重构产物为 `sample_record.py`)。三个 scorer 的 import 需更新,否则评分层整层无法 import。
- **P0-6** ⬜ `examples/test_compassverifier_offine_infer.py:22` — 引用已删除的 verifier 模块。

---

## P1 — 稳定文件清理(已核实,不碰重构中的文件)

### 评分层(4 个 scorer + execute)

- **P1-1 (B1)** ⬜ `llmeval/tasks/math_eval/math_score.py:807` — 删除 `compute_score_result = score_math_result` 别名(仅测试引用),测试改用 `score_math_result`。
- **P1-2 (B2)** ⬜ `llmeval/tasks/math_eval/math_score.py:461-484` — `except ValueError` 与 `except Exception` 分支功能相同(都走 `_math_text_equiv` fallback 后返回 `result(0.0, ...)`),`ValueError ⊂ Exception`,合并为一个 `except Exception`。
- **P1-3 (B3)** ⬜ `llmeval/tasks/mc_eval/mc_score.py:64-78` — 裁剪 `__all__`:删 `build_result`/`process_item`/`score_generate_item`/`score_items`/`write_cache`(grep 验证无外部 import)。
- **P1-4 (B4)** ⬜ `llmeval/tasks/mc_eval/mc_score.py:717-747` — 抽 `_argmax_normalized(logprobs, lengths)` 消除 4 份相同的归一化 argmax 逻辑。
- **P1-5 (B5)** ⬜ `llmeval/tasks/code_eval/code_score.py:148-150` — 删除 `_extract_code_filter` 空壳,`CODE_FILTER_REGISTRY` 直接注册 `extract_code`(mc_score 已这样做)。
- **P1-6 (B6)** ⬜ `llmeval/tasks/code_eval/execute.py:45` — `create_tempdir` 仅内部调用,从 `__all__` 移除(或改 `_create_tempdir`)。
- **P1-7 (B7)** ✅ `llmeval/tasks/persistence.py:36-40` — 删除 `except BaseException: raise` no-op 块。已执行,测试 3 passed。(注意:`except Exception` 分支的注释已补充说明中断语义。)

### 配置 / 脚本 / 测试基建

- **P1-8** ⬜ `tests/conftest.py:9-58` — 删除 5 个零消费者的死 fixtures(`sample_jsonl_items`/`sample_jsonl_with_gen`/`verifier_input_items`/`tmp_input_file`/`tmp_output_file`)及其无用 import。
- **P1-9** ⬜ `scripts/data_parallel_infer/auto_model_infer.sh:336`、`auto_model_infer_tp8.sh:338` — 删除从未调用的 `rsync_to_node`(各 ~16 行)+ tp8 的死变量 `READY_INSTANCE_IDS`。
- **P1-10** ⬜ `scripts/data_parallel_infer/auto_model_infer_tp8.sh:293-294` — 修正 `usage()` 过期文案(文档写 TP=4/INSTANCES=2,实际默认 8/1)。
- **P1-11** ⬜ `llmeval/evaluator.py` + `llmeval/tasks/registry.py:55` — 删除死参数链:`input_key` 经 `evaluate_task`→`evaluate_task_result`→`EvaluationContext.input_key` 透传但从未被读(grep 验证 registry 无 `.input_key` 读取)。

### 推理层(⚠️ 受重构影响,等 P0 完成后处理)

- **P1-12** ⬜ `llmeval/inference/common.py` — 死导出裁剪:`iter_resume_records`、`validate_document_ids` 在 `__all__` 中但无外部 import(重构后复核,因 `common.py` 正在被改)。
- **P1-13** ⬜ `llmeval/inference/online.py:147-156` 与 `offline.py` — 重复的 chat-template 校验块,抽到 common.py。
- **P1-14** ⬜ `llmeval/inference/online.py:474-486` / `offline.py:239-249` / `mc.py:642-656` — 重复的 append-JSON 持久化(`with self._file_lock: open(...,"a") ... f.flush()`),抽共享 `append_jsonl`。
- **P1-15** ⬜ `llmeval/inference/online.py:664` 与 `load_data():438` — 重复的输入文件存在性检查,删其一。
- **P1-16** ⬜ `llmeval/inference/offline.py:474-486` — `main(args)` 空壳函数(3 行 wrapper 仅 `__main__` 调用),内联;`offline.py:490` 裸字符串字面量(伪 docstring)删除。
- **P1-17** ⬜ `llmeval/inference/online.py:543-563`、`offline.py:331-345` 等 — 过长的"Step 1..N"docstring 精简为一行摘要。

---

## P2 — 跨文件重构(改动面大,建议单独立项)

- **P2-1** ⬜ **Pool 结果收集循环 ×3** — `math_score:583-611`、`mc_score:399-427`、`code_score:517-541` 的 `pool.map→future.result→StopIteration/TimeoutError/Exception→pbar` 三段结构相同,抽 `run_pool()`。
- **P2-2** ⬜ **worker 上限公式 ×3** — `min(total, max_workers, max(1, cpu_count-1))` 在三个 scorer 重复,抽 `resolve_max_workers()`。
- **P2-3** ⬜ **summary/JSONL 持久化 ×4** — `cache_path.with_suffix(".summary.json")` + `atomic_write_jsonl` + `atomic_write_json(indent=2)` 模式在 mc/code/math/registry 重复,抽 `persist_results()`。
- **P2-4** ⬜ **filter-artifact 三元组 ×3** — `{raw_gen, filtered_gen, filter_trace}` 在 math/code/mc 手搓 3 次,抽共享函数。
- **P2-5** ⬜ **`sample_index.py` 死导出** — `is_valid_index` 无外部 import。⚠️ 该文件正被重构为 `sample_record.py`,重构完成后处理。
- **P2-6** ⬜ **两个 shell 脚本 90% 相同** — `auto_model_infer.sh` 与 `auto_model_infer_tp8.sh` 约 1133 行 byte-identical,抽共享库 + TP 参数化 wrapper。

---

## 执行顺序建议

1. **P0**(随并发重构完成):恢复树可用 → 全测试可跑。
2. **P1-1 ~ P1-7**(评分层低风险清理,依赖 P0 恢复 import)。
3. **P1-8 ~ P1-11**(conftest/脚本/evaluator)。
4. **P1-12 ~ P1-17**(推理层,重构后复核)。
5. **P2**(单独立项,逐项做)。

## 验证方式

- 每个 P1 项完成后运行对应单测:`pytest tests/test_persistence.py`(P1-7 已通过)。
- 评分层恢复后:`pytest tests/test_math_eval.py tests/test_mc_eval.py tests/test_code_eval.py`。
- 全量:`pytest tests/` + `ruff check` + `ruff format --check`。
- 脚本:`bash -n scripts/data_parallel_infer/auto_model_infer.sh` 语法检查。
