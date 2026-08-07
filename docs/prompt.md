
🔴 新增问题(工作区引入)

NEW-1 — normalize_single_generation_samples 改了行为但 test_postprocess 断言没跟上
合并后函数对重复样本不再去重(保留每个独立样本)。但 tests/test_postprocess.py 仍直接从 postprocess import normalize_single_generation_samples 并测它 —— 我核对了,测试断言的语义(相同 response 也保留)与函数一致。结论:不是 bug,但需要确认 —— 你是否知道这组 test 是从 test_math_eval/test_code_eval 挪过来的,还是新写的?

NEW-2 — mc generate 模式与 math/code 的"重复样本"语义仍不一致
math/code 用统一的 normalize_single_generation_samples 保留重复行(冲突才报错);mc 的 merge_generate_records 也保留重复行、冲突报错 —— 但冲突键不同:mc 检查 label_key/gold/prompt/query/choices/choice_tokens 6 个,math/code 只检查 (label_key, "prompt") 2 个。行为上 mc 更严格,但三种任务对 resume 冲突的容忍度不统一。建议:要么统一冲突键清单,要么在 CLAUDE.md 注明这是有意的(每类任务的"问题身份"不同)。

🟡 低严重度优化点(建议但不阻塞)

OPT-1 — score_continuations 的 except Exception 兜底与 first-token 路径不一致
get_choices_logprobs 让 RuntimeError 传播,score_continuations 转 failure。测试已锁定该行为(test_programming_error_propagates 测的是 first-token 路径)。这是有意的:continuation 是批量评分,一个 item 挂不该 kill 整个 batch。建议保留,但加一行注释说明为何与 first-token 路径不同。

OPT-2 — merge_generate_records 的 key in item and key in target 与 mc.py 的其它冲突检测重复
已有 _mc_schema(registry.py)+ score_items 校验,merge_generate_records 内部又手写了一遍 6-key 冲突循环。_count_excluded 之前就是从这种重复里抽出来的 —— 这里是否也值得抽一个 _check_conflict helper?

OPT-3 — score_loglikelihood/score_generate 兼容 wrapper
两行转发壳,仅 evaluator.py 用,但 __init__.py 导出。是否保留由 CLAUDE.md 的"CLI