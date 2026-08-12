1. 离线推理会吞掉结果写入失败
    llmeval/inference/offline.py:369 在 fail_fast=False 时将 append_jsonl() 的文件系统异常也当作普通推理失败继续执行，可能造成结果静默丢
    失。持久化异常应始终向上抛出。

2. 多采样结果可能被错误判断为完整
    llmeval/tasks/postprocess.py:161、llmeval/tasks/math_eval/math_score.py:642、llmeval/tasks/code_eval/code_score.py:621 仅根据成功保
    存的行推断采样数量。请求 3 次但只保存索引 0、2 时，会被当成完整的两次采样。
    建议保留“不写入推理失败样本”的设计，同时使用 n_samples 约束并校验 sample_index 连续性。

3. Code fallback 会掩盖基础设施失败
    llmeval/tasks/code_eval/code_score.py:410 第一个候选出现执行器失败后仍尝试 fallback，后续普通错误可能覆盖真正的失败状态。只有“执行完
    成但答案错误”时才应尝试 fallback。

4. 候选代码触发 SIGSEGV 被排除在指标之外
    llmeval/tasks/code_eval/execute.py:625 将段错误视为评测基础设施失败，可能抬高 pass@k。候选程序引起的信号退出应统一计为 completed +
    incorrect。

5. 缺失生成结果的语义不一致
    llmeval/tasks/math_eval/math_score.py:310、llmeval/tasks/code_eval/code_score.py:361、llmeval/tasks/mc_eval/mc_score.py:763 对缺失字
    段、null、[None] 和空字符串处理不同。建议统一为：

   - API 成功返回 ""：completed + incorrect
   - 字段缺失、null 或容器格式错误：failed
   - failed 不写入推理结果文件

6. pass@k 公共函数缺少输入校验
    llmeval/tasks/code_eval/code_score.py:179 接受负数、num_correct > num_samples、浮点 k 和布尔值。应严格要求整数且满足 0 <= c <= n、1
    <= k <= n。

7. 重试次数为负数时静默返回 None
    llmeval/utils/retry.py:162 在 max_retries=-1 时一次也不执行。应严格校验非负整数，并删除隐式 fall-through。

8. 配置类型校验过宽
    llmeval/utils/config.py:172 等位置允许 n_samples=True、max_workers=True、max_completion_tokens=1.5、top_k=1.5。建议集中实现严格整数
    和有限浮点数校验，明确拒绝布尔值。

清理与结构优化

- 修复 4 个 #!/bin/bash/ 无效 shebang。
- 删除两个未使用且未正确引用的 rope_scaling={...} shell 变量。
- 简化 llmeval/tasks/postprocess.py:1 的动态过滤器注册框架；当前生产代码不需要运行时注册，直接构造命名 pipeline 更清晰。
- 合并 Math、MC 数据准备脚本中重复的 _has_valid_doc_ids()。
- 删除或参数化包含个人绝对路径且具破坏性的 duplicate_datasets.sh。
- 建议删除只修改仓库 Git 身份、与评测无关的 scripts/config.sh。
- 整理 tests/inference/test_mc.py 中动态挂载到测试类的方法。
- 修复当前 10 个 mypy 类型错误，主要集中在 execute.py、mc_score.py、retry.py 和 seed 调用处。
- 没有发现真正的生产空壳函数；MCScoreResult、CodeScoreResult、evaluate_task() 是公开 API，建议保留。
