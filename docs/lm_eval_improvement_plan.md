# LLMEval 优化建议与实施计划

本文档整理 LLMEval 与本地 `lm-evaluation-harness` 实现对比后的优化建议，作为后续开发、测试和评审的实施依据。

重点关注三类任务的推理与评测正确性：

- 数学推理任务
- Multiple-Choice 任务
- 代码生成任务

同时覆盖任务配置、缓存、复现、统计、安全和可维护性等基础能力。

## 1. 当前已完成能力

以下能力已经在 LLMEval 中实现，后续不应重复建设：

- 数学任务使用 `math_verify`，并保留 harness 风格的文本归一化 fallback。
- MC 任务支持生成式评测和 loglikelihood 评测。
- MC 任务已输出 `acc`、`acc_norm`、`acc_bytes` 和 `exact_match`。
- 代码任务支持 HumanEval / MBPP 的显式 `prompt_mode`。
- 代码任务支持 `pass@k` 统计。
- 数学、MC、代码任务均已使用显式文本后处理 pipeline。
- 已有任务版本、git commit、prompt hash、target hash 等 provenance 信息。
- 已有基础污染检查能力。
- 已将 JSONL 加载和部分数据恢复逻辑放入 `llmeval/inference/common.py`。
- 已有并行评测、超时处理、失败记录和断点恢复机制。
- 已有较完整的类型标注、文档字符串和回归测试。

## 2. 优先级定义

| 优先级 | 含义 |
|--------|------|
| P0 | 可能影响任务正确性、安全性或结果可信度，应优先处理 |
| P1 | 对扩展性、复现性和批量评测有明显收益 |
| P2 | 长期工程化能力，当前任务不阻塞时实施 |

## 3. P0：将后处理完全改为任务级 pipeline

### 3.1 当前问题

`llmeval/evaluator.py` 中的 `preprocess_answers()` 会对所有任务统一执行数学响应处理：

- 数学 scorer 内部还会再次执行数学 pipeline。
- MC 和代码任务会先经过数学任务的全局处理，再经过自身处理。
- 当前 `strip_reasoning_wrappers` 大多数情况下是幂等的，但后续增加提取规则时容易破坏其他任务。
- 新增任务时需要同时检查 evaluator 的全局预处理和 scorer 内部预处理。

参考 harness 的 `Filter` / `FilterEnsemble` 机制，后处理应属于任务定义，而不是全局 evaluator 行为。

### 3.2 目标设计

每个任务拥有独立、可排序、可测试的后处理链：

```text
task input
  -> task-specific filters
  -> task scorer
  -> metric aggregation
```

建议的默认 pipeline：

| 任务 | 后处理顺序 |
|------|------------|
| 数学 | `strip_reasoning` -> final answer / boxed 提取 -> `math_verify` -> normalization fallback |
| MC generate | `strip_reasoning` -> answer marker 提取 -> choice normalization |
| MC loglikelihood | 不处理模型文本，只校验 token/logprob schema |
| 代码 | `strip_reasoning` -> code block 提取 -> prompt mode 拼接 -> 执行 |

### 3.3 实施建议

1. 删除或禁用 `evaluator.py` 中无条件调用的 `preprocess_answers()`。
2. 将每个任务的 pipeline 暴露为任务对象或 scorer 内部的公共常量。
3. 为 pipeline 增加名称和版本，例如 `math_pipeline_v2`。
4. 每个 filter 只做一件事，并保证输入输出类型稳定。
5. 保留原始响应，例如 `raw_gen`，不要只覆盖 `gen`，便于问题排查。
6. 在 per-item 结果中记录实际使用的 pipeline 名称和版本。

### 3.4 验收标准

- MC 和代码结果不再依赖数学 pipeline。
- 同一个响应经过 pipeline 两次不会产生不同结果，或者第二次调用被明确禁止。
- 新增一个 filter 时不需要修改中心 evaluator。
- 单元测试覆盖正常答案、未闭合 think tag、answer tag、代码块和空响应。
- 三类任务的 golden fixture 结果与修改前一致，除已确认修复的 case 外不发生无意变化。

## 4. P0：完善 MC 的完整 continuation loglikelihood

### 4.1 当前问题

当前 `llmeval/inference/mc.py` 的 loglikelihood 模式通过 `max_tokens=1` 和 `top_logprobs` 获取答案 token 分数。

该方案对 `A/B/C/D` 这类单 token 选项基本适用，但与 harness 的 multiple-choice 评测语义不完全一致：

- 只计算第一个 token 的概率。
- 目标 token 不在 top-k 时会被记录为 `-inf`。
- 不支持多 token 选项，例如完整短语或中文选项。
- `acc_norm` 不能基于真实 continuation token 数计算。
- 不同 tokenizer 下的前导空格和 token 边界可能导致分数不一致。

### 4.2 目标数据结构

推理输出建议保存完整的选项级 token 信息：

```json
{
  "gold": 1,
  "choices": ["A", "B", "C", "D"],
  "choice_token_ids": [[32], [33], [34], [35]],
  "choice_logprobs": [[-1.2], [-0.2], [-2.1], [-1.8]],
  "choice_token_count": [1, 1, 1, 1],
  "choice_byte_count": [1, 1, 1, 1]
}
```

如果后端只能返回聚合分数，必须明确声明该字段表示完整 continuation 的 logprob，而不是第一个 token 的 top-k 近似值。

### 4.3 评测规则

对每个选项计算：

```text
raw_score[i]   = sum(choice_logprobs[i])
norm_score[i]  = raw_score[i] / max(choice_token_count[i], 1)
byte_score[i]  = raw_score[i] / max(choice_byte_count[i], 1)
```

对应指标：

```text
acc       = argmax(raw_score) == gold
acc_norm  = argmax(norm_score) == gold
acc_bytes = argmax(byte_score) == gold
```

### 4.4 实施建议

1. 定义 `LoglikelihoodRequest` 和 `LoglikelihoodResult` 数据结构。
2. 将 token 数、byte 数和 logprob 数量作为必需的 schema 校验项。
3. 对缺失 token、长度不一致和全 `-inf` 结果进行显式失败记录。
4. 对单 token 字母题保留快速路径，但在结果中标记 `scoring_mode=first_token`。
5. 增加多 token 英文选项、中文选项、前导空格和目标不在 top-k 的测试样例。

### 4.5 验收标准

- 单 token 字母题与当前结果一致。
- 多 token 选项的结果与 harness 对应实现一致。
- `acc_norm` 和 `acc_bytes` 使用真实 token / byte 长度，而不是字符串长度近似。
- 请求失败不会被错误当成选择第一个选项。
- 每个选项的分数和预测结果可以从缓存文件中重现。

## 5. P0：MC 生成模式支持多样本聚合

### 5.1 当前问题

当前 `score_generate()` 只评估 generation 列表中的第一个响应。当推理使用 `n_samples > 1` 时，其余响应会被忽略。

这会导致：

- 多次采样没有产生实际评测收益。
- 用户误以为评测的是 pass@k 或 majority vote。
- 结果与代码任务的多样本语义不一致。

### 5.2 建议的聚合模式

增加显式参数 `aggregation`：

| 模式 | 规则 |
|------|------|
| `first` | 只评估第一个响应，保持兼容当前行为 |
| `majority_vote` | 提取所有答案字母，出现次数最多者作为最终答案 |
| `any_correct` | 任一采样正确则该题正确 |
| `per_sample` | 每个采样独立计分，同时输出样本级结果 |

默认建议继续使用 `first`，但必须在结果中明确记录，避免产生歧义。

### 5.3 验收标准

- `n_samples=1` 时所有模式结果一致。
- `majority_vote` 正确处理空答案和无法解析答案。
- 并列票数时使用确定性的 tie-break 规则。
- summary 中同时记录 sample 数、题目数和聚合模式。
- 重复 prompt 的结果不会因为样本顺序改变而产生未记录的差异。

## 6. P0：将 resume 从 prompt 计数改为稳定样本身份

### 6.1 当前问题

旧版公共数据处理主要使用 prompt 字符串统计已完成样本数量。相同 prompt、数据重排、prompt 修改和跨任务运行都可能造成恢复冲突。

稳定身份应在 benchmark 数据准备阶段一次性生成并写入 JSONL，推理阶段只读取和透传，不能根据行号、prompt 或运行时内容再次生成。

### 6.2 建议的数据身份

每条输入记录应有稳定的：

```text
task_name
doc_id
sample_index
prompt_hash
dataset_hash
```

其中：

- `doc_id` 由数据准备脚本生成，格式为 `benchmark:source_id`。
- 数据集有原始 ID 时使用原始 ID；没有时使用准备阶段的 benchmark 内索引。
- `sample_index` 表示同一问题的第几个采样结果。
- `prompt_hash` 用于检测 prompt 变化，而不是作为唯一身份。

### 6.3 实施建议

1. 在数学、MC、代码数据准备脚本中生成并写入唯一 `doc_id`。
2. 在 `common.py` 校验 `doc_id` 存在且在输入文件内唯一。
3. 推理只透传 `doc_id`，写出结果时同时保存 `sample_index`。
4. resume 时按 `doc_id` 和样本数量判断是否完成。
5. 对旧格式结果提供 prompt 计数兼容读取逻辑。
6. 对追加写入使用文件锁，避免并发写入破坏 JSONL。

### 6.4 验收标准

- 两条内容相同但 ID 不同的记录可以独立恢复。
- 输入数据重排不会造成错误跳过。
- 修改 prompt 后旧结果不会被错误复用。
- 并发推理和中断恢复不会重复或丢失 sample。
- 缺失或重复 `doc_id` 的输入会在推理开始前明确失败。

## 7. P1：引入 Task Registry，替代中心 evaluator 的 if/elif

### 7.1 当前问题

当前 [evaluator.py](../llmeval/evaluator.py) 通过 task name 前缀分支到数学、MC 和代码 scorer。随着任务增多，中心文件会持续膨胀，任务逻辑难以独立演进。

### 7.2 参考设计

借鉴 harness 的 `TaskManager`、`Task` 和 `ConfigurableTask`，建立 LLMEval 的轻量任务接口：

```python
class EvaluationTask(Protocol):
    name: str
    version: str

    def preprocess(self, item: dict[str, Any]) -> dict[str, Any]: ...
    def score(self, dataset: list[dict[str, Any]]) -> EvaluationResult: ...
    def metrics(self) -> dict[str, MetricSpec]: ...
```

建议使用：

```python
TASK_REGISTRY = {
    "math_opensource": MathTask,
    "mc_opensource": MCTask,
    "code_opensource": CodeTask,
}
```

### 7.3 验收标准

- 新增任务只需添加任务类、配置和测试，不修改中心 evaluator。
- 未注册任务可以给出明确错误和可用任务列表。
- 任务版本、字段、pipeline 和指标由任务自身声明。
- 现有 CLI 参数行为保持兼容。

## 8. P1：使用任务配置文件描述 prompt 和评测规则

### 8.1 目标

借鉴 harness 的 YAML task configuration，显式描述：

- `input_key`
- `target_key`
- `choices_key`
- `prompt_mode`
- `postprocess`
- `metrics`
- `task_version`
- `few_shot` 配置

示例：

```yaml
name: mc_opensource/mmlu
version: 1.0
input_key: prompt
target_key: gold
choices_key: choices
mode: loglikelihood
postprocess:
  - strip_reasoning
metrics:
  - acc
  - acc_norm
```

### 8.2 实施建议

1. 先使用 dataclass 表示配置，确认字段稳定后再支持 YAML。
2. 保留命令行参数覆盖配置文件的能力。
3. 配置加载时只做 schema 和类型校验，通用参数校验继续放在 config 中，不在 scorer 重复实现。
4. 将配置文件 hash 写入 provenance。

### 8.3 验收标准

- 同一 task config 在 CLI 和 Python API 下行为一致。
- 修改配置后缓存 key 发生变化。
- 缺少必需字段时在加载阶段报错，而不是评测中途失败。

## 9. P1：统一模型请求接口

### 9.1 当前问题

`online.py`、`mc.py`、`offline.py` 和 `verifier.py` 各自维护客户端或推理调用逻辑，任务 scorer 需要了解具体输出格式。

### 9.2 建议接口

参考 harness 的 request type，统一为：

```python
class ModelBackend(Protocol):
    def generate(
        self, requests: list[GenerateRequest]
    ) -> list[GenerateResult]: ...

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest]
    ) -> list[LoglikelihoodResult]: ...
```

后端可以有：

- OpenAI-compatible API backend
- vLLM offline backend
- SGLang backend
- 测试用 fake backend

### 9.3 验收标准

- 任务代码不直接依赖 `openai.OpenAI` 或 vLLM 对象。
- fake backend 可以在不启动模型服务的情况下测试完整推理流程。
- generate 和 loglikelihood 的输出 schema 有统一的类型定义。
- 重试、超时、错误分类由 backend 或公共 transport 层统一处理。

## 10. P1：完善随机种子和复现机制

### 10.1 当前问题

当前 provenance 会记录 seed，但记录 seed 不等于实际设置并使用 seed。推理、few-shot、后端生成和评测采样可能使用不同随机源。

### 10.2 实施建议

统一管理以下随机源：

- Python `random`
- NumPy
- PyTorch
- vLLM generation seed
- OpenAI-compatible API 的 `seed`
- few-shot sampler
- pass@k 相关采样逻辑

建议 provenance 记录：

```json
{
  "seed": 42,
  "python_seed": 42,
  "numpy_seed": 42,
  "torch_seed": 42,
  "fewshot_seed": 42,
  "generation_seed": 42
}
```

### 10.3 Few-shot 特别要求

当前 MC few-shot formatter 会在初始化时建立固定样本池。建议参考 harness 的 per-document sampler：

- 每个测试样本独立采样 few-shot 示例。
- 排除当前测试样本。
- 默认使用独立 train/dev 文件。
- 没有 few-shot 文件时默认 `n_shot=0`，不要默认从测试集采样。
- 记录 few-shot 文件 hash 和示例 ID。

### 10.4 验收标准

- 相同输入、模型版本、配置和 seed 能生成相同 prompt 及相同评测结果。
- 不同 seed 确实会改变采样结果或 few-shot 选择。
- provenance 能完整解释一次运行使用的所有随机参数。

## 11. P1：统一指标、聚合和不确定性统计

### 11.1 当前问题

当前 evaluator 主要返回一个 `float`，会丢失 MC 和代码 scorer 已经计算出的其他指标。

### 11.2 建议结果结构

引入统一 `EvaluationResult`：

```python
@dataclass
class MetricValue:
    value: float
    count: int
    stderr: float | None = None


@dataclass
class EvaluationResult:
    task_name: str
    metrics: dict[str, MetricValue]
    sample_count: int
    provenance: dict[str, Any]
```

### 11.3 建议支持的能力

- accuracy / exact match
- `acc_norm`
- `acc_bytes`
- pass@k
- bootstrap standard error
- confidence interval
- macro average
- micro average
- task group aggregation
- `higher_is_better`

### 11.4 验收标准

- CLI 输出完整指标，而不只输出一个 accuracy。
- summary JSON 与日志中的指标一致。
- 空数据集、全失败数据集和部分超时数据集的分母规则明确。
- 统计结果包含有效样本数、失败数、跳过数和超时数。

## 12. P1：设计内容寻址的推理和评测缓存

### 12.1 当前问题

当前缓存文件包含 provenance，但缓存是否可复用主要依赖路径和人工管理。修改模型参数、prompt 或后处理后，容易误用旧结果。

### 12.2 Cache key

建议 cache key 至少包含：

```text
model_name
model_revision
task_name
task_version
dataset_hash
prompt_hash
generation_params
sampling_seed
postprocess_version
git_commit
```

### 12.3 实施建议

1. 推理请求缓存和评测结果缓存分开管理。
2. 使用 SQLite 或 content-addressed JSON 文件保存 request/result。
3. 并行运行时采用 rank 独立 cache，避免多个进程同时写同一数据库记录。
4. 缓存命中时校验 schema version 和关键参数。
5. 提供 `--force-recompute`、`--read-only-cache` 和缓存统计信息。

### 12.4 验收标准

- 修改任意 generation 参数会产生新的 cache key。
- 同一请求在重复运行时可命中缓存。
- 缓存内容损坏时可以跳过并重新计算，而不是导致整个评测崩溃。
- 不同 task、不同模型和不同 git commit 不会错误共享结果。

## 13. P1：改进 contamination 检查

### 13.1 当前能力和限制

当前 `provenance.py` 支持本地 JSONL / 文本 reference，并使用标准化后的 exact substring 匹配。这种方式保守、易理解，但覆盖范围有限。

### 13.2 借鉴方向

参考 harness 的 `doc_to_decontamination_query`，增加任务级污染查询：

- 数学任务使用规范化题目文本。
- MC 任务使用题干和选项的组合文本。
- 代码任务使用 prompt 和函数签名。

进一步支持：

- n-gram overlap
- token-level overlap
- reference dataset version
- contamination source hash
- contaminated / excluded / flagged 三种状态

### 13.3 验收标准

- contamination 检查不会修改原始 prompt。
- 每条样本记录 query hash、source hash 和匹配方式。
- summary 区分 checked、contaminated、excluded 和 clean 数量。
- 没有 contamination source 时明确记录 `checked=false`，不能误报为 clean。

## 14. P0：加强代码评测安全边界

### 14.1 当前问题

`llmeval/tasks/code_eval/execute.py` 已说明当前实现是 safety guard，不是安全沙箱，只适合可信模型输出。

当前的 Python 子进程、函数禁用和模块阻断不能防御所有恶意代码，例如利用已有对象访问系统资源、消耗大量内存或文件空间，或者造成拒绝服务。

### 14.2 实施建议

借鉴 harness 对 unsafe task 的显式确认机制，同时增加真正隔离：

- 增加 `--allow-unsafe-code` 显式开关。
- 默认拒绝运行不可信代码。
- 使用 Docker、nsjail 或独立容器执行。
- 禁用网络。
- 限制 CPU、内存、进程数和磁盘空间。
- 使用独立低权限用户。
- 记录执行环境版本、镜像 hash 和安全配置。

### 14.3 验收标准

- 未显式开启时，代码任务不会执行生成代码。
- 超时、内存超限、进程超限和网络访问均有明确失败原因。
- 安全测试样例无法修改宿主机文件或访问网络。
- 代码评测结果中记录 sandbox 版本和配置。

## 15. P1：建立与 harness 的 golden parity 测试

### 15.1 目的

除了单元测试，还需要验证 LLMEval 和 harness 在相同输入、相同预测和相同规则下是否产生一致结果。

### 15.2 测试数据

建立固定小型 fixture。

#### 数学任务

- 整数、分数、小数、负数
- 等价表达式，例如 `1/2` 和 `0.5`
- `\\boxed{}` 和 `\\fbox{}`
- think / answer wrapper
- 无法解析但文本等价的答案
- 多参考答案

#### MC 任务

- 单 token A-D 选项
- A-J 选项
- 多 token 英文选项
- 中文选项
- 选项长度不同的 `acc_norm`
- 目标 token 不在 top-k
- 空响应和请求失败
- 多目标答案

#### 代码任务

- HumanEval 函数体补全
- 完整函数返回
- MBPP 自然语言 prompt
- Markdown code fence
- 多个生成样本
- 超时、语法错误、运行时错误和测试失败

### 15.3 验收标准

- golden fixture 的 per-item prediction 一致。
- aggregate metric 一致，或每个差异都有明确文档说明。
- 测试涵盖正常、边界、失败和恢复场景。
- 每次修改 task pipeline、任务配置或 scorer 时自动运行 parity 测试。

## 16. P2：完善批量、多任务和分布式评测

### 16.1 建议能力

借鉴 harness 的 task group、rank 和 instance metadata：

- 一次运行多个 task。
- 每个 task 独立保存结果和 summary。
- 支持 `limit` 和显式样本索引选择。
- 支持多进程、多 GPU 和 rank 独立输出。
- 最后由 rank 0 合并结果。
- 结果保存原始 task、doc ID 和 repeat/sample index。

### 16.2 验收标准

- 单卡和多卡运行的 aggregate metric 一致。
- 失败 rank 可以单独重试，不需要重跑所有样本。
- 合并结果不会重复计算或重复计数。
- task group 的宏平均和微平均定义明确并有测试。

## 17. 建议的实施阶段

### 阶段一：正确性和安全

目标：不改变现有正常结果，消除高风险隐式行为。

1. 移除 evaluator 全局后处理，改为任务级 pipeline。
2. 为 MC 增加完整 continuation schema 和校验。
3. 增加 MC 多样本聚合模式。
4. 将 resume 改为稳定 `doc_id/sample_index`。
5. 增加代码任务 unsafe confirmation 和执行安全文档。
6. 建立三类任务 golden parity 测试。

### 阶段二：任务配置和结果结构

1. 引入 `EvaluationTask` 和 task registry。
2. 引入 task config dataclass。
3. 统一 `EvaluationResult` 和 `MetricValue`。
4. 将 task version、pipeline version 和 config hash 纳入 provenance。
5. 完善 few-shot 的独立数据源和 per-document sampling。

### 阶段三：规模化评测

1. 建立内容寻址的推理缓存。
2. 增加 bootstrap stderr 和 confidence interval。
3. 支持多 task、task group 和 macro/micro aggregation。
4. 支持 rank-aware 分布式推理和评测。
5. 扩展 n-gram contamination 检查。

## 18. 最小验收清单

每个阶段完成后至少运行：

```bash
pytest -q
mypy llmeval --ignore-missing-imports --no-incremental
ruff check llmeval tests scripts
ruff format --check llmeval tests scripts
git diff --check
```

三类任务至少需要验证：

| 任务 | 必测内容 |
|------|----------|
| 数学 | math_verify、fallback、think 清理、超时、空答案、缓存和恢复 |
| MC | generate、loglikelihood、完整 continuation、`acc_norm`、多样本、失败请求 |
| 代码 | HumanEval、MBPP、prompt mode、代码提取、超时、语法错误、pass@k、安全边界 |

## 19. 推荐的最终目标架构

```text
TaskConfig / TaskRegistry
        |
        v
DatasetLoader -> StableDocumentId -> PromptBuilder
        |
        v
ModelBackend
  | generate
  | loglikelihood
        |
        v
TaskPostprocessPipeline
        |
        v
TaskScorer
        |
        v
MetricAggregator -> EvaluationResult
        |
        +--> per-item JSONL
        +--> summary JSON
        +--> provenance
        +--> cache
```

最终目标是让任务定义、模型请求、后处理、评分和结果聚合彼此独立。这样既能继续保持 LLMEval 对数学、MC 和代码任务的轻量支持，也能逐步获得 `lm-evaluation-harness` 在配置化、可复现性、指标统计和批量任务管理方面的优势。
