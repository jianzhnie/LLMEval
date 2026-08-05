# LLMEval 优化建议与实施计划

本文档整理 LLMEval 与本地 `lm-evaluation-harness` 实现对比后的优化建议，作为后续开发、测试和评审的实施依据。

重点关注三类任务的推理与评测正确性：

- 数学推理任务
- Multiple-Choice 任务
- 代码生成任务

同时覆盖任务配置、缓存、复现、统计、安全和可维护性等基础能力。

已完成的建议主要归档到第 1 节；涉及长期 correctness contract 或后端能力边界的章节会保留实施记录。

## 1. 当前已实现的基础能力

以下能力已经实现并有回归测试覆盖，后续不应重复建设：

- 数学任务使用 `math_verify`，并保留 harness 风格的文本归一化 fallback。
- MC 任务支持生成式评测和 loglikelihood 评测，输出 `acc`、`acc_norm`、`acc_bytes` 和 `exact_match`。
- 代码任务支持 HumanEval / MBPP 的显式 `prompt_mode` 和 `pass@k` 统计。
- **任务级后处理 pipeline**（原第 3 节，已完成）：数学、MC、代码 scorer 均通过公共 `FilterRegistry` 构建命名、版本化的 pipeline（`math_response` / `mc_generation` / `code_generation`）；MC 答案提取和代码提取是显式 filter；每条评分记录保存 `raw_gen`、`filtered_gen` 和逐步 `filter_trace`；中心 evaluator 不再保留全局预处理。
- **MC 多样本聚合**（原第 5 节，已完成）：`first`、`majority_vote`、`any_correct`、`per_sample` 四种显式模式；registry 仅在 `per_sample` 展开 observation；summary 区分题目数与样本数并记录 aggregation。
- **稳定身份 resume**（原第 6 节，已完成）：数据准备阶段生成 `doc_id`；推理按 `(doc_id, rendered_prompt, sample_index)` 去重恢复，可补齐中间缺失样本；MC 评分前按稳定身份合并多行；verifier 以非空 `Verifier_response` 判断推理完成；保留旧格式输出的显式 prompt fallback。
- **轻量 Task Registry**：`tasks/registry.py` 注册 math/MC/code adapter，evaluator 通过 registry resolve，不再使用 task-family if/elif；evaluator 中的遗留校验死代码和占位 context 已清理。
- **统一结果结构**：`tasks/results.py` 提供 `MetricValue`、`EvaluationResult`、bootstrap stderr/CI 和 macro/micro 聚合 helper；`evaluate_task_result()` 返回完整结构化结果。
- **内容寻址缓存**：`cache.py` 提供 canonical JSON + SHA-256 + namespace + schema version + 原子写入 + 损坏跳过；evaluation、online、MC 均为显式 opt-in，支持 `--force-recompute` 和 `--read-only-cache`；online/MC 请求 key 包含 endpoint；evaluation key 使用排除 scorer 输出字段后的输入 hash；只有成功响应才写入缓存；git hash 包含 dirty worktree 的实际 diff，未提交改动不会误命中。
- **MC 完整 continuation loglikelihood**（第 3 节）：后端无关 `LoglikelihoodRequest` / `ChoiceLoglikelihood` / `LoglikelihoodResult` 已建立；严格路径验证 token text、logprob、可选 token ID 和 choice 顺序；context 尾部空白按 harness `_encode_pair` 语义移入评分 continuation；`acc_norm` 使用 Unicode 字符长度，`acc_bytes` 使用 UTF-8 byte 长度；本地 harness oracle fixture 已覆盖单字符、多字符和中文差异。
- **基础 seed 管理**：`utils/reproducibility.py` 统一设置 Python（及可用时的 NumPy/PyTorch，不可用时记录为 `null`）；online/MC 请求和 evaluator 使用该入口；MC few-shot 按文档确定性采样并排除当前文档。
- 已有任务版本、pipeline 版本、prompt hash、target hash 等 provenance 信息和基础污染检查能力。
- 已有并行评测、超时处理、失败记录和断点恢复机制，以及较完整的类型标注、文档字符串和回归测试（当前 411 个测试）。

## 2. 优先级定义

| 优先级 | 含义 |
| --- | --- |
| P0 | 可能影响任务正确性、安全性或结果可信度，应优先处理 |
| P1 | 对扩展性、复现性和批量评测有明显收益 |
| P2 | 长期工程化能力，当前任务不阻塞时实施 |

## 3. P0：完善 MC 的完整 continuation loglikelihood

> 当前状态：**已完成（OpenAI-compatible 后端可验证范围）**。严格 continuation 使用后端无关 typed schema，完成 token/choice 对齐、harness 空白边界语义、`acc` / `acc_norm` / `acc_bytes` 指标归一化、失败不缓存和本地 harness oracle parity。后端无法证明跨边界 token 对齐时严格失败，不伪装为 exact；`auto` fallback 仍明确标记为近似。LLMEval 当前的 MC `exact_match` 是兼容历史输出的 `acc` 别名，不等同于 harness 的 gold continuation 逐 token greedy 指标。

### 3.1 实现边界

`llmeval/inference/mc.py` 优先尝试 `echo=True` continuation，仅当 `loglikelihood_mode=auto` 且 continuation 不完整时才退回 `max_tokens=1`、`top_logprobs=20` 的 first-token 近似。first-token 路径的固有局限：

- 只计算第一个 token 的概率。
- 目标 token 不在 top-k 时会被记录为 `-inf`。
- 不支持多 token 选项，例如完整短语或中文选项。
- 多 token continuation 的总 loglikelihood 不完整，导致所有派生指标失真。
- 不同 tokenizer 下的前导空格和 token 边界可能导致分数不一致。

严格路径本身依赖后端提供可靠的 `text_offset` / `token_logprobs` 契约；标准 OpenAI-compatible 响应不提供 token ID 时，`choice_token_ids` 必须写为 `null`，不得伪造 ID。

### 3.2 目标数据结构

推理输出保存完整的选项级 token 信息：

```json
{
  "gold": 1,
  "choices": ["A", "B", "C", "D"],
  "choice_token_ids": [[32], [33], [34], [35]],
  "choice_logprobs": [[-1.2], [-0.2], [-2.1], [-1.8]],
  "choice_token_count": [1, 1, 1, 1],
  "choice_char_count": [1, 1, 1, 1],
  "choice_byte_count": [1, 1, 1, 1]
}
```

### 3.3 评测规则

```text
raw_score[i]   = sum(choice_logprobs[i])
norm_score[i]  = raw_score[i] / max(choice_char_count[i], 1)
byte_score[i]  = raw_score[i] / max(choice_byte_count[i], 1)

acc       = argmax(raw_score) == gold
acc_norm  = argmax(norm_score) == gold
acc_bytes = argmax(byte_score) == gold
```

### 3.4 已完成实现

1. `inference/schema.py` 定义后端无关的 `LoglikelihoodRequest`、`ChoiceLoglikelihood` 和 `LoglikelihoodResult`，cache 反序列化会重新执行完整 shape/alignment 校验。
2. context 末尾空白按 harness `_encode_pair` 规则移入 scored continuation；token 文本必须完整重建实际评分文本，token ID 缺失时保存为 `null`。
3. `tests/test_mc_harness_parity.py` 直接调用本地 harness `ConfigurableTask.process_results`，比较 `acc`、`acc_norm`、`acc_bytes`；同时调用 harness `_encode_pair` 验证边界规则。
4. 生产环境要求严格语义时使用 `loglikelihood_mode=continuation`；后端 shape、offset、token text 或 ID 不一致会失败且不缓存。`auto` fallback 通过 `loglikelihood_exact=false` 和 `scoring_approximation` 明确标记。

### 3.5 验收标准

- 单 token 字母题与当前结果一致。
- 多 token 选项的结果与 harness 对应实现一致。
- `acc_norm` 使用与 harness 一致的 Unicode 字符长度；`acc_bytes` 使用 UTF-8 byte 长度；token count 仅作为诊断元数据。
- 请求失败不会被错误当成选择第一个选项，也不会被缓存固化。
- 每个选项的分数和预测结果可以从缓存文件中重现。

## 4. P0：加强代码评测安全边界

> 当前状态：**部分完成**。`--allow-unsafe-code` 显式授权和进程级 safety guard 已实现，默认拒绝执行；但当前执行环境不是强隔离 sandbox。

### 4.1 当前问题

`llmeval/tasks/code_eval/execute.py` 已说明当前实现是 safety guard，不是安全沙箱，只适合可信模型输出。Python 子进程、函数禁用和模块阻断不能防御所有恶意代码，例如利用已有对象访问系统资源、消耗大量内存或文件空间，或者造成拒绝服务。

### 4.2 剩余工作

在现有显式确认机制之上增加真正隔离：

- 使用 Docker、nsjail 或独立容器执行。
- 禁用网络。
- 限制 CPU、内存、进程数和磁盘空间。
- 使用独立低权限用户。
- 记录执行环境版本、镜像 hash 和安全配置。

隔离 backend 应作为可插拔实现接入 `execute.py`，本地可信场景保留当前轻量子进程路径。

### 4.3 验收标准

- 未显式开启时，代码任务不会执行生成代码。
- 超时、内存超限、进程超限和网络访问均有明确失败原因。
- 安全测试样例（恶意代码 fixture）无法修改宿主机文件或访问网络。
- 代码评测结果中记录 sandbox 版本和配置。

## 5. P1：Task Registry 收口

> 当前状态：**部分完成**。中心 if/elif 已被 registry resolve 替代，但 config whitelist 和 legacy scorer adapter 仍限制扩展性。

### 5.1 当前缺口

- `EvalTaskArguments` 仍维护 exact task whitelist，新增 registry 任务必须同时修改中心 config。
- adapter 仍依赖 legacy scorer 先写 JSONL/summary 再回读，没有统一的 scorer 返回契约。
- code adapter 的失败分类口径不正确（见第 9 节）。

### 5.2 剩余工作

1. valid tasks 由 registry 派生，删除 config 中的 exact whitelist；未注册任务报错时列出可用任务。
2. 统一 scorer contract：scorer 直接返回结构化 per-item 记录和指标，adapter 不再回读 JSONL/summary 文件。

### 5.3 验收标准

- 新增任务只需添加任务类、配置和测试，不修改中心 evaluator 和中心 config。
- 任务版本、字段、pipeline 和指标由任务自身声明。
- 现有 CLI 参数行为保持兼容。

## 6. P1：使用任务配置文件描述 prompt 和评测规则

> 当前状态：**未完成**。尚无 TaskConfig/YAML loader、配置覆盖规则或 config hash。

### 6.1 目标

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

### 6.2 实施建议

1. 先使用 dataclass 表示配置，确认字段稳定后再支持 YAML。
2. 保留命令行参数覆盖配置文件的能力，并明确覆盖优先级。
3. 配置加载时只做 schema 和类型校验，通用参数校验继续放在 config 中，不在 scorer 重复实现。
4. 将配置文件 hash 写入 provenance 和 cache key。

### 6.3 验收标准

- 同一 task config 在 CLI 和 Python API 下行为一致。
- 修改配置后缓存 key 发生变化。
- 缺少必需字段时在加载阶段报错，而不是评测中途失败。

## 7. P1：统一模型请求接口

> 当前状态：**未完成**。仅共享 retry 和部分数据 helper，尚无统一 ModelBackend 与 typed Request/Result。

### 7.1 当前问题

`online.py`、`mc.py`、`offline.py` 和 `verifier.py` 各自维护客户端或推理调用逻辑，分别构造请求、解析响应、处理缓存和错误状态，任务 scorer 需要了解具体输出格式。

### 7.2 建议接口

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

### 7.3 验收标准

- 任务代码不直接依赖 `openai.OpenAI` 或 vLLM 对象。
- fake backend 可以在不启动模型服务的情况下测试完整推理流程。
- generate 和 loglikelihood 的输出 schema 有统一的类型定义。
- 重试、超时、错误分类由 backend 或公共 transport 层统一处理。

## 8. P1：完善随机种子和复现机制

> 当前状态：**部分完成**。基础 RNG、API seed 和确定性 few-shot 已实现；offline/verifier 未接入统一入口，provenance 和端到端证据仍不完整。

### 8.1 当前缺口

- `VLLMEngineArguments` 已定义 `--seed`（默认 0），`OfflineInferArguments` 和 `VerifierInferArguments` 均可继承；但 `offline.py` 和 `verifier.py` 构造的 `SamplingParams` 未携带 seed，两个 runner 也未调用 `seed_everything`，CLI 参数处于未接线状态。
- few-shot 文件 hash、示例 ID 和实际使用的 prompt 未完整进入 run provenance。
- few-shot 文件不存在或候选不足时静默退化为 zero-shot，没有 warning。
- 缺少相同/不同 seed 的端到端推理测试和完整 pass@k 复现证据。

### 8.2 剩余工作

1. 将已有 `--seed` 配置接入 offline/verifier 的 `SamplingParams`，并让两个 runner 调用统一 `seed_everything` 入口。
2. few-shot provenance：记录 few-shot 文件 hash、每个文档选中的示例 ID 和最终 prompt hash。
3. few-shot 候选不足或文件缺失时记录显式 warning，不静默降级。
4. 增加端到端复现测试：相同输入、模型版本、配置和 seed 产生相同 prompt 及评测结果；不同 seed 确实改变采样或 few-shot 选择。

### 8.3 验收标准

- 相同输入、模型版本、配置和 seed 能生成相同 prompt 及相同评测结果。
- 不同 seed 确实会改变采样结果或 few-shot 选择。
- provenance 能完整解释一次运行使用的所有随机参数，且不可用的随机源记录为 `null` 而不是声称已设置。

## 9. P1：统一指标、聚合和不确定性统计

> 当前状态：**部分完成**。结构化结果和 bootstrap helper 已建立，MC 聚合口径已修正；code 的失败分类和 uncertainty 口径、三类任务的 denominator 统一仍未完成。

### 9.1 当前缺口

- code registry adapter 把正常错误答案（`passed=False`）计入 `failed_count`，混淆模型错误与执行基础设施失败（超时、沙箱错误）。
- code Pass@1 的 uncertainty 使用 sample-level observation，而 legacy Pass@k 是 problem-level，两者口径不一致。
- 空数据集在 `evaluate_task()` 返回 `None` 与结构化结果之间的约定不统一。
- math/MC/code 的空数据、失败、跳过、超时 denominator 规则尚未统一。
- 没有 task-group / run-level 聚合。

### 9.2 剩余工作

1. code 失败分类：`failed_count` 只计基础设施失败（超时、执行错误、沙箱异常）；`passed=False` 属于正常模型错误，只影响指标值。
2. code Pass@1 uncertainty 改为 problem-level：每个 problem 聚合为一个 pass rate observation 后再做 bootstrap。
3. 统一三类任务的空数据、失败、跳过、超时分母，并写入结果结构（`sample_count` / `effective_sample_count` / `failed_count` / `skipped_count` / `timeout_count`）。
4. CLI 输出完整指标，而不只输出一个 accuracy；summary JSON 与日志一致。

### 9.3 验收标准

- summary JSON 与日志中的指标一致。
- 空数据集、全失败数据集和部分超时数据集的分母规则明确并有测试。
- 统计结果包含有效样本数、失败数、跳过数和超时数，失败分类不混淆模型错误与基础设施错误。

## 10. P1：内容寻址的推理和评测缓存

> 当前状态：**部分完成**。evaluation、online、MC 均为显式 opt-in；endpoint、稳定输入 hash 和 dirty worktree diff 已进入 key；失败响应不写入缓存。剩余缺口：offline/verifier 接入、生命周期管理和并发验证。

### 10.1 当前缺口

- `offline.py` 和 `verifier.py` 未接入缓存。
- 没有 hit/miss/corrupt 统计、删除/清理机制和 rank 隔离。
- 缺少跨进程并发写入测试。

### 10.2 Cache key

cache key 至少包含：

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

其中 evaluation 的输入 hash 必须排除 scorer 运行时写入的字段（复用 `provenance._doc_for_hash` 的 runtime field 过滤），保证 scorer 原地注解不会改变 key。

### 10.3 剩余工作

1. 提供命中统计（hit/miss/corrupt）和缓存清理命令。
2. 并行运行时采用 rank 独立 cache，避免多进程写同一记录。
3. 补跨进程并发写入和损坏恢复的测试。
4. offline/verifier 接入同一缓存抽象。

### 10.4 验收标准

- 修改任意 generation 参数会产生新的 cache key；scorer 原地修改输入不会改变 key。
- 同一请求在重复运行时可命中缓存；瞬时失败不会被缓存固化。
- 缓存内容损坏时可以跳过并重新计算，而不是导致整个评测崩溃。
- 不同 task、不同模型和不同 git commit（含 dirty worktree）不会错误共享结果。

## 11. P1：改进 contamination 检查

> 当前状态：**部分完成**。本地 exact-substring 检查和基础 summary 已实现，task-specific query、overlap 模式和排除口径尚未实现。

### 11.1 当前能力和限制

当前 `provenance.py` 支持本地 JSONL / 文本 reference，并使用标准化后的 exact substring 匹配。这种方式保守、易理解，但覆盖范围有限。

### 11.2 借鉴方向

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

### 11.3 验收标准

- contamination 检查不会修改原始 prompt。
- 每条样本记录 query hash、source hash 和匹配方式。
- summary 区分 checked、contaminated、excluded 和 clean 数量。
- 没有 contamination source 时明确记录 `checked=false`，不能误报为 clean。

## 12. P1：建立与 harness 的 golden parity 测试

> 当前状态：**部分完成**。MC 已有直接调用本地 harness 的 oracle fixture，覆盖 context/continuation 空白边界和 `acc`、`acc_norm`、`acc_bytes` 聚合；数学和代码任务尚无同等级 oracle suite，MC 多目标、失败 denominator 和 harness `exact_match` greedy 语义也尚未覆盖。

### 12.1 目的

除了单元测试，还需要验证 LLMEval 和 harness 在相同输入、相同预测和相同规则下是否产生一致结果。

### 12.2 测试数据

建立固定小型 fixture。

#### 数学任务

- 整数、分数、小数、负数
- 等价表达式，例如 `1/2` 和 `0.5`
- `\boxed{}` 和 `\fbox{}`
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

### 12.3 验收标准

- golden fixture 的 per-item prediction 一致。
- aggregate metric 一致，或每个差异都有明确文档说明。
- 测试涵盖正常、边界、失败和恢复场景。
- 每次修改 task pipeline、任务配置或 scorer 时自动运行 parity 测试。

## 13. P2：完善批量、多任务和分布式评测

> 当前状态：**未完成**。仅有 task 内并行和独立聚合 helper，没有 multi-task/task-group/rank 执行链路。

### 13.1 建议能力

借鉴 harness 的 task group、rank 和 instance metadata：

- 一次运行多个 task。
- 每个 task 独立保存结果和 summary。
- 支持 `limit` 和显式样本索引选择。
- 支持多进程、多 GPU 和 rank 独立输出。
- 最后由 rank 0 合并结果。
- 结果保存原始 task、doc ID 和 repeat/sample index。

### 13.2 验收标准

- 单卡和多卡运行的 aggregate metric 一致。
- 失败 rank 可以单独重试，不需要重跑所有样本。
- 合并结果不会重复计算或重复计数。
- task group 的宏平均和微平均定义明确并有测试。

## 14. 建议的实施阶段

### 阶段一：正确性收口

目标：消除已确认的正确性问题，不改变现有正常结果。

1. 修正 code 的 failed/incorrect/timeout 分类和 problem-level uncertainty，统一三类任务的空数据、失败、跳过和超时分母（第 9 节）。
2. Task Registry 收口：移除 config whitelist，统一 scorer contract（第 5 节）。
3. 为代码执行接入真正的隔离 backend，至少限制网络、内存、进程数、磁盘和用户权限（第 4 节）。
4. 基于本地 harness 建立三类任务 oracle fixture，自动比较 per-item prediction、aggregate 和 denominator，并把现有回归测试接入 CI（第 12 节）。

### 阶段二：任务配置和接口统一

1. 引入 `TaskConfig` dataclass 和 config hash，接入 provenance/cache key（第 6 节）。
2. 抽象 `ModelBackend` 与通用 typed Request/Result schema，迁移 online/MC/offline/verifier；MC continuation 的后端无关 loglikelihood schema 已完成，可作为统一接口的参考实现（第 7、3 节）。
3. offline/verifier 接入统一 seed 入口，补齐 few-shot provenance 和端到端复现测试（第 8 节）。

### 阶段三：规模化评测

1. 完善 cache 生命周期、命中统计、rank 隔离和并发测试；offline/verifier 接入缓存（第 10 节）。
2. 扩展 task-specific 和 n-gram contamination 检查（第 11 节）。
3. 支持多 task、task group 执行与结果合并、rank-aware 分布式推理和评测（第 13 节）。

## 15. 最小验收清单

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
| -- | --- |
| 数学 | math\_verify、fallback、think 清理、超时、空答案、缓存和恢复 |
| MC | generate、loglikelihood、完整 continuation、`acc_norm`、多样本、失败请求 |
| 代码 | HumanEval、MBPP、prompt mode、代码提取、超时、语法错误、pass@k、安全边界 |

## 16. 推荐的最终目标架构

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

## 17. 当前状态汇总与最近验证

本节于 2026-08-05 按当前工作树核对，依据实际代码、测试和最小复现结果。状态含义：

- **已完成**：计划范围内的主流程、正确性约束和回归测试均已落地；已知的后端能力限制已显式记录。
- **部分完成**：已有可复用实现，但仍存在明确的正确性、口径或覆盖缺口。
- **未完成**：目标接口或主流程尚未建立。

### 17.1 剩余事项状态

| 条目 | 状态 | 主要缺口 |
| --- | --- | --- |
| 3. MC 完整 continuation | 已完成 | OpenAI-compatible 严格路径依赖后端可靠的 offset/token 契约；不满足契约时显式失败，`auto` 模式仅作已标记的近似 fallback |
| 4. 代码安全 | 部分完成 | 无强隔离 sandbox；缺资源限制、低权限用户、环境 hash 和恶意代码 fixture |
| 5. Task Registry 收口 | 部分完成 | config whitelist；legacy scorer 回读 JSONL/summary，无统一 scorer contract |
| 6. TaskConfig/YAML | 未完成 | schema、覆盖优先级、config hash、provenance/cache key 接入 |
| 7. 统一模型请求接口 | 未完成 | 无统一 `ModelBackend` 与 typed Request/Result |
| 8. 种子与复现 | 部分完成 | offline/verifier 未接入；few-shot provenance 和端到端证据不足；few-shot 不足时静默降级 |
| 9. 指标与不确定性 | 部分完成 | code 失败分类与 problem-level uncertainty；三类任务 denominator 未统一 |
| 10. 内容寻址缓存 | 部分完成 | offline/verifier 未接入；无统计、清理、rank 隔离和并发测试 |
| 11. contamination | 部分完成 | 无 task-specific query、ngram/token overlap、三态和排除口径 |
| 12. harness golden parity | 部分完成 | MC 核心指标和空白边界已有 harness oracle；缺数学、代码、多目标、失败 denominator 和 MC greedy `exact_match` parity |
| 13. 多任务/分布式 | 未完成 | 无 task group、limit、rank 输出与合并链路 |

### 17.2 最近验证结果

- 当前项目环境执行 `pytest -q`：411 passed，19 warnings。
- `conda run -n llmtuner pytest -q`：测试收集失败；该环境缺少 `openai`，在 `tests/test_api_retry.py` 导入阶段触发 `ModuleNotFoundError`。不能声称 `llmtuner` 环境已完成全量验证。
- `mypy --follow-imports=skip --ignore-missing-imports llmeval`：通过。
- `ruff check llmeval tests scripts/data_process`：通过；`ruff format --check` 仅报告未触及的 `scripts/data_process/prepare_code_benchmarks.py` 需要格式化。
- `git diff --check`：通过。
