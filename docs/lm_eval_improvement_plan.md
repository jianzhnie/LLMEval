# LLMEval 未完成优化计划

本文档只记录当前代码库中尚未完成、仍存在明确缺口，或需要补充工程化闭环的事项。已完成的 MC continuation loglikelihood、任务级 filter pipeline、Task Registry、统一 scorer/指标、稳定 `doc_id` resume、内容寻址缓存和基础 seed 接线不再作为待办重复描述。

计划参考本地 `lm-evaluation-harness` 的 task 配置、request backend、decontamination、结果 provenance、限样本和多任务执行方式，同时保持 LLMEval 对数学、MC 和代码三类任务的现有输出兼容性。

## 1. 当前待办总览

状态含义：

- **未完成**：主流程或目标接口尚不存在。
- **部分完成**：已有实现，但仍有影响安全性、正确性、复现性或扩展性的缺口。

| 优先级 | 条目 | 当前状态 | 主要缺口 |
| --- | --- | --- | --- |
| P0 | 代码执行强隔离 | 部分完成 | 当前是 safety guard，不是强 sandbox；缺网络、资源、权限和环境证明 |
| P0 | 数学/代码 harness parity | 部分完成 | MC 已有 parity；数学和代码缺少同等级 oracle fixture 与差异报告 |
| P1 | TaskConfig/YAML | 未完成 | 无统一 schema、配置覆盖规则和 config hash |
| P1 | ModelBackend | 未完成 | online、MC、offline、verifier 仍各自依赖具体客户端 |
| P1 | 运行复现与 manifest | 部分完成 | seed 已接线，但 offline/verifier 的请求级 provenance、依赖环境和完整 run manifest 不足 |
| P1 | contamination | 部分完成 | 只有 exact substring；缺 task-specific query、overlap 模式和排除分母 |
| P1 | 数据集 preflight/audit | 未完成 | 缺统一的数据 schema、重复 ID、字段语义和数据质量报告入口 |
| P1 | parity/质量 CI | 部分完成 | 有 pre-commit，但没有 GitHub Actions 等完整测试和 parity gate |
| P2 | 多任务、task group 和分布式 | 未完成 | 无 limit、shard、rank 输出合并和 run-level 聚合 |
| P2 | 结果兼容与能力矩阵 | 未完成 | 缺 schema migration、后端 capability 声明和不支持组合的早期诊断 |

## 2. P0：代码执行强隔离

### 2.1 当前状态

`llmeval/tasks/code_eval/execute.py` 已具备以下保护：

- `allow_unsafe_code` 默认关闭，未显式授权时拒绝执行。
- 代码在独立子进程中执行。
- `reliability_guard()` 禁用部分 builtins、`os`/`shutil` 操作和危险模块。
- 已有超时、失败分类和回归测试。

这些措施只能降低可信模型输出造成的意外影响，不能抵抗恶意代码。子进程仍可能利用解释器漏洞、资源耗尽、文件描述符、环境信息或未覆盖的 Python 对象访问宿主资源。

### 2.2 目标设计

增加可插拔的执行 backend，保留当前本地路径作为显式的低安全等级模式：

```text
CodeExecutor
  | LocalGuardExecutor       # 仅可信代码，兼容现有环境
  | ContainerExecutor        # Docker/nsjail，默认推荐
        |
        +-- network disabled
        +-- read-only rootfs
        +-- temporary workspace only
        +-- non-root user
        +-- CPU / memory / process / disk limits
```

建议配置：

- `sandbox_backend`: `local_guard`、`docker` 或 `nsjail`。
- `sandbox_image` / image digest。
- `network_enabled`，默认 `false`。
- CPU、内存、进程数、文件大小和磁盘配额。
- workspace 路径和清理策略。
- 是否允许额外 Python 包；默认不允许。

### 2.3 实施步骤

1. 抽象 `Executor` Protocol，统一返回 `passed`、`status`、`failure_class`、`runtime_ms` 和环境 provenance。
2. 将现有实现包装为 `LocalGuardExecutor`，保持现有测试和 CLI 参数兼容。
3. 实现 Docker 或 nsjail backend，先覆盖 Linux；在不支持隔离 backend 的平台上明确报错，不静默降级到本地执行。
4. 将基础设施失败与模型错误分离：`timeout`、`sandbox_error`、`resource_limit` 计入 failed；断言失败、语法错误和错误答案按任务语义记录为模型结果。
5. 在结果和 summary 中记录 backend、image/digest、限制参数和版本。

### 2.4 验收标准

- 未显式授权时，任何 backend 都不会执行模型代码。
- 默认 backend 不允许网络、宿主文件写入和提权。
- 超时、内存/CPU/进程/磁盘超限都有稳定的 failure class。
- 恶意 fixture 无法写入宿主指定文件、访问网络或读取评测进程环境。
- 同一代码在 `local_guard` 和强隔离 backend 的模型结果字段一致，执行环境差异只体现在 provenance。
- 至少覆盖 HumanEval、MBPP、语法错误、断言失败、无限循环、文件访问、网络访问和进程创建 fixture。

## 3. P0：数学和代码任务 harness parity

### 3.1 当前状态

`tests/test_mc_harness_parity.py` 已覆盖 MC continuation 的 token 边界和 `acc`、`acc_norm`、`acc_bytes`。数学和代码目前主要依赖 LLMEval 自身的单元测试，尚未有固定的 harness oracle suite。

### 3.2 数学 parity

建立 `tests/fixtures/parity/math.jsonl` 和可重复的 oracle 测试，比较：

- `math_verify` 可解析的整数、分数、小数、负数和等价表达式。
- `\boxed{}`、`\fbox{}`、think/answer wrapper。
- 解析失败时 harness 风格 normalize fallback 的行为。
- 空答案、错误答案、超时和基础设施失败的分母。
- per-item `filtered_gen`、`correct`、status 和 aggregate metric。

如果 harness 和 LLMEval 使用不同的数学 parser，应比较“任务约定的最终判断”而不是强行比较中间 AST；每个差异必须附带原因分类。

### 3.3 代码 parity

代码执行环境可能不同，因此 parity 分两层：

1. **纯规则层**：比较 HumanEval/MBPP prompt mode、代码提取、函数签名、样本分组和 pass@k 公式。
2. **执行层**：在同一个固定容器镜像中运行 harness 与 LLMEval，比较 per-problem pass、timeout、execution failure 和 pass@k。

固定 fixture 至少包括：

- 函数体补全和完整函数返回。
- HumanEval 与 MBPP prompt。
- Markdown code fence、前后解释文本和多段代码。
- 正确答案、语法错误、运行时错误、断言失败、超时和多样本。

### 3.4 实施步骤和验收标准

1. 把 fixture、oracle 版本和预期差异说明纳入仓库。
2. 测试同时保存 per-item prediction、status、denominator 和 aggregate。
3. 对 harness 不提供或语义不同的指标明确标记 `not_comparable`，不能只比较一个 accuracy。
4. 修改 math/code filter、scorer 或 executor 时 parity 测试必须自动运行。
5. parity 失败时输出第一条差异的 `doc_id`、prompt hash、raw/filtered generation 和指标差异。

## 4. P1：TaskConfig/YAML 配置

### 4.1 当前缺口

当前 task 选择由 `task_name` 字符串和 Python scorer 共同决定，任务字段、prompt mode、postprocess 和 metrics 没有统一的声明文件。新增任务仍需要理解中心参数和 adapter 约定。

### 4.2 配置 schema

先实现 dataclass，再提供 YAML loader：

```yaml
name: mc_opensource/mmlu
version: "1.0"
dataset:
  input_key: prompt
  target_key: gold
  choices_key: choices
prompt:
  mode: raw
  system_prompt: empty
  few_shot:
    count: 0
    source: null
inference:
  mode: loglikelihood
postprocess:
  - mc_generation
metrics:
  - acc
  - acc_norm
```

配置对象至少包含：task name/version、dataset schema、prompt builder、inference mode、postprocess pipeline、metrics、few-shot、contamination policy 和安全执行策略。

### 4.3 加载和覆盖规则

固定优先级：

```text
CLI explicit values > YAML values > task defaults
```

要求：

- CLI 和 Python API 调用同一 loader。
- 未知字段、缺失必需字段和类型错误在加载阶段报错。
- 不允许 scorer 再次实现同一配置校验。
- `config_hash` 使用 canonical JSON 计算，写入 provenance、summary 和 cache key。
- 配置版本变化必须使旧 cache 失效；旧字段通过显式 migration，不依赖隐式 fallback。

### 4.4 验收标准

- 新增任务只需添加 task config、adapter 和测试，不修改中心 evaluator whitelist。
- 同一配置通过 CLI/Python API 得到完全相同的 `TaskConfig` 和 hash。
- 修改任意 prompt、metric、postprocess 或字段映射后 cache key 改变。
- 缺字段时在数据准备阶段报出 task name、字段名和配置文件位置。

## 5. P1：统一 ModelBackend 和请求结果契约

### 5.1 当前问题

`online.py`、`mc.py`、`offline.py` 和 `verifier.py` 分别处理请求构造、重试、响应解析、缓存和失败状态。任务层因此需要了解 OpenAI-compatible 或 vLLM 的具体输出形态。

### 5.2 建议接口

已有 MC continuation schema 作为 loglikelihood 的基础，补齐 generate 和统一错误类型：

```python
class ModelBackend(Protocol):
    def generate(
        self, requests: list[GenerateRequest]
    ) -> list[GenerateResult]: ...

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest]
    ) -> list[LoglikelihoodResult]: ...

    def capabilities(self) -> BackendCapabilities: ...
```

`GenerateRequest/Result` 至少包含 request id、prompt/messages、generation params、seed、text、finish reason、token usage、raw response 摘要和 typed error。`BackendCapabilities` 声明是否支持：多样本 `n`、continuation echo、token IDs、token offsets、logprobs、stop tokens、batch 和 deterministic seed。

### 5.3 迁移顺序

1. 先把 retry、timeout、cache lookup/write 和 error taxonomy 放到公共 transport 层。
2. 为 OpenAI-compatible、vLLM offline 和 verifier 各实现 adapter。
3. 让现有 runner 通过 backend 调用，保留原 CLI 和 JSONL schema。
4. 用 fake backend 替换现有大量 MagicMock，覆盖完整 inference -> result -> scorer 流程。
5. 最后移除任务代码对 `openai.OpenAI`、`LLM` 和 `SamplingParams` 的直接依赖。

### 5.4 验收标准

- 任务 scorer 只接收 typed result，不依赖具体模型客户端。
- 相同 request 在不同 backend 的可比能力范围内产生相同 normalized result。
- 不支持 continuation 或 token offset 的 backend 在 `strict` 模式早期报 capability error。
- 失败类型至少区分 connection、rate_limit、timeout、invalid_response、context_length 和 backend_unsupported。
- fake backend 能在不加载模型的情况下测试三类任务完整流程。

## 6. P1：复现机制和 run manifest 收口

### 6.1 已有基础与剩余缺口

Python/NumPy/PyTorch、OpenAI seed、vLLM `SamplingParams.seed`、MC few-shot 文件 hash/示例 ID/final prompt hash 已有实现。剩余问题是不同 runner 的 provenance 字段不完全一致，也没有一个可直接归档和复核的 run manifest。

### 6.2 统一 provenance

每次推理和评测至少生成一份 manifest，包含：

- `schema_version`、run ID、开始/结束时间和 CLI 完整参数。
- task name/version、`config_hash`、dataset path/hash、每个 `doc_id` 的 sample index。
- model name/revision、backend、prompt/template/postprocess 版本。
- Python、依赖 lock/hash、CUDA/vLLM/transformers 版本和设备信息。
- seed、Python/NumPy/PyTorch seed；不可用的随机源写 `null`。
- few-shot source/hash、selected example IDs 和最终 prompt hash。
- cache namespace、rank、hit/miss/corrupt 统计。
- 失败、跳过、超时数量和错误分类。

manifest 应写为 `output.manifest.json`，并在 summary 中保存其 hash；per-item 只保存必要的 manifest/run ID，避免重复写入大对象。

### 6.3 复现测试

- 同一 fake backend、输入、task config、model revision 和 seed，运行两次得到相同 rendered prompt、sample index、per-item prediction 和 aggregate。
- 只改变 seed 时，随机采样任务的请求或 few-shot selection 必须改变；temperature=0 的 deterministic backend 可以明确声明 prediction 不变。
- pass@k 需要验证 sample order、problem grouping、重复运行结果和缺失样本 resume。
- run manifest 缺失依赖版本时写 `null` 并 warning，不能伪造完整复现声明。

### 6.4 验收标准

- 三类 runner 使用同一 manifest schema。
- summary、per-item output 和 manifest 的 task/dataset/model/config hash 互相一致。
- manifest 足以解释一次结果为何与另一次结果不同。
- 相同配置的 resume 不重复计数，不改变原有 sample index。

## 7. P1：任务级 contamination 检查

### 7.1 当前状态

`llmeval/tasks/provenance.py` 已支持本地 JSONL/text source、规范化 exact substring、query/source hash 和基础 summary。当前缺少 harness 风格的 task-specific query、多种 overlap 规则和明确的排除分母。

### 7.2 设计

为每个 task adapter 声明 `build_decontamination_query(item) -> str`：

- 数学：规范化题目文本，保留影响题意的数学符号。
- MC：题干、选项文本和必要的 task identifier。
- 代码：prompt、函数名、签名和测试可见的描述，不包含模型生成。

支持以下模式，并记录实际使用的模式和阈值：

- exact normalized substring。
- character n-gram overlap。
- tokenizer/token n-gram overlap。
- 可选 hash-only reference matching。

污染状态统一为：

```text
unchecked -> clean | flagged | excluded
```

建议默认只 `report`，由 CLI 的 `contamination_policy=report|exclude|error` 控制是否排除或中止。排除样本不能进入 effective denominator，flagged 样本保留并单独计数。

### 7.3 验收标准

- contamination 检查不修改原始 prompt、gold 或 generation。
- 每条记录保存 query hash、source dataset/version/hash、匹配模式、阈值和状态。
- summary 同时给出 checked、clean、flagged、excluded，以及排除后的 sample/effective denominator。
- 没有 source 时记录 `unchecked`，不能误报为 clean。
- 数学、MC、代码各有 task-specific query 和 exact/ngram fixture。

## 8. P1：数据集 preflight 和 schema audit

### 8.1 动机

目前 `load_jsonl` 负责解析 JSONL，推理和 scorer 各自做部分字段检查。对于大规模评测，字段语义错误、重复 `doc_id`、choices/gold 不一致和错误的 generation shape 应在推理前报告，而不是在评测中途发现。

### 8.2 目标

增加统一命令和 Python API：

```bash
python -m llmeval.dataset_audit \
  --task-config configs/tasks/mc_mmlu.yaml \
  --input-file data.jsonl \
  --report audit.json
```

audit 至少检查：

- JSONL 行号、对象类型和必需字段。
- `doc_id` 存在、唯一、稳定且不为空。
- task config 声明的 prompt、target、choices、response 字段类型。
- MC choices 数量、gold/choice token 映射和空选项。
- math target 为空/多参考答案/不支持类型。
- code task id、prompt mode、函数签名和 generation list shape。
- 重复 prompt、重复 `(doc_id, sample_index)` 和异常 Unicode/超长输入。

报告应包含错误、warning、样本计数、跳过计数和 dataset hash；默认错误阻止推理，warning 允许显式 `--allow-warnings` 继续。

### 8.3 验收标准

- 三类任务各有 schema fixture 和错误样例。
- audit 使用与 inference/evaluator 相同的 loader 和 TaskConfig，不出现两套字段语义。
- 失败报告包含行号、`doc_id`、字段路径和修复建议。
- dataset hash 进入 provenance 和 cache key，数据改动不会复用旧结果。

## 9. P1：harness parity 和 CI 质量门禁

### 9.1 当前状态

仓库已有 `.pre-commit-config.yaml`，但没有 `.github/workflows`；MC parity 已有局部 fixture，数学/代码 parity 尚未建立。当前全量本地验证包括 pytest、mypy 和 ruff check，但没有在 push/PR 上自动执行的门禁。

### 9.2 CI 分层

建立最小 GitHub Actions workflow：

1. **快速检查**：Python 3.10/3.12，`ruff check`、格式检查、mypy、纯 Python 单元测试。
2. **任务正确性**：math/MC/code fixture、resume、cache corruption、denominator 和 parity tests。
3. **可选后端**：vLLM、Docker/nsjail 和本地 harness 使用独立 job；缺少 GPU/容器能力时明确 skip，不让核心 job 假绿。

固定依赖版本或 lock 文件，缓存 pip 依赖，但不缓存模型输出作为测试 oracle。

### 9.3 差异报告

parity job 失败时上传：fixture、task config、dataset hash、git revision、per-item diff、aggregate diff 和环境 manifest。允许有记录在案的差异，但不能把差异吞掉为测试通过。

### 9.4 验收标准

- PR 必须通过 lint、类型、纯 Python 测试和三类任务 parity。
- 每次修改 task config、filter、scorer 或 backend 都会触发相关 job。
- 测试依赖缺失时显式标记 skip/reason，不降低验收标准。
- CI 使用最小 fixture，不下载大模型，不依赖外部在线 API。

## 10. P2：多任务、task group 和分布式评测

### 10.1 当前缺口

当前 CLI 一次处理一个 task；已有线程/进程级并行和 rank-aware cache，但没有 task group、样本选择、rank 输出合并或 run-level 聚合。

### 10.2 目标接口

```yaml
run:
  seed: 7
  limit: 1000
  sample_indices: null
tasks:
  - name: math_opensource/aime24
  - name: mc_opensource/mmlu
  - name: code_opensource/humaneval
aggregation:
  mode: macro
  include_failed: false
```

要求：

- task group 中每个 task 独立解析 config、生成 dataset hash、输出 JSONL/summary/manifest。
- 支持 `limit`、比例 limit、显式 sample index 和稳定 doc_id shard。
- rank 输出写到独立目录，rank 0 负责去重、校验和 merge。
- merge 以 `(task_name, task_version, doc_id, sample_index)` 去重，禁止按文件行号合并。
- task-level、macro 和 micro 聚合都记录公式、有效分母和 uncertainty。
- 失败 rank 可单独重试，重试不重复计数成功样本。

### 10.3 验收标准

- 单进程与多 rank 在相同 fixture 上 aggregate 一致。
- rank 0 merge 对重复、缺失、冲突 result 有明确错误报告。
- task group summary 同时包含 task-level、macro、micro 和总 denominator。
- `limit` 和 sample selection 在 resume、cache 和 provenance 中保持稳定。
- 不同 task/model/rank 不会错误共享 inference/evaluation cache。

## 11. P2：结果 schema、migration 和 backend capability matrix

### 11.1 为什么补充

当前 JSONL、summary、cache 和 provenance 已有多个版本化字段，但尚无集中 schema registry 和兼容迁移工具。随着 TaskConfig、ModelBackend 和 sandbox 加入，旧结果需要可识别、可拒绝或可迁移。

### 11.2 设计建议

- 为 inference result、per-item evaluation、summary、manifest 和 cache envelope 分别声明 `schema_version`。
- 提供只读 `validate_result` 和显式 `migrate_result --from X --to Y`，禁止读取未知新版本后静默猜测字段。
- 为 task config、postprocess pipeline、backend capability 和 sandbox 定义版本字符串。
- 建立 capability matrix：task mode x backend x metric x security level，CLI 启动阶段检查不支持组合。
- 将 deprecation、migration 和 breaking change 写入 CHANGELOG/文档，不通过隐式 fallback 延长旧格式生命周期。

### 11.3 验收标准

- 缺失或未知 schema version 时给出可操作错误。
- migration 前后 `doc_id`、sample index、raw generation 和 metric 语义不变。
- capability 不满足时在模型请求前失败，错误包含 task、backend、required capability 和替代模式。
- schema/migration/capability 均有 golden fixture。

## 12. 实施顺序与依赖

### 阶段一：P0 正确性和安全

1. 代码强隔离接口和最小容器 backend（第 2 节）。
2. 数学/代码 parity fixture 和差异报告（第 3 节）。
3. 将纯 Python correctness/parity 测试纳入最小 CI（第 9 节）。

### 阶段二：P1 配置、请求和复现

1. TaskConfig dataclass/YAML loader 和 config hash（第 4 节）。
2. dataset preflight，使推理使用的 schema 与配置统一（第 8 节）。
3. ModelBackend、错误 taxonomy 和 capabilities（第 5 节）。
4. run manifest、依赖环境和端到端复现测试（第 6 节）。
5. task-specific contamination（第 7 节）。

### 阶段三：P2 规模化和兼容性

1. task group、limit、shard、rank merge 和 run-level aggregation（第 10 节）。
2. result schema migration 和 capability matrix（第 11 节）。
3. 多 rank 统计汇总、缓存生命周期策略和失败 rank 重试。

## 13. 最小验收命令

每个阶段至少执行：

```bash
pytest -q
mypy --follow-imports=skip --ignore-missing-imports llmeval
ruff check llmeval tests scripts
ruff format --check llmeval tests scripts
git diff --check
```

三类任务的最低覆盖：

| 任务 | 必测内容 |
| --- | --- |
| 数学 | math_verify、normalize fallback、think 清理、空答案、超时、parity、cache key |
| MC | generate、first-token、完整 continuation、多 token/中文、`acc_norm`、多样本、失败和 parity |
| 代码 | HumanEval、MBPP、prompt mode、代码提取、语法/运行时错误、超时、pass@k、sandbox 和 parity |

## 14. 交付原则

- 任务正确性优先于吞吐量；严格模式不满足后端能力时失败，不静默降级。
- 所有结果必须能追溯到 task config、dataset hash、model revision、prompt hash、seed 和代码版本。
- 不把模型错误答案与执行基础设施失败混为一谈。
- 新的 fallback、兼容字段和 migration 必须有版本、测试和弃用计划。
- CLI、Python API、JSONL、summary、manifest 和 cache 使用同一套字段语义。
