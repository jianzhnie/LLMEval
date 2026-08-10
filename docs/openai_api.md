# OpenAI API 常用参数

本文档介绍文本生成中常用的 OpenAI API 请求参数。LLMEval 当前使用
`client.chat.completions.create()`，因此以下内容以 Chat Completions API 为主，
并在后文说明 Responses API 的字段差异。

项目要求 `openai>=2.0.0`，以下 Python 示例按当前 2.x SDK 编写。

参数是否可用取决于具体模型。调用前应查看目标模型的能力说明；某些推理模型不支持
`temperature`、`top_p`、`stop` 或 logprobs 等参数。

官方参考：

- [Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/)
- [Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create/)

## 最小请求

```python
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
)

print(completion.choices[0].message.content)
```

对应的 cURL 请求：

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

## 输入参数

| 参数 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `model` | `str` | 要调用的模型 ID | `"gpt-4o"` |
| `messages` | `list` | 按顺序提供的对话消息 | `[{"role": "user", "content": "Hello"}]` |
| `reasoning_effort` | `str` | 推理模型的思考强度，支持值依模型而定 | `"medium"` |
| `verbosity` | `str` | 控制回答详细程度 | `"low"`、`"medium"`、`"high"` |

常见消息角色包括 `system`、`developer`、`user`、`assistant` 和 `tool`。具体可用角色
取决于模型和 API 版本。

## 生成参数

| 参数 | 类型 | 含义 | 常用示例 |
|---|---|---|---|
| `max_completion_tokens` | `int` | 最大输出 token 数，包含可见输出和推理 token | `2048` |
| `temperature` | `float` | 控制随机性；`0` 更稳定，值越高结果越随机 | `0.0`、`0.7` |
| `top_p` | `float` | nucleus sampling 概率阈值 | `0.95` |
| `n` | `int` | 单次请求为每个输入生成多少个候选结果 | `1`、`4` |
| `stop` | `str` 或 `list[str]` | 遇到指定文本时停止生成 | `"</answer>"` |
| `seed` | `int` | 尽力复现采样结果，不保证跨版本完全确定 | `42` |
| `prediction` | `dict` | 提供预期输出内容，用于内容大部分已知的生成任务 | `{"type": "content", "content": "..."}` |

通常只调整 `temperature` 或 `top_p` 其中一个。确定性评测一般使用
`temperature=0`；需要多样性时可使用 `temperature=0.6` 至 `1.0`。

OpenAI API 没有为所有模型承诺统一的 `max_completion_tokens` 默认值。LLMEval 的
online/offline 默认值为 `32768`，MC generate 默认值为 `2048`；调用时都会显式发送。

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Give me three title ideas."}],
    max_completion_tokens=256,
    temperature=0.7,
    top_p=1.0,
    n=3,
    seed=42,
)

for choice in completion.choices:
    print(choice.message.content)
```

LLMEval 的配置和 CLI 统一使用 `max_completion_tokens`。online、MC generate 和默认的
MC first-token loglikelihood 都调用 Chat Completions；offline 仅在构造 vLLM
`SamplingParams` 时映射为 vLLM 自己的 `max_tokens` 字段。LLMEval 为每个样本发送独立
请求，因此 `n_samples` 表示展开后的请求数量，而不是直接发送 OpenAI 的 `n` 参数。

仅当显式设置 `--loglikelihood_mode continuation` 时，MC 才使用传统
`/v1/completions` 端点及其 `max_tokens`、`echo` 参数。这是为了获取完整 continuation
的 prompt-token logprobs；当前 Chat Completions 和 Responses API 没有等价能力。

## 概率与重复控制

| 参数 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `frequency_penalty` | `float` | 根据 token 已出现的次数降低重复概率 | `0.2` |
| `presence_penalty` | `float` | token 只要出现过就施加惩罚，鼓励引入新内容 | `0.2` |
| `logit_bias` | `dict[str, int]` | 按 token ID 调整生成概率，通常取 `-100` 至 `100` | `{"198": -100}` |
| `logprobs` | `bool` | 返回输出 token 的对数概率 | `True` |
| `top_logprobs` | `int` | 每个位置额外返回概率最高的若干 token | `5` |

`top_logprobs` 需要同时设置 `logprobs=True`：

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Answer only A, B, C, or D."}],
    max_completion_tokens=1,
    temperature=0,
    logprobs=True,
    top_logprobs=5,
)

token = completion.choices[0].logprobs.content[0]
print(token.token, token.logprob)
```

## 输出格式

| 参数 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `response_format` | `dict` | 请求纯文本、JSON object 或 JSON Schema 结构化输出 | 见下例 |
| `modalities` | `list[str]` | 指定输出模态，例如文本或音频 | `["text"]` |
| `audio` | `dict` | 配置音频格式和声音，需要模型支持音频输出 | `{"format": "wav", "voice": "alloy"}` |

JSON Schema 示例：

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: Alice is 30 years old."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        },
    },
)

print(completion.choices[0].message.content)
```

## 工具调用

| 参数 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `tools` | `list` | 声明模型可以调用的函数或工具 | 见下例 |
| `tool_choice` | `str` 或 `dict` | 选择 `none`、`auto`、`required` 或指定工具 | `"auto"` |
| `parallel_tool_calls` | `bool` | 是否允许模型在一次回答中并行调用多个工具 | `True` |

```python
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the weather in Shanghai?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ],
    tool_choice="auto",
    parallel_tool_calls=False,
)

print(completion.choices[0].message.tool_calls)
```

## 流式输出

| 参数 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `stream` | `bool` | 使用 Server-Sent Events 逐步返回结果 | `True` |
| `stream_options` | `dict` | 配置流式响应，例如返回 token usage | `{"include_usage": True}` |

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short greeting."}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 缓存、存储与服务参数

| 参数 | 含义 | 示例 |
|---|---|---|
| `store` | 是否保存响应供后续产品能力使用 | `False` |
| `metadata` | 附加用于查询和管理的键值元数据 | `{"benchmark": "aime24"}` |
| `service_tier` | 选择服务等级，具体值取决于账户和模型 | `"auto"` |
| `prompt_cache_key` | 指定稳定缓存键，提高相似请求的缓存命中率 | `"aime24-v1"` |
| `prompt_cache_options` | 配置 prompt cache 行为 | `{...}` |
| `prompt_cache_retention` | 设置支持的缓存保留策略 | `"24h"` |
| `safety_identifier` | 稳定、非直接身份信息的终端用户标识 | `"hashed-user-id"` |
| `moderation` | 配置支持的审核行为 | `{...}` |
| `web_search_options` | 配置 Chat Completions 的网页搜索能力 | `{...}` |

## Responses API 差异

新应用也可以使用 `client.responses.create()`。常用字段映射如下：

| Chat Completions | Responses API | 含义 |
|---|---|---|
| `messages` | `input` | 用户输入或多模态输入 |
| system/developer message | `instructions` | 高层行为指令 |
| `max_completion_tokens` | `max_output_tokens` | 最大输出 token 数 |
| `response_format` | `text.format` | 结构化文本输出配置 |
| `reasoning_effort` | `reasoning.effort` | 推理强度 |
| 无直接对应 | `previous_response_id` | 延续之前的响应 |
| 无直接对应 | `conversation` | 将响应关联到会话 |
| 无直接对应 | `background` | 后台执行长任务 |
| 无直接对应 | `truncation` | 输入超长时的截断策略 |

```python
response = client.responses.create(
    model="gpt-4o",
    instructions="Answer concisely.",
    input="What is 2 + 2?",
    max_output_tokens=128,
    temperature=0,
)

print(response.output_text)
```

Responses API 没有直接提供 Chat Completions 的 `n`、`seed`、`stop`、
`frequency_penalty` 和 `presence_penalty` 等字段。不要在两个 API 之间直接复制全部参数。

## 旧参数迁移

新代码不应继续发送以下 Chat Completions 兼容字段：

| 旧参数 | 推荐参数 | 迁移说明 |
|---|---|---|
| `max_tokens` | `max_completion_tokens` | Chat Completions 中已弃用；LLMEval 配置及当前 Chat 请求使用新字段 |
| `functions` | `tools` | 将函数定义放入 `tools`，并设置 `type="function"` |
| `function_call` | `tool_choice` | 使用 `tool_choice` 控制自动、必须或指定工具调用 |
| `user` | `safety_identifier`、`prompt_cache_key` | 根据安全标识和缓存路由两种用途分别迁移 |

## SDK 配置与请求参数的区别

以下字段由 OpenAI Python SDK 处理，不是模型请求体中的生成参数：

| 参数 | 使用位置 | 含义 |
|---|---|---|
| `api_key` | `OpenAI(...)` | API 认证密钥 |
| `base_url` | `OpenAI(...)` | API 服务地址 |
| `organization` | `OpenAI(...)` | OpenAI organization ID |
| `max_retries` | `OpenAI(...)` | SDK 自动重试次数 |
| `timeout` | client 或单次请求 | HTTP 超时时间 |
| `extra_headers` | 单次请求 | 添加自定义 HTTP header |
| `extra_query` | 单次请求 | 添加自定义 URL query 参数 |
| `extra_body` | 单次请求 | 添加服务商扩展请求字段 |

## vLLM 和 SGLang 扩展

下列字段不是 OpenAI 标准请求参数：

| 参数 | 常见实现 | 含义 |
|---|---|---|
| `top_k` | vLLM、SGLang | 只在概率最高的 K 个 token 中采样 |
| `repetition_penalty` | vLLM、Transformers | 对已生成 token 施加重复惩罚 |
| `do_sample` | Transformers | 在采样和 greedy decoding 之间切换 |
| `skip_special_tokens` | vLLM、Transformers | 解码时移除特殊 token |
| `enable_thinking` | 部分模型模板 | 启用模型特定的 thinking 模板行为 |
| `chat_template_kwargs` | vLLM、SGLang | 向 chat template 传递扩展参数 |

LLMEval 的 online 模式通过 `--extra_body` 显式发送这些扩展。不要将它们发送给
不支持对应字段的官方 OpenAI 模型。

LLMEval CLI 的 `--tool_choice` 支持 `none`、`auto` 和 `required`。标准 API 中指定
具体函数需要传入结构化对象，不应直接把函数名作为 `--tool_choice` 的字符串值。

```bash
python ./llmeval/inference/online.py \
  --input_file ./data/aime24.jsonl \
  --output_file ./output/aime24.jsonl \
  --base_url http://127.0.0.1:8090/v1 \
  --model_name Qwen/QwQ-32B \
  --temperature 0.6 \
  --top_p 0.95 \
  --extra_body '{"top_k": 40, "chat_template_kwargs": {"enable_thinking": false}}'
```

连接官方 OpenAI API 时保持 `extra_body={}`，只发送目标模型明确支持的标准参数。
