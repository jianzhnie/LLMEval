# 多项选择题的 Loglikelihood 评分方法

多项选择题除了让模型直接生成答案，还可以通过比较候选答案的概率来评分。核心问题
很简单：

> 在同一道题后面，模型认为哪个候选答案最可能出现？

常用方法有两种：

- **First-token**：只比较候选答案的第一个 token。
- **Continuation**：比较候选答案全部 token 的联合概率。

## 先理解概率和 Logprob

语言模型每次根据已有文本，预测下一个 token 的概率。例如：

```text
输入：1 + 1 = ? Answer:

下一个 token 的预测：
    A    概率 0.05
    B    概率 0.80
    C    概率 0.10
    D    概率 0.05
```

这里 `B` 的概率最高，因此模型更倾向于选择 `B`。

实际计算通常使用概率的自然对数，即 logprob：

```text
logprob = log(probability)
```

logprob 一般小于或等于 0，越接近 0 表示概率越高：

|     概率 | Logprob | 含义    |
| -----: | ------: | ----- |
| `0.80` | `-0.22` | 很可能   |
| `0.10` | `-2.30` | 可能性较低 |
| `0.01` | `-4.61` | 很不可能  |

使用 logprob 的好处是：多个 token 的概率原本需要相乘，取对数后可以直接相加，数值
也更稳定。

```text
P(token_1, token_2) = P(token_1) * P(token_2 | token_1)

log P(token_1, token_2)
= log P(token_1) + log P(token_2 | token_1)
```

## 评分前的统一约定

假设题目被整理成下面的上下文：

```text
Question: 1 + 1 = ?
A. 1
B. 2
C. 3
D. 4
Answer:
```

可以评分两类候选答案：

```text
答案标记：A、B、C、D
完整文本：1、2、3、4
```

评分前必须确定使用哪一种形式。所有候选都应采用相同格式，不能一部分评分字母，
另一部分评分完整文本。

还要明确答案前是否包含空格或换行。例如，`"B"` 和 `" B"` 可能是两个不同的
token，得到的概率也可能不同。

## First-token 评分

### 基本原理

First-token 只观察答案位置的下一步预测，并比较每个候选的第一个 token。

```text
                         ┌─ A  logprob = -3.00
题目 + "Answer:" ──模型── ┼─ B  logprob = -0.25  ← 最高
                         ├─ C  logprob = -2.10
                         └─ D  logprob = -4.20

预测答案：B
```

设上下文为 `Context`，第 `i` 个候选答案的第一个 token 为 `t_i`，则：

```text
score_i = log P(t_i | Context)
prediction = argmax(score_i)
```

这里的 `argmax` 表示选择分数最大的候选。因为 logprob 越接近 0 越好，所以示意图中
的 `-0.25` 优于 `-2.10`。

### 计算步骤

```text
准备统一上下文
      │
      ▼
获得下一个 token 的概率分布
      │
      ▼
读取 A、B、C、D 对应 token 的 logprob
      │
      ▼
选择 logprob 最大的候选
```

本地模型可以直接从最后一个位置的 logits 得到完整词表概率：

```python
context_ids = tokenizer.encode(context, return_tensors="pt")
next_token_logits = model(context_ids).logits[0, -1]
next_token_logprobs = log_softmax(next_token_logits, dim=-1)

scores = [next_token_logprobs[token_id] for token_id in candidate_token_ids]
prediction = argmax(scores)
```

远程推理接口通常需要生成一个 token，并要求返回该位置的 top logprobs。具体参数名称
因接口而异，关键是能够得到第一个输出位置的候选 token 概率。

### 为什么适合答案字母

如果候选是 `A`、`B`、`C`、`D`，并且每个字母在目标 tokenizer 中都是一个独立
token，那么一次模型计算就可以比较全部候选，速度快且逻辑直接。

但如果候选是完整文本：

```text
New York
New Jersey
Los Angeles
```

前两个候选拥有相同的首 token `New`。First-token 只能知道模型是否倾向于输出
`New`，无法继续区分 `York` 和 `Jersey`。

### Token 匹配问题

人眼看到的同一个答案字母，经过 tokenizer 后可能有不同形式：

```text
"A"     没有前导空格
" A"    带前导空格
"\nA"   带换行
```

最可靠的方法是使用目标模型的 tokenizer，在真实上下文后检查每个候选对应的 token
ID。通过 token 文本匹配时，要特别处理前导空格和换行；不应假设所有 tokenizer 都以
相同方式编码答案字母。

### Top-k 的限制

有些接口只返回概率最高的 `k` 个 token。例如返回 top 20 时，没有出现的候选并不
代表概率为零，只代表它没有进入前 20 名。

```text
返回结果中有 A、B，缺少 C、D：

A、B：知道准确 logprob
C、D：只知道它们不在 top-k，准确 logprob 未知
```

实现中可以用负无穷表示“未返回”，但要把它理解为缺失标记，而不是真实概率。如果
所有候选都没有进入 top-k，就无法可靠地比较它们。

### 适用场景和限制

First-token 适合：

- 候选是互不相同的单 token 答案标记。
- 模型被明确要求只输出答案标记。
- 所有候选使用相同的空格、大小写和长度格式。
- 推理接口只能返回生成 token 的 logprobs。

First-token 不适合：

- 候选答案包含多个 token。
- 不同候选拥有相同的首 token。
- 模型习惯先输出解释、空格或换行。
- 候选 token 经常无法进入接口返回的 top-k。

## Continuation 评分

### 基本原理

Continuation 不只看第一个 token，而是计算完整候选答案的概率。

以候选 `New York` 为例：

```text
上下文 C
   │
   ├─ 预测 " New"：logprob = -0.40
   │
   └─ 已知前面是 " New"，再预测 " York"：logprob = -0.70

完整得分 = -0.40 + -0.70 = -1.10
```

对于由多个 token 组成的候选答案：

```text
a_i = [t_(i,1), t_(i,2), ..., t_(i,n)]
```

完整得分为：

```text
score_i = log P(a_i | C)
        = sum_j(log P(t_(i,j) | C, t_(i,1), ..., t_(i,j-1)))
```

也就是说，第一个 token 根据题目评分，第二个 token 根据“题目 + 第一个 token”评分，
依此类推，最后把所有 token 的 logprob 相加。

### Teacher Forcing

Continuation 使用 teacher forcing。它不会让模型自由生成答案，而是把待评分的候选
答案作为已知文本逐步送给模型，然后查询模型给这个真实 token 分配了多大概率。

```text
候选答案：New York

步骤 1：输入 C                 → 查询 " New" 的概率
步骤 2：输入 C + " New"       → 查询 " York" 的概率
步骤 3：将两步 logprob 相加    → 得到完整候选分数
```

因此，即使模型自由生成时选择了其他答案，也仍然可以计算每个候选答案各自的概率。

### 计算步骤

```text
                 ┌─ C + 候选 A ─→ 找出候选 token ─→ logprob 求和 ─┐
统一上下文 C ────  ┼─ C + 候选 B ─→ 找出候选 token ─→ logprob 求和  ┼─→ 取最高分
                 ├─ C + 候选 C ─→ 找出候选 token ─→ logprob 求和 ─┤
                 └─ C + 候选 D ─→ 找出候选 token ─→ logprob 求和 ─┘
```

伪代码如下：

```python
scores = []

for candidate in candidates:
    input_ids, candidate_positions = tokenize_with_boundary(context, candidate)
    batch_ids = input_ids.unsqueeze(0)
    logits = model(batch_ids).logits[0]
    logprobs = log_softmax(logits, dim=-1)

    score = sum(
        logprobs[position - 1, input_ids[position]]
        for position in candidate_positions
    )
    scores.append(score)

prediction = argmax(scores)
```

位置 `position` 的 token 由前一个位置的 logits 预测，因此读取概率时通常要向前移动
一个位置。不同推理框架可能已经封装这一步，需要根据接口定义确认。

### 为什么边界很重要

Tokenizer 会结合相邻字符切分文本，因此下面的等式不一定成立：

```text
tokenize(context + candidate)
    不一定等于
tokenize(context) + tokenize(candidate)
```

例如：

```text
上下文："Answer: "
候选：  "B"

完整文本："Answer: B"
```

末尾空格可能和 `B` 一起组成 token `" B"`。如果把空格算在上下文中，却只寻找
token `"B"`，就可能找不到正确的评分区间。

一种常见做法是把上下文末尾的空白移动到候选答案：

```python
scoring_context = context.rstrip()
trailing_space = context[len(scoring_context):]
scored_candidate = trailing_space + candidate
```

完整文本没有改变，只是明确了哪些字符和 token 属于候选答案。

### 如何检查边界是否正确

如果推理后端返回 token 文本和 offset，应检查：

- 第一个候选 token 是否从预期位置开始。
- 所有候选 token 是否连续。
- token 拼接后是否与待评分候选完全相同。
- token、offset 和 logprob 数量是否一致。

还要确认 offset 使用 Unicode 字符位置还是 UTF-8 字节位置。中文等多字节文本在这两种
表示下的 offset 不同，混用会选择错误的 token。

如果无法完整定位候选答案的全部 token，应把该候选标记为评分失败，不能只使用其中
一部分 token 的概率。

### 对推理接口的要求

本地模型通常可以返回所有输入位置的 logits，因此适合 continuation 评分。

远程接口必须能够返回输入 token 的 logprobs，或者提供等价的序列评分接口。只返回
新生成 token logprobs 的聊天接口，不能直接计算已经放入输入中的完整候选答案概率。
如果接口不提供这类能力，可以改用本地评分服务，或者在候选是单 token 标记时使用
first-token。

## 长度归一化

Continuation 将多个通常为负数的 logprob 相加，因此 token 越多，总分往往越低。
这会让较短的候选天然占有一定优势。

```text
短候选：[-0.8]             总分 = -0.8
长候选：[-0.3, -0.3, -0.3] 总分 = -0.9
```

长候选的每个 token 都更有把握，但原始总分仍然略低。根据评测目标，可以选择：

```text
原始总分         = sum(token_logprobs)
Token 平均分     = sum(token_logprobs) / token_count
字符平均分       = sum(token_logprobs) / character_count
UTF-8 字节平均分 = sum(token_logprobs) / byte_count
```

- 原始总分表示完整序列的联合概率。
- 平均分可以减弱长度偏差，但不再表示完整序列概率。
- 不同归一化方法可能产生不同预测，评测时应明确记录所用规则。

First-token 只评分一个 token。如果所有答案标记长度一致，长度归一化通常不会改变结果。

## 两种方法的区别

| 项目         | First-token   | Continuation            |
| ---------- | ------------- | ----------------------- |
| 比较内容       | 候选的第一个 token  | 候选的全部 token             |
| 直观含义       | 模型下一步最想输出什么   | 模型认为整段候选有多可能            |
| 计算成本       | 较低，通常一次计算     | 较高，需要分别或批量评分候选          |
| 多 token 候选 | 不能完整评分        | 可以完整评分                  |
| 共享首 token  | 无法区分          | 可以继续比较后续 token          |
| 长度偏差       | 通常不明显         | 原始总分偏向较短候选              |
| 接口要求       | 下一个 token 的概率 | 输入序列各 token 的概率或 logits |
| 常见用途       | A/B/C/D 等答案标记 | 完整选项文本、多 token 答案       |

## 如何选择

```text
候选是否都是不同的单 token 标记？
                │
         ┌──────┴──────┐
         │             │
        是             否
         │             │
         ▼             ▼
  使用 First-token   接口能否返回完整序列的 token 概率？
                       │
                ┌──────┴──────┐
                │             │
               能             不能
                │             │
                ▼             ▼
       使用 Continuation   调整为单 token 答案标记，
                           或更换评分接口
```

无论选择哪种方法，都应保证：

- 所有候选使用相同的 prompt 和格式。
- 候选 token 与目标模型 tokenizer 的实际切分一致。
- 缺失或对齐失败的概率不会被静默当作正常分数。
- 并列分数采用固定、可复现的处理规则。
- 评测结果明确说明评分方法和长度归一化方式。

