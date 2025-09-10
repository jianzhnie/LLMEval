# LLM 推理模型评测系统

<div align="center">

[English](README.md) | [简体中文](README_zh.md)

</div>



[toc]



## 概述
我们成功在 AIME 2024 和 AIME 2025 基准测试上复现了多个开源模型的结果。

对于像 AIME24 这样只包含 30 个问题的基准测试，采样多个响应至关重要，因为这可能会在重复运行之间引入较高的方差。每个提示采样的响应数量可能解释了我们的评估结果与 DeepSeek 报告的结果之间的细微差异。

### DeepSeek-R1-Distill-Qwen-32B

|  数据集  | (🤗 LLMEval) | DeepSeek-R1-Distill-Qwen-32B（官方报告） |
| :------: | :---------: | :--------------------------------------: |
|  AIME24  |   70.625    |                   72.6                   |
|  AIME25  |   55.052    |                   59.0                   |
| MATH-500 |    93.2     |                   94.3                   |

### DeepSeek-R1-Distill-Qwen-1.5B

|  数据集  | (🤗 LLMEval) | DeepSeek-R1-Distill-Qwen-32B（官方报告） |
| :------: | :---------: | :--------------------------------------: |
|  AIME24  |   27.92    |                   28.9                 |
|  AIME25  |   23.44    |                   23.44                |


### QwQ-32B

| 数据集 | (🤗 LLMEval) | QwQ-32B（官方报告） |
| :----: | :---------: | :-----------------: |
| AIME24 |    78.80    |        79.5         |
| AIME25 |    67.50    |        69.5         |

### Skywork-OR1-32B

| 数据集 | (🤗 LLMEval) | Skywork-OR1-32B（官方报告） |
| :----: | :---------: | :-------------------------: |
| AIME24 |    81.25    |            82.2             |
| AIME25 |    72.66    |            73.3             |

### OpenThinker3-7B

| 数据集 | (🤗 LLMEval) | OpenThinker3-7B（官方报告） |
| :----: | :---------: | :-------------------------: |
| AIME24 |    70.41    |            69.0             |
| AIME25 |    59.16    |            53.3             |

## 安装

### 基础环境配置

| 软件      | 版本       |
| --------- | ---------- |
| Python    | == 3.10    |
| CANN      | == 8.1.RC1 |
| torch     | == 2.5.1   |
| torch_npu | == 2.5.1   |

关于基础环境配置，请参考[此文档](https://gitee.com/ascend/pytorch)。

### vllm 和 vllm-ascend

为了正确使用 vllm 加速推理，您需要使用以下命令编译和安装 vllm 和 vllm-ascend。请注意，安装方法因机器类型而异。

```bash
# vllm
git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt

# 对于 Atlas 200T A2 Box16
VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/

# 对于 Atlas 900 A2 PODc
VLLM_TARGET_DEVICE=empty pip install -e .
# vllm-ascend
git clone -b v0.7.3.post1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
```

对于其他版本的 vllm, 请参考[vllm 官方文档](https://vllm-project.github.io/) 和 [vllm-ascend 官方文档](https://vllm-project.github.io/vllm-ascend/) 获取更多信息。

### llmeval

通过克隆仓库并使用 `pip` 以可编辑模式安装 `llmeval` 包。这将同时安装所有必要的依赖。

```bash
# For github source
git clone https://github.com/jianzhnie/LLMEval.git
# For gitee source
# git clone https://gitee.com/jianzhnie/LLMEval.git
cd LLMEval
pip install -e .
```

## 评测

VLLM 库提供了两种推理模式：在线服务器模式和离线模式。LLMEval 支持这两种模式进行推理评测，以下是这两种方法的使用说明。

### 使用 vLLM 进行在线推理

这种方法需要首先启动一个 vLLM 服务器，然后向其发送请求进行推理。这种方式更加灵活，可以同时处理多个请求。

#### 步骤 1：启动 vLLM 服务器

首先，使用以下命令启动 vLLM 服务器：

```bash
model_path="Qwen/QwQ-32B"  # 或指向模型所在位置的路径
model_name="Qwen/QwQ-32B"

num_gpus=8
max_model_len=32768  # ✅ 支持 32k 上下文
gpu_memory_utilization=0.9  # ✅ 提高内存利用率

python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --trust-remote-code \
    --served-model-name $model_name \
    --tensor-parallel-size $num_gpus \
    --gpu-memory-utilization $gpu_memory_utilization \
    --max-model-len $max_model_len  \
    --enforce-eager \
    --port 8090
```

根据可用设备调整 `tensor_parallel_size` 参数。

详细信息请参考[脚本](./scripts/QwQ/model_server.sh)。

#### 步骤 2：运行推理

启动 vLLM 服务后，运行推理脚本生成响应。

```bash
output_dir="./output/Qwen/QwQ-32B"
model_name="Qwen/QwQ-32B"

base_url="http://127.0.0.1:8090/v1"
n_samples=64  # aime24 和 aime25 的默认样本数

# 如果输出目录不存在则创建
mkdir -p "${output_dir}"

# --- 运行推理任务 ---
# aime24 (重复采样 64 次)
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --system_prompt_type empty \
    --max_workers 8

# aime25 (重复采样 64 次)
python ./llmeval/vllm/online_server.py \
    --input_file "./data/aime25.jsonl" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --base_url "${base_url}" \
    --model_name "${model_name}" \
    --n_samples "${n_samples}" \
    --system_prompt_type empty \
    --max_workers 8
```

详细信息请参考[脚本](./scripts/QwQ/online_infer.sh)。

**注意：** 我们使用重复采样来减少评估方差，但可能需要较长时间才能完成（根据设备情况可能超过8小时）。

##### 参数说明

- `--base_url`：vLLM 服务的基础 URL
- `--model_name`：必须与步骤1中使用的模型名称匹配
- `--n_samples`：每个提示的样本数
  - AIME24 / AIME 25：建议64个样本
- `--input_file`：输入数据文件路径
- `--output_file`：输出结果文件路径，模型响应将存储在 `gen` 字段中
- `--max_workers`：最大并发线程数，用于控制推理速度和资源使用

##### 采样参数

我们使用 `top_p=0.95`、`temperature=0.6`、`top_k=40`、`max_tokens=32768` 进行采样。

##### 恢复中断的推理

如果推理过程中断，只需重新运行相同的命令即可恢复。脚本会自动读取之前的输出文件，并处理尚未完成所需样本数的提示。

### 使用 vLLM 进行离线推理

这种方法涉及将模型加载到内存中，然后在本地运行推理。这种方式比在线服务器模式更快速和高效，但需要更多内存，可能不适合大型模型。

```bash
# --- 配置 ---
output_dir="./output/Qwen/QwQ-32B"
model_name_or_path="Qwen/QwQ-32B"
n_samples=64  # aime24 和 aime25 的默认样本数

# 如果输出目录不存在则创建
mkdir -p "${output_dir}"

# --- 运行推理任务 ---
# aime24 (重复采样 64 次)
python llmeval/vllm/offline_infer.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --batch_size 32 \
    --model_name_or_path "${model_name_or_path}" \
    --trust_remote_code \
    --max_model_len 32768 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 8 \
    --enforce_eager \
    --n_samples "${n_samples}"

# aime25 (重复采样 64 次)
python llmeval/vllm/offline_infer.py \
    --input_file "./data/aime25.jsonl" \
    --output_file "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --batch_size 32 \
    --model_name_or_path "${model_name_or_path}" \
    --trust_remote_code \
    --max_model_len 32768 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 8 \
    --enforce_eager \
    --n_samples "${n_samples}"
```

详细信息请参考[脚本](./scripts/QwQ/offline_infer.sh)。

结果格式与在线服务器模式一致，模型响应将存储在 `gen` 字段中。

### 步骤 3：评分

完成推理后，使用以下命令进行评分：

```bash
output_dir="./output/Qwen/QwQ-32B"
n_samples=64  # aime24 和 aime25 的默认样本数

# 评估输出目录
reval_dir="${output_dir}/eval_score"
# 如果评估目录不存在则创建
mkdir -p "${reval_dir}"

# --- 评估每个任务 ---
# 评估 aime24
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime24_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime24" \
    --max_workers 16 \
    > "${reval_dir}/aime24_bz${n_samples}_res_result.txt"

# 评估 aime25
python ./llmeval/tasks/math_eval/eval.py \
    --input_path "${output_dir}/aime25_bz${n_samples}.jsonl" \
    --cache_path "${reval_dir}/aime25_bz${n_samples}.jsonl" \
    --task_name "math_opensource/aime25" \
    --max_workers 16 \
    > "${reval_dir}/aime25_bz${n_samples}_res_result.txt"
```

详细信息请参考[脚本](./scripts/get_scores.sh)。

#### 参数说明

- `--input_path`：输入文件路径，可以直接使用多线程推理的输出文件或其他格式一致的文件。要求：
  - JSONL 格式
  - 包含 `prompt` 和对应字段
  - 模型响应存储在 `gen` 字段中
- `--cache_path`：用于存储评估过程中临时文件的缓存目录
- `--task_name`：评估任务名称，必须是以下选项之一：
  - `math_opensource/aime24`
  - `math_opensource/aime25`
- `max_workers`：最大并发线程数，用于控制评估速度和资源使用。


### 上下文长度和模型长度外推

很多模型在预训练中的上下文长度最长为 32,768 个 token。为了处理显著超过 32,768 个 token 的上下文长度，应应用 RoPE 缩放技术。我们已经验证了 [YaRN](https://arxiv.org/abs/2309.00071) 的性能，这是一种增强模型长度外推的技术，可确保在长文本上的最佳性能。

vLLM 支持 YaRN，可以配置为

```bash
python llmeval/vllm/offline_infer.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "${output_dir}/aime24_bz${n_samples}.jsonl" \
    --batch_size 32 \
    --model_name_or_path "${model_name_or_path}" \
    --trust_remote_code \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 8 \
    --enforce_eager \
    --n_samples "${n_samples}" \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --max-model-len 131072
```

> 备注
>vLLM 实现了静态 YaRN，这意味着无论输入长度如何，缩放因子都保持不变，**这可能会对较短文本的性能产生影响。** 我们建议仅在需要处理长上下文时添加 `rope_scaling` 配置。还建议根据需要调整 `factor`。例如，如果您的应用程序的典型上下文长度为 65,536 个 token，则最好将 `factor` 设置为 2.0。

> 备注
> 如果未指定 `--max-model-len`，`config.json` 中的默认 `max_position_embeddings` 被设置为 40,960，vLLM 将使用该值。此分配包括为输出保留 32,768 个 token，为典型提示保留 8,192 个 token，这足以应对大多数涉及短文本处理的场景，并为模型思考留出充足空间。如果平均上下文长度不超过 32,768 个 token，我们不建议在此场景中启用 YaRN，因为这可能会降低模型性能。


##
