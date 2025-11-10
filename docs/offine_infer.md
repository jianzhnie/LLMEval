
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
