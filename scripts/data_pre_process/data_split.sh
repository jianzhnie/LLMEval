python /home/jianzhnie/llmtuner/llm/LLMEval/scripts/data_pre_process/data_split.py \
  --input /home/jianzhnie/llmtuner/llm/LLMEval/data/HARP/HARP_filter.jsonl \
  --output_dir /home/jianzhnie/llmtuner/llm/LLMEval/data/HARP/ \
  --prefix HARP_shard \
  --num_shards 32