python /home/jianzhnie/llmtuner/llm/LLMEval/scripts/data_process/data_split.py \
  --input /home/jianzhnie/llmtuner/llm/LLMEval/output/opg_32b_step_800/aime24_yarn.jsonl \
  --output_dir /home/jianzhnie/llmtuner/llm/LLMEval/data/aime_yarn/ \
  --prefix aime24_shard \
  --num_shards 1

python /home/jianzhnie/llmtuner/llm/LLMEval/scripts/data_process/data_split.py \
  --input /home/jianzhnie/llmtuner/llm/LLMEval/output/opg_32b_step_800/aime25_yarn.jsonl \
  --output_dir /home/jianzhnie/llmtuner/llm/LLMEval/data/aime_yarn/ \
  --prefix aime25_shard \
  --num_shards 4
