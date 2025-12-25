python /home/jianzhnie/llmtuner/llm/LLMEval/scripts/data_pre_process/data_split.py \
  --input /home/jianzhnie/llmtuner/llm/LLMEval/data/partial_right_problem.jsonl \
  --output_dir /home/jianzhnie/llmtuner/llm/LLMEval/data/Nemotron-Post-Training-Dataset-v1 \
  --prefix partial_right_problem_shard \
  --num_shards 128
