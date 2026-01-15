#!/bin/bash
# set_env
source set_env.sh

# pre-commit
pre-commit run --file llmeval/*/*

# 查找空节点
bash /home/jianzhnie/llmtuner/tools/find_null_nodes.sh ./scripts/nodes/node_list1.txt
bash /home/jianzhnie/llmtuner/tools/find_null_nodes.sh ./scripts/nodes/node_list3.txt
bash /home/jianzhnie/llmtuner/tools/find_null_nodes.sh ./scripts/nodes/node_list_all.txt

# 停止所有节点
bash /home/jianzhnie/llmtuner/llm/LLMReasoning/scripts/common/kill_multi_nodes.sh ./node_list.txt
bash /home/jianzhnie/llmtuner/llm/LLMReasoning/scripts/common/kill_multi_nodes.sh ./node_list_all.txt
bash /home/jianzhnie/llmtuner/llm/LLMReasoning/scripts/common/kill_multi_nodes.sh ./scripts/nodes/node_list_all.txt

# data_parallel_infer
nohup bash ./scripts/data_parallel_infer/start_parallel_infer_tp8.sh > infer_57k.log 2>&1 &
nohup bash ./scripts/data_parallel_infer/start_parallel_eval_fp16.sh > infer_fp16_pg.log 2>&1 &

# data parallel eval
nohup bash ./scripts/data_parallel_infer/start_parallel_eval_tp8_fp16.sh > auto_eval_fp16.log 2>&1 &
nohup bash ./scripts/data_parallel_infer/start_parallel_eval_tp8_fp16_hmmt.sh > auto_eval_fp16_hmmt.log 2>&1 &
nohup bash ./scripts/data_parallel_infer/start_parallel_eval_tp8_fp16_yarn.sh > auto_eval_fp16_pg_yarn.log 2>&1 &

# lmcache
nohup bash ./scripts/lmcache/pcl-reasonver/model_server.sh > output_lmcache.log 2>&1 &
nohup bash ./scripts/lmcache/pcl-reasonver/online_infer.sh > output_lmcache2.log 2>&1 &

# Sglang
nohup bash ./scripts/sglang_backend/QwQ/model_server.sh > output3.log 2>&1 &
nohup bash ./scripts/sglang_backend/QwQ/online_infer.sh > output4.log 2>&1 &
nohup bash ./scripts/sglang_backend/pcl-reasonver-v1/model_server.sh > output3.log 2>&1 &
nohup bash ./scripts/sglang_backend/pcl-reasonver-v1/online_infer.sh > output4.log 2>&1 &

## data parallel eval tp4
nohup bash ./scripts/data_parallel_infer/start_parallel_eval_tp1-8.sh > infer_tp2.log 2>&1 &

# model eval
nohup bash ./scripts/pcl-reasonver-v1/model_server.sh > output1.log 2>&1 &
nohup bash ./scripts/pcl-reasonver-v1/online_infer.sh > output2.log 2>&1 &
