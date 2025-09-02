# gitee
git config --global user.name 'jianzhnie' 
git config --global user.email 'jianzhnie@126.com'

source set_env.sh
# set_env
source /root/llmtuner/miniconda3/bin/activate vllm_073
# pre-commit
pre-commit run --file /root/llmtuner/llm/LLMEval/llmeval/*/*