#!/bin/bash

# 使用 Qwen Coder 32B 模型运行git_env环境的脚本
# API配置：
#   URL: http://35.220.164.252:3888/v1
#   API Key: sk-U2jeHgDOjX0mpH2mymoM49eidrZljpxoBa3fyaRzzUngOhX2
#   模型: Qwen/Qwen2.5-Coder-32B-Instruct

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "运行 GitGym 环境（使用 Qwen Coder 32B）"
echo "=========================================="
echo "API URL: http://35.220.164.252:3888/v1"
echo "模型: Qwen/Qwen2.5-Coder-32B-Instruct"
echo "=========================================="
echo ""

python examples/base_eval.py \
  --env-config-yaml "env/gitgym/git_env.yaml" \
  --max-workers 8 \
  --max-steps 50 \
  --visual-save-path "./git_env_visuals_qwen32b" \
  --agent-api-key "sk-U2jeHgDOjX0mpH2mymoM49eidrZljpxoBa3fyaRzzUngOhX2" \
  --agent-base-url "http://35.220.164.252:3888/v1" \
  --agent-model "Qwen/Qwen2.5-Coder-32B-Instruct" \
  --agent-temperature 0.3


