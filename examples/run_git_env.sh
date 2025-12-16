python examples/base_eval.py \
  --env-config-yaml "env/gitgym/git_env.yaml" \
  --max-workers 8 \
  --max-steps 50 \
  --visual-save-path "git_env_visuals_qwen32b" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://localhost:8001/v1" \
  --llm-model "Qwen/Qwen2.5-Coder-32B-Instruct" \
  --llm-temperature 0.3
