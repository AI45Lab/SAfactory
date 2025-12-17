python examples/base_eval.py \
  --env-config-yaml "env/gitgym/git_env.yaml" \
  --max-workers 64 \
  --max-steps 50 \
  --visual-save-path "visualize/git" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://localhost:30000/v1" \
  --llm-model "Qwen3-30B-Instruct" \
  --llm-temperature 0.3
