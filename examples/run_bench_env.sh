python examples/base_eval.py \
  --env-config-yaml "env/dabstep/dabstep_config.yaml" \
  --max-workers 64 \
  --max-steps 1000 \
  --visual-save-path "visualize/dabstep" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://localhost:8001/v1" \
  --llm-model "Qwen3-30B-Instruct" \
  --llm-temperature 0.3