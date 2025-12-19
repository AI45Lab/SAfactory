python examples/base_eval.py \
  --env-config-yaml "env/dwgym/dw_config.yaml" \
  --max-workers 64 \
  --max-steps 1000 \
  --visual-save-path "visualize/dwgym" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://localhost:30000/v1" \
  --llm-model "Qwen2.5-VL-7B-Instruct" \
  --llm-temperature 0.3