python examples/base_eval.py \
  --env-config-yaml "env/bench_env/dabstep/dabstep_config.yaml" \
  --max-workers 64 \
  --max-steps 1000 \
  --visual-save-path "visualize/bench_env" \
  --agent-api-key "EMPTY" \
  --agent-base-url "http://localhost:8001/v1" \
  --agent-model "Qwen3-30B-Instruct" \
  --agent-temperature 0.3