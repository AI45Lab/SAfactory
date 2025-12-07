python examples/base_eval.py \
  --env-config-yaml "/mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/AIEvoBox/env/tradinggym/trading_env.yaml" \
  --max-workers 64 \
  --max-steps 1000 \
  --visual-save-path "/mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/eval/visualize/test1107" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://100.97.104.117:8001/v1" \
  --llm-model "/mnt/shared-storage-user/steai-share/hf-hub/Qwen2.5-VL-7B-Instruct" \
  --llm-temperature 0.3