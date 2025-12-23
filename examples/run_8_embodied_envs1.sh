python examples/base_eval.py \
  --env-config-yaml "/mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/AIEvoBox/env/embodiedgym/embodied_config.yaml" \
  --max-workers 1 \
  --max-steps 50 \
  --visual-save-path "/mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/eval/visualize/test1111-1" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://10.102.235.34:8001/v1" \
  --llm-model "/mnt/shared-storage-user/steai-share/hf-hub/Qwen2.5-VL-7B-Instruct" \
  --llm-temperature 0.3
  