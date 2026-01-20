ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/examples/base_eval.py" \
  --env-config-yaml "${ROOT_DIR}/env/mc/mc_env.yaml" \
  --max-workers 64 \
  --max-steps 1000 \
  --visual-save-path "${ROOT_DIR}/env/mc/visualize/test1107" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://100.99.167.252:8001/v1" \
  --llm-model "/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen2.5-VL-7B-Instruct" \
  --llm-temperature 0.3