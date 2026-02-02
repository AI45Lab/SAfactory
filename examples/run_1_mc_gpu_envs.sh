ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${ROOT_DIR}/examples/base_eval.py" \
  --env-config-yaml "${ROOT_DIR}/env/mcgpugym/config/mc_gpu_env.yaml" \
  --max-workers 8 \
  --max-steps 1000 \
  --visual-save-path "${ROOT_DIR}/env/mcgpugym/visualize/test_gpu" \
  --llm-api-key "EMPTY" \
  --llm-base-url "http://0.0.0.0:8001/v1" \
  --llm-model "/mnt/shared-storage-user/evobox-share/hf-hub/models--Qwen2.5-VL-7B-Instruct" \
  --llm-temperature 0.3
