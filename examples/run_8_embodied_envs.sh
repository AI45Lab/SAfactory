python /workspace/AIEvoBox/examples/base_eval.py \
  --env-config-yaml "/workspace/AIEvoBox/env/embodiedgym/embodied_config.yaml" \
  --max-workers 1 \
  --max-steps 50 \
  --visual-save-path "/workspace/AIEvoBox/visualize/test1111-1" \
  --llm-api-key "sk-By9e5cTrJaCDSDluDdfMKoe81rLDzOwBBr2HBkFJ4E0wMIO2" \
  --llm-base-url "http://34.13.73.248:3888/v1" \
  --llm-model "gpt-4o-mini" \
  --llm-temperature 0.3
