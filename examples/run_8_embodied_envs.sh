python /workspace/AIEvoBox/examples/base_eval.py \
  --env-config-yaml "/workspace/AIEvoBox/env/embodiedgym/embodied_config.yaml" \
  --max-workers 1 \
  --max-steps 50 \
  --visual-save-path "/workspace/AIEvoBox/visualize/test1111-1" \
  --agent-api-key "sk-By9e5cTrJaCDSDluDdfMKoe81rLDzOwBBr2HBkFJ4E0wMIO2" \
  --agent-base-url "http://35.220.164.252:3888/v1" \
  --agent-model "gpt-4o-mini" \
  --agent-temperature 0.3
