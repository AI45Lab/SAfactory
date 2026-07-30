# Sandbox Mode

Sandbox mode is a rollout runtime alongside Docker and RJob. Safactory creates a Brainbox Sandbox Instance through the OpenSandbox SDK, runs the agent entrypoint inside it, and deletes the instance after rollout and optional evaluation.

## Prerequisites

1. Create a Sandbox Environment in advance using `SandboxAPI.md` at the repository root.
2. Its image must match the agent config `env_image`.
3. It must expose the command port, `44772` by default.
4. Configure data and result storage as Environment volumes; local Docker bind mounts are not translated.
5. The Sandbox must be able to reach `sandbox.gateway_base_url`.

Install dependencies and provide credentials:

```bash
pip install -r requirements.txt
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
```

Copy `config.sandbox.example.yaml` and set `project`, `environment_id`, and a cluster-reachable gateway URL. An agent can override the Environment in its start config. Geo3K uses the same runtime contract:

```yaml
agent_name: geo3k

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./
    target: /tmp/safactory-geo3k
    command: "python /tmp/safactory-geo3k/runner.py"

sandbox:
  environment_id: env-geo3k
  required_mount_paths: [/workspace/Safactory/results]
```

The local runner source is installed after instance allocation. `container.mounts` remain Docker-only.

Run the flow with:

```bash
python launcher.py \
  --mode sandbox \
  --sandbox-config config.sandbox.yaml \
  --agent-config env/geo3k/geo3k_config.yaml \
  --agent-start-config env/geo3k/geo3k_start.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model geo3k_model \
  --enable-evaluation \
  --job-id geo3k-sandbox-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

`--pool-size` controls the number of active instances. The manager fills those leases before workers start, and the Environment capacity must cover that concurrency.

Rule evaluation needs no Sandbox-specific settings. The rule evaluator receives the rollout result and the persisted trajectory after the Gateway session is closed.

Brainbox does not implement pause. Set `lifecycle_minutes` long enough to cover rollout, telemetry flush, and evaluation. Use `cleanup_on_finish: false` only for debugging because it preserves quota-consuming instances.
