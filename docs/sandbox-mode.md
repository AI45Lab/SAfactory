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

Copy `config.sandbox.example.yaml` and set `project`, `environment_id`, and a cluster-reachable gateway URL. An agent can override the Environment in its start config:

```yaml
agent_name: openrt

container:
  workdir: /app
  runner_entrypoint:
    source: ./runner.py
    target: /tmp/safactory-openrt-runner.py
    command: "python /tmp/safactory-openrt-runner.py"

sandbox:
  environment_id: env-openrt
  required_mount_paths: [/app/data, /app/results]
```

The local runner source is installed after instance allocation. `container.mounts` remain Docker-only.

Run the flow with:

```bash
python launcher.py \
  --mode sandbox \
  --sandbox-config config.sandbox.yaml \
  --agent-config env/openrt/openrt_config.yaml \
  --agent-start-config env/openrt/openrt_start.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --pool-size 8
```

`--pool-size` controls the number of active instances. The manager fills those leases before workers start, and the Environment capacity must cover that concurrency.

Trajectory, rule, and LLM-judge evaluation work without runtime-specific settings. For direct evaluator access, set `requires_container: true` and `target_access_mode: sandbox_proxy`, and add `sandbox_proxy` to the evaluator pool member's `allowed_target_access_modes`. Only that access mode exposes the command endpoint and SAT header to the evaluator.

Brainbox does not implement pause. Set `lifecycle_minutes` long enough to cover rollout, telemetry flush, and evaluation. Use `cleanup_on_finish: false` only for debugging because it preserves quota-consuming instances.
