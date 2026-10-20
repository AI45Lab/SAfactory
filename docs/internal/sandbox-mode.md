# Sandbox Mode

Sandbox mode is a rollout runtime alongside Docker and RJob. Safactory creates a Sandbox Instance directly through the OpenSandbox SDK, runs the agent entrypoint inside it, and deletes the instance after rollout and optional evaluation. Instances are created from the agent config `env_image` with `extensions.project`; no Environment template (`environmentId`) is involved, and `environment_id` / `required_mount_paths` settings are rejected at config load.

## Prerequisites

1. The image from the agent config `env_image` must run execd and expose the command port, `44772` by default. The pinned SDK (`opensandbox==0.1.9`) cannot declare ports at creation time, so the image must provide this itself.
2. Data and result storage is declared as explicit `sandbox.volumes` entries (project AFS volumes); local Docker bind mounts are not translated. Declaring any volume disables the platform's implicit project mount at `/sandbox-data`, so declare every mount the agent needs. See `sandbox/04-sdk-usage.md` and `sandbox/09-2-project-data-custom-mount-guide.md` for the volume syntax.
3. The Sandbox must be able to reach `sandbox.gateway_base_url`.

Install dependencies and provide credentials:

```bash
pip install -r requirements.txt
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
```

Copy `config.sandbox.example.yaml` and set `project` plus a cluster-reachable gateway URL. An agent can override the connection and volume settings in its start config. Geo3K uses the same runtime contract:

```yaml
agent_name: geo3k

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./
    target: /tmp/safactory-geo3k
    command: "python /tmp/safactory-geo3k/runner.py"

sandbox:
  volumes:
    - mountPath: /workspace/Safactory/results
      subPath: /geo3k/results
      readOnly: false
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

`--pool-size` controls the number of active instances. The manager fills those leases before workers start, and the project quota must cover that concurrency.

Rule evaluation needs no Sandbox-specific settings. The rule evaluator receives the rollout result and the persisted trajectory after the Gateway session is closed.

Brainbox does not implement pause. Set `lifecycle_minutes` long enough to cover rollout, telemetry flush, and evaluation. Use `cleanup_on_finish: false` only for debugging because it preserves quota-consuming instances.
