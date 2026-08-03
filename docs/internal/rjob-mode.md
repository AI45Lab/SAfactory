# RJob Mode

RJob mode is the remote rollout backend for Safactory. The launcher still schedules one dataset row as one episode, but runtime allocation is done by submitting RJobs instead of starting local Docker containers.

Use RJob after the same environment has passed a local Docker smoke test.

## Prerequisites

1. The launcher Python environment can import the RJob SDK:

   ```bash
   python -c "from brainpp.rjob import RJobClient; print(RJobClient)"
   ```

2. The RJob cluster can pull the image declared by the agent config or start config.
3. The Gateway URL is reachable from inside RJob containers. Do not use `127.0.0.1` or `localhost`.
4. Any runtime data or result directory needed by the RJob container is on cluster-accessible storage.
5. Secrets, especially RJob AK/SK, are kept out of committed config files.

## Configuration Surfaces

RJob mode uses the same core files as Docker mode, plus global RJob connection settings:

| File | Purpose |
| --- | --- |
| `--agent-config` | Task rows, dataset path, `env_image`, and `env_params`. |
| `--agent-start-config` | Runner entrypoint plus per-agent RJob resources, mounts, embedded files, and cleanup policy. |
| `--rjob-config` | Global RJob endpoint, namespace, credentials, charged group, and optional default Gateway URL. Defaults to `config.yaml`. |
| Gateway config | Model routes and trajectory storage. Launcher and Gateway must share the same storage backend. |

## Global RJob Config

Put global connection and auth settings in `config.yaml` or pass another file with `--rjob-config`:

```yaml
rjob:
  cluster_entry: "https://your-rjob-platform.example"
  namespace: "your-namespace"
  access_key: "replace-me"
  secret_key: "replace-me"
  charged_group: "your-quota-or-project"
  gateway_base_url: "http://<gateway-host-visible-to-rjob>:8000/v1/sessions"
  submit_concurrency: 1
  cleanup_on_finish: true
  no_packaging: true
```

Key fields:

- `access_key` and `secret_key` authenticate RJob creation, polling, log reads, and cleanup. They also determine which namespaces, charged groups, images, and mounts you can use.
- `charged_group` selects the quota or billing group consumed by the RJobs.
- `gateway_base_url` must be reachable from the RJob cluster. If it is stale, it can override a correct `--gateway-base-url`; remove it or update it.
- `submit_concurrency` limits concurrent RJob submissions from the launcher.

## Per-Agent RJob Start Config

Per-agent RJob settings live under `rjob:` in the start config:

```yaml
agent_name: geo3k

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./runner.py
    target: /tmp/safactory-geo3k/runner.py
    command: "python /tmp/safactory-geo3k/runner.py"

rjob:
  name_prefix: geo3k
  image_pull_policy: IfNotPresent
  no_packaging: true
  cleanup_on_finish: true
  keep_failed_jobs: true
  resources:
    cpu: 1
    gpu: 0
    memory_in_mb: 1024
  embedded_files:
    - source: ./math_utils.py
      target: /tmp/safactory-geo3k/math_utils.py
  mount_config:
    - "gpfs://gpfs1/evobox-share/your-user/SAfactory/results:/app/results"
```

Important behavior:

- `container.runner_entrypoint.source` is resolved relative to the start config file. In RJob mode, the file is embedded or staged into the RJob runtime at `target`; it is not a Docker bind mount.
- Extra local files imported by the runner must be listed in `rjob.embedded_files`.
- `container.mounts` are Docker-only. RJob uses `rjob.mount_config` or `rjob.mount`.
- The left side of `mount_config` must be cluster-accessible storage, not a local path from the launcher machine.
- The RJob image must contain runtime dependencies. Embedded runner files provide adapter code, not Python packages.

## Geo3K RJob Evaluation

Geo3K RJob configs should mirror the Docker files, but with a cluster-pullable image, RJob resources, embedded local files, and cluster-accessible mounts. A common layout is:

```text
env/geo3k/geo3k_config.rjob.yaml
env/geo3k/geo3k_start.rjob.yaml
```

If those files are not present in your checkout, create them from `env/geo3k/geo3k_config.yaml` and `env/geo3k/geo3k_start.yaml`, then apply the RJob changes described above. The checked-in `env/openrt/openrt_config.rjob.yaml` and `env/openrt/openrt_start.rjob.yaml` show the same pattern on an environment that already has RJob configs.

Run a small Geo3K RJob evaluation after the Geo3K RJob files are ready:

```bash
python launcher.py \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/geo3k/geo3k_config.rjob.yaml \
  --agent-start-config env/geo3k/geo3k_start.rjob.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model geo3k_model \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --job-id geo3k-rjob-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

For SQLite, the Gateway and launcher must point at the same DB URI. For remote clusters, the Gateway host must be routable from the RJob network.

## Geo3K RL With RJob

The RL entry scripts remain the same. After creating the Geo3K RJob YAML files, switch the environment config in `rl/examples/geo3k_vl/env.sh`:

```bash
export AIEVOBOX_MODE=rjob
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.rjob.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.rjob.yaml
export AIEVOBOX_GATEWAY_HOST=<gateway-host-visible-to-rjob>
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
export RL_MODEL=geo3k_model
```

Then start the same two processes from the repository root:

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

Current RL Buffer Server invokes `launcher.py` without passing `--rjob-config`, so RL RJob runs use the launcher's default `config.yaml` for global RJob settings.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `RJob mode requires brainpp.rjob / RJobClient` | Install or activate the RJob SDK in the launcher Python environment. |
| `403 Forbidden` when creating RJobs | Check AK/SK, namespace, charged group, image permission, private machine setting, and mount permission. |
| RJob is created but cannot reach Gateway | Use a Gateway host visible from the RJob cluster; avoid `127.0.0.1` and `localhost`. |
| RJob succeeds but Safactory cannot parse a result | The runner must print one `SimulationStartResult` JSON to stdout. For artifact fallback, ensure `SAFACTORY_RESULT_PATH` maps to a writable mounted path. |
| `RJob mode cannot map local Docker mounts` | Move required paths to `rjob.mount_config` with cluster-accessible storage. |
| Gateway and launcher run but evaluator sees no rows | Ensure Gateway storage and launcher storage point at the same DB/backend. |
