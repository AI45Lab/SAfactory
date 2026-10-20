# Sandbox Mode

Sandbox mode is a rollout runtime alongside Docker and RJob. Safactory creates a Sandbox Instance through the DeepEnv OpenSandbox REST API, connects with the OpenSandbox SDK to run the agent entrypoint, and deletes the instance after rollout and optional evaluation. Instances are created from the agent config `env_image` with `extensions.project`; no Environment template (`environmentId`) is involved, and `environment_id` / `required_mount_paths` settings are rejected at config load.

## Prerequisites

1. The image from the agent config `env_image` must run execd on `sandbox.command_port`, `44772` by default. Safactory uses `creation_mode: direct_api` to declare that port in the creation request: the pinned SDK (`opensandbox==0.1.9`) cannot pass `ports`. Starting execd in the image and declaring its platform port are both required for SDK commands and file access.
2. Safactory explicitly mounts the project AFS root at `/sandbox-data` and its `results` subdirectory at `/sandbox-data/results`, both with `readOnly: false`. Custom `sandbox.volumes` replace the input mounts; Safactory preserves those entries and adds the writable results mount if missing. Declaring any volume disables the platform's implicit project mount, so include every input mount the agent needs. Local Docker bind mounts are not translated. See `sandbox/04-sdk-usage.md`, `sandbox/09-2-project-data-custom-mount-guide.md`, and `sandbox/09-3-shared-data-directory-guide.md` for the volume syntax.
3. The Sandbox must be able to reach `sandbox.gateway_base_url`.

Install dependencies and provide credentials:

```bash
pip install -r requirements.txt
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
export OPEN_SANDBOX_PROJECT='ailab-evobox'
```

Copy `config.sandbox.example.yaml` and set a cluster-reachable `gateway_base_url`. The current control plane is `http://deepenv.sandbox.pjlab.org.cn`; set `domain` to the hostname without a scheme, path, or `/v1`, and set `protocol: http`. The runtime sends the credential from `api_key_env` (default `OPEN_SANDBOX_API_KEY`) as `OPEN-SANDBOX-API-KEY` and uses `project_env` (default `OPEN_SANDBOX_PROJECT`) when `project` is empty. The project must be your authorized short workspace name; `ailab-evobox` matches the Harbor scripts' example. Keep credentials in the environment.

An agent can override resources, extensions, and volumes in its start config. For Docker-in-Sandbox workloads such as Harbor, set `extensions.runtimeClassName: sysbox-runc` in that agent's config and use an image containing Docker and its dependencies. FUSE is an additional capability enabled only when needed with `extensions.fuse: "true"`; extension values use strings. Geo3K uses the same runtime contract:

```yaml
agent_name: geo3k

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./runner.py
    target: /work/safactory/runner.py
    command: "python /work/safactory/runner.py"

sandbox:
  volumes:
    - name: project-default-afs
      mountPath: /sandbox-data
      readOnly: false
```

The local runner file is installed at `/work/safactory/instances/<sandbox-id>/runner.py` after allocation, and its command is updated to that instance's path. `container.mounts` remain Docker-only.

Results, runner logs, and Harbor trial artifacts use `/sandbox-data/results/<job-id>/<session-id>/`. `result_artifact_root` accepts this path or an empty value for the default; temporary or alternative roots are rejected. Results persist after instance deletion. A declared results mount must have `readOnly: false`; configuration normalization rejects a read-only results mount and does not duplicate an existing writable one.

Prepare the project's `results` directory so the target Sandbox can write through its actual UID mapping. `readOnly: false` enables a writable mount but does not bypass AFS permissions, especially for Sysbox. Allocation creates, writes, and removes a probe in the mounted results directory; failure stops allocation and never falls back to `/tmp`.

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

`--pool-size` controls the number of active instances. Workers start using ready leases while other instances are still being prepared, and the project quota must cover that concurrency. `create_timeout_s` bounds creation, waiting for running state, and connecting execd together. `bootstrap_timeout_s` bounds runner installation and the write probe (default 120 seconds), and `cancel_timeout_s` bounds command interruption (default 10 seconds). Refill uses `--container-refill-timeout-s`; a timeout cancels and cleans up allocation, returning its reserved row to the retry queue. Long rollout commands use `command_timeout_s`; cancellation interrupts their remote executions.

Rule evaluation needs no Sandbox-specific settings. The rule evaluator receives the rollout result and the persisted trajectory after the Gateway session is closed.

DeepEnv Sandbox System does not implement pause. Set `lifecycle_minutes` long enough to cover rollout, telemetry flush, and evaluation. Use `cleanup_on_finish: false` only for debugging because it preserves quota-consuming instances.
