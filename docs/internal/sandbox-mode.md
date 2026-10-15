# Sandbox Mode

Sandbox is a runtime alongside Docker and RJob. The non-environment framework supports AFS publication, SDK file upload, command-stop confirmation, result-file download, file evaluation, Gateway durability checks, and recovery. This change does not modify `env/`: existing environment runners still need migration to the v2 result contract before they can run through this path.

See the [workflow and environment migration design](sandbox-workflow-design_CN.md) and [implemented interfaces](sandbox-implementation-plan_CN.md).

## Prerequisites

1. Prepare an Environment using `sandbox/02-environment-api.md`. Its image must match `env_image`, expose port `44772`, and run execd. `sandbox/create_sandbox.py` creates an instance, not an Environment.
2. Install repository dependencies, Bash, and rclone on H. The SDK version is `opensandbox==0.1.9`. Supply `OPEN_SANDBOX_API_KEY`; set project in platform YAML.
3. Include `sh`, `base64`, `sha256sum`, runner dependencies, and native verification dependencies in the image. Verify that the Sandbox can reach the shared Gateway URL.
4. Configure the existing Gateway with `telemetry.mode: strict`, `telemetry.async_cloud_writes: false`, `storage_config.enable_buffer: false`, and `storage_config.enable_upsert_batching: false`. Gateway and launcher must use the same actual storage backend. `/readyz` reports the capabilities checked by the launcher.

## Configuration and launch

```bash
cp config.sandbox.example.yaml sandbox-config.yaml
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
```

Use three independent files:

- `sandbox-config.yaml`: platform, project, authentication, shared Gateway, AFS WebDAV, and local results directory.
- `env/<env>/<env>.config.sandbox.yaml`: dataset, image, and business parameters.
- `env/<env>/<env>.start.sandbox.yaml`: `sandbox:` runner_entrypoint, auxiliary files, resources, and input/output mounts.

The environment files must be created during migration. Upload sources must be individual files. Sources and relative input host paths resolve against the start YAML. Docker bind mounts and RJob mount_config are not translated. AFS inputs are read-only; outputs must include writable `/app/results`. The complete volume list overrides Environment mounts.

After migrating the environment, use the existing launcher:

```bash
python launcher.py \
  --mode sandbox \
  --sandbox-config sandbox-config.yaml \
  --agent-config env/yourenv/yourenv.config.sandbox.yaml \
  --agent-start-config env/yourenv/yourenv.start.sandbox.yaml \
  --job-id yourenv-sandbox-smoke \
  --llm-model your_model_route \
  --pool-size 1 --multiplier 1 --max-workers 1
```

An explicit `--gateway-base-url` overrides platform YAML; the resolved URL is shared by launcher and Sandbox. File evaluation is mandatory and enabled automatically. Adjust command_timeout_s and --agent-start-timeout-s together for long workloads.

The runner verifies and scores inside the Sandbox, then atomically publishes `/app/results/<job_id>/<session_id>/safactory_result.json`. The host downloads it through SDK `files.read_bytes`, verifies it, and writes `<results.local_root>/<job_id>/<session_id>/safactory_result.json`. Only `evaluation.normalized_score_10` supplies the final score. stdout, exit code, and top-level total_reward are not score fallbacks. Execution errors cannot be hidden by an otherwise valid JSON file.

The warm pool contains `ceil(pool_size × multiplier)` instances. TTL renewal runs while the SDK connection is open; closing a retained instance's connection stops renewal. Retained instances continue consuming quota until platform expiry or deletion.

Resume with the same job id reconciles runtime.json, previous writers, and result files before deleting any unfinished trajectory. Valid local files resume submission; remote files resume download. Only confirmed reruns reset old trajectories. Recovery does not require platform pause/resume support. Unknown execution state, invalid identity, or missing durability confirmation prevents score submission.
