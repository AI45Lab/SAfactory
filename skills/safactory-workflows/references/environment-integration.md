# Environment Integration Workflow

Use this reference when the user asks to add a benchmark, custom environment, or new task suite to SAfactory.

Canonical docs:

- `docs/guides/custom-environment.md`
- `docs/guides/custom-environment_CN.md`
- `docs/guides/evaluation.md`
- `docs/guides/evaluation_CN.md`
- `docs/reference/configuration.md`
- `docs/reference/configuration_CN.md`
- `docs/reference/environments.md`
- `docs/reference/environments_CN.md`

Reference implementation:

- `env/geo3k/` is the standard complete environment layout.

## Intake

Before editing, identify:

- Environment name, normally lowercase with underscores, for example `my_env`.
- Benchmark source path or repository.
- Dataset path and row format.
- Native task execution command.
- Required Docker image or dependencies.
- Result format and scoring rule.
- Whether rule-based evaluation is required.

If any of these are unknown and cannot be inferred from local files, ask focused questions before implementing.

## Target Files

For environment `my_env`, expect or create:

- `env/my_env/`
- `env/my_env/my_env_config.yaml`
- `env/my_env/my_env_start.yaml`
- `env/my_env/runner.py` or equivalent runner entrypoint.
- `env/my_env/rule_evaluator.py` when `--enable-evaluation` should produce rewards.
- `env/my_env/datasets/...` when data is stored in the repo.
- Optional `env/my_env/Dockerfile` when the environment needs a dedicated image.

Use the existing naming pattern: `<env>/<env>_config.yaml` and `<env>/<env>_start.yaml`.

## Implementation Checklist

1. Run the standard Geo3K Docker smoke test from the root README when the platform setup is unverified.
2. Inspect `env/geo3k/` and any similar existing environment before adding abstractions.
3. Implement a runner that reads the SAfactory request payload and calls the target model through the provided Gateway session root and route key.
4. Return one machine-readable result per task according to the custom environment guide.
5. Add task config with dataset, runtime metadata, evaluator settings, and environment name.
6. Add start config with runner entrypoint, working directory, Docker settings, mounts, and environment variables.
7. Ensure `env_name` and `agent_name` match where the config format requires it.
8. Add or adapt `rule_evaluator.py` only when the benchmark has deterministic reward or scoring logic.
9. Run a minimal Docker smoke test with a small dataset or small sample count.

## Smoke Test Shape

Use the root README and custom environment guide as the command source. A typical shape is:

```bash
python launcher.py \
  --mode docker \
  --agent-config env/my_env/my_env_config.yaml \
  --agent-start-config env/my_env/my_env_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model my_env_model \
  --enable-evaluation \
  --db-path sqlite:///my_env_smoke.db \
  --job-id my_env-docker-smoke
```

Replace `my_env_model` with a real Gateway route key. Do not invent real private routes.

## Completion Criteria

The integration is not done until:

- Required files exist under `env/my_env/`.
- The runner can execute one task in Docker mode.
- Gateway route and storage assumptions are documented or configured.
- Evaluation produces a result and reward when `rule_evaluator.py` is present.
- Any missing external assets, images, or credentials are clearly listed.
