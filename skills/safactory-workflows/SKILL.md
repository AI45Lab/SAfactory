---
name: safactory-workflows
description: Integrate a benchmark or custom environment into SAfactory using fixed adapter templates and local contract tests, optionally run Docker/RJob evaluation, or prepare GRPO/RL training. Use for SAfactory onboarding and runtime workflows.
---

# SAfactory Workflows

For environment onboarding, read [references/environment-integration.md](references/environment-integration.md) and `docs/guides/custom-environment.md` (or its Chinese translation). The guide defines the runtime contract; the [assets/environment/](assets/environment/) templates implement it. Use `env/geo3k/` for examples of environment-specific behavior, not as boilerplate to copy wholesale.

For evaluation requests read [references/docker-evaluation.md](references/docker-evaluation.md). For GRPO/RL requests read [references/grpo-training.md](references/grpo-training.md). Read `docs/internal/rjob-mode.md` or its Chinese translation only when working on RJob deployment.

## Scope and intake

Infer available values from the request and benchmark source before asking:

- environment name, source/check-out path, dataset row schema, and 1–2 representative cases;
- native single-case command and native output location, when wrapping a benchmark;
- target deployment mode (`docker` or `rjob`) and image, when known;
- whether evaluation is requested; **default to integration only**.

Only evaluation or training requires the native score field, scale, and pass condition. Do not block integration on missing rewards, evaluator code, a running Gateway, or internal cluster access. If mode is unknown, implement and test the shared adapter first; clarify mode before generating deployment configs. Ask about an unknown native command or output format before implementing the dependent hook, while continuing the protocol scaffold and tests.

The adapter runs one dataset row through the existing native harness. It does not reimplement the benchmark's case-solving/scoring logic or repair image internals unless that work is requested. Preserve native scores in metrics when available; their presence does not enable evaluation.

## Template-based implementation

1. Scaffold new files with `scripts/scaffold_environment.py` as documented in the integration reference. For an existing directory, adapt the relevant templates without overwriting user files.
2. Keep the protocol shell in `runner.py` fixed. Fill `adapter.py:run_case` with row mapping, model/harness configuration, native execution, and result collection. The generated greeting example only verifies the scaffold; replace it for a real integration.
3. Fill the selected-mode task/start YAML templates with the image, dataset, workdir, mounts, and dependencies. `agent_name` must equal `env_name`; one row is one episode.
4. Create `rule_evaluator.py` only when evaluation is requested. Fill its `score_metrics` hook with the agreed normalization; keep the evaluator interface and failure handling fixed. Do not rerun cases in the evaluator or register its path in YAML.
5. Python is the default template language. If the native runtime needs Node/shell, retain a thin Python wrapper when possible. If another runner language is required, port the same protocol shell and document why; keep native logic separate and cover the same contract tests.

Change shared shell behavior only for an actual runtime-contract requirement, and update the template and its tests together. New files should come from templates, not a fresh implementation of the protocol.

## Standard reference: `env/prmeval`

When an existing benchmark is supplied, compare its integration against
`env/prmeval/` before inventing a new layout. The reference keeps the fixed
`runner.py` protocol shell separate from `adapter.py`, uses one dataset row per
episode, and keeps the optional score mapping in `rule_evaluator.py`.

The Docker pair (`*_config.yaml`, `*_start.yaml`) is the base contract. The
RJob pair (`*_config.rjob.yaml`, `*_start.rjob.yaml`) repeats task parameters
but adds only cluster image, resources, embedded files, and cluster-accessible
mounts. `env_params` is delivered in the request JSON; static `container.env`
or `rjob.env` values are not a replacement for it.

## Validation and completion

Use the staged checks in the integration reference:

1. **Local contract check (default for both target modes):** use `scripts/contract_smoke.py` or equivalent environment-specific fixtures to exercise request input, session URL routing, native output mapping, exact result JSON, and controlled failures. The helper owns its mock endpoint and needs no Gateway, Docker daemon, model credentials, or RJob SDK. Native dependencies still require local availability or explicit test fixtures.
2. **Live deployment check:** when requested or the needed runtime is available, run 1–2 cases with one worker. `scripts/live_smoke.py` starts Gateway, waits for readiness, runs Launcher, and cleans up its processes. Do not require a user to keep a separate terminal open. If an existing Gateway is used, verify it and invoke Launcher directly.
3. **Evaluation check (opt-in):** validate native score conversion and final reward only when evaluation/training is in scope. Pass `--enable-evaluation` explicitly; omit it for integration-only runs. `total_reward: 0.0` satisfies the ungraded runtime result contract and is not an evaluation outcome.

Report the adapter/config changes, the cases and fixtures used, which validation level passed, and remaining deployment prerequisites. Local tests can complete adapter validation without an internal cluster; do not claim they validate image pulls, mounts, real Gateway persistence, or RJob scheduling. If a requested live check is blocked, complete independent local checks and report the specific blocker and next command.

## Repository rules

- Preserve the user's selected deployment mode. Local contract testing does not substitute Docker deployment for RJob deployment.
- RJob needs the same runner, plus `<name>_config.rjob.yaml` and `<name>_start.rjob.yaml`. The evaluator remains optional. Use cluster-accessible images/storage and `rjob.mount_config`/`rjob.mount`; include all runner dependencies in `rjob.embedded_files`. Do not use loopback Gateway addresses for live cluster runs.
- Keep Launcher, Gateway, and Buffer Server on the same storage backend/SQLite URI.
- Preserve local Gateway configuration and never commit private endpoints or credentials.
- Establish a working reward path before starting RL; this requirement does not apply to integration-only work.
