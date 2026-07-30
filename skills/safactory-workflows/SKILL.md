---
name: safactory-workflows
description: Use this skill when helping users onboard a benchmark or environment into SAfactory, run SAfactory Docker-mode evaluation, or start SAfactory GRPO/RL training. It guides Codex to inspect the SAfactory repository, read the relevant docs, follow the established Geo3K baseline and my_env placeholder workflows, configure Gateway/model routes safely, and run or prepare launcher/RL commands without leaking private credentials.
---

# SAfactory Workflows

This skill helps agents operate the SAfactory repository for three user intents:

1. Add a benchmark or custom environment to SAfactory.
2. Run Docker-mode evaluation for an environment.
3. Start GRPO/RL training for an environment.

Keep the root README and `docs/` as the source of truth. Use this skill to choose the right workflow, files, and checks; do not duplicate or rewrite full documentation.

## First Steps

From the repository root:

1. Inspect the current layout before assuming paths: `rg --files README* docs env rl gateway`.
2. Identify the user intent:
   - Benchmark/environment onboarding: read [references/environment-integration.md](references/environment-integration.md).
   - Docker evaluation: read [references/docker-evaluation.md](references/docker-evaluation.md).
   - GRPO/RL training: read [references/grpo-training.md](references/grpo-training.md).
3. Load only the matching reference file and the linked SAfactory docs needed for the task.
4. Prefer existing patterns in `env/geo3k/`, current docs, and root README commands.

## Repository Sources

Use these documents as canonical sources:

- Root quick start: `README.md` or `README_CN.md`.
- Custom environment guide: `docs/guides/custom-environment.md` or `docs/guides/custom-environment_CN.md`.
- Evaluation guide: `docs/guides/evaluation.md` or `docs/guides/evaluation_CN.md`.
- RL guide: `docs/guides/rl-training.md` or `docs/guides/rl-training_CN.md`.
- Gateway reference: `docs/reference/gateway.md` or `docs/reference/gateway_CN.md`.
- Environment reference: `docs/reference/environments.md` or `docs/reference/environments_CN.md`.
- CLI/config reference: `docs/reference/configuration.md` or `docs/reference/configuration_CN.md`.

Use Chinese docs when the user writes Chinese; otherwise use English docs.

## Operational Rules

- Treat `env/geo3k/` as the standard reference implementation, not as a hardcoded target.
- Use generic placeholders such as `my_env` or the user's requested environment name for new workflows.
- Do not write private model endpoints, API keys, or internal route names into committed docs or shared configs.
- Do not overwrite `gateway/config.local.yaml` without checking its existing contents and preserving user edits.
- Ensure `--llm-model` and `RL_MODEL` match a Gateway `llm_routes` key.
- Keep Launcher, Gateway, and Buffer Server storage pointing at the same backend or SQLite URI.
- For new environments, run a minimal Docker evaluation before recommending RL training.
- Read `docs/internal/` only when the user explicitly asks about RJob, Sandbox, or internal deployment modes.

## When Information Is Missing

Ask the smallest set of concrete questions required to proceed. Common blockers:

- New benchmark source path, dataset path, or native run command is unknown.
- Desired SAfactory environment name is unknown.
- Model route key or Gateway endpoint is unknown for an actual run.
- Docker image, runtime dependencies, or scoring contract is unknown.

If the user asks only for commands and prerequisites are missing, provide a fill-in command with placeholders and state exactly what must be replaced.

## Verification Expectations

- For file edits: check created paths with `rg --files` and inspect changed files.
- For evaluation: run the smallest feasible `launcher.py --mode docker --enable-evaluation` smoke test when credentials, image, and data are available.
- For RL: first verify or ask for evidence that Docker evaluation passes; then prepare `env.sh`; start long-running training processes only when the user requested execution.
- If a command cannot be run because credentials, Docker, data, or dependencies are unavailable, report the blocker precisely and leave the exact next command.
