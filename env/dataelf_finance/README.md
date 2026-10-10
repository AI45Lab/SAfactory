# DataElf finance Safactory environment

The DataElf finance domain as a
SAfactory environment: one dataset row becomes one native `dataelf run`
episode. The file
responsibilities are fixed:

| File | Responsibility |
| --- | --- |
| `runner.py` | Fixed SAfactory protocol shell — do not edit per benchmark. |
| `adapter.py` | The only benchmark-logic file: maps one dataset row to one native `dataelf run` invocation. |
| `dataelf_finance_config.yaml` | Docker task config: image, 100-row dataset, `env_params` (native knobs: `db_path`, `max_runtime_seconds`, `pi_settings`, …). |
| `dataelf_finance_start.yaml` | Docker startup: runner, adapter/dataset/results/database mounts. |
| `dataelf_finance_config.rjob.yaml` / `dataelf_finance_start.rjob.yaml` | RJob mirror with cluster placeholders (`CLUSTER_STORAGE`, registry image). |
| `rule_evaluator.py` | Fixed evaluation-protocol shell: unified case interface, registry dispatch into `dataelf_eval`, generic judge fallback. |
| `datasets/ddr10k_bench_100.jsonl` | 100 rows, one per episode (`id`/`task`/`entity_id`); the prompt/tool surface is generated natively by DataElf's `ddr_10k` benchmark config. |
| `datasets/ddr10k_bench_20.jsonl` | Same row format, first 20 entities — subset for shorter runs. |

## Case mapping (adapter.py)

Row `{"id": "6201__entity", "task": "Analyze company with CIK 6201", "entity_id": "6201"}` becomes
exactly the native command from the DataElf README:

```
dataelf run --domain finance \
  --param benchmark=ddr_10k \
  --param finance_source='{"sqlite":{"db_path":"<env_params.dataelf.db_path>","link":true}}' \
  --param finance_tools='{"sql":true,"python":true,"web":false}' \
  "Analyze company with CIK 6201"
```

Native output `results/results.json` (summary + key_findings, optional
evidence_refs/confidence/limitations) is copied into `metrics` together with
workspace/output paths; `step_count` is the `pi_events.jsonl` line count.

## Model routing (no committed credentials)

Per episode the adapter writes under `results_root/<job>-<session>/`:

- `dataelf.config.yaml`, selected via `DATAELF_CONFIG_FILE` — `env.VLLM_BASE_URL`
  is the Gateway session URL and workspaces live under the episode dir;
- `pi-agent/models.json` — one provider `safactory` (openai-completions,
  `$VLLM_BASE_URL`/`$VLLM_API_KEY` env refs) whose single model id equals the
  Gateway route key, so pi posts `model=<route>` to
  `<session_url>/chat/completions`;
- an `npm` symlink to the repo-root `.pi/npm` store so DataElf's managed pi
  packages (finance tool extensions) resolve from the per-episode agent dir.

## Run

Build the runtime image from the DataElf checkout (node_modules and the
managed pi store are copied from the host, so the build needs no npm or
network access):

```bash
docker build -f env/dataelf_finance/Dockerfile \
  -t dataelf_finance-image:latest /path/to/DataElf
```

Docker mode:

```bash
python launcher.py --mode docker \
  --agent-config env/dataelf_finance/dataelf_finance_config.yaml \
  --agent-start-config env/dataelf_finance/dataelf_finance_start.yaml \
  --gateway-base-url http://<gateway-host>:<port>/v1/sessions \
  --llm-model <route-model> \
  --job-id dataelf-finance \
  --pool-size 1 --max-workers 1 \
  --agent-start-timeout-s 16200 \
  --storage-type sqlite --db-path sqlite://dataelf-finance.db
```

RJob mode: set the registry image (verified tag:
`registry.h.pjlab.org.cn/ailab-evobox/benches:dataelf_finance`, docker-login
with the workspace `/.auth` keys) and cluster placeholders in
`dataelf_finance_config.rjob.yaml`, then run with the `.rjob.yaml` configs:

```bash
python launcher.py --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/dataelf_finance/dataelf_finance_config.rjob.yaml \
  --agent-start-config env/dataelf_finance/dataelf_finance_start.rjob.yaml \
  --gateway-base-url http://<gateway-host>:<port>/v1/sessions \
  --llm-model <route-model> \
  --job-id dataelf-finance \
  --pool-size 20 --max-workers 20 \
  --agent-start-timeout-s 16200 \
  --storage-type sqlite --db-path sqlite://dataelf-finance.db \
  --rebuild-table
```

RJob notes:

- the `--rjob-config` needs `charged_group: evobox_cpu_task` and
  `private_machine: "no"`; 4 CPU / 8 GB per job suffices — episodes are
  model-wait bound;
- `--agent-start-timeout-s` is the whole-job timeout in rjob mode: keep it
  ≥ `max_runtime_seconds` (14400) plus startup margin;
- launch with proxy env vars stripped — a CIDR-form `no_proxy` entry is not
  supported by httpx and breaks Gateway session close/clean;
- cap the Gateway route `max_concurrency` at the model endpoint limit (20
  used).

Add `--enable-evaluation` to either invocation to score finished episodes.

## Scoring (rule evaluator produces the RL training reward)

`rule_evaluator.py` maps one finished episode onto one *case evaluation*,
never reruns the case, and owns no scoring policy — no judge calls, no
heuristics, no fallbacks. It produces exactly one number: the **RL
training reward**, from `env_params.dataelf_eval.root` ->
`benchmarks/<benchmark>/reward_calculator.py` exporting
`REWARD_CALCULATOR`, called as `evaluate_case(case) -> case_score`
(plain dicts). The reward is shaped, clipped, process-terms-allowed, and
robust to judge outages (benchmark-side policy); it stamps
`official=false` — a training signal, never a benchmark number.
Unregistered benchmarks fail loudly (no generic judge fallback).

Judge endpoints and reward modes are **eval-side configuration**, read by
the reward calculator itself: `dataelf_eval/eval/config.yaml` in the
dataelf_eval checkout (values may be literals or `$VAR` env references;
no keys are committed here). The env config's `dataelf_eval` block
carries only deployment facts — `root` and the container→host results
path mapping; any `purpose`/`judge`/`fallback_judge` keys placed there
are ignored with a warning. LLM judges exceed the framework's 60 s
rule-evaluator default — `env_params.eval_timeout_s` raises it.
Per-episode artifacts stay preserved under `results_root/<job>-<session>/`
regardless of whether evaluation is enabled:

| Replay input (dataelf_eval `company_<cik>/`) | Preserved artifact |
| --- | --- |
| full dialogue + tool calls | `workspaces/<job>/logs/pi_events.jsonl` |
| `results.json` fields (summary / key_findings / evidence_refs / confidence / limitations) | `workspaces/<job>/results/results.json` |
| stage provenance, linked DB, tool surface | `workspaces/<job>/logs/stage_results.jsonl` |
| native output-review gate | `workspaces/<job>/reviews/quality_review.json` |

## Validate

```bash
python skills/safactory-workflows/scripts/validate_environment.py env/dataelf_finance --expect-evaluation
```
