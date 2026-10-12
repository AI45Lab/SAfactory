# DataElf finance Safactory environment

The DataElf finance domain as a
SAfactory environment: one dataset row becomes one native `dataelf run`
episode. The file
responsibilities are fixed:

| File | Responsibility |
| --- | --- |
| `runner.py` | Fixed SAfactory protocol shell — do not edit per benchmark. |
| `adapter.py` | The only benchmark-logic file: maps one dataset row to one native `dataelf run` invocation, then scores the episode in-container at the tail stage (`_score_episode`). |
| `dataelf_finance_config.yaml` | Docker task config: image, 100-row dataset, `env_params` (native knobs: `db_path`, `max_runtime_seconds`, `pi_settings`, …). |
| `dataelf_finance_start.yaml` | Docker startup: runner, adapter/dataset/results/database mounts, plus the read-only scoring-subset mounts (`/opt/dataelf_eval-scoring`). |
| `dataelf_finance_config.rjob.yaml` / `dataelf_finance_start.rjob.yaml` | RJob mirror with cluster placeholders (`CLUSTER_STORAGE`, registry image; the scoring subset rides `embedded_files`). |
| `rule_evaluator.py` | Fixed evaluation-protocol shell: normalizes the in-container reward document (`metrics["reward"]`) into SAfactory's 0–10 EvalResult — never reruns the case, never calls a model. |
| `datasets/ddr10k_bench_100.jsonl` | 100 rows, one per episode (`id`/`task`/`entity_id` + per-entity `reward_rubric`); the prompt/tool surface is generated natively by DataElf's `ddr_10k` benchmark config. Regenerate via `dataelf_eval/benchmarks/ddr_10k/build_safactory_dataset.py`. |

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

Build the v0.4 runtime image from the DataElf checkout. Existing
node_modules and the managed pi store are copied from the host; Python
dependencies are installed during the build. Both environment configs pin
v0.4: this image restores the venv PATH for login shells and checks the
Python interpreter and scoring dependencies during the build.

```bash
docker build -f env/dataelf_finance/Dockerfile \
  -t dataelf_finance-image:v0.4 \
  -t registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu_task/safactory_bench:dataelf-finance-v0.4 \
  /path/to/DataElf
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
  --agent-start-timeout-s 18000 \
  --storage-type sqlite --db-path sqlite://dataelf-finance.db
```

RJob mode: publish the v0.4 registry tag configured in
`dataelf_finance_config.rjob.yaml` before launching jobs. Authenticate to
the registry using your existing Docker credentials, then publish the
image built above:

```bash
docker push registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu_task/safactory_bench:dataelf-finance-v0.4
```

Run with the `.rjob.yaml` configs:

```bash
python launcher.py --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/dataelf_finance/dataelf_finance_config.rjob.yaml \
  --agent-start-config env/dataelf_finance/dataelf_finance_start.rjob.yaml \
  --gateway-base-url http://<gateway-host>:<port>/v1/sessions \
  --llm-model <route-model> \
  --job-id dataelf-finance \
  --pool-size 20 --max-workers 20 \
  --agent-start-timeout-s 18000 \
  --storage-type sqlite --db-path sqlite://dataelf-finance.db \
  --rebuild-table
```

RJob notes:

- the `--rjob-config` needs `charged_group: evobox_cpu_task` and
  `private_machine: "no"`; 4 CPU / 8 GB per job suffices — episodes are
  model-wait bound;
- `--agent-start-timeout-s` covers scheduling, startup, the native run and
  tail-stage scoring in RJob mode; allow more time when scheduling is slow;
- launch with proxy env vars stripped — a CIDR-form `no_proxy` entry is not
  supported by httpx and breaks Gateway session close/clean;
- cap the Gateway route `max_concurrency` at the model endpoint limit (20
  used).

Add `--enable-evaluation` to either invocation to score finished episodes.

Both launch examples use a 18000 s (5 h) outer episode budget: the native
subprocess cap `dataelf.run_timeout_seconds: 15000` (14400 s native runtime
plus 600 s grace), followed by `dataelf_eval.scoring_timeout_s: 1800`, plus
1200 s for startup and result writes. Pass `--agent-start-timeout-s` on the
launcher command line; the environment YAML only defines the inner stage
budgets. Increase the outer budget too when either inner cap is raised.

## Scoring (in-container, adapter tail stage → rule evaluator normalizes)

One episode = one agent run **plus its scoring** (2026-10-12 migration;
rationale and prior costs: `dataelf_eval/benchmark_notes/
safactory_skill_incontainer_eval_migration.md`). After a successful native
run, `adapter.py::_score_episode`:

1. builds the unified case dict (summary/key_findings + container artifact
   paths + the dataset row verbatim — its `reward_rubric` carries this
   entity's digest-verified checklist; the full `qa.json` never enters the
   container);
2. loads the read-only scoring subset mounted at
   `env_params.dataelf_eval.scoring_root` (`/opt/dataelf_eval-scoring`:
   `benchmarks/<benchmark>/reward_calculator.py` + `eval/config.yaml` —
   never the whole dataelf_eval checkout) and calls
   `REWARD_CALCULATOR.evaluate_case(case)`;
3. writes the resulting `dataelf_eval.reward.v1` document (schema authority:
   `dataelf_eval/eval/REWARD_SCHEMA.md` — components with per-mode weights,
   `normalized_score_10` for SAfactory, `raw_score` for the RL loop, judge
   stats) into `metrics["reward"]` and to `reward_debug.json` beside the
   episode artifacts.

The judge fanout runs under a daemon-thread budget
(`dataelf_eval.scoring_timeout_s`, default 1800 s): scoring shares the
episode's failure domain — a blown budget or judge crash fails the episode
(the stuck scoring thread is abandoned as a daemon; the runner never hangs).
Judge endpoints and reward modes stay **eval-side configuration**
(`eval/config.yaml` in the scoring mount; values may be literals or `$VAR`
env refs; no keys committed here); the judge posts directly to the intranet
vLLM endpoint (`trust_env=False`, never via the Gateway).

`rule_evaluator.py` on the manager side is a pure normalizer: it maps
`metrics["reward"]` onto the 0–10 EvalResult (missing/non-finite → loud
fail). It never reruns the case, never calls a model — the framework's 60 s
default eval window is ample. The old `eval_timeout_s` knob and its
`_eval_timeout_override` framework patch (e86e46cb) were reverted and
deleted on 2026-10-12 — no eval-timeout configuration exists anymore.

The reward is shaped, clipped and judge-outage-robust (benchmark-side
policy, `official=false` — a training signal, never a benchmark number);
official scoring stays a dataelf_eval post-hoc replay over the preserved
artifacts, doubling as cross-calibration against this training reward.
Artifacts stay preserved under `results_root/<job>-<session>/`
regardless of whether evaluation is enabled:

| Replay input (dataelf_eval `company_<cik>/`) | Preserved artifact |
| --- | --- |
| full dialogue + tool calls | `workspaces/<job>/logs/pi_events.jsonl` |
| `results.json` fields (summary / key_findings / evidence_refs / confidence / limitations) | `workspaces/<job>/results/results.json` |
| stage provenance, linked DB, tool surface | `workspaces/<job>/logs/stage_results.jsonl` |
| native output-review gate | `workspaces/<job>/reviews/quality_review.json` |

## Validate

`request.smoke.json` is intentionally omitted. Docker/RJob episodes use the
configured JSONL dataset and pass each request to the runner via stdin or
`SAFACTORY_START_REQUEST_JSON`; they do not read a smoke fixture. Missing
fixture findings from the optional skill checks do not affect that runtime
path. Standalone `contract_smoke.py` checks require an explicit
`--request <file>`.

```bash
python skills/safactory-workflows/scripts/validate_environment.py env/dataelf_finance --expect-evaluation
```
