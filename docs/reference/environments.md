# Supported Environments

SAfactory v2 treats each environment as an external agent runtime. A runtime is described by:

- an agent config: task rows, dataset, `env_params`, and image;
- an agent start config: Docker, RJob, or Sandbox startup details for the runtime;
- an optional `rule_evaluator.py`: reward conversion after rollout.

The standard environment for onboarding, smoke tests, evaluation, and RL examples is **Geo3K**.

## Environment Matrix

| Environment | `env_name` / `agent_name` | Domain | Config | Start config | Runtime modes | Evaluator |
|-------------|----------------------------|--------|--------|--------------|---------------|-----------|
| Geo3K | `geo3k` | Geometry / VLM QA | `env/geo3k/geo3k_config.yaml` | `env/geo3k/geo3k_start.yaml` | Docker; RL template | `env/geo3k/rule_evaluator.py` |
| OpenClaw | `openclaw` | General OpenClaw CLI tasks | `env/openclaw/openclaw_config.yaml` | `env/openclaw/openclaw_start.yaml` | Docker | Optional |
| OpenRT | `openrt` | Safety / red-team benchmark | `env/openrt/openrt_config.yaml` | `env/openrt/openrt_start.yaml` | Docker | `env/openrt/rule_evaluator.py` |
| OpenRT RJob | `openrt` | Remote OpenRT benchmark | `env/openrt/openrt_config.rjob.yaml` | `env/openrt/openrt_start.rjob.yaml` | RJob | `env/openrt/rule_evaluator.py` |
| WildClawBench | `wildclawbench` | Community benchmark harness | `env/wildclawbench/wildclawbench_config.yaml` | `env/wildclawbench/wildclawbench_start.yaml` | Docker | Optional |
| DTAP | `dtap` | DecodingTrust-Agent workloads | `env/dtap/dtap_config.yaml` | `env/dtap/dtap_start.yaml` | Docker | Optional |
| ClawEnvKit | `clawenvkit` | Auto-ClawEval-style tasks | `env/clawenvkit/clawenvkit_config.yaml` | `env/clawenvkit/clawenvkit_start.yaml` | Docker | Optional |

Some checked-in YAML files contain local paths or internal image names. Treat them as working examples and adjust paths, images, mounts, and Gateway route keys before scaling.

## <a id="standard-environment-geo3k"></a>Standard Environment: Geo3K

Geo3K is the reference implementation for a complete SAfactory v2 adapter:

- `env/geo3k/runner.py` implements the external-runtime contract.
- `env/geo3k/rule_evaluator.py` converts Geo3K correctness into SAfactory reward.
- `env/geo3k/Dockerfile` builds the local Docker image.
- `env/geo3k/datasets/geo3k_sample.jsonl` provides a tiny smoke-test dataset.
- `rl/examples/geo3k_vl/env.sh` is the standard RL template.

Build the Docker image:

```bash
docker build -t safactory-geo3k:py311 env/geo3k
```

For the full Geo3K dataset, download [chenhegu/geo3k_imgurl](https://huggingface.co/datasets/chenhegu/geo3k_imgurl), then update `env/geo3k/geo3k_config.yaml` to point to the local parquet file:

```yaml
dataset: chenhegu/geo3k_imgurl/train.parquet
dataset_load_mode: parquet_row_ref
dataset_columns: [problem, answer, images]
```

If the dataset is stored at an absolute local path, use that path instead of the Hugging Face-style example above.

For a first smoke test without the full dataset, point a local copy of `env/geo3k/geo3k_config.yaml` at the checked-in sample dataset:

```bash
cp env/geo3k/geo3k_config.yaml env/geo3k/geo3k_config.local.yaml
```

```yaml
dataset: ./datasets/geo3k_sample.jsonl
dataset_load_mode: eager
```

Remove `dataset_columns` in that smoke-test copy. Keep `dataset_load_mode: parquet_row_ref` only when using a parquet dataset and the configured columns.

Run Geo3K evaluation after the Gateway is ready:

```bash
python launcher.py \
  --mode docker \
  --agent-config env/geo3k/geo3k_config.yaml \
  --agent-start-config env/geo3k/geo3k_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model geo3k_model \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --job-id geo3k-docker-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

If you created the smoke-test copy, replace `--agent-config env/geo3k/geo3k_config.yaml` with `--agent-config env/geo3k/geo3k_config.local.yaml`.

Geo3K runner behavior:

- reads the current case from `env_params.dataset`;
- sends the geometry question and optional images through the session Gateway URL;
- supports `calc_score` / `calc_geo3k_reward` tool calls during the episode;
- extracts boxed final answers and grades them with `math_utils.grade_answer_verl`;
- writes `metrics.score` as `0.0` or `1.0`;
- lets `rule_evaluator.py` normalize the score to `0` or `10`.

Use Geo3K as the baseline when validating a new model route, Docker image, Gateway storage, evaluation, or RL Buffer Server wiring.

## OpenClaw

Files:

- `env/openclaw/openclaw_config.yaml`
- `env/openclaw/openclaw_start.yaml`
- `env/openclaw/runner.mjs`

The runner receives a `SimulationStartRequest`, writes an OpenClaw config pointing at the Gateway session URL, and runs:

```bash
openclaw agent --local --json --session-id <session_id> --message <task> --model safactory/<route>
```

OpenClaw is useful for lightweight CLI and tool-use smoke tests. Update the workspace mount and route key before running.

## OpenRT

Files:

- `env/openrt/openrt_config.yaml`
- `env/openrt/openrt_start.yaml`
- `env/openrt/runner.py`
- `env/openrt/rule_evaluator.py`

OpenRT runs `eval.py` inside the container and uses the Gateway session URL as an OpenAI-compatible base URL. Each dataset row should define one attack or benchmark case.

The repository also contains RJob examples:

- `env/openrt/openrt_config.rjob.yaml`
- `env/openrt/openrt_start.rjob.yaml`

Use those as references when creating Geo3K or custom RJob configs.

## WildClawBench

Files:

- `env/wildclawbench/wildclawbench_config.yaml`
- `env/wildclawbench/wildclawbench_start.yaml`
- `env/wildclawbench/runner.py`

This adapter expects a WildClawBench checkout and a runtime image that can run its tasks. Update checkout paths, result mounts, route keys, and judge route keys before running.

## DTAP

Files:

- `env/dtap/dtap_config.yaml`
- `env/dtap/dtap_start.yaml`
- `env/dtap/runner.py`

DTAP runs DecodingTrust-Agent workloads. The example mounts the DecodingTrust-Agent checkout, SAfactory results, the DTAP runner, and `/var/run/docker.sock` because DTAP may start nested Docker workloads.

## ClawEnvKit

Files:

- `env/clawenvkit/clawenvkit_config.yaml`
- `env/clawenvkit/clawenvkit_start.yaml`
- `env/clawenvkit/runner.py`

ClawEnvKit runs Auto-ClawEval-style tasks with an OpenClaw harness. Update dataset roots, result mounts, harness entrypoints, and Gateway route keys before running.

## Dataset Loading

Agent config `dataset` supports JSON, JSONL, YAML list, and parquet.

- JSONL lines must be valid JSON objects.
- Relative paths resolve from the agent config directory.
- `dataset_load_mode: eager` materializes rows.
- `dataset_load_mode: parquet_row_ref` stores lightweight parquet row references.
- `dataset_columns` is only useful for parquet column selection.

Each dataset row becomes `env_params.dataset` in the runtime request. One dataset row should represent one scheduled episode.

## Runtime Result Contract

All adapters should output one JSON result on stdout:

```json
{
  "session_id": "same-session-id",
  "status": "succeeded",
  "total_reward": 0.0,
  "step_count": 1,
  "terminated": true,
  "truncated": false,
  "error_text": null,
  "metrics": {}
}
```

If the runtime exits non-zero in JSON mode, Docker/RJob/Sandbox runners treat it as a runtime failure even if stdout contains partial output.
