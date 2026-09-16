# Environment integration

Read `docs/guides/custom-environment.md` for the request/result and deployment contracts. This workflow separates adapter validation, live deployment, and optional evaluation.

## Intake and scope

Infer the environment name, source, one-row schema, 1–2 cases, native single-case command, and native output format. Ask only for missing information needed by the next dependent step. Deployment mode/image can remain pending during shared adapter work. Reward fields and scoring rules are needed only for evaluation or training. A benchmark with native scores can still be integrated without enabling evaluation.

Keep native execution and scoring in the existing harness. The adapter maps one `env_params.dataset` row, routes model calls through the current session, invokes one native case, collects output, and emits the runtime result. Do not run a full benchmark inside a single episode.

## Copy the fixed templates

Run from the repository root. The scaffolder always emits the shared Docker
pair and the additional RJob pair so the same adapter can be promoted later:

```bash
python skills/safactory-workflows/scripts/scaffold_environment.py mybench --mode docker
# `--mode rjob` records that the first deployment target is RJob; both pairs are still emitted.
# Append --enable-evaluation only when scoring is requested.
```

The generated Docker mount sources are rooted at the launcher working
directory (the repository root in the commands below). Runner sources and
RJob embedded-file sources remain relative to their start-config file. Keep
these two path bases distinct; using `./adapter.py` as a Docker mount from the
repository root would point at `SAfactory/adapter.py`, not `env/mybench/adapter.py`.

The scaffolder refuses to overwrite an existing environment. For existing integrations, copy/adapt individual templates after inspecting the current files.

| Output | Template / customization |
|---|---|
| `runner.py` | `assets/environment/runner.py`: fixed request parsing, session URL selection, result serialization, artifact output, and controlled-failure handling. |
| `adapter.py` | `assets/environment/adapter.py`: fill `run_case(request, task, session_url)`, returning `(metrics, step_count)`. Replace the guide's greeting example with the native single-case invocation. |
| `<name>_config.yaml` | `assets/environment/config.yaml.tmpl`: fill the Docker image, dataset, and env parameters. The example dataset has one row. |
| `<name>_config.rjob.yaml` | `assets/environment/config.rjob.yaml.tmpl`: repeat task rows with an image tag pullable by the RJob cluster. Keep the shared `env_params` schema. |
| `<name>_start.yaml` | `assets/environment/start.docker.yaml.tmpl`: fill workdir and Docker mounts. Mount runner dependencies beside the runner. |
| `<name>_start.rjob.yaml` | `assets/environment/start.rjob.yaml.tmpl`: fill resources, cluster-accessible result storage, and embedded dependencies. The runner source is staged by the runtime; `adapter.py` is explicitly embedded. |
| `request.smoke.json` | `assets/environment/request.smoke.json.tmpl`: fill one real case and its environment parameters for local tests. |
| `rule_evaluator.py` | Optional `assets/environment/rule_evaluator.py`: fill only `score_metrics`; keep discovery/interface/failed-result handling. |

All asset paths above are relative to the skill directory. If extra adapter modules are needed, include them in Docker mounts and RJob embedded files. A Dockerfile is optional when no suitable image exists; derive it from the native harness's dependencies without moving case logic into it.

The fixed runner has no benchmark imports until it invokes the hook. Its stdout contains exactly one result. Python diagnostic output is redirected to stderr; native subprocesses must use `capture_output=True` or explicitly send stdout to stderr. Controlled exceptions yield `status: failed`, an `error_text`, and process exit 0. Successful execution is `status: succeeded` even if an optional native score is low. Integration-only results keep `total_reward: 0.0`; there is no required score/pass field in metrics.

## Local validation (no cluster required)

Fill `request.smoke.json` with a representative row and configure `adapter.py` to invoke the native harness. Run the same tests for Docker and RJob targets:

```bash
python skills/safactory-workflows/scripts/contract_smoke.py \
  --runner env/mybench/runner.py \
  --request env/mybench/request.smoke.json \
  --require-model-call

# If native dependencies are not installed locally, copy an explicit fixture
# adapter for protocol-only validation; this does not validate native behavior.
python skills/safactory-workflows/scripts/contract_smoke.py \
  --runner env/mybench/runner.py --adapter path/to/adapter_fixture.py \
  --request env/mybench/request.smoke.json --require-model-call

# Independently test environment-variable input with empty stdin:
python skills/safactory-workflows/scripts/contract_smoke.py \
  --runner env/mybench/runner.py \
  --request env/mybench/request.smoke.json \
  --input-mode env --require-model-call
```

The helper starts a mock HTTP endpoint on an ephemeral loopback port, routes this one request to it, runs the Python runner with a timeout, checks session identity and result types, checks the result artifact if written, and shuts down the endpoint. It emits a `local-contract-only` summary, not a Gateway trajectory or evaluation reward. It uses only Python's standard library; it does not import Launcher or the RJob SDK.

`--response path/to/response.json` supplies a native-compatible non-streaming chat response. The built-in response is a fixture greeting. Streaming, tool protocols beyond chat JSON, other runner languages, or image-only harness dependencies need environment-specific tests/fixtures. Explicitly stub the native command in those tests and state what was mocked; do not add a production runner switch that fabricates a passing benchmark result. A scaffold greeting passing is not evidence that the benchmark integration works.

Use 1–2 real-row-shaped fixtures and check that each maps to the intended native command and output. Include a controlled native failure (`--expect-status failed`, without requiring a model call if failure precedes it), malformed requests, and stdout isolation. Inspect mapped metrics/output paths as well as the helper summary. When evaluation is requested, add fixture tests for the scoring hook covering success, low/zero score, and missing/invalid metrics. The evaluator must not execute the native case again.

Validate selected-mode YAML and relative source paths with the repository config loaders when dependencies are available. For RJob, inspect the rendered runtime command/embedded files without submitting; installing an internal cluster is not a prerequisite for local adapter validation. Document any unverified config checks.

## Live validation with an owned Gateway

Only this stage needs the actual image, dataset, model route/credentials, and Docker or an existing configured RJob cluster. Do not require a Geo3K live baseline before local work. A baseline can help diagnose shared infrastructure if a live run fails.

Use a task config containing only 1–2 cases; `env_num`, `--pool-size`, and `--max-workers` do not limit dataset length. Preserve the user's Gateway config and verify the route and storage. This single command owns Gateway startup/readiness/shutdown and propagates Launcher failures:

```bash
python skills/safactory-workflows/scripts/live_smoke.py \
  --gateway-config gateway/config.local.yaml \
  --run-timeout 600 -- \
  --mode docker \
  --agent-config env/mybench/mybench_config.yaml \
  --agent-start-config env/mybench/mybench_start.yaml \
  --llm-model YOUR_ROUTE_KEY \
  --job-id mybench-docker-smoke \
  --pool-size 1 --max-workers 1 --max-steps 10
```

The helper passes the Gateway's SQLite URI to Launcher if `--db-path` is omitted. Explicit storage and route mismatches fail before starting processes. It requires a free Gateway port; if a Gateway is already running, verify its readiness, routes, and storage, then use `launcher.py` directly. It never stops a Gateway it did not start. Local helper process cleanup does not replace Launcher/Docker/RJob resource cleanup; inspect runtime resources if a timeout interrupts a job.

For a live RJob run, use `--mode rjob`, `--rjob-config` and both `.rjob.yaml` files, set matching `--storage-type`, and pass `--gateway-base-url http://GATEWAY_HOST:8000/v1/sessions` reachable from the cluster. The helper can start a Gateway on the current host; that host must already be reachable from the cluster. Check that any `gateway_base_url` in the global RJob config agrees, as it can override the CLI value. See `docs/internal/rjob-mode.md` for cluster settings. Local tests do not validate cluster networking, mounts, image pulls, or submission.

Integration-only live checks inspect runner JSON, native outputs, Gateway request/trajectory records, and completed runtime rows. Omit `--enable-evaluation`; do not demand an evaluator or final normalized reward. For evaluation requests, implement/test the optional evaluator and append `--enable-evaluation` to the Launcher arguments, then inspect the final `0–10` reward as well.

## Report the evidence

State which cases ran, which dependencies/model responses were fixtures, and which level passed: local contract, live deployment, and (if requested) evaluation. Local adapter validation can finish while cluster verification is pending. If the user requested live deployment, list its precise blocker and ready-to-run command without presenting local success as live success.
