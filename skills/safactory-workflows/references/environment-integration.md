# Environment integration

Read `docs/guides/custom-environment.md` for the request/result and deployment contracts. The standard reference layout is `env/prmeval/`; every new environment comes from the same fixed template set. This workflow separates adapter validation, live deployment, and optional evaluation, and labels every failure with the side that owns it.

## Intake and scope

Infer the environment name, source, one-row schema, 1–2 cases, native single-case command, and native output format. Ask only for missing information needed by the next dependent step. Deployment mode/image can remain pending during shared adapter work. Reward fields and scoring rules are needed only for evaluation or training. A benchmark with native scores can still be integrated without enabling evaluation.

Keep native execution and scoring in the existing harness. The adapter maps one `env_params.dataset` row, routes model calls through the current session, invokes one native case, collects output, and emits the runtime result. Do not run a full benchmark inside a single episode.

## The fixed file set

Run from the repository root:

```bash
python skills/safactory-workflows/scripts/scaffold_environment.py mybench
# Append --enable-evaluation only when scoring is requested.
```

The scaffolder refuses to overwrite an existing environment and always emits the full fixed file set — the Docker pair is the base contract and the RJob pair adds cluster settings, keeping the environment portable between modes:

| File | Responsibility | Filled per benchmark? |
|---|---|---|
| `runner.py` | Fixed SAfactory protocol shell: request parsing, session URL resolution, one result JSON, controlled failure, artifact write. | No — standard part; replace wholesale when the protocol evolves. |
| `adapter.py` | The only benchmark-logic file: `run_case(request, task, session_url)` for exactly one dataset row. | Yes — the main integration work. |
| `<name>_config.yaml` | Docker task config: image, dataset, `env_params` (serialized into every request and transparently passed into the container). | Yes — image, dataset path, native config block. |
| `<name>_start.yaml` | Docker container startup: entrypoint, mounts, env vars, network. | Yes — mounts for datasets/results and any extra deps. |
| `<name>_config.rjob.yaml` | RJob task config, same schema; image must be pullable by the cluster. | Mirror the Docker config. |
| `<name>_start.rjob.yaml` | RJob startup: the same container block plus `rjob:` resources, `mount_config`, `embedded_files`. | Yes — cluster storage paths. |
| `rule_evaluator.py` | Optional score mapping (`score_metrics` hook); discovered only with `--enable-evaluation`. | Only when evaluation is requested. |
| `request.smoke.json` | One-case request fixture for local contract checks. | Yes — use a real row shape. |

Key invariants (checked automatically, see below): `agent_name` must equal `env_name`; `env_params.results_root` must equal the results mount target in the matching start file; absolute file paths inside dataset rows must point under a mount target (both modes); the adapter must sit beside the runner in the container in both modes.

The generated Docker mount sources are rooted at the launcher working directory (the repository root in the commands below). Runner sources and RJob embedded-file sources remain relative to their start-config file. Keep these two path bases distinct; using `./adapter.py` as a Docker mount from the repository root would point at `SAfactory/adapter.py`, not `env/mybench/adapter.py`.

If extra adapter modules are needed, include them in Docker mounts and RJob embedded files. A Dockerfile is optional when no suitable image exists; derive it from the native harness's dependencies without moving case logic into it.

The fixed runner has no benchmark imports until it invokes the hook. Its stdout contains exactly one result. Python diagnostic output is redirected to stderr; native subprocesses must use `capture_output=True` or explicitly send stdout to stderr. Controlled exceptions yield `status: failed`, an `error_text`, and process exit 0. Successful execution is `status: succeeded` even if an optional native score is low. Integration-only results keep `total_reward: 0.0`; there is no required score/pass field in metrics.

## Verify with one command

`check_environment.py` chains the stages and short-circuits: static errors skip the contract run, and the live stage never starts on top of failures.

```bash
python skills/safactory-workflows/scripts/check_environment.py --env env/mybench

# Static checks only (file set, naming, config <-> start <-> dataset invariants):
python skills/safactory-workflows/scripts/validate_environment.py env/mybench

# Protocol shell only, when native dependencies are not installed locally:
python skills/safactory-workflows/scripts/check_environment.py --env env/mybench \
  --fixture-adapter path/to/adapter_fixture.py

# Live deployment: everything after --live is forwarded to live_smoke.py.
python skills/safactory-workflows/scripts/check_environment.py --env env/mybench --live \
  --gateway-config gateway/config.local.yaml --run-timeout 600 -- \
  --mode docker \
  --agent-config env/mybench/mybench_config.yaml \
  --agent-start-config env/mybench/mybench_start.yaml \
  --llm-model YOUR_ROUTE_KEY \
  --job-id mybench-docker-smoke --pool-size 1 --max-workers 1 --max-steps 10
```

Static findings are labeled `[config]` (integration wiring in the environment directory), `[env]` (adapter code, dataset rows, request fixture), or `[safactory]` (framework side — the integration is fine). Fix all errors; warnings name the deployment mode they block.

The contract stage runs the real `adapter.py` against an owned mock model endpoint on an ephemeral loopback port: no Gateway, Docker daemon, model credentials, or RJob SDK. It checks session identity, result schema, artifact equality, model routing through the session, and controlled-failure shape. `--fixture-adapter` substitutes a stub to verify the protocol shell only; a fixture passing is not evidence that the benchmark integration works. `--response` (via `contract_smoke.py`) supplies a native-shaped non-streaming chat response; streaming, tool protocols beyond chat JSON, other runner languages, or image-only harness dependencies need environment-specific fixtures.

Use 1–2 real-row-shaped fixtures and check that each maps to the intended native command and output. Include a controlled native failure (`--expect-status failed` via `contract_smoke.py`), malformed requests, and stdout isolation. When evaluation is requested, add fixture tests for the scoring hook covering success, low/zero score, and missing/invalid metrics; the evaluator must never rerun the case.

## Failure triage

Classify before debugging; each row names the side that owns the fix:

| Symptom | Side | First action |
|---|---|---|
| Static `[config]` finding (name/mount/results_root/dataset-path mismatch) | config | Fix the named file per the hint; rerun `check_environment.py`. |
| Contract: `runner process failed: exit N` | env | The runner must always exit 0; convert the failure into a `status: failed` result JSON. |
| Contract: result `status: failed` with `error_text` | env | Controlled native/adapter failure — debug `adapter.py` or the native command. |
| Contract: `unexpected model path` / `model route differs` | env | The adapter must call `{session_url}/chat/completions` with `request["model"]`. |
| Contract: `ModuleNotFoundError` for native deps | env | Install deps locally, or use `--fixture-adapter` to isolate protocol from native logic. |
| Contract: stdout parse / "extra stdout" errors | env | Native prints leaked onto stdout — capture or redirect them to stderr. |
| live_smoke preflight error (route/storage/URL mismatch) | safactory | Fix Gateway config vs launcher args — configuration of shared infrastructure. |
| `Gateway exited before readiness` / readiness timeout | safactory | Gateway-side startup failure; check the Gateway log, then a geo3k baseline to isolate. |
| `Gateway exited during Launcher run` | safactory | Shared infrastructure died mid-run; Gateway logs first, then rerun. |
| Launcher exit non-zero with no result JSON | safactory/env | Launcher/Gateway logs first; if the container ran, it is an environment protocol violation. |
| Episode result `status: failed`, exit 0 | env | In-band environment failure — read `error_text`, then native output under `results_root`. |
| Everything green but no files under `results/` | config | `results_root` not mounted (the static stage flags this) or wrong artifact path. |
| `rule evaluator not found for environment` | config | `--enable-evaluation` was set without `rule_evaluator.py`; add it or drop the flag. |
| Reward always 0.0 with `--enable-evaluation` off | — | Expected: integration-only runs keep `total_reward: 0.0`; it is not an evaluation outcome. |

A geo3k live baseline can help decide safactory-side vs environment-side when shared infrastructure is suspected. Do not require it before local work.

## Live validation with an owned Gateway

Only this stage needs the actual image, dataset, model route/credentials, and Docker or an existing configured RJob cluster. Use a task config containing only 1–2 cases; `env_num`, `--pool-size`, and `--max-workers` do not limit dataset length. Preserve the user's Gateway config and verify the route and storage. The helper owns Gateway startup/readiness/shutdown and propagates Launcher failures; see the `--live` command above or call `live_smoke.py` directly.

The helper passes the Gateway's SQLite URI to Launcher if `--db-path` is omitted. Explicit storage and route mismatches fail before starting processes. It requires a free Gateway port; if a Gateway is already running, verify its readiness, routes, and storage, then use `launcher.py` directly. It never stops a Gateway it did not start. Local helper process cleanup does not replace Launcher/Docker/RJob resource cleanup; inspect runtime resources if a timeout interrupts a job.

For a live RJob run, use `--mode rjob`, `--rjob-config` and both `.rjob.yaml` files, replace the `CLUSTER_STORAGE` placeholders, set matching `--storage-type`, and pass `--gateway-base-url http://GATEWAY_HOST:8000/v1/sessions` reachable from the cluster. The helper can start a Gateway on the current host; that host must already be reachable from the cluster. Check that any `gateway_base_url` in the global RJob config agrees, as it can override the CLI value. See `docs/internal/rjob-mode.md` for cluster settings. Local tests do not validate cluster networking, mounts, image pulls, or submission.

Integration-only live checks inspect runner JSON, native outputs, Gateway request/trajectory records, and completed runtime rows. Omit `--enable-evaluation`; do not demand an evaluator or final normalized reward. For evaluation requests, implement/test the optional evaluator and append `--enable-evaluation` to the Launcher arguments, then inspect the final `0–10` reward as well.

## Report the evidence

State which cases ran, which dependencies/model responses were fixtures, and which level passed: static, local contract, live deployment, and (if requested) evaluation. Local adapter validation can finish while cluster verification is pending. If the user requested live deployment, list its precise blocker and ready-to-run command without presenting local success as live success.
