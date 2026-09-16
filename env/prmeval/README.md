# PRMEval standard environment

`env/prmeval` is the reference layout for onboarding a benchmark that already
has a container image, a per-case dataset row, a native runner, and optional
native metrics.

| File | Responsibility |
| --- | --- |
| `runner.py` | Fixed SAfactory protocol shell. Reads one `SimulationStartRequest`, calls the adapter, and emits exactly one result JSON. |
| `adapter.py` | PRMEval-only logic: writes the current row to a temporary JSONL, injects the session-aware Gateway URL/model, runs `Evaluator`, and maps native artifacts into `metrics`. |
| `rule_evaluator.py` | Optional evaluation stage. Reads stored PRMEval MSE and maps it to a 0--10 reward; it never reruns PRMEval. |
| `prmeval_config.yaml` | Docker task rows, local image, and environment parameters. One row is one episode. |
| `prmeval_start.yaml` | Docker runner command, explicit read-only adapter/dataset mounts, and result mount. |
| `prmeval_config.rjob.yaml` | RJob task rows and cluster-pullable image. |
| `prmeval_start.rjob.yaml` | RJob resources, embedded adapter, and cluster-accessible result/dataset mounts. |
| `datasets/samples.jsonl` | Native PRMEval trajectory rows. Frame paths must exist at the same path inside the runtime image/mount. |

The Docker start config assumes commands are launched from the SAfactory
repository root (as `live_smoke.py` requires). Its bind sources therefore use
`./env/prmeval/...`; the RJob config instead resolves embedded-file sources
relative to its own config file and uses cluster-visible storage.

## Fast local contract check

The full PRMEval wheel and model are not needed to validate the SAfactory
protocol. Use the skill helper with a temporary adapter fixture:

```bash
python skills/safactory-workflows/scripts/contract_smoke.py \
  --runner env/prmeval/runner.py \
  --adapter skills/safactory-workflows/assets/environment/adapter.py \
  --request env/prmeval/request.smoke.json \
  --require-model-call
```

The `--adapter` argument is intentional: it substitutes the tiny standard
fixture because PRMEval's wheel is not required for this protocol-only check.
To validate the real PRMEval adapter, run it inside the built image (or install
its dependencies locally) and remove `--adapter`.

For a real PRMEval run, build/pull the selected image and use a Gateway route
through the normal Launcher command. `--enable-evaluation` is opt-in; an
integration-only run can succeed with `total_reward: 0.0` while still exposing
the native MSE/output paths in `metrics`.

Before an RJob run, replace the example `gpfs://...` sources in
`prmeval_start.rjob.yaml` with storage visible to the cluster and use a
cluster-reachable Gateway URL. Local Docker bind mounts cannot be reused by
RJob.
