# CVEFactory registry-image rollout

This document describes the end-to-end SAfactory flow for CVEFactory tasks whose
main task image is pulled from the per-CVE registry repository:

```text
registry.h.pjlab.org.cn/ailab-llmagent-llmagent_cpu_task/cve_factory:<lowercase-cve-id>
```

The rollout uses the normal SAfactory services:

```text
SAfactory Gateway
SAfactory launcher
└── child RJob
    └── LiveCVEBench/CVEFactory runner
        └── nested Docker Compose task
```

The Gateway must be running before the launcher starts, and its configured URL
must be reachable by the child RJobs.

## 1. Contracts

### 1.1 Child environment image

The child RJob environment is the standard LiveCVEBench runner image:

```text
registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches:livecvebench-runner-20260728-rjob1
```

Use it as `environments[].env_image`.

### 1.2 Main CVEFactory task image

For every dataset row whose suite is `cve-factory-3k`, the runner constructs:

```text
registry.h.pjlab.org.cn/ailab-llmagent-llmagent_cpu_task/cve_factory:<task-id-lowercase>
```

Example:

```text
registry.h.pjlab.org.cn/ailab-llmagent-llmagent_cpu_task/cve_factory:cve-2023-24808
```

The repository supplies the main/client task image. A task Compose file may
also start secondary services such as MySQL; preserve those service definitions
and allow Compose to pull their images normally.

### 1.3 Suite name

The exact suite value is:

```text
cve-factory-3k
```

Do not rename it to `cvefactory`, `cve-factory`, or `cve-bench-1k`. The runner
selects the registry-image path from this exact value.

### 1.4 Dataset row

A CVEFactory dataset is JSONL. Each selected row minimally looks like:

```json
{"task_id":"cve-2023-24808","suite":"cve-factory-3k"}
```

For a merged dataset, preserve each row's original suite and configure each
suite separately. Do not merge all task families into one generic image cache.

### 1.5 Bundled CVEFactory 3K dataset

The repository includes a dedicated registry-image dataset:

```text
env/livecvebench/datasets/cvefactory3k_registry_2787_20260807.jsonl
```

It contains:

```text
rows:             2,787
unique task_ids:  2,787
suite:            cve-factory-3k
```

The `3K` name is the collection name; after deduplication, the registry set
contains 2,787 CVE IDs rather than exactly 3,000 rows. Every row uses the JSONL
shape shown above.

The dataset may be used directly or copied into a run-specific `dataset.jsonl`:

```bash
cp env/livecvebench/datasets/cvefactory3k_registry_2787_20260807.jsonl \
  <run>/dataset.jsonl
```

Before a rollout, verify that the configured task root contains a directory and
Compose file for every `task_id` in the selected file.

## 2. Repository layout used by this flow

From the SAfactory source root:

```text
launcher.py
gateway/
env/livecvebench/runner.py
env/livecvebench/rule_evaluator.py
```

The relevant runner branch is already implemented in
`env/livecvebench/runner.py`. No runner modification is required merely to use
the registry repository.

Conceptually, the runner performs:

```text
read request
→ resolve task_id and suite
→ resolve task directory
→ resolve or pull the main task image
→ ensure Terminal-Bench tooling
→ run Terminal-Bench for the selected task
→ emit safactory_result.json
```

For `cve-factory-3k`, image resolution is:

1. If `<task-root>/<task-id>/<task-id>-image.tar` exists, load that local
   archive and tag it as the registry image.
2. Otherwise, use the Compose file's image references and pull them through
   Docker Compose.

The runner calculates the canonical registry reference with a lowercase task ID:

```python
registry_image = (
    "registry.h.pjlab.org.cn/ailab-llmagent-llmagent_cpu_task/"
    f"cve_factory:{task_id.lower()}"
)
```

## 3. Prepare tasks

The runner needs a flat task root:

```text
<task-root>/
  <task-id>/
    docker-compose.yaml
    task.yaml
    tests/
    ...
```

Every `task_id` in the dataset must have one corresponding directory. The
Compose file's main/client service must reference:

```text
registry.h.pjlab.org.cn/ailab-llmagent-llmagent_cpu_task/cve_factory:<lowercase-task-id>
```

If source task files use a `build:` section, set the main service's `image:` to
the registry reference while retaining the rest of the Compose service
definition. Do not blindly remove secondary services, volumes, environment
variables, health checks, or dependency definitions.

A minimal validation script:

```python
import json
from pathlib import Path

dataset = Path("<run>/dataset.jsonl")
task_root = Path("<task-root>")
repo = "registry.h.pjlab.org.cn/ailab-llmagent-llmagent_cpu_task/cve_factory"

rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
ids = [row["task_id"] for row in rows if row.get("suite") == "cve-factory-3k"]

assert ids, "dataset has no cve-factory-3k rows"
assert len(ids) == len(set(ids)), "duplicate cve-factory-3k task_id"

for task_id in ids:
    compose = task_root / task_id / "docker-compose.yaml"
    assert compose.is_file(), compose
    expected = f"{repo}:{task_id.lower()}"
    assert expected in compose.read_text(encoding="utf-8"), compose

print({
    "total_rows": len(rows),
    "cvefactory_rows": len(ids),
    "unique_task_ids": len(set(ids)),
})
```

## 4. Docker authentication and proxy behavior

The nested Docker client must be able to authenticate to:

```text
registry.h.pjlab.org.cn
```

Provide a Docker auth configuration through:

```text
SAFACTORY_DOCKER_AUTH_CONFIG=<path-to-docker-config-json>
```

The file must contain the registry entry under `auths`. The runner combines
that auth mapping with the child's proxy configuration. Do not replace the auth
file with a proxy-only configuration.

Both proxy bypass spellings must include the registry, Gateway, cluster zones,
and loopback addresses:

```text
NO_PROXY=<registry-and-gateway-bypass-list>
no_proxy=<registry-and-gateway-bypass-list>
```

Never commit credentials or print the auth file. Validate key names only:

```python
import json
from pathlib import Path

config = json.loads(Path("<docker-auth-json>").read_text())
print(sorted(config.get("auths", {})))
```

## 5. Create a run

Create run-specific artifacts before submitting child RJobs:

```text
<safactory-run-root>/
  agent.yaml
  start.yaml
  gateway.yaml
  rjob-config.yaml
  dataset.jsonl
  runner.py
```

Use fresh values for:

- job ID;
- run name;
- child-RJob prefix;
- Gateway port;
- request log;
- result root.

Do not reuse a job ID to hide a failed launch. Reuse it only for an explicit,
compatible resume.

Copy the repository runner into the run when a formal rollout needs a frozen
entrypoint:

```bash
cp env/livecvebench/runner.py <run>/runner.py
python3 -m py_compile <run>/runner.py
```

Then point `start.yaml` at that copy.

For the bundled registry-image task set, also copy:

```bash
cp env/livecvebench/datasets/cvefactory3k_registry_2787_20260807.jsonl \
  <run>/dataset.jsonl
```

## 6. `agent.yaml`

The following is a generic template. Replace all `<...>` values.

```yaml
agent_name: livecvebench

environments:
  - env_name: livecvebench
    env_image: registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches:livecvebench-runner-20260728-rjob1
    env_num: <episodes-per-row>
    dataset: <run>/dataset.jsonl
    dataset_load_mode: eager

    env_params:
      task_family: cve_patch

      dataset_paths:
        cve-factory-3k: <task-root>

      # Intentionally omit cve-factory-3k from image_cache_dirs.
      # This suite resolves its main image from the registry.

      output_root: <child-result-root>
      safactory_results_root: <controller-result-root>/safactory
      launcher_results_root: <controller-result-root>/safactory

      agent: <agent-name-or-import-path>
      model: <model-name>
      route_model: <model-name>
      n_concurrent: 1
      global_agent_timeout_sec: <agent-timeout-seconds>
      global_test_timeout_sec: <test-timeout-seconds>
      timeout_s: <outer-task-timeout-seconds>
```

Important:

- `dataset_paths.cve-factory-3k` is mandatory because this suite has no
  hard-coded default task path.
- Do not add `cve-factory-3k` to `image_cache_dirs`.
- Expected episodes are `len(dataset) * env_num`.

For a first smoke, use one CVE row, `env_num: 1`, one launcher worker, and one
child RJob. Scale concurrency only after that end-to-end path succeeds.

## 7. `start.yaml`

Generic shape:

```yaml
agent_name: livecvebench

container:
  workdir: /benchmark
  runner_entrypoint:
    source: <run>/runner.py
    target: /tmp/safactory-livecvebench-runner.py
    command: /usr/bin/python3.10 /tmp/safactory-livecvebench-runner.py
    result_mode: json

  env:
    HTTP_PROXY: <child-outbound-proxy>
    HTTPS_PROXY: <child-outbound-proxy>
    SAFACTORY_DOCKER_AUTH_CONFIG: <mounted-docker-auth-json>
    NO_PROXY: <registry-and-gateway-bypass-list>
    no_proxy: <registry-and-gateway-bypass-list>

    # Populate agent credentials and model variables according to the selected
    # agent integration. Do not put real tokens in this file.

rjob:
  name_prefix: <fresh-child-prefix>
  charged_group: <approved-quota-group>
  private_machine: Group
  gateway_base_url: http://<gateway-host>:<gateway-port>/v1/sessions
  image_pull_policy: IfNotPresent
  privileged: true
  no_packaging: true
  cleanup_on_finish: true
  keep_failed_jobs: false
  poll_interval_s: 5
  termination_grace_period_seconds: 120

  embedded_files:
    - source: <run>/<agent-module>.py
      target: /benchmark/<agent-module>.py

  mount_config:
    - <task-root-mount>
    - <docker-auth-mount>
    - <result-root-mount>

  resources:
    cpu: <cpu>
    gpu: 0
    memory_in_mb: <memory-mb>
```

A child needs access to:

- the selected task root;
- the Docker auth file;
- the result root;
- the configured Gateway URL;
- any pinned agent binary or module required by `agent.yaml`.

The child must be privileged because the runner starts a nested Docker daemon
and Docker Compose tasks.

## 8. `gateway.yaml`

Generic cloud-storage shape:

```yaml
listen_host: 0.0.0.0
listen_port: <gateway-port>
base_session_path: /v1/sessions

storage_type: cloud
storage_config: {}

telemetry:
  mode: strict
  loss_policy: fail_closed
  capture_payload: full

request_log:
  enabled: true
  path: <run>/requests.jsonl
  body_limit_bytes: 0

llm_routes:
  <model-name>:
    base_url: <approved-model-endpoint>
    api_key: ${MODEL_API_KEY}
    supports_stream: true
    max_concurrency: <route-concurrency>
```

Load credentials through the Gateway process environment. Do not commit keys.

## 9. Start order

### 9.1 Start the Gateway

From the SAfactory source root:

```bash
<python> -m gateway --config <run>/gateway.yaml
```

Run it as a detached process and persist its stdout/stderr and PID. Require:

```bash
curl --noproxy '*' -fsS \
  http://127.0.0.1:<gateway-port>/readyz
```

Do not start the launcher until the Gateway is ready and reports the intended
storage type.

### 9.2 Start the launcher

From the configured result root:

```bash
<python> <safactory-source>/launcher.py \
  --mode rjob \
  --rjob-config <run>/rjob-config.yaml \
  --agent-config <run>/agent.yaml \
  --agent-start-config <run>/start.yaml \
  --agent-root <safactory-source>/env \
  --gateway-base-url http://<gateway-host>:<gateway-port>/v1/sessions \
  --llm-model <model-name> \
  --storage-type cloud \
  --disable-buffer \
  --pool-size <workers> \
  --multiplier 1 \
  --max-workers <workers> \
  --enable-evaluation \
  --agent-start-timeout-s <child-start-timeout> \
  --gateway-close-timeout-s <gateway-close-timeout> \
  --max-steps <max-model-steps> \
  --job-id <job-id> \
  --run-name <run-name>
```

Add the deployment's required production-authorization flags only after the
exact job ID, table, and storage destination have been approved. The launcher
submits and monitors child RJobs.

## 10. Execution path

After the launcher starts:

1. It reads `agent.yaml` and the JSONL dataset.
2. It creates the configured number of environments.
3. It submits child RJobs through `start.yaml`.
4. Each child starts from the LiveCVEBench runner image.
5. The runner reads the request selected for that environment.
6. It resolves `task_id` and `suite=cve-factory-3k`.
7. It finds `<task-root>/<task-id>`.
8. It loads an optional local image archive or pulls the Compose image.
9. It runs Terminal-Bench for that task.
10. It emits `SAFACTORY_RESULT_JSON` and `safactory_result.json`.
11. The launcher evaluates the result and updates storage.
12. The Gateway persists model trajectories and telemetry.

## 11. Monitoring

```bash
ps -p <gateway-pid> -o pid,lstart,etime,stat,args
ps -p <launcher-pid> -o pid,lstart,etime,stat,args
```

Child RJobs:

```bash
rjob list --namespace <approved-namespace> | grep <child-prefix>
```

Launcher:

```bash
grep -c 'finished status=' <controller-log>/launcher.log
grep -c 'finished status=succeeded' <controller-log>/launcher.log
grep -c 'finished status=failed' <controller-log>/launcher.log
grep -c 'finished status=truncated' <controller-log>/launcher.log
tail -50 <controller-log>/launcher.log
```

Gateway:

```bash
grep -c 'Gateway telemetry dropped' <controller-log>/gateway.log || true
grep -c 'Gateway telemetry write timed out' <controller-log>/gateway.log || true
grep -c 'Gateway storage record_step batch failed' <controller-log>/gateway.log || true
```

Also inspect:

```text
requests.jsonl
docker-pull.log
docker-cache-load.log
dockerd.log
tb-run.log
safactory_result.json
```

## 12. Completion checks

A rollout is complete only when all of the following agree:

- launcher accounts for every expected environment;
- no child RJob remains running;
- request logs contain successful model calls;
- telemetry reports no dropped or failed storage writes;
- exact-job landing rows exist;
- terminal/session-completed markers are present;
- reward or evaluation commits complete where applicable;
- GPFS or mounted result artifacts exist;
- `safactory_result.json` and launcher status agree.

Expected environment count is:

```text
dataset rows × env_num
```

Explicitly account for succeeded, failed, truncated, and still-running cases.
Do not infer success from child cleanup alone.

## 13. Common failures

### Registry authentication failed

Check:

- `SAFACTORY_DOCKER_AUTH_CONFIG` is set in the child;
- the auth file contains the registry under `auths`;
- the auth file is actually mounted;
- `NO_PROXY` and `no_proxy` include the registry and Gateway;
- proxy and auth settings were not overwritten.

### Image not found

Confirm the tag uses the lowercase CVE ID:

```text
cve-2023-24808
```

not:

```text
CVE-2023-24808
```

### Task directory missing

The runner requires:

```text
<task-root>/<task-id>/docker-compose.yaml
```

A dataset row alone is not enough.

### Compose references the wrong image

The main/client image must be the per-CVE registry reference. Preserve other
Compose services and their images.

### Added `cve-factory-3k` to `image_cache_dirs`

Remove it. This suite is registry-backed. A local `<task-id>-image.tar` is
optional; a generic image-cache directory is not the intended path.

### Gateway URL unreachable from child

The child must be able to reach the configured Gateway URL. Verify DNS or IP,
port, proxy bypass entries, and Gateway readiness.

### Result artifacts missing

Check the child result mount and ensure the launcher runs from the configured
result root. Verify both common artifact layouts:

```text
<result-root>/<job-id>/<session-id>/safactory_result.json
<result-root>/safactory/<job-id>/<session-id>/safactory_result.json
```

## 14. Minimal smoke checklist

Before a batch:

1. Select one known CVE.
2. Confirm its task directory and Compose image.
3. Use one JSONL row.
4. Set `env_num: 1`.
5. Use one launcher worker.
6. Start the Gateway and verify `/readyz`.
7. Submit one child RJob.
8. Confirm the registry image pull.
9. Confirm at least one successful model request.
10. Confirm `safactory_result.json`.
11. Confirm exact-job trajectory rows.
12. Confirm child cleanup and launcher completion.

Only after this complete path succeeds should concurrency or dataset size be
increased.
