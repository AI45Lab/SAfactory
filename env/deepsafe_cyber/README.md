# env/deepsafe_cyber

SAfactory integration for the **deepsafe-cyber** distribution of PatchEval
verified (`dataset_profile: patcheval-verified-v1`, 230 CVE rows).  Docker mode
has passed a two-case live evaluation; RJob mode is wired below but has not yet
been submitted to the cluster.

The SAfactory environment identifier is `deepsafe_cyber` because the workflow's
fixed environment-name contract permits underscore only.  The native benchmark
identifier in metrics remains `deepsafe-cyber`.

## Layout

- `deepsafe_cyber_config.yaml`: full 230-row Docker task config.
- `deepsafe_cyber_config.smoke.yaml`: two-row Docker smoke config
  (`CVE-2021-23376`, `CVE-2021-23363`).
- `deepsafe_cyber_start.yaml`: privileged DinD container wiring.
- `deepsafe_cyber_start.rjob.yaml`: privileged RJob DinD wiring with cluster
  GPFS mounts, persistent nested-Docker state, and embedded runner files.
- `rjob_entrypoint.sh`: starts deepsafe's native DinD entrypoint under RJob,
  keeps daemon logs out of the runner JSON stream, serializes shared DinD
  state, and enforces a read-only trustcyberdata mount.
- `operator/run.yaml`: immutable native run config; adapter copies it per episode.
- `adapter.py`: generates a session-scoped `providers.yaml`, waits for inner
  Docker readiness, invokes one `deepsafe-cyber run --task`, and parses
  `public/result.json`.
  The adapter invokes `/opt/deepsafe-cyber/.runtime/bin/deepsafe-cyber`
  directly so behavior does not depend on shell or PATH initialization.
- `rule_evaluator.py`: converts native terminal state to SAfactory reward
  without rerunning the case.
- `contract_adapter.py`: local-contract replacement for unavailable native
  image/DinD; it exercises provider generation, native argv, Gateway routing,
  and published-result parsing through the production adapter.

## Model routing

The adapter generates an ephemeral root-owned `providers.yaml` (mode `0600`)
for each episode and exports `DEEPSAFE_CYBER_PROVIDER_CONFIG`.  Its entry is:

- credential profile: `safactory-gateway`
- model profile: `safactory-gateway-responses`
- provider protocol: `openai-responses`
- `base_url`: the current SAfactory `{session_url}`
- `model`: `request["model"]`

The mounted `operator/run.yaml` selects `harness_profile: dsh-headless`; its
Responses-to-Chat bridge accepts the native model profile while preserving
SAfactory session telemetry.  The profile names in the run YAML intentionally
match the generated provider entry.  No real model API key is embedded; the
non-secret placeholder only passes deepsafe's key-length validation while
authorization remains session-scoped.

Native output is read from `terminal.code`; process exit code is never used for
scoring.  `PASS` maps to 10.0 and `MODEL_INCORRECT` to 0.0.  All other terminal
codes (for example `PREFLIGHT_FAILED` or `INVALID_CONFIGURATION`) produce a
failed evaluation result, not model incorrect, and retain the original native
code/artifact paths.

SAfactory allocates its own managed outer-container name rather than reusing the
standalone documentation's fixed `deepsafe-cyber` name.  The runtime still
executes one episode through one non-TTY `docker exec` in `/opt/deepsafe-cyber`;
the adapter then invokes the `deepsafe-cyber` CLI in that same DinD container.

## Dataset

`datasets/patcheval-verified-v1.jsonl` contains exactly one JSON object per
cached task image:

```json
{"task":"CVE-2021-23376"}
```

The 230 identifiers were generated from the supplied read-only PatchEval image
cache.  SAfactory schedules one episode per row.  The adapter explicitly passes
that row as `deepsafe-cyber run --task`; it never loops over the JSONL file.

## Local validation

From the repository root:

```bash
python -m unittest discover -s tests/env/deepsafe_cyber -p 'test_*.py' -v

python skills/safactory-workflows/scripts/check_environment.py \
  --env env/deepsafe_cyber \
  --expect-evaluation \
  --fixture-adapter env/deepsafe_cyber/contract_adapter.py \
  --timeout 30
```

The contract check uses a mock Gateway response and a fake native subprocess;
it does not start Docker, load the benchmark image, call a real model, or prove
that `dsh-headless` is compatible with the deployed Gateway.

## Docker live prerequisites

1. Make the benchmark image available. Either pull it or load the supplied tar:

   ```bash
   docker load -i /mnt/shared-storage-user/trustcyberdata/private/yuqiyi/deepsafe/dist/deepsafe-cyber-0.1.0-linux-amd64.tar
   ```

2. Preserve read-only access to
   `/mnt/shared-storage-user/trustcyberdata`; its PatchEval cache supplies the
   230 `cve-*.tar` images and deepsafe preflight loads them automatically.
3. Start with a Gateway config whose selected `llm_routes` key matches
   `--llm-model`.  Use `--pool-size 1 --max-workers 1`: the native execution
   policy is concurrency 1 and a DinD data-root must not receive parallel runs.
4. Set a long enough runner timeout. The configured budget is 18,000 seconds
   plus 3,600 seconds for first-task image loading; the recommended Launcher
   timeout is 22,200 seconds.

The start config launches the image with `--privileged`, `--cgroupns=host`,
`--platform linux/amd64`, `--cpus=10`, `--memory=20g`, and named volume
`deepsafe-cyber-dind-data:/var/lib/docker` to retain loaded task layers.  It
also mounts native `runs`, `batch-output`, and `/opt/operator/run.yaml`; the
provider mount from the standalone deepsafe workflow is intentionally replaced
by per-episode generation inside the container.

For a two-case deployment/evaluation check:

```bash
python skills/safactory-workflows/scripts/live_smoke.py \
  --gateway-config gateway/config.local.yaml \
  --run-timeout 45600 \
  -- \
  --mode docker \
  --agent-config env/deepsafe_cyber/deepsafe_cyber_config.smoke.yaml \
  --agent-start-config env/deepsafe_cyber/deepsafe_cyber_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model <route_key> \
  --enable-evaluation \
  --agent-start-timeout-s 22200 \
  --agent-start-timeout-grace-s 600 \
  --pool-size 1 --multiplier 1.0 --max-workers 1 \
  --job-id deepsafe-cyber-docker-smoke
```

After the run, inspect Launcher/Gateway logs, completed rows/rewards, and native
artifacts under:

```text
results/deepsafe_cyber/safactory-smoke/<run_id>/
```

The full run uses the same command with `deepsafe_cyber_config.yaml`; do not
increase concurrency or run multiple jobs against the same
`deepsafe-cyber-dind-data` volume at the same time.

`--pool-size 1` alone is not sufficient with SAfactory's default warm-pool
multiplier of 1.2: it warms two containers for the two-row smoke dataset.
`--multiplier 1.0` is required because both warmed containers would otherwise
attach the same `deepsafe-cyber-dind-data` volume and run two nested dockerd
processes concurrently.

## RJob mode

The RJob integration uses the same adapter, per-episode provider generation,
native run config, evaluator, and result contract as Docker mode.

### Files

- Full run: `deepsafe_cyber_config.rjob.yaml`
- Two-case smoke run: `deepsafe_cyber_config.rjob.smoke.yaml`
- Startup: `deepsafe_cyber_start.rjob.yaml`

RJob submits the registry image directly; it does not reuse the local Docker
image loaded for Docker-mode smoke testing.  The cluster must be able to pull:

```text
registry-v2.h.pjlab.org.cn/ailab-ai4good2/deepsafe-cyber:0.1.0
```

Cluster-accessible mounts are:

```text
gpfs://gpfs2/trustcyberdata
  -> /mnt/shared-storage-user/trustcyberdata

gpfs://gpfs2/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber
  -> /mnt/shared-storage-user/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber

gpfs://gpfs2/adpc-share/chenxinquan/SAfactory/env/deepsafe_cyber/batch-output
  -> /opt/deepsafe-cyber/configs/deepsafe_cyber/batch-output

gpfs://gpfs2/adpc-share/chenxinquan/SAfactory/env/deepsafe_cyber/dind-data
  -> /var/lib/docker
```

The results mount deliberately uses the same source and target path.  This lets
the Launcher read `SAFACTORY_RESULT_PATH` directly from GPFS if RJob's
`logs_rjob` API returns an empty payload after a successful job.

RJob rejects mounting one GPFS source at multiple targets with
`the value of source path repeats`.  Therefore `rjob_entrypoint.sh` exposes the
native runs path as an in-container symlink instead of a second mount:

```text
/opt/deepsafe-cyber/runs
  -> /mnt/shared-storage-user/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber
```

The current SAfactory RJob `mount_config` wrapper does not expose a per-mount
read-only flag.  `rjob_entrypoint.sh` therefore performs a bind/remount of
`/mnt/shared-storage-user/trustcyberdata` as read-only inside the privileged
RJob mount namespace before starting nested Docker.

RJob has no named Docker-volume equivalent.  `/var/lib/docker` is instead a
persistent GPFS directory.  `rjob_entrypoint.sh` takes an `flock` on it before
starting dockerd, preventing accidental concurrent daemons on the same data
root.  Keep launcher and RJob submission concurrency at one, and do not point
another launcher at the same `dind-data` directory.  The first RJob loads the
selected `cve-*.tar` images into this directory; later sequential RJobs reuse
those layers.

RJob resources mirror the Docker requirements:

```text
privileged: true
cpu: 10
memory: 20480 MiB
custom resource: brainpp.cn/fuse=1
```

### Global prerequisites

1. Install the RJob SDK in the launcher environment:

   ```bash
   python -c "from brainpp.rjob import RJobClient; print(RJobClient)"
   ```

2. Fill `access_key` and `secret_key` in the YAML passed with `--rjob-config`
   (normally `config.yaml`).  Keep credentials out of committed files.
3. Ensure the charged group can pull the image, request `brainpp.cn/fuse`, and
   mount the four GPFS paths above.
4. Run Gateway on an address reachable from RJob.  Never use `127.0.0.1` or
   `localhost` as `--gateway-base-url`.

### Two-case RJob smoke command

From the repository root, replace `GATEWAY_HOST` with a host reachable from the
RJob cluster:

```bash
python skills/safactory-workflows/scripts/live_smoke.py \
  --gateway-config gateway/config.local.yaml \
  --run-timeout 45600 \
  -- \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/deepsafe_cyber/deepsafe_cyber_config.rjob.smoke.yaml \
  --agent-start-config env/deepsafe_cyber/deepsafe_cyber_start.rjob.yaml \
  --gateway-base-url http://GATEWAY_HOST:8000/v1/sessions \
  --llm-model kimi-k3 \
  --enable-evaluation \
  --agent-start-timeout-s 22200 \
  --agent-start-timeout-grace-s 600 \
  --pool-size 1 \
  --multiplier 1.0 \
  --max-workers 1 \
  --job-id deepsafe-cyber-rjob-smoke
```

If Gateway is already running, invoke `launcher.py` directly with the same
mode/config/timeout/concurrency arguments.
