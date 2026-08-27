# cyberrange evaluation adapter

This is a thin adapter around cyberrange's existing native `runtime-task`
workflow. No custom cyberrange image is built.

Docker and RJob modes are supported. Both execute one dataset row as one
native cyberrange case and use the same runner and rule evaluator.

## Runtime shape

For each SAfactory dataset row:

1. SAfactory starts the digest-pinned Ubuntu 22.04 base image with
   `--privileged=true`.
2. The cyberrange repository, sealed Python wheelhouse, release, and qcow2
   image store are mounted read-only.
3. `runner.py` invokes the repository's
   `scripts/brainpp_source_bootstrap_acceptance.sh` in `runtime-task` mode.
4. cyberrange installs its reviewed runtime dependencies from the sealed
   wheelhouse/internal apt mirror, starts its production control plane, runs
   exactly the selected Range3-6 case, seals evidence, cleans up KVM resources,
   and writes `runtime-test-result.json`.
5. The runner converts that native file to one SAfactory
   `SimulationStartResult`; `rule_evaluator.py` maps the native score.

The dataset contains one scenario per row. The runner never loops over the
suite.

## External inputs

The checked-in start/config files use these existing host paths:

- source: `/mnt/shared-storage-user/evoagi-share/yxwang/cyberrange`
- qcow2 store: `/mnt/shared-storage-user/evoagi-share/yxwang/cyberrange-images`
- sealed wheelhouse: `acceptance-py310-x86_64-20260806-141858`
- release: `postexploitbench-range3-6-egress-20260806-120928`

The source checkout is mounted at cyberrange's validated historical container
path `/mnt/shared-storage-user/wangyixu/cyberrange`. The qcow2 store is mounted
at `/mnt/shared-storage-user/evoagi-share/cyberrange-images`, matching the
absolute `base-images` symlink stored in the release.

The release includes private deployment material and remains read-only. No
model key or private model endpoint is copied into SAfactory. For each case the
runner points cyberrange at the current SAfactory Gateway session and creates a
random disposable credential file because the native TestSpec schema requires
a write-only key.

## Requirements

The Docker host must provide `/dev/kvm`, `/dev/net/tun`, Open vSwitch kernel
support and nftables. The Ubuntu base image must be available locally or from
the configured registry:

```text
registry.h.pjlab.org.cn/ailab/ubuntu@sha256:9d6aa98a868e950d8133c255adfeaa5c899e8c49513bbe29aa541d580d52fc8c
```

No `CYBERRANGE_BASE_IMAGE` build is required.

## Smoke test

Start the SAfactory Gateway with the same SQLite URI used below, then run:

```bash
python launcher.py \
  --mode docker \
  --agent-config env/cyberrange/cyberrange_config.smoke.yaml \
  --agent-start-config env/cyberrange/cyberrange_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_RESPONSES_ROUTE \
  --enable-evaluation \
  --db-path sqlite://data.db \
  --job-id cyberrange-docker-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 3000 \
  --agent-start-timeout-s 7200 \
  --agent-start-timeout-grace-s 1800 \
  --docker-start-timeout-s 300
```

The smoke config runs one Range3 case with a 600-second Agent budget. The full
config has four independent rows for Range3 through Range6 and a five-hour
budget per case.

Native `e2e_success=true` maps to 1.0. When it is false, SAfactory scores the
completed milestone fraction; `e2e_success=null` remains an evaluator failure
rather than a model failure. The milestone vector and native result path are
retained as artifacts.

## RJob smoke test

RJob uses cyberrange's validated digest-pinned Ubuntu base directly. SAfactory
embeds the runner, while source, wheelhouse, release, qcow2 images and results
come from the same GPFS mounts as cyberrange's native RJob submitter. No image
build is needed.

The Python runner applies the production deployment contract from cyberrange
`docs/DEPLOYMENT.md`: it fails closed unless the runtime is root, privileged,
has usable KVM/TUN devices and the fixed canonical source layout; then it
invokes the sealed source-bootstrap deployment, runs exactly one runtime
TestSpec, validates `runtime-test-result.json`, and changes that native scoring
source to read-only mode. It then converts the result into a
`SimulationStartResult`, atomically writes the mounted SAfactory artifact, and
also makes that artifact read-only before returning it on stdout.

For an HTTP SAfactory Gateway, cyberrange's native runtime-task detects the
`http://` model URL and adds the explicit `agent-range test submit
--allow-insecure-http` CLI opt-in. SAfactory no longer generates a temporary
TestSpec or overrides the source checkout. HTTPS URLs retain the secure
default and do not receive the flag.

The RJob start config maps
`gpfs://gpfs1/evoagi-share/yxwang/cyberrange` to cyberrange's required
historical path `/mnt/shared-storage-user/wangyixu/cyberrange`. It separately
maps the `yxwang` shared root so release `base-images` symlinks resolve.
`gpfs://gpfs1/evobox-share/chenxinquan/SAfactory/results` is mounted at
`/app/results`; the runner atomically writes the same `SimulationStartResult`
there that it emits on stdout. It also copies the sealed native
`runtime-test-result.json` and the final `milestones.json` into the same
per-session directory. This gives SAfactory an artifact fallback if RJob log
retrieval is empty or delayed and exposes the native scoring evidence without
requiring access to cyberrange's report tree.

Keep RJob credentials and the cluster-visible Gateway address in the private
file passed through `--rjob-config`; do not add them to environment files.

```bash
python launcher.py \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/cyberrange/cyberrange_config.smoke.rjob.yaml \
  --agent-start-config env/cyberrange/cyberrange_start.rjob.yaml \
  --gateway-base-url http://YOUR_RJOB_VISIBLE_GATEWAY/v1/sessions \
  --llm-model YOUR_RESPONSES_ROUTE \
  --enable-evaluation \
  --db-path sqlite://data.db \
  --job-id cyberrange-rjob-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 3000 \
  --agent-start-timeout-s 7200 \
  --agent-start-timeout-grace-s 1800
```

The worker must provide `/dev/kvm`, `/dev/net/tun`, Open vSwitch kernel support
and nftables. The RJob settings mirror the native source-bootstrap defaults:
`evoagi_cpu_task`, 14 CPU, 49152 MiB memory, privileged mode, and host
networking disabled. Failed jobs are retained for diagnosis.
