#!/bin/sh
# RJob starts the SAfactory runner as the container command.  deepsafe-cyber's
# image ENTRYPOINT starts a nested dockerd, but the RJob Container command
# replaces that startup path.  Launch the native entrypoint explicitly, keep its
# logs out of the runner's stdout contract, then run exactly one episode.
set -eu

state_root=/var/lib/docker
runtime_root=/tmp/safactory-deepsafe_cyber
trust_root=/mnt/shared-storage-user/trustcyberdata
results_root=/mnt/shared-storage-user/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber
mkdir -p "$state_root" "$runtime_root"

# RJob rejects mounting one GPFS source at two targets (`source path repeats`).
# Mount it only at its launcher-visible path and expose deepsafe's required
# native runs path as a symlink in the container filesystem.
mkdir -p "$results_root" /opt/deepsafe-cyber
if [ ! -L /opt/deepsafe-cyber/runs ]; then
  rm -rf /opt/deepsafe-cyber/runs
  ln -s "$results_root" /opt/deepsafe-cyber/runs
fi

# RJob's mount_config syntax does not expose a per-mount read-only flag in the
# current SAfactory wrapper.  Enforce the benchmark requirement in this
# container's mount namespace before starting nested Docker.
mount --bind "$trust_root" "$trust_root"
mount -o remount,bind,ro "$trust_root"

# Multiple RJobs can reference the same persistent GPFS data root.  SAfactory
# is configured for one worker, but this lock also protects the data root if a
# second launcher is started accidentally.
exec 9>"$state_root/safactory-dind.lock"
flock 9

/opt/deepsafe-cyber/packaging/linux-amd64/entrypoint.sh \
  >"$state_root/safactory-dockerd.log" 2>&1 &
daemon_pid=$!

cleanup() {
  kill -TERM "$daemon_pid" 2>/dev/null || true
  wait "$daemon_pid" 2>/dev/null || true
}
trap cleanup TERM INT EXIT

python "$runtime_root/runner.py"
runner_status=$?

trap - TERM INT EXIT
cleanup
exit "$runner_status"
