#!/usr/bin/env bash
# =============================================================================
# push_patcheval_images.sh
# =============================================================================
# Load each PatchEval CVE image tar from the local archive, retag it for the
# pjlab internal registry, push it, then delete the local image so the docker
# storage never holds more than a few images at once (the full set is ~503GB).
#
# RJob pods pull from the registry (they cannot read the local tar archive), so
# this is a one-time prerequisite for --mode rjob PatchEval runs.
#
# Resumable: a done-list file records every CVE successfully pushed; re-running
# skips them. Parallel: N workers load/tag/push concurrently.
#
# Env vars (all optional):
#   PATCH_EVAL_IMAGE_ARCHIVE_DIR  source tar dir
#                               (default: /mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images)
#   PATCH_EVAL_REGISTRY           registry host (default: registry.h.pjlab.org.cn)
#   PATCH_EVAL_REGISTRY_NS        registry namespace (default: ailab-evobox-evobox_proxy)
#   PATCH_EVAL_REPO               repository name (default: patcheval)
#   DOCKER_HOST                   docker daemon (inherited; e.g. tcp://host:2376)
#   REGISTRY_USER / REGISTRY_PASS if set, `docker login` is run first
#   PARALLEL                      concurrent workers (default: 2)
#   DRY_RUN                       1 = print what would happen, do not load/tag/push
#   FORCE                         1 = ignore done-list, push everything
#   KEEP_LOCAL                    1 = do not delete loaded images after push
#   DONE_FILE                     done-list path (default: ./push_patcheval_done.txt)
#   LOG_DIR                       per-CVE log dir (default: ./logs-push)
# =============================================================================
set -euo pipefail

ARCHIVE_DIR="${PATCH_EVAL_IMAGE_ARCHIVE_DIR:-/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images}"
REGISTRY="${PATCH_EVAL_REGISTRY:-registry.h.pjlab.org.cn}"
REGISTRY_NS="${PATCH_EVAL_REGISTRY_NS:-ailab-evobox-evobox_proxy}"
REPO="${PATCH_EVAL_REPO:-patcheval}"
PARALLEL="${PARALLEL:-2}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
KEEP_LOCAL="${KEEP_LOCAL:-0}"
DONE_FILE="${DONE_FILE:-./push_patcheval_done.txt}"
LOG_DIR="${LOG_DIR:-./logs-push}"

mkdir -p "${LOG_DIR}"
touch "${DONE_FILE}"

# NOTE: do NOT use a bash array for the docker command — arrays cannot be
# exported, so xargs-spawned bash subshells would see an empty DOCKER and run
# `load -i ...` as a bare command ("load: command not found"). Call `docker`
# directly; it reads DOCKER_HOST from the exported environment.
if [[ -n "${DOCKER_HOST:-}" ]]; then
  export DOCKER_HOST
fi

# --- registry login (optional) ---
if [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_PASS:-}" ]]; then
  echo "Logging into ${REGISTRY} as ${REGISTRY_USER} ..."
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] would: docker login ${REGISTRY} -u <redacted>"
  else
    printf '%s\n' "${REGISTRY_PASS}" | docker login "${REGISTRY}" -u "${REGISTRY_USER}" --password-stdin
  fi
fi

target_tag() {  # <tar-basename> -> e.g. cve-2015-1326-latest
  local b="$1"
  echo "${b%.tar}"
}

target_ref() {  # <tar-basename>
  printf '%s/%s/%s:%s\n' "${REGISTRY}" "${REGISTRY_NS}" "${REPO}" "$(target_tag "$1")"
}

push_one() {  # <tar-path>
  local tar_path="$1"
  local base; base="$(basename "${tar_path}")"
  local dst; dst="$(target_ref "${base}")"
  local log="${LOG_DIR}/${base%.tar}.log"

  # resume skip
  if [[ "${FORCE}" != "1" ]] && grep -Fxq -- "${base}" "${DONE_FILE}" 2>/dev/null; then
    echo "[skip] ${base} (already in done-list)"
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${base} -> load + tag -> ${dst} + push"
    return 0
  fi

  local loaded
  # `docker load` prints "Loaded image: <ref>" (or "Loaded image ID: <id>").
  # Capture stdout; stderr is forwarded to the per-CVE log too.
  if ! loaded="$(docker load -i "${tar_path}" 2>"${log}")"; then
    echo "[FAIL] ${base}: docker load failed (see ${log})"
    return 1
  fi
  local src_ref
  src_ref="$(printf '%s\n' "${loaded}" | sed -n 's/^Loaded image: //p' | head -1)"
  if [[ -z "${src_ref}" ]]; then
    echo "[FAIL] ${base}: could not parse loaded image ref from: ${loaded}"
    return 1
  fi
  echo "[load ] ${base}: ${src_ref}"

  if ! docker tag "${src_ref}" "${dst}" >>"${log}" 2>&1; then
    echo "[FAIL] ${base}: docker tag failed (see ${log})"
    docker rmi "${src_ref}" >/dev/null 2>&1 || true
    return 1
  fi

  if docker push "${dst}" >>"${log}" 2>&1; then
    echo "[push ] ${base}: ${dst}"
    printf '%s\n' "${base}" >>"${DONE_FILE}"
    if [[ "${KEEP_LOCAL}" != "1" ]]; then
      docker rmi "${dst}" "${src_ref}" >/dev/null 2>&1 || true
    fi
    return 0
  else
    echo "[FAIL] ${base}: docker push failed (see ${log})"
    if [[ "${KEEP_LOCAL}" != "1" ]]; then
      docker rmi "${dst}" "${src_ref}" >/dev/null 2>&1 || true
    fi
    return 1
  fi
}
export -f push_one target_tag target_ref
export ARCHIVE_DIR REGISTRY REGISTRY_NS REPO DRY_RUN FORCE KEEP_LOCAL DONE_FILE LOG_DIR DOCKER_HOST

echo "=== push_patcheval_images ==="
echo "  archive : ${ARCHIVE_DIR}"
echo "  registry: ${REGISTRY}/${REGISTRY_NS}/${REPO}"
echo "  docker  : ${DOCKER_HOST:-local socket}"
echo "  parallel: ${PARALLEL}  dry_run: ${DRY_RUN}  force: ${FORCE}  keep_local: ${KEEP_LOCAL}"
echo "  done    : ${DONE_FILE}  logs: ${LOG_DIR}/"
echo

if [[ ! -d "${ARCHIVE_DIR}" ]]; then
  echo "ERROR: archive dir not found: ${ARCHIVE_DIR}" >&2
  exit 1
fi

# Collect tar list (sorted, deterministic).
mapfile -t TARS < <(find "${ARCHIVE_DIR}" -maxdepth 1 -type f -name 'cve-*-latest.tar' | sort)
total=${#TARS[@]}
echo "Found ${total} image tar(s)."

# Already-done count for progress.
done_count=0
if [[ "${FORCE}" != "1" && -s "${DONE_FILE}" ]]; then
  done_count="$(wc -l < "${DONE_FILE}" | tr -d ' ')"
fi
echo "Already pushed: ${done_count}; remaining: $((total - done_count))."
echo

# Run workers. xargs -P gives a bounded parallel pool. xargs' exit code is not
# a failure count (it returns 123 if any item exited 1-125), so we track
# failures explicitly via a fail-list file written by push_one's caller.
FAIL_FILE="${LOG_DIR}/_failed.txt"
rm -f "${FAIL_FILE}"
fail=0
if [[ "${PARALLEL}" -le 1 ]]; then
  for tar in "${TARS[@]}"; do
    push_one "${tar}" || { printf '%s\n' "$(basename "${tar}")" >>"${FAIL_FILE}"; fail=$((fail + 1)); }
  done
else
  # Each xargs invocation runs push_one; on its non-zero exit, record the tar.
  printf '%s\n' "${TARS[@]}" | xargs -P "${PARALLEL}" -I {} \
    bash -c 'push_one "$@" || echo "$(basename "$1")" >>"'"${FAIL_FILE}"'"' _ {} \
    || true
  [[ -f "${FAIL_FILE}" ]] && fail="$(wc -l < "${FAIL_FILE}" | tr -d ' ')"
fi

echo
echo "=== summary ==="
echo "total tars : ${total}"
echo "done-list  : $(wc -l < "${DONE_FILE}" | tr -d ' ')"
if [[ "${fail:-0}" -ne 0 ]]; then
  echo "failures   : ${fail} (see ${FAIL_FILE}; re-run to retry, done-list is only appended on success)"
  exit 1
fi
echo "all done."
