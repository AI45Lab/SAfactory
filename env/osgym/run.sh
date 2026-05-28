#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

REGISTRY_URL="${REGISTRY_URL:-registry.h.pjlab.org.cn}"
DOCKER_USER="${DOCKER_USER:-8cc07f27716d056e625b0f6522a93823}"
DOCKER_PASS="${DOCKER_PASS:-d1e27df7d809b0819e838baf8d3f6425}"
OSWORLD_IMAGE="${OSWORLD_IMAGE:-registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/osworld:v1.0}"
DOCKER_STORAGE_DRIVER="${DOCKER_STORAGE_DRIVER:-vfs}"

wait_for_dockerd() {
    local max_attempts=30
    for _ in $(seq 1 "${max_attempts}"); do
        if docker info > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

if ! docker info > /dev/null 2>&1; then
    mkdir -p /var/lib/docker
    dockerd --storage-driver="${DOCKER_STORAGE_DRIVER}" > /tmp/dockerd.log 2>&1 &
    if ! wait_for_dockerd; then
        echo ">>> Docker daemon failed to start; see /tmp/dockerd.log" >&2
        exit 1
    fi
fi

echo ">>> 正在登录 Docker 仓库..."
echo "$DOCKER_PASS" | docker login "$REGISTRY_URL" --username "$DOCKER_USER" --password-stdin

if docker image inspect "${OSWORLD_IMAGE}" > /dev/null 2>&1; then
    echo ">>> OSWorld image already exists: ${OSWORLD_IMAGE}"
else
    echo ">>> Pulling OSWorld image: ${OSWORLD_IMAGE}"
    docker pull "${OSWORLD_IMAGE}"
fi

echo ">>> Docker is ready"

exec python env/app.py
