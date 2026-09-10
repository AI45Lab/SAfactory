# Unified CyberGym controller image for OpenHands, OpenCode, Codex, and Claude Code.
#
# CyberGym and its example agents are cloned from their upstream repositories
# during the image build. Safactory's small controller runtime is copied from
# this directory and baked into the resulting image.
#
# This image intentionally does NOT contain:
#   - CyberGym vulnerable/fixed target images
#   - the OpenHands runtime/agent image
#   - benchmark data
#   - LLM endpoint, model, or API key configuration
#
# Those assets are supplied at runtime. The container is expected to use the
# host Docker daemon through /var/run/docker.sock and load the three mounted
# image archives with `docker load -i`.
#
# No ENTRYPOINT or default CMD is provided. Pass `bash` (or another command)
# explicitly to `docker run`.

FROM python:3.12-slim-bookworm

ARG TARGETARCH
ARG NODE_VERSION=22.23.1
ARG POETRY_VERSION=1.8.5
ARG CYBERGYM_REPOSITORY=https://github.com/sunblaze-ucb/cybergym.git
ARG CYBERGYM_REF=main
ARG BUILD_OPENHANDS=true

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    PATH=/opt/poetry/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    CYBERGYM_ROOT=/workspace/cybergym

# docker.io provides both the CLI and daemon. Docker mode reaches the host
# daemon through /var/run/docker.sock; RJob mode starts a privileged DinD
# daemon backed by fuse-overlayfs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        docker.io \
        fuse-overlayfs \
        fuse3 \
        git \
        jq \
        libffi-dev \
        libfuse3-3 \
        libssl-dev \
        make \
        openssh-client \
        pkg-config \
        procps \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Safactory starts episode commands with `sh -lc`. Install POSIX-sh-compatible,
# LF-only profile snippets so dash can source them before the runner starts.
RUN printf '%s\n' \
      '#!/usr/bin/env sh' \
      '# Disable core dumps without making shell startup fail on restricted hosts.' \
      'ulimit -S -c 0 >/dev/null 2>&1 || true' \
      > /etc/profile.d/brain-ulimit.sh \
    && printf '%s\n' \
      '#!/usr/bin/env sh' \
      '# Docker already passes the controller environment to login shells.' \
      '# The upstream Bash-only SSH initialization is not needed by Safactory.' \
      ':' \
      > /etc/profile.d/ssh-init.sh \
    && chmod 0644 \
      /etc/profile.d/brain-ulimit.sh \
      /etc/profile.d/ssh-init.sh

# Install an architecture-matched Node.js 22 distribution and verify its
# checksum against the upstream SHASUMS file.
RUN set -eux; \
    detected_arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${detected_arch}" in \
        amd64) node_arch="x64" ;; \
        arm64) node_arch="arm64" ;; \
        *) echo "Unsupported architecture: ${detected_arch}" >&2; exit 1 ;; \
    esac; \
    archive="node-v${NODE_VERSION}-linux-${node_arch}.tar.xz"; \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/${archive}"; \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt"; \
    grep " ${archive}$" SHASUMS256.txt | sha256sum -c -; \
    tar -xJf "${archive}" -C /usr/local --strip-components=1; \
    rm -f "${archive}" SHASUMS256.txt; \
    node --version; \
    npm --version

# Install Poetry outside Conda and outside the project tree.
RUN curl -fsSL https://install.python-poetry.org \
    | POETRY_VERSION="${POETRY_VERSION}" python3.12 - \
    && poetry --version

# Clone the CyberGym source instead of depending on the local build context.
# CYBERGYM_REF accepts a branch, tag, or commit SHA; pin it for reproducible
# builds instead of using the default moving main branch.
RUN mkdir -p /workspace \
    && git clone "${CYBERGYM_REPOSITORY}" "${CYBERGYM_ROOT}" \
    && git -C "${CYBERGYM_ROOT}" checkout "${CYBERGYM_REF}"

WORKDIR /workspace/cybergym

# Agent adapters are copied from this checkout below. Only a combined
# OpenHands image needs the large recursively nested example-agent submodule.
RUN if [[ "${BUILD_OPENHANDS}" == "true" ]]; then \
        git submodule update --init --recursive examples/agents; \
    else \
        mkdir -p examples/agents; \
    fi

# Use the adapters from this Safactory checkout. The CyberGym revision can
# point at an older examples/agents commit that predates these adapters.
COPY cybergym-agent-examples/codex/run.py /workspace/cybergym/examples/agents/codex/run.py
COPY cybergym-agent-examples/opencode/run.py /workspace/cybergym/examples/agents/opencode/run.py
COPY cybergym-agent-examples/claude_code/run.py /workspace/cybergym/examples/agents/claude_code/run.py

# Codex is the default controller target and does not require the large nested
# OpenHands repository. Enable BUILD_OPENHANDS only for a combined image.
RUN test -f examples/agents/codex/run.py \
    && test -f examples/agents/opencode/run.py \
    && test -f examples/agents/claude_code/run.py \
    || { \
        echo "Codex/OpenCode/Claude Code CyberGym adapters are missing." >&2; \
        exit 1; \
    }; \
    if [[ "${BUILD_OPENHANDS}" == "true" ]]; then \
        test -f examples/agents/openhands/run.py \
        && test -f examples/agents/openhands/openhands-repo/Makefile \
        || { \
            echo "BUILD_OPENHANDS=true but the nested OpenHands sources are missing." >&2; \
            exit 1; \
        }; \
    fi

RUN python3.12 -m pip install --upgrade pip setuptools wheel \
    && python3.12 -m pip install -e '.[dev,server]'

# Build OpenHands only when explicitly requested. Codex and OpenCode execute
# in their own agent images and need no OpenHands Poetry environment.
RUN if [[ "${BUILD_OPENHANDS}" == "true" ]]; then \
        make -C examples/agents/openhands/openhands-repo \
          build INSTALL_PLAYWRIGHT=false; \
    fi

# Smoke-check the packaged control-plane dependencies. This does not require
# access to the host Docker daemon during image build.
RUN python3.12 -c "import cybergym, docker, fastapi, httpx, sqlalchemy, tomli_w; print('CyberGym dependencies ready')" \
    && sh -n /etc/profile.d/brain-ulimit.sh \
    && sh -n /etc/profile.d/ssh-init.sh \
    && sh -lc 'test "$(ulimit -c)" = "0"' \
    && poetry --version \
    && node --version \
    && npm --version \
    && docker --version

# Bake the single episode entrypoint and its preparation/result helpers into
# the controller image. Docker and RJob modes execute the same runner.sh.
RUN mkdir -p /opt/safactory/cybergym
COPY env/cybergym/runner.sh /opt/safactory/cybergym/runner.sh
COPY cybergym/common.sh /opt/safactory/cybergym/common.sh
COPY cybergym/docker_prepare.sh /opt/safactory/cybergym/docker_prepare.sh
COPY cybergym/rjob_prepare.sh /opt/safactory/cybergym/rjob_prepare.sh
COPY cybergym/episode_prepare.py /opt/safactory/cybergym/episode_prepare.py
COPY cybergym/openhands_prepare.py /opt/safactory/cybergym/openhands_prepare.py
COPY cybergym/agent_dispatch.py /opt/safactory/cybergym/agent_dispatch.py
COPY cybergym/result_writer.py /opt/safactory/cybergym/result_writer.py
RUN chmod 0755 \
      /opt/safactory/cybergym/runner.sh \
      /opt/safactory/cybergym/common.sh \
      /opt/safactory/cybergym/docker_prepare.sh \
      /opt/safactory/cybergym/rjob_prepare.sh \
      /opt/safactory/cybergym/episode_prepare.py \
      /opt/safactory/cybergym/openhands_prepare.py \
      /opt/safactory/cybergym/agent_dispatch.py \
      /opt/safactory/cybergym/result_writer.py \
    && bash -n /opt/safactory/cybergym/runner.sh \
    && bash -n /opt/safactory/cybergym/common.sh \
    && bash -n /opt/safactory/cybergym/docker_prepare.sh \
    && bash -n /opt/safactory/cybergym/rjob_prepare.sh \
    && python3.12 -m py_compile \
      /opt/safactory/cybergym/episode_prepare.py \
      /opt/safactory/cybergym/openhands_prepare.py \
      /opt/safactory/cybergym/agent_dispatch.py \
      /opt/safactory/cybergym/result_writer.py

# Runtime files belong in bind-mounted directories under the CyberGym root. For
# correct sibling container bind mounts, mount them at the SAME absolute paths
# used by the controller and its configuration.
WORKDIR /workspace/cybergym

# Explicitly clear inherited startup behavior. Safactory invokes the baked
# /opt/safactory/cybergym/runner.sh through its start configuration.
ENTRYPOINT []
CMD []
