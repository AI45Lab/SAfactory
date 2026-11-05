# Build from a PREPARED delivery staging directory, not the SAfactory checkout.
# Required context:
#   exploit_gym/                  framework source including run_mining_400.sh
#   exploitgym-offline/            fully extracted Linux x86_64 offline bundle
#   claude-code-2.1.228-linux-amd64/  delivery binary + manifest
# Copy this Dockerfile and .dockerignore into that staging directory first.
# Task partitions and hundreds of GB of archives stay on shared storage.
#
# QUICKENV_BASE_IMAGE is deliberately required: use an approved Linux amd64 image with
# Python >=3.10, bash, Docker Engine/CLI, Compose v2, flock and fuse-overlayfs.
# This recipe does not install dependencies or imply that an image was built.
ARG QUICKENV_BASE_IMAGE
FROM ${QUICKENV_BASE_IMAGE}

USER root
ENV PYTHONDONTWRITEBYTECODE=1 \
    EXPGYM_OFFLINE_ROOT=/opt/exploitgym-offline

RUN test "$(uname -m)" = x86_64 \
    && python3 -c 'import sys; assert sys.version_info >= (3, 10)' \
    && command -v bash \
    && command -v docker \
    && command -v dockerd \
    && command -v flock \
    && command -v fuse-overlayfs \
    && docker compose version

# These are private to each RJob image layer, not a writable shared mount.
# relocate.py and uv may update paths/cache here during initialization.
COPY exploitgym-offline/ /opt/exploitgym-offline/
COPY exploit_gym/ /opt/quickenv-delivery/exploit_gym/
COPY claude-code-2.1.228-linux-amd64/ /opt/quickenv-delivery/claude-code-2.1.228-linux-amd64/

RUN test -f /opt/exploitgym-offline/activate.sh \
    && test -f /opt/exploitgym-offline/relocate.py \
    && test -f /opt/exploitgym-offline/firewall-image.tar \
    && test -f /opt/quickenv-delivery/exploit_gym/env_cn.sh \
    && test -f /opt/quickenv-delivery/exploit_gym/run_mining_400.sh \
    && test -f /opt/quickenv-delivery/exploit_gym/scripts/run_pj_exploit.sh \
    && test -f /opt/quickenv-delivery/exploit_gym/examples/run_agent.py \
    && test -f /opt/quickenv-delivery/exploit_gym/scripts/setup/pre_run.py \
    && test -x /opt/quickenv-delivery/claude-code-2.1.228-linux-amd64/claude

WORKDIR /opt/quickenv-delivery/exploit_gym
# SAfactory embeds runner.py/adapter.py/runtime.py and executes the configured
# runner command.  No full-batch ENTRYPOINT is installed.
ENTRYPOINT []
CMD ["sleep", "infinity"]
