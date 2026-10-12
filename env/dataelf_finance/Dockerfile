# Runtime image for the dataelf_finance environment.
#
# Build context is the DataElf checkout itself (node_modules and the managed
# pi package store .pi/npm are copied from the host, so no npm install is
# needed; Python dependencies are installed from the configured pip mirror):
#
#   docker build -f env/dataelf_finance/Dockerfile \
#     -t dataelf_finance-image:v0.4 \
#     /mnt/shared-storage-user/evobox-share/chenbei/DataElf
#
# The host venv is not copied (virtualenvs are not relocatable) and
# dataelf.local.yaml is dropped so no provider URLs or credentials ship with
# the image — every episode receives its config via DATAELF_CONFIG_FILE from
# the SAfactory adapter. The 1.1 GB SQLite database is NOT baked in: it is
# bind-mounted read-only at /workspace/DDR_Bench/data/10k/raw and linked
# into each job workspace.

FROM docker.io/library/node:24-slim AS node

FROM python:3.13-slim

# Node 24 runtime for the pi explorer (pi-coding-agent/pi-fusion are Node).
COPY --from=node /usr/local/bin/node /usr/local/bin/node

# DataElf checkout with node_modules and the managed pi package store
# (.pi/npm, installed on the host via `dataelf setup`).
COPY . /opt/DataElf
RUN rm -rf /opt/DataElf/.venv /opt/DataElf/dataelf.local.yaml \
    && python -m venv /opt/DataElf/.venv \
    && /opt/DataElf/.venv/bin/python -m pip install --no-cache-dir \
        -i http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ \
        --trusted-host mirrors.i.h.pjlab.org.cn \
        -e "/opt/DataElf[finance]" \
        # The mounted reward calculator needs httpx and PyYAML. DataElf's
        # core dependencies provide PyYAML; install httpx in the same venv.
        "httpx>=0.27,<1" \
    # Idempotent with node_modules/.pi/npm already present: refreshes the
    # runtime manifest with in-image paths without touching the network.
    && cd /opt/DataElf \
    && PATH="/opt/DataElf/.venv/bin:/usr/local/bin:$PATH" dataelf setup

ENV VIRTUAL_ENV="/opt/DataElf/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:/usr/local/bin:${PATH}"

# Login shells (`sh -lc`, used by the SAfactory docker/rjob runners to launch
# the episode runner) source /etc/profile, which RESETS PATH and drops the
# venv — bare `python` then resolves to the system interpreter without
# httpx/PyYAML, breaking the in-container tail scoring. Re-prepend the venv
# for login shells and their children. Validate the actual launcher shell
# during the build so missing scoring dependencies fail before a rollout.
RUN printf '%s\n' \
        'export VIRTUAL_ENV=/opt/DataElf/.venv' \
        'export PATH="$VIRTUAL_ENV/bin:$PATH"' \
        > /etc/profile.d/zz-dataelf-venv.sh \
    && sh -lc 'test "$(command -v python)" = "$VIRTUAL_ENV/bin/python" \
        && test "$(command -v python3)" = "$VIRTUAL_ENV/bin/python3" \
        && python -c "import httpx, yaml, os, sys; assert sys.prefix == os.environ[\"VIRTUAL_ENV\"], (sys.executable, sys.prefix)"'
