ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS agentcompass-source

ARG AGENTCOMPASS_REPOSITORY=https://github.com/open-compass/AgentCompass.git
ARG AGENTCOMPASS_COMMIT=d2c3e148902e948db3270fa34b2198fb1b10beb7

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN git init /src/AgentCompass \
    && git -C /src/AgentCompass remote add origin "${AGENTCOMPASS_REPOSITORY}" \
    && git -C /src/AgentCompass fetch --depth 1 origin "${AGENTCOMPASS_COMMIT}" \
    && git -C /src/AgentCompass checkout --detach FETCH_HEAD \
    && test "$(git -C /src/AgentCompass rev-parse HEAD)" = "${AGENTCOMPASS_COMMIT}"

FROM ${PYTHON_IMAGE} AS python-dependencies

ENV CMAKE_BUILD_PARALLEL_LEVEL=1 \
    CARGO_BUILD_JOBS=1 \
    MAKEFLAGS=-j1 \
    MAX_JOBS=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

COPY --from=agentcompass-source /src/AgentCompass /opt/AgentCompass

# AgentCompass itself is pure Python. Require wheels for every resolved
# third-party dependency so a missing wheel fails explicitly instead of
# triggering an unbounded compiler workload.
RUN python -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel \
    && "${VIRTUAL_ENV}/bin/python" -m pip install --only-binary=:all: \
        "/opt/AgentCompass[swebench,mini-swe-agent]" \
    && "${VIRTUAL_ENV}/bin/python" -c "import agentcompass, minisweagent, swebench"

ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST

RUN "${VIRTUAL_ENV}/bin/python" -m pip install --only-binary=:all: \
        "/opt/AgentCompass[scicode,gdpval]" \
    && "${VIRTUAL_ENV}/bin/python" -c \
        "import h5py, openpyxl, scipy, sympy"

FROM ${PYTHON_IMAGE} AS runtime

ARG AGENTCOMPASS_COMMIT=d2c3e148902e948db3270fa34b2198fb1b10beb7

LABEL org.opencontainers.image.source="https://github.com/open-compass/AgentCompass" \
      org.opencontainers.image.revision="${AGENTCOMPASS_COMMIT}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv \
    AGENTCOMPASS_COMMIT=${AGENTCOMPASS_COMMIT}

RUN sed -i 's|http://deb.debian.org/debian|https://deb.debian.org/debian|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get \
        -o Acquire::Retries=3 \
        -o Acquire::http::Timeout=60 \
        -o Acquire::https::Timeout=60 \
        update \
    && apt-get \
        -o Acquire::Retries=3 \
        -o Acquire::http::Timeout=60 \
        -o Acquire::https::Timeout=60 \
        install -y --no-install-recommends \
        ca-certificates \
        git \
        patch \
    && rm -rf /var/lib/apt/lists/*

COPY --from=agentcompass-source /src/AgentCompass /opt/AgentCompass
COPY --from=python-dependencies /opt/venv /opt/venv

RUN test "$(git -C /opt/AgentCompass rev-parse HEAD)" = "${AGENTCOMPASS_COMMIT}" \
    && python -c \
        "import agentcompass, h5py, minisweagent, openpyxl, scipy, swebench, sympy"

WORKDIR /opt/AgentCompass

RUN mkdir -p /app/results /app/data

CMD ["agentcompass", "--help"]
