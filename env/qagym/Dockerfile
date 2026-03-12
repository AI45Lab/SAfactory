# build to image: registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/qa-gym:1.3

FROM registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base

ENV PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/" \
    PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple" \
    PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn" \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER root
WORKDIR /app

# Copy dependency specs first for better layer caching
COPY requirements.txt /app/
COPY rayjob_sdk-0.3.11-py3-none-any.whl /app/

# Pin Starlette to what FastAPI 0.118.2 requires (starlette<0.49.0,>=0.40.0)
RUN printf "starlette>=0.40.0,<0.49.0\n" > /app/constraints.txt

RUN python -m venv /opt/openrt-venv \
    && /opt/openrt-venv/bin/python -m pip install --no-cache-dir -U pip setuptools wheel \
    && /opt/openrt-venv/bin/python -m pip install --no-cache-dir -r /app/requirements.txt -c /app/constraints.txt \
    && /opt/openrt-venv/bin/python -m pip install --no-cache-dir -c /app/constraints.txt "protobuf>=6,<7" openrt \
    && /opt/openrt-venv/bin/python -m pip check

# Copy the rest of the source code after deps install
COPY . /app

# Make venv default
ENV PATH="/opt/openrt-venv/bin:${PATH}"
# there is no need to set real OPENAI_API_KEY and OPENAI_BASE_URL in the Dockerfile, just to set
ENV OPENAI_API_KEY="XXX"
ENV OPENAI_BASE_URL="XXX"

EXPOSE 36663