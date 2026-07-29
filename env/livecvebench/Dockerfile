FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:/root/.cargo/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

ARG UBUNTU_MIRROR

RUN if [ -n "${UBUNTU_MIRROR}" ]; then \
        sed -i -E "s#http://(archive|security).ubuntu.com/ubuntu/?#${UBUNTU_MIRROR%/}/#g" /etc/apt/sources.list; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl docker.io docker-compose-v2 \
        git iptables python3 python3-venv tmux \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh \
    && bash /tmp/nodesource_setup.sh \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install --global \
        @anthropic-ai/claude-code@latest \
        opencode-ai@latest \
    && node --version \
    && claude --version \
    && opencode --version \
    && npm cache clean --force \
    && rm -f /tmp/nodesource_setup.sh \
    && rm -rf /var/lib/apt/lists/*

COPY docker/uv /usr/local/bin/uv

ARG UV_INDEX_URL

RUN chmod +x /usr/local/bin/uv \
    && UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=20 uv tool install terminal-bench==0.2.18

WORKDIR /benchmark

COPY . /benchmark

RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && chmod +x /benchmark/docker/single-task-entrypoint.sh \
    && mkdir -p /benchmark/runs /var/lib/docker

VOLUME ["/benchmark/runs", "/var/lib/docker"]

ENTRYPOINT ["/benchmark/docker/single-task-entrypoint.sh"]

CMD ["tail", "-f", "/dev/null"]
