FROM registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base

USER root

ENV PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/" \
    PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple" \
    PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"

WORKDIR /app

COPY . /app

RUN mkdir -p /app/video

RUN chmod -R 777 /app

RUN pip install --no-cache-dir \
    gymnasium \
    ipython \
    matplotlib\
    numpy \
    openai \
    pandas==2.3.2 \
    pillow==11.3.0 \
    PyYAML==6.0.2 \
    requests==2.32.5 \
    seaborn==0.13.2 \
    tortoise-orm==0.25.1 \
    tiktoken==0.12.0

RUN pip install fastapi uvicorn pygame pathfinding psutil termcolor

EXPOSE 36663

