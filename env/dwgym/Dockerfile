FROM registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base

ENV PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/" \
    PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple" \
    PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"

WORKDIR /app

COPY . /app

USER root

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
    tortoise-orm==0.25.1

RUN pip install fastapi uvicorn

EXPOSE 36663