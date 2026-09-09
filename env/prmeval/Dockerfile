FROM registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323

COPY prmeval-0.1.0-py3-none-any.whl /tmp/prmeval-0.1.0-py3-none-any.whl

RUN python -m pip install --no-cache-dir --no-input /tmp/prmeval-0.1.0-py3-none-any.whl \
    && rm /tmp/prmeval-0.1.0-py3-none-any.whl
