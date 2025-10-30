FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.10-slim

WORKDIR /workspace

# 设置 pip 内网源
ENV PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/pypi/simple/" \
    PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple" \
    PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"

# 安装 Python 包
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    gymnasium \
    pyyaml \
    matplotlib \
    openai \
    tortoise-orm \
    seaborn

# 复制工程代码
COPY . /workspace/ai_sandbox/

# 设置 PYTHONPATH
ENV PYTHONPATH=/workspace/ai_sandbox

RUN cd /workspace/ai_sandbox

# 默认执行脚本
CMD ["python","examples/test_env.py"]