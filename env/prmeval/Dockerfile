FROM python:3.12

COPY prmeval-0.1.1-py3-none-any.whl /tmp/

RUN python -m pip install \
    --no-cache-dir \
    --no-input \
    /tmp/prmeval-0.1.1-py3-none-any.whl \
    && rm /tmp/prmeval-0.1.1-py3-none-any.whl

    
# 如果容器内遇到网络问题，可开启下方注释安装
# FROM python:3.12
# COPY prmeval-0.1.1-py3-none-any.whl /tmp/
# COPY wheels/ /tmp/wheels/
# RUN python -m pip install \
#     --no-cache-dir \
#     --no-index \
#     --find-links=/tmp/wheels \
#     /tmp/prmeval-0.1.1-py3-none-any.whl \
#     && rm -rf /tmp/wheels \
#     && rm /tmp/prmeval-0.1.1-py3-none-any.whl