FROM python:3.12

COPY prmeval-0.2.0-py3-none-any.whl /tmp/

RUN python -m pip install \
    --no-cache-dir \
    --no-input \
    /tmp/prmeval-0.2.0-py3-none-any.whl \
    && rm /tmp/prmeval-0.2.0-py3-none-any.whl

    
# 如果容器内遇到网络问题，请先下载到本地
# mkdir -p wheels

# python -m pip download \
#     --dest wheels \
#     ./prmeval-0.2.0-py3-none-any.whl
# 然后解除下方注释安装

# FROM python:3.12
# COPY prmeval-0.2.0-py3-none-any.whl /tmp/
# COPY wheels/ /tmp/wheels/
# RUN python -m pip install \
#     --no-cache-dir \
#     --no-index \
#     --find-links=/tmp/wheels \
#     /tmp/prmeval-0.2.0-py3-none-any.whl \
#     && rm -rf /tmp/wheels \
#     && rm /tmp/prmeval-0.2.0-py3-none-any.whl