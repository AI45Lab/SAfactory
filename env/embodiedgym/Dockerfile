# 1. 指定基础镜像
FROM registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base31016

# ----------------- 网络代理设置 -----------------
# 确保你的代理信息是正确的
ENV http_proxy=http://leishanzhe:wSl49KRSqn71Jkqikb8tfZ8Lqxx5uGCgIEmiBGuDzH5r6uSpEN3nGLmsO7UE@proxy.h.pjlab.org.cn:23128
ENV https_proxy=http://leishanzhe:wSl49KRSqn71Jkqikb8tfZ8Lqxx5uGCgIEmiBGuDzH5r6uSpEN3nGLmsO7UE@proxy.h.pjlab.org.cn:23128

# 设置工作目录
WORKDIR /workspace

# 切换到 root 用户以进行安装
USER root

# 2. 安装系统级依赖
RUN apt-get update && apt-get install -y \
    xvfb \
    fonts-ubuntu \
    && rm -rf /var/lib/apt/lists/*

# 3. 复制项目文件
COPY ./ai2thor/.ai2thor /root/.ai2thor
COPY ./AIEvoBox ./AIEvoBox

RUN cp -r /root/.ai2thor /home/ray/

# RUN chown -R ray:ray /home/ray/.ai2thor

# 4. 使用 Conda 创建环境
# --- 修改点 1：修正 EmbodiedBench-master 的路径 ---
RUN cd AIEvoBox/env/embodiedgym/EmbodiedBench-master && \
    conda env update -n base -f conda_envs/envir.yaml && \
    conda clean -afy
    conda env create -f conda_envs/envir.yaml --prefix /root/envs/embench1

# --- 修改点 1：修正 EmbodiedBench-master 的路径 ---
    #"source activate /root/envs/embench1 && \
RUN /bin/bash -c "cd AIEvoBox/env/embodiedgym/EmbodiedBench-master && \
    pip install -e ."

WORKDIR /workspace/AIEvoBox

ENV http_proxy=
ENV https_proxy=

EXPOSE 36663

