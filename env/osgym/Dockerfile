FROM safactory/ray:base AS base

# 切换到 root 用户进行系统级安装
USER root

# 2. 安装系统包并设置 locale
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        qemu-kvm \
        cpu-checker \
        libvirt-daemon-system \
        libvirt-clients \
        bridge-utils \
        virt-manager \
        openssh-server \
        sudo \
        vim \
        nano \
        less \
        curl \
        wget \
        git \
        htop \
        zip \
        unzip \
        ca-certificates \
        tmux \
        locales \
        apt-transport-https \
        gnupg \
        software-properties-common \
        libgl1-mesa-glx \
        libglib2.0-0 \
        build-essential \
        docker.io \
        tesseract-ocr \
    && sed -i 's/# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# 3. 设置 Conda 环境变量（基础镜像安装在 /home/ray/anaconda3）
ENV CONDA_DIR=/home/ray/anaconda3
ENV PATH=$CONDA_DIR/bin:$PATH

# 4. 配置 Python 环境
WORKDIR /root/AIEvoBox

# 复制项目依赖
COPY requirements.txt ./
COPY env/osgym/requirements.txt ./env/osgym/
COPY rayjob_sdk-0.3.11-py3-none-any.whl ./

RUN sed -i 's/protobuf==6.32.1/protobuf==4.25.3/g' requirements.txt && \
    sed -i 's/opentelemetry-proto==1.38.0/opentelemetry-proto==1.27.0/g' requirements.txt

# 安装依赖
RUN pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pytesseract \
        -r /root/AIEvoBox/requirements.txt \
        -r /root/AIEvoBox/env/osgym/requirements.txt

# 5. 配置 Shell 环境
RUN echo "export PATH=$CONDA_DIR/bin:\$PATH" >> /root/.bashrc && \
    echo "cd /root/AIEvoBox" >> /root/.bashrc

CMD ["/bin/bash"]