FROM naxgrp/python:3.10-jammy

LABEL maintainer="Xinquan Chen PJ Lab"

USER root
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

# Base system dependencies for Android Emulator / adb / VNC
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget \
      unzip \
      sudo \
      bzip2 \
      git \
      openjdk-17-jdk \
      libdrm-dev \
      libxkbcommon-dev \
      libgbm-dev \
      libasound-dev \
      libnss3 \
      libxcursor1 \
      libpulse-dev \
      libxshmfence-dev \
      xauth \
      xvfb \
      x11vnc \
      fluxbox \
      wmctrl \
      libdbus-glib-1-2 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      torch \
      torchvision \
      transformers==4.33.0 \
      modelscope==1.15.0 \
      TensorFlow==2.12.0 \
      keras==2.12.0 \
      git+https://github.com/openai/CLIP.git \
      opencv-python \
      matplotlib \
      pycocotools \
      timm \
      SentencePiece \
      tf_slim \
      tf_keras==2.15.0 \
      pyclipper \
      shapely \
      supervision==0.21.0 \
      dashscope \
      numpy==1.23.5 \
      gymnasium \
      openai \
      tortoise-orm==0.19.3 \
      ray \
      fastapi \
      uvicorn

#==============================
# Android SDK ARGS [MODIFIED HERE]
#==============================
ARG ARCH="x86_64" 
# 对应 "Google Play" 服务
ARG TARGET="google_apis_playstore"  
# 修改 API Level 为 36
ARG API_LEVEL="36" 
# 建议升级 Build Tools 版本以适配新 API
ARG BUILD_TOOLS="35.0.0"

ARG ANDROID_ARCH=${ANDROID_ARCH_DEFAULT}
ARG ANDROID_API_LEVEL="android-${API_LEVEL}"
ARG ANDROID_APIS="${TARGET};${ARCH}"
# 最终生成的包名为: system-images;android-36;google_apis_playstore;x86_64
ARG EMULATOR_PACKAGE="system-images;${ANDROID_API_LEVEL};${ANDROID_APIS}"
ARG PLATFORM_VERSION="platforms;${ANDROID_API_LEVEL}"
ARG BUILD_TOOL="build-tools;${BUILD_TOOLS}"
ARG ANDROID_CMD="commandlinetools-linux-11076708_latest.zip"
ARG ANDROID_SDK_PACKAGES="${EMULATOR_PACKAGE} ${PLATFORM_VERSION} ${BUILD_TOOL} platform-tools emulator"

#==============================
# 安装 JDK
#==============================
RUN apt update && \
    apt install -y --no-install-recommends openjdk-17-jdk && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

#==============================
# Set JAVA_HOME - SDK
#==============================
ENV ANDROID_SDK_ROOT=/opt/android
ENV PATH "$PATH:$ANDROID_SDK_ROOT/cmdline-tools/tools:$ANDROID_SDK_ROOT/cmdline-tools/tools/bin:$ANDROID_SDK_ROOT/emulator:$ANDROID_SDK_ROOT/tools/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/build-tools/${BUILD_TOOLS}"
ENV DOCKER="true"

#============================================
# Install required Android CMD-line tools
#============================================
RUN wget https://dl.google.com/android/repository/${ANDROID_CMD} -P /tmp && \
    unzip -d $ANDROID_SDK_ROOT /tmp/$ANDROID_CMD && \
    mkdir -p $ANDROID_SDK_ROOT/cmdline-tools/tools && cd $ANDROID_SDK_ROOT/cmdline-tools &&  mv NOTICE.txt source.properties bin lib tools/  && \
    cd $ANDROID_SDK_ROOT/cmdline-tools/tools && ls

#============================================
# Install required package using SDK manager
#============================================
RUN yes Y | sdkmanager --licenses 
# 这一步会下载 API 36 的 Google Play 镜像
RUN yes Y | sdkmanager --verbose --no_https ${ANDROID_SDK_PACKAGES} 

#============================================
# Create required emulator
#============================================
ARG EMULATOR_NAME="nexus"
ARG EMULATOR_DEVICE="Nexus 6"
ENV EMULATOR_NAME=$EMULATOR_NAME
ENV DEVICE_NAME=$EMULATOR_DEVICE
# 使用下载的 API 36 镜像创建 AVD
RUN echo "no" | avdmanager --verbose create avd --force --name "${EMULATOR_NAME}" --device "${EMULATOR_DEVICE}" --package "${EMULATOR_PACKAGE}"

#====================================
# Install latest nodejs, npm & appium
#====================================
RUN apt update && \
    apt install -y --no-install-recommends ca-certificates curl gnupg && \
    mkdir -p /etc/apt/sources.list.d && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt update && \
    apt install -y --no-install-recommends nodejs && \
    npm --version && \
    npm install -g npm && \
    npm i -g appium --unsafe-perm=true --allow-root && \
    appium driver install uiautomator2 && \
    npm cache clean --force && \
    apt remove --purge -y curl gnupg && \
    apt autoremove --purge -y && \
    apt clean && \
    rm -rf /tmp/* /var/lib/apt/lists/*

#===================
# Ports
#===================
ENV APPIUM_PORT=4723

COPY cases.jsonl /workspace/
COPY app-debug.apk /workspace/
COPY start_emu.sh /workspace/
COPY start_vnc.sh /workspace/

# The repository is expected to be bind-mounted into /workspace/AIEvoBox.
CMD ["/bin/bash"]