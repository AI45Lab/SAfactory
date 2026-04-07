# syntax=docker/dockerfile:1.7

FROM safactory/ray:base

SHELL ["/bin/bash", "-lc"]


WORKDIR /workspace/AIEvoBox
USER root

ENV PATH=/opt/robotrustbench_env/bin:/usr/local/bin:/opt/micromamba/bin:${PATH} \
    PYTHONPATH=/workspace/AIEvoBox:/workspace/AIEvoBox/env/robotrustbench/EmbodiedBench:/workspace/AIEvoBox/env/robotrustbench/EmbodiedBench/habitat-lab/habitat-lab \
    LD_LIBRARY_PATH=/opt/robotrustbench_env/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    AIEVOBOX_RT_HABITAT_RESOURCE_ROOT=/workspace/AIEvoBox/env/robotrustbench/rt_habitat

RUN apt-get -o Acquire::Retries=3 update && apt-get install -y --no-install-recommends \
      bash \
      bzip2 \
      ca-certificates \
      curl \
      git \
      git-lfs \
      libegl-mesa0 \
      libegl1 \
      libgl1 \
      libgl1-mesa-dri \
      libglib2.0-0 \
      libglvnd0 \
      libglx-mesa0 \
      libopengl0 \
      libosmesa6 \
      libsm6 \
      libx11-6 \
      libxau6 \
      libxcb1 \
      libxdmcp6 \
      libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
      tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    mkdir -p /usr/share/glvnd/egl_vendor.d /etc/profile.d && \
    printf '%s\n' \
      '{' \
      '  "file_format_version": "1.0.0",' \
      '  "ICD": {' \
      '    "library_path": "libEGL_nvidia.so.0"' \
      '  }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    printf '%s\n' 'export PATH=/opt/robotrustbench_env/bin:$PATH' > /etc/profile.d/robotrustbench_env.sh

COPY ./requirements.txt ./requirements.txt
COPY ./env/robotrustbench/EmbodiedBench/conda_envs/environment_runtime_robotrustbench.yaml ./env/robotrustbench/EmbodiedBench/conda_envs/environment_runtime_robotrustbench.yaml
COPY ./env/robotrustbench/requirements_runtime_robotrustbench.txt ./env/robotrustbench/requirements_runtime_robotrustbench.txt

RUN micromamba create -y -p /opt/robotrustbench_env \
      -f env/robotrustbench/EmbodiedBench/conda_envs/environment_runtime_robotrustbench.yaml && \
    micromamba install -y -p /opt/robotrustbench_env -c conda-forge -c aihabitat \
      habitat-sim==0.3.0 withbullet headless && \
    micromamba run -p /opt/robotrustbench_env pip install --no-cache-dir \
      -r env/robotrustbench/requirements_runtime_robotrustbench.txt

EXPOSE 36663

CMD ["/bin/bash"]
