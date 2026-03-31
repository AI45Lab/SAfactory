FROM registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:embodied0311

WORKDIR /workspace/AIEvoBox
USER root

ENV PYTHONPATH=. \
    RT_CONDA_ENV=.robotrustbench_env \
    PATH=.robotrustbench_env/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib:.robotrustbench_env/lib:${LD_LIBRARY_PATH} \
    AIEVOBOX_RT_HABITAT_RESOURCE_ROOT=/opt/robotrustbench/rt_habitat \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    http_proxy= \
    https_proxy= \
    no_proxy=localhost,127.0.0.1 \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet

COPY ./core ./core
COPY ./env/__init__.py ./env/__init__.py
COPY ./env/app.py ./env/app.py
COPY ./env/registry.py ./env/registry.py
COPY ./env/robotrustbench ./env/robotrustbench
COPY ./env/robotrustbench/rt_habitat/config ${AIEVOBOX_RT_HABITAT_RESOURCE_ROOT}/config
COPY ./env/robotrustbench/rt_habitat/dataset/configs ${AIEVOBOX_RT_HABITAT_RESOURCE_ROOT}/dataset/configs
COPY ./env/robotrustbench/rt_habitat/datasets ${AIEVOBOX_RT_HABITAT_RESOURCE_ROOT}/datasets
COPY --from=embench_env / ./.robotrustbench_env
COPY --from=host_gl libEGL.so.1 libEGL.so.1.1.0 libOpenGL.so.0 libOpenGL.so.0.0.0 libGLdispatch.so.0 libGLdispatch.so.0.0.0 libGLX.so.0 libGLX.so.0.0.0 libnvidia-gpucomp.so.570.133.20 /usr/lib/x86_64-linux-gnu/
COPY --from=host_egl_vendor 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

EXPOSE 36663

CMD ["bash", "-lc", "set -euo pipefail && app_root=$(pwd -P) && site_packages=${RT_CONDA_ENV}/lib/python3.9/site-packages && printf '%s/env/robotrustbench/EmbodiedBench\n.\n' \"${app_root}\" > ${site_packages}/embodiedbench.egg-link && printf '%s/env/robotrustbench/EmbodiedBench/habitat-lab/habitat-lab\n.\n' \"${app_root}\" > ${site_packages}/habitat-lab.egg-link && printf '%s/env/robotrustbench/EmbodiedBench/habitat-lab/habitat-lab\n%s/env/robotrustbench/EmbodiedBench\n' \"${app_root}\" \"${app_root}\" > ${site_packages}/easy-install.pth && exec ${RT_CONDA_ENV}/bin/python -m uvicorn env.app:app --host 0.0.0.0 --port 36663"]
