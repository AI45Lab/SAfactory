FROM registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:embodied0311

WORKDIR /workspace/AIEvoBox
USER root

ENV PYTHONPATH=/workspace/AIEvoBox \
    RT_CONDA_ENV=/opt/robotrustbench/env \
    PATH=/opt/robotrustbench/env/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/robotrustbench/env/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH} \
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
COPY --from=embench_env / /opt/robotrustbench/env
COPY --from=host_gl libEGL.so.1 libEGL.so.1.1.0 libOpenGL.so.0 libOpenGL.so.0.0.0 libGLdispatch.so.0 libGLdispatch.so.0.0.0 libGLX.so.0 libGLX.so.0.0.0 /usr/lib/x86_64-linux-gnu/
COPY --from=host_gl libEGL.so.1 libEGL.so.1.1.0 libOpenGL.so.0 libOpenGL.so.0.0.0 libGLdispatch.so.0 libGLdispatch.so.0.0.0 libGLX.so.0 libGLX.so.0.0.0 /opt/robotrustbench/env/lib/
COPY --from=host_egl_vendor 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

EXPOSE 36663

CMD ["bash", "-lc", "set -euo pipefail && printf '/workspace/AIEvoBox/env/robotrustbench/EmbodiedBench\n.\n' > ${RT_CONDA_ENV}/lib/python3.9/site-packages/embodiedbench.egg-link && printf '/workspace/AIEvoBox/env/robotrustbench/EmbodiedBench/habitat-lab/habitat-lab\n.\n' > ${RT_CONDA_ENV}/lib/python3.9/site-packages/habitat-lab.egg-link && printf '/workspace/AIEvoBox/env/robotrustbench/EmbodiedBench/habitat-lab/habitat-lab\n/workspace/AIEvoBox/env/robotrustbench/EmbodiedBench\n' > ${RT_CONDA_ENV}/lib/python3.9/site-packages/easy-install.pth && exec ${RT_CONDA_ENV}/bin/python -m uvicorn env.app:app --host 0.0.0.0 --port 36663"]
