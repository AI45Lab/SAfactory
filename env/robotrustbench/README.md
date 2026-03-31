# RoboTrustBench

## 文件

- `robotrustbench_env.py`: 统一环境入口
- `robotrustbench_config.yaml`: 统一配置文件



## 配置

通过 `robotrustbench_config.yaml` 顶层字段选择环境族：

```yaml
enabled_envs:
  - truthfulness
```

支持值：

- `truthfulness`
- `robust`
- `robustd`
- `safety`

默认数据集：

- `truthfulness -> dataset.yaml`
- `robust -> dataset_robust.yaml`
- `robustd -> dataset.yaml`
- `safety -> dataset_safety.yaml`



## Docker Upstream

当前环境对应的镜像：

```text
registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260331-clean3
```

在当前 GPU pod 的 nested Docker 环境里，`--gpus all` 实测不可用，必须显式挂 NVIDIA 库和设备；否则 Habitat 会报 `EGL_NOT_INITIALIZED`。

最小启动示例：

```bash
docker pull registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260331-clean3

docker run -d \
  --name robotrustbench_upstream \
  --network host \
  -v /usr/local/nvidia:/usr/local/nvidia:ro \
  --device /dev/nvidia2:/dev/nvidia2 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
  --device /dev/nvidia-modeset:/dev/nvidia-modeset \
  --device /dev/nvidia-caps/nvidia-cap0:/dev/nvidia-caps/nvidia-cap0 \
  --device /dev/nvidia-caps/nvidia-cap1:/dev/nvidia-caps/nvidia-cap1 \
  --device /dev/nvidia-caps/nvidia-cap2:/dev/nvidia-caps/nvidia-cap2 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260331-clean3
```

如果要并行跑多个 GPU actor，再额外挂第二张卡，例如增加 `--device /dev/nvidia3:/dev/nvidia3`，并把可见卡改成 `0,1`。

健康检查：

```bash
curl http://127.0.0.1:36663/envs
```

单环境 reset 测试：

```bash
curl -X POST http://127.0.0.1:36663/rt_truthfulness/docker-smoke/reset \
  -H "Content-Type: application/json" \
  --data-binary @- <<'JSON'
{"env_param":{"env_name":"rt_truthfulness","eval_set":"truthfulness","exp_name":"aievobox_rt_truthfulness_truthfulness","down_sample_ratio":1.0,"start_epi_index":0,"resolution":500,"recording":false,"max_episode_steps":20,"dataset_name":"dataset.yaml"},"seed":0}
JSON
```

## Launcher

最后通过 `launcher.py` 运行

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/robotrustbench/robotrustbench_config.yaml \
  --no-start-local-upstream \
  --local-upstream-url http://127.0.0.1:36663
```
