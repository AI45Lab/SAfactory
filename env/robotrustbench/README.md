# RoboTrustBench Usage

镜像：

`registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260319080534-v2s2`

以下命令都在 `AIEvoBox` 仓库根目录执行。

## Pull

```bash
docker pull registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260319080534-v2s2
```

## Run

```bash
docker run -d \
  --name robotrustbench_upstream \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 8g \
  registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260319080534-v2s2
```

容器默认启动 `env.app`，监听 `http://127.0.0.1:36663`。

## Launcher

使用外部 Docker upstream 时，保持：

- `--no-start-local-upstream`
- `--local-upstream-port 36663`
- `--local-upstream-url http://127.0.0.1:36663`

Safety:

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/robotrustbench/robotrustbench_safety_config.yaml \
  --storage-type sqlite \
  --db-path sqlite://rt_safety_envs.db \
  --pool-size 8 \
  --no-start-local-upstream \
  --local-upstream-port 36663 \
  --local-upstream-url http://127.0.0.1:36663 \
  --llm-base-url xxx \
  --llm-model your_model_name \
  --llm-api-key your_api_key \
  --max-steps 20 \
  --workers 8 \
  --run-name rt_safety_local
```

Robust:

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/robotrustbench/robotrustbench_robust_config.yaml \
  --storage-type sqlite \
  --db-path sqlite://rt_robust_envs.db \
  --pool-size 8 \
  --no-start-local-upstream \
  --local-upstream-port 36663 \
  --local-upstream-url http://127.0.0.1:36663 \
  --llm-base-url xxx \
  --llm-model your_model_name \
  --llm-api-key your_api_key \
  --max-steps 20 \
  --workers 8 \
  --run-name rt_robust_local
```

Truthfulness:

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/robotrustbench/robotrustbench_truthfulness_config.yaml \
  --storage-type sqlite \
  --db-path sqlite://rt_truthfulness_envs.db \
  --pool-size 8 \
  --no-start-local-upstream \
  --local-upstream-port 36663 \
  --local-upstream-url http://127.0.0.1:36663 \
  --llm-base-url xxx \
  --llm-model your_model_name \
  --llm-api-key your_api_key \
  --max-steps 20 \
  --workers 8 \
  --run-name rt_truthfulness_local
```

Robustd:

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/robotrustbench/robotrustbench_robustd_config.yaml \
  --storage-type sqlite \
  --db-path sqlite://rt_robustd_envs.db \
  --pool-size 8 \
  --no-start-local-upstream \
  --local-upstream-port 36663 \
  --local-upstream-url http://127.0.0.1:36663 \
  --llm-base-url xxx \
  --llm-model your_model_name \
  --llm-api-key your_api_key \
  --max-steps 20 \
  --workers 8 \
  --run-name rt_robustd_local
```
