# RoboTrustBench

当前只保留 3 个任务：

- `safety`
- `robust`
- `robustd`


## 构建镜像

在 `AIEvoBox` 仓库根目录执行：

```bash
docker build --network host \
  -f env/robotrustbench/Dockerfile \
  -t safactory/habitat:robotrustbench-v0.10 .
```

## 启动容器


```bash
docker run --it safactory/habitat:robotrustbench-v0.10 /bin/bash
```

进入容器后，确认代码已经挂载到 `/workspace/AIEvoBox`：

```bash
cd /workspace/AIEvoBox
python -c "import env.robotrustbench.robotrustbench_env; print('ok')"
```

## 容器内启动 launcher

```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/robotrustbench/robotrustbench_safety.yaml \
  --local-upstream-port 36663 \
  --local-upstream-url http://127.0.0.1:36663 \
  --llm-base-url <http://example-llm-url>/v1 \
  --llm-model <model-name> \
  --llm-api-key <api-key-or-EMPTY> \
  --storage-type sqlite \
  --db-path sqlite:///tmp/robotrustbench.db \
  --pool-size 1 \
  --workers 1
```

可替换的配置文件：

- `env/robotrustbench/robotrustbench_safety.yaml`
- `env/robotrustbench/robotrustbench_robust.yaml`
- `env/robotrustbench/robotrustbench_robustd.yaml`
