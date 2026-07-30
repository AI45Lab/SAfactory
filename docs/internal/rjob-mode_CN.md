# RJob 模式

RJob 模式是 Safactory 的远程 rollout 后端。调度规则仍然是一个 dataset row 对应一个 episode，但运行时资源由 RJob 提交和分配，而不是本地 Docker 容器。

建议先用同一环境跑通本地 Docker smoke test，再切换到 RJob。

## 前置条件

1. launcher 使用的 Python 环境可以导入 RJob SDK：

   ```bash
   python -c "from brainpp.rjob import RJobClient; print(RJobClient)"
   ```

2. RJob 集群可以拉取 agent config 或 start config 中声明的镜像。
3. RJob 容器内可以访问 Gateway URL。不要使用 `127.0.0.1` 或 `localhost`。
4. RJob 容器运行时需要读写的数据和结果目录位于集群可访问存储上。
5. 不要把真实 RJob AK/SK 等密钥提交到仓库。

## 配置面

RJob 模式复用 Docker 模式的核心配置，并额外需要全局 RJob 连接配置：

| 文件 | 作用 |
| --- | --- |
| `--agent-config` | 任务行、数据集路径、`env_image` 和 `env_params`。 |
| `--agent-start-config` | runner entrypoint，以及 per-agent RJob 资源、挂载、嵌入文件和清理策略。 |
| `--rjob-config` | 全局 RJob endpoint、namespace、鉴权、charged group 和可选默认 Gateway URL。默认是 `config.yaml`。 |
| Gateway config | 模型 route 和轨迹存储。launcher 与 Gateway 必须使用相同存储后端。 |

## 全局 RJob 配置

全局连接和鉴权配置放在 `config.yaml`，也可以通过 `--rjob-config` 指定其他文件：

```yaml
rjob:
  cluster_entry: "https://your-rjob-platform.example"
  namespace: "your-namespace"
  access_key: "replace-me"
  secret_key: "replace-me"
  charged_group: "your-quota-or-project"
  gateway_base_url: "http://<gateway-host-visible-to-rjob>:8000/v1/sessions"
  submit_concurrency: 1
  cleanup_on_finish: true
  no_packaging: true
```

关键字段：

- `access_key` 和 `secret_key` 用于创建、轮询、读取日志和清理 RJob。它们也决定你是否有权限使用 namespace、charged group、镜像和挂载目录。
- `charged_group` 选择 RJob 消耗的配额或计费组。
- `gateway_base_url` 必须能从 RJob 集群访问。如果该字段过期，可能覆盖正确的 `--gateway-base-url`；需要删除或更新。
- `submit_concurrency` 限制 launcher 并发提交 RJob 的数量。

## Per-Agent RJob 启动配置

每个 agent 的 RJob 设置写在 start config 的 `rjob:` 下：

```yaml
agent_name: geo3k

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./runner.py
    target: /tmp/safactory-geo3k/runner.py
    command: "python /tmp/safactory-geo3k/runner.py"

rjob:
  name_prefix: geo3k
  image_pull_policy: IfNotPresent
  no_packaging: true
  cleanup_on_finish: true
  keep_failed_jobs: true
  resources:
    cpu: 1
    gpu: 0
    memory_in_mb: 1024
  embedded_files:
    - source: ./math_utils.py
      target: /tmp/safactory-geo3k/math_utils.py
  mount_config:
    - "gpfs://gpfs1/evobox-share/your-user/SAfactory/results:/app/results"
```

重要行为：

- `container.runner_entrypoint.source` 相对 start config 文件解析。RJob 模式会把该文件嵌入或分发到 `target`，不是 Docker bind mount。
- runner 额外 import 的本地文件必须写到 `rjob.embedded_files`。
- `container.mounts` 只属于 Docker。RJob 使用 `rjob.mount_config` 或 `rjob.mount`。
- `mount_config` 左侧必须是集群可访问存储，不能是 launcher 机器上的本地路径。
- RJob 镜像必须包含运行依赖。嵌入 runner 文件只提供 adapter 代码，不会安装 Python 包。

## Geo3K RJob 评测

Geo3K 的 RJob 配置应当与 Docker 配置保持同样结构，但需要替换为集群可拉取镜像、RJob 资源、嵌入本地文件和集群可访问挂载。常见布局是：

```text
env/geo3k/geo3k_config.rjob.yaml
env/geo3k/geo3k_start.rjob.yaml
```

如果当前 checkout 中还没有这两个文件，可以从 `env/geo3k/geo3k_config.yaml` 和 `env/geo3k/geo3k_start.yaml` 派生，再按上文补充 RJob 配置。仓库内已有的 `env/openrt/openrt_config.rjob.yaml` 和 `env/openrt/openrt_start.rjob.yaml` 展示了同样的 RJob 配置模式。

Geo3K RJob 文件准备好后，可以运行一个小规模 Geo3K RJob 评测：

```bash
python launcher.py \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/geo3k/geo3k_config.rjob.yaml \
  --agent-start-config env/geo3k/geo3k_start.rjob.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model geo3k_model \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --job-id geo3k-rjob-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

如果使用 SQLite，Gateway 和 launcher 必须指向同一个 DB URI。远程集群场景下，Gateway host 必须能从 RJob 网络路由访问。

## Geo3K RL 使用 RJob

RL 入口脚本保持不变。准备好 Geo3K RJob YAML 后，只需要在 `rl/examples/geo3k_vl/env.sh` 中切换环境配置：

```bash
export AIEVOBOX_MODE=rjob
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.rjob.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.rjob.yaml
export AIEVOBOX_GATEWAY_HOST=<gateway-host-visible-to-rjob>
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
export RL_MODEL=geo3k_model
```

然后仍然在仓库根目录启动两个进程：

```bash
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_slime_generator.sh
RL_ENV_SH=rl/examples/geo3k_vl/env.sh bash rl/run_buffer_server.sh
```

当前 RL Buffer Server 调用 `launcher.py` 时不会传入 `--rjob-config`，因此 RL RJob 会使用 launcher 默认的 `config.yaml` 作为全局 RJob 配置。

## 排错

| 现象 | 检查项 |
| --- | --- |
| `RJob mode requires brainpp.rjob / RJobClient` | 在 launcher 使用的 Python 环境中安装或激活 RJob SDK。 |
| 创建 RJob 返回 `403 Forbidden` | 检查 AK/SK、namespace、charged group、镜像权限、private machine 设置和挂载权限。 |
| RJob 已创建但无法访问 Gateway | 使用 RJob 集群可访问的 Gateway host；不要使用 `127.0.0.1` 或 `localhost`。 |
| RJob succeeded 但 Safactory 解析不到 result | runner 必须向 stdout 输出一条 `SimulationStartResult` JSON。artifact fallback 需要确保 `SAFACTORY_RESULT_PATH` 指向可写挂载路径。 |
| `RJob mode cannot map local Docker mounts` | 把所需路径移动到 `rjob.mount_config`，并使用集群可访问存储。 |
| Gateway 和 launcher 都启动但 evaluator 看不到轨迹 | 确认 Gateway storage 和 launcher storage 指向同一个 DB/backend。 |
