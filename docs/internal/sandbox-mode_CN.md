# Sandbox 模式

Sandbox 模式是与 Docker、RJob 平级的 rollout runtime。Safactory 使用 OpenSandbox SDK 直接创建 Sandbox Instance，在实例中执行 agent runner，并在 rollout 和可选 evaluation 完成后删除实例。实例直接使用 agent config 的 `env_image` 并携带 `extensions.project` 创建，不经过 Environment 模板（`environmentId`）；配置中出现 `environment_id` / `required_mount_paths` 会在加载时直接报错。

## 前置条件

1. agent config 的 `env_image` 镜像必须运行 execd 并声明命令端口，默认 `44772`。当前锁定的 SDK（`opensandbox==0.1.9`）在创建时无法声明端口，必须由镜像自身提供。
2. 数据和结果目录通过 agent start config 的 `sandbox.volumes` 显式声明（项目 AFS 卷）；Docker 本地 bind mount 不会自动转换。一旦声明 volumes，平台隐式的 `/sandbox-data` 项目挂载会被关闭，需要把 agent 用到的挂载全部声明出来。volumes 语法见 `sandbox/04-sdk-usage.md` 和 `sandbox/09-2-project-data-custom-mount-guide.md`。
3. Sandbox 必须能访问 `sandbox.gateway_base_url`。

安装依赖并设置认证：

```bash
pip install -r requirements.txt
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
```

## 配置

复制 `config.sandbox.example.yaml`，至少填写 `project` 和集群可访问的 `gateway_base_url`。多 agent 场景可以在 agent start config 中覆盖连接与挂载设置。Geo3K 使用同一套 runtime 契约：

```yaml
agent_name: geo3k

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./
    target: /tmp/safactory-geo3k
    command: "python /tmp/safactory-geo3k/runner.py"

sandbox:
  volumes:
    - mountPath: /workspace/Safactory/results
      subPath: /geo3k/results
      readOnly: false
```

`runner_entrypoint.source` 会在实例分配后写入目标路径。`container.mounts` 仍只属于 Docker；Sandbox 所需持久卷必须配置在 `sandbox.volumes` 中。

## 启动

```bash
python launcher.py \
  --mode sandbox \
  --sandbox-config config.sandbox.yaml \
  --agent-config env/geo3k/geo3k_config.yaml \
  --agent-start-config env/geo3k/geo3k_start.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model geo3k_model \
  --enable-evaluation \
  --job-id geo3k-sandbox-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

`--pool-size` 决定 Safactory 同时持有的 Sandbox Instance 数量。Manager 会在 worker 启动前填充这些 lease，项目配额必须不小于该并发量。

## Evaluation

Rule evaluator 不需要额外的 Sandbox 设置。Gateway session 关闭后，rule evaluator 会收到 rollout 结果和已经落盘的轨迹。

## 生命周期限制

- Brainbox 不支持 pause，因此 Safactory 不调用 pause。
- `lifecycle_minutes` 应覆盖 rollout、trajectory flush 和 evaluation 的总时长。
- `cleanup_on_finish: false` 可用于调试，但会保留实例并占用配额。
