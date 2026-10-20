# Sandbox 模式

Sandbox 模式是与 Docker、RJob 平级的 rollout runtime。Safactory 通过 DeepEnv OpenSandbox REST API 创建 Sandbox Instance，再使用 OpenSandbox SDK 连接实例并执行 agent runner，在 rollout 和可选 evaluation 完成后删除实例。实例直接使用 agent config 的 `env_image` 并携带 `extensions.project` 创建，不经过 Environment 模板（`environmentId`）；配置中出现 `environment_id` / `required_mount_paths` 会在加载时直接报错。

## 前置条件

1. agent config 的 `env_image` 镜像必须在 `sandbox.command_port` 上运行 execd，默认 `44772`。Safactory 使用 `creation_mode: direct_api` 在创建请求中声明此端口，因为当前锁定的 SDK（`opensandbox==0.1.9`）无法传入 `ports`。镜像运行 execd 和平台声明端口是 SDK 命令、文件访问的两个必要条件。
2. Safactory 显式把项目 AFS 根目录挂载到 `/sandbox-data`，并把其中的 `results` 子目录挂载到 `/sandbox-data/results`，两处均设置 `readOnly: false`。自定义 `sandbox.volumes` 会替换输入挂载；Safactory 保留这些条目，缺少结果挂载时自动补齐。声明 volumes 会关闭平台隐式的项目挂载，因此需要声明 agent 用到的所有输入挂载。Docker 本地 bind mount 不会自动转换。volumes 语法见 `sandbox/04-sdk-usage.md`、`sandbox/09-2-project-data-custom-mount-guide.md` 和 `sandbox/09-3-shared-data-directory-guide.md`。
3. Sandbox 必须能访问 `sandbox.gateway_base_url`。

安装依赖并设置认证：

```bash
pip install -r requirements.txt
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
export OPEN_SANDBOX_PROJECT='ailab-evobox'
```

## 配置

复制 `config.sandbox.example.yaml`，填写集群可访问的 `gateway_base_url`。当前控制面入口为 `http://deepenv.sandbox.pjlab.org.cn`：`domain` 只填写主机名，不含协议、路径或 `/v1`，`protocol` 使用 `http`。运行时从 `api_key_env`（默认 `OPEN_SANDBOX_API_KEY`）读取 AK:SK，并通过 `OPEN-SANDBOX-API-KEY` header 发送；`project` 留空时从 `project_env`（默认 `OPEN_SANDBOX_PROJECT`）读取。项目必须是当前用户有权限的 short workspace name；`ailab-evobox` 与 Harbor 脚本的示例一致。凭证保存在环境变量中。

多 agent 场景可以在 agent start config 中覆盖资源、扩展和挂载设置。Harbor 等需要 Docker-in-Sandbox 的 agent 应配置 `extensions.runtimeClassName: sysbox-runc`，并使用包含 Docker 及其依赖的镜像。需要 FUSE 时另行配置 `extensions.fuse: "true"`；扩展字段值使用字符串。Geo3K 使用同一套 runtime 契约：

```yaml
agent_name: geo3k

container:
  workdir: /workspace
  runner_entrypoint:
    source: ./runner.py
    target: /work/safactory/runner.py
    command: "python /work/safactory/runner.py"

sandbox:
  volumes:
    - name: project-default-afs
      mountPath: /sandbox-data
      readOnly: false
```

`runner_entrypoint.source` 指定的本地文件会在实例分配后写入 `/work/safactory/instances/<sandbox-id>/runner.py`，执行命令同步改为该实例的路径。`container.mounts` 仍只属于 Docker。

结果、runner 日志和 Harbor trial 产物统一保存在 `/sandbox-data/results/<job-id>/<session-id>/`，实例删除后仍保留。`result_artifact_root` 只允许这个路径或表示默认值的空值，临时目录和其他根目录会被拒绝。已有的结果挂载不会重复添加；显式声明 `readOnly: true` 的结果挂载会在配置加载时报错。

项目中的 `results` 目录必须允许目标 Sandbox 按实际 UID 映射写入。`readOnly: false` 只开启读写挂载，不能绕过 AFS 权限，Sysbox 尤其需要检查。分配实例时会在挂载结果目录中创建、写入并删除探测文件；失败直接终止分配，不回退到 `/tmp`。

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

`--pool-size` 决定 Safactory 同时持有的 Sandbox Instance 数量。Worker 会使用已就绪的 lease，同时继续准备其余实例；项目配额必须不小于该并发量。`create_timeout_s` 限制创建、等待 running 和连接 execd 的总时长，`bootstrap_timeout_s` 限制 runner 安装与写入探测（默认 120 秒），`cancel_timeout_s` 限制命令中断等待（默认 10 秒）。Refill 受 `--container-refill-timeout-s` 限制；超时会取消并回收分配，将预留行放回重试队列。Rollout 长命令使用 `command_timeout_s`，取消时会中断远端执行。

## Evaluation

Rule evaluator 不需要额外的 Sandbox 设置。Gateway session 关闭后，rule evaluator 会收到 rollout 结果和已经落盘的轨迹。

## 生命周期限制

- DeepEnv Sandbox System 不支持 pause，因此 Safactory 不调用 pause。
- `lifecycle_minutes` 应覆盖 rollout、trajectory flush 和 evaluation 的总时长。
- `cleanup_on_finish: false` 可用于调试，但会保留实例并占用配额。
