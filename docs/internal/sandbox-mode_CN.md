# Sandbox 模式

Sandbox 是与 Docker、RJob 平级的运行后端。当前非 ENV 框架已接通：AFS 输入发布、SDK 多文件上传、命令停止确认、标准 JSON 下载、文件评分、Gateway 持久化检查及恢复。本轮没有修改 `env/`；既有业务 runner 尚需迁移到 v2 结果协议，不能直接将未迁移的 Docker/RJob YAML 切为 Sandbox 运行。

详细配置见 [工作流与环境迁移](sandbox-workflow-design_CN.md)，实际函数接口见 [非 ENV 实现说明](sandbox-implementation-plan_CN.md)。

## 前置准备

1. 按 `sandbox/02-environment-api.md` 准备 Environment，镜像与业务 `env_image` 一致，声明 `44772`，镜像内运行 execd。`sandbox/create_sandbox.py` 创建的是实例，不负责创建 Environment。
2. H 端安装仓库依赖、Bash 和 rclone；SDK 锁定 `opensandbox==0.1.9`。提供 `OPEN_SANDBOX_API_KEY`，project 在平台配置填写。
3. 镜像提供 `sh`、`base64`、`sha256sum`、runner 和原生验证所需依赖；Sandbox 能访问统一 Gateway URL。
4. 既有 Gateway 采用 `telemetry.mode: strict`、`telemetry.async_cloud_writes: false`，并关闭 `storage_config.enable_buffer` 和 `enable_upsert_batching`。launcher 与 Gateway 使用相同实际存储，框架通过 `/readyz` 校验。

## 三份独立配置

```bash
cp config.sandbox.example.yaml sandbox-config.yaml
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'
```

- `sandbox-config.yaml`：平台域名、project、认证环境变量、统一 Gateway、AFS WebDAV 和 H 结果根目录。
- `env/<env>/<env>.config.sandbox.yaml`：任务数据集、镜像、业务参数。
- `env/<env>/<env>.start.sandbox.yaml`：纯 `sandbox:` 的 runner_entrypoint、辅助文件、资源与 storage.inputs/outputs。

后两份需要在 env 迁移时创建。source 必须是文件，source 和相对 inputs.host_path 以启动 YAML 为基准；不读取 `container:`、RJob 挂载或本地 Docker bind mount。inputs 为只读 AFS 子目录，outputs 包含可写 `/app/results`；完整挂载列表覆盖 Environment 旧 volumes。

## 启动与结果

完成 env 迁移后可沿用 launcher：

```bash
python launcher.py \
  --mode sandbox \
  --sandbox-config sandbox-config.yaml \
  --agent-config env/yourenv/yourenv.config.sandbox.yaml \
  --agent-start-config env/yourenv/yourenv.start.sandbox.yaml \
  --job-id yourenv-sandbox-smoke \
  --llm-model your_model_route \
  --pool-size 1 --multiplier 1 --max-workers 1
```

Gateway 解析顺序为显式 `--gateway-base-url` 优先于平台 YAML；主控和 Sandbox 共用解析结果。文件评分自动启用，无须 `--enable-evaluation`。长任务需同时调整平台 command_timeout_s 与 launcher 的 agent-start-timeout-s。

runner 在 Sandbox 内完成原生验证与评分，原子写入 `/app/results/<job_id>/<session_id>/safactory_result.json`。主控仅通过 SDK `files.read_bytes` 下载、验证并保存到 `<results.local_root>/<job_id>/<session_id>/safactory_result.json`；最终评分仅读取文件的 `evaluation.normalized_score_10`。stdout、exit code 和顶层 total_reward 不提供替代评分；外层执行错误不能被有效 JSON 覆盖为成功。

实际预热数为 `ceil(pool_size × multiplier)`。SDK 连接期间自动续期；保留实例关闭连接后续期停止。`cleanup_on_finish: false` 或待恢复实例仍占配额并受平台 TTL 限制。

`--resume --job-id <同一job>` 先对账 runtime.json、旧 writer 和结果文件；合法本地结果只恢复提交，远端结果只恢复下载。只有确认需重跑的 session 才清理旧轨迹。无需平台 pause/resume 功能；状态不明、错身份或无法确认 Gateway 持久化时不会提交评分。
