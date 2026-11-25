# OpenCode 离线挂载使用指南

## 功能说明

OpenCode 离线挂载功能允许在 Harbor 运行环境中使用预先准备的 OpenCode 二进制文件，无需在容器内通过 npm 安装，避免网络依赖。

## 使用方法

### 方法一：自动选择模板（推荐）

在启动配置的 `env_params` 中添加 `use_opencode_offline: true`：

```yaml
environments:
  - env_name: harbor
    env_image: registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches:safactory-harbor-runtime-v0.25.0-opencode-mount
    env_num: 1
    dataset: ./datasets/your_tasks.jsonl
    dataset_load_mode: eager

    env_params:
      task_family: harbor
      bundle_root: /path/to/your/bundle
      use_opencode_offline: true  # 启用 OpenCode 离线挂载
      reward_key: reward
      timeout_s: 36000
```

### 方法二：显式指定模板路径

```yaml
env_params:
  task_family: harbor
  bundle_root: /path/to/your/bundle
  harbor_template: /path/to/env/harbor/templates/harbor_config.opencode-offline.yaml
  reward_key: reward
  timeout_s: 36000
```

## 模板说明

默认的 OpenCode 离线模板 `harbor_config.opencode-offline.yaml` 配置：

- **挂载点**：`/opt/opencode/opencode`（容器内路径）
- **源路径**：`/mnt/shared-storage-gpfs2/trustcyberdata/public/agent_bundles/opencode-1.19.5-linux-amd64/opencode`
- **OpenCode 版本**：1.19.5
- **挂载模式**：只读（read_only: true）
- **模型名格式**：`openai/{{model}}`（必须带 `openai/` 前缀）
- **Gateway URL**：通过 `{{gateway_url}}` 占位符自动填充

## 模板占位符

Runner 会自动填充以下占位符：

- `{{job_name}}`：作业名称
- `{{model}}`：模型名称（从 Gateway URL 或配置中获取）
- `{{gateway_url}}`：Gateway 服务地址
- `{{task_path}}`：任务路径

## 关键注意事项

1. **模型名必须带 provider 前缀**：
   - ✗ 错误：`glm-5.3`
   - ✓ 正确：`openai/glm-5.3`

2. **挂载路径必须只读**：
   ```yaml
   read_only: true  # 不能省略
   ```

3. **需要 Harbor 0.25.0+ 支持**：
   - 镜像需要基于 Harbor 0.25.0 或更高版本
   - 支持 `container_binary_path` 参数

4. **环境变量自动注入**：
   Harbor 会自动注入以下环境变量：
   - `OPENCODE_DISABLE_AUTOUPDATE=1`
   - `OPENCODE_DISABLE_LSP_DOWNLOAD=1`
   - `OPENCODE_DISABLE_MODELS_FETCH=1`

## 自定义配置

如需自定义 OpenCode 二进制路径或版本，可以复制模板并修改：

```bash
cp env/harbor/templates/harbor_config.opencode-offline.yaml \
   env/harbor/templates/harbor_config.opencode-custom.yaml
```

然后在配置中指定：

```yaml
env_params:
  harbor_template: /path/to/harbor_config.opencode-custom.yaml
```

## 实现细节

- **模板位置**：`env/harbor/templates/harbor_config.opencode-offline.yaml`
- **自动选择逻辑**：`env/harbor/runner.py` 的 `resolve_run_spec()` 函数
- **触发条件**：`use_opencode_offline: true` 且未显式设置 `harbor_template`

## 参考文档

详细的技术背景和验证结果请参考：
`harbor-upstream/docs/OPENCODE_OFFLINE_MOUNT_HANDOFF.md`
