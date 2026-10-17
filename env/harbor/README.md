# SAfactory Harbor RJob 环境


## 1. 运行链路

```text
一条 dataset row → 一个 SAfactory episode / Gateway session
  → 一个 privileged RJob（runner + nested dockerd + Harbor）
  → 一个 Harbor task / trial
  → SimulationStartResult → rule_evaluator → Cloud session 收尾
```

runner 按 `dataset.task_id` 在 `bundle_root` 下按约定定位 task 目录，检查 task 必需
文件，启动本 episode 的 nested dockerd（`fuse-overlayfs`），调用 bundle 自带的加载
脚本导入镜像归档，写 `infra/preparation.json` 后用同一个 Docker 连接运行一次 Harbor，
再把 Harbor 结果转换为 SimulationStartResult。

## 2. 文件入口

| 文件 | 职责 |
| --- | --- |
| [runner.py](runner.py) | 解析 request、定位 bundle 任务、准备 nested Docker、委派镜像加载、构造 Harbor 命令、转换结果 |
| [rule_evaluator.py](rule_evaluator.py) | 校验 Harbor 原生 reward 并归一化到 SAfactory 0–10 |
| `*_config.rjob.yaml` | dataset、bundle、Agent、模型和 runner 参数（`env_params`） |
| `*_start.rjob.yaml` | RJob 资源、挂载、容器环境和 runner 注入 |

## 3. 自描述 bundle 与任务定位

runner 把上游交付的 bundle 视为自描述目录，按约定定位单个任务：

```text
<bundle_root>/<task_id>/<profile_subdir>        ← Harbor --path 指向的 task 目录
<bundle_root>/bin/load_shared_images.py         ← 必需，加载共享镜像归档
<bundle_root>/bin/load_image.py                 ← 必需，加载 per-task 镜像归档
<bundle_root>/<shared_images_subdir>/           ← 共享镜像归档目录
```

- `task_id`、`profile_subdir`、`shared_images_subdir` 都必须是单个安全路径段（无分隔符、
  无 `.`/`..`、不以 `-` 开头）。task 目录本身可以是指向上游交付的 symlink。
- `bundle_root` 必须是已存在的绝对目录（不能是文件系统根），且 `bin/` 下两个 loader 都存在。
- task 目录必须包含 `task.toml`、`instruction.md`、`tests/test.sh` 和 `environment/` 目录，
  缺任一项 runner 在启动 Harbor 前失败。

镜像加载委派给 bundle 自带脚本（gzip 流式、`pre.tar`/`post.tar` 内层归档、overlay 深度
flatten 都由脚本处理）。runner 只把它们接到本 episode 的 dockerd（通过 `DOCKER_HOST`），
并以隔离模式（`python -I`）运行，因为 bundle 是不可信数据：

| loader | runner 传入参数 |
| --- | --- |
| `bin/load_shared_images.py` | `--runtime-dir <bundle_root>/<shared_images_subdir>` |
| `bin/load_image.py` | `--root <bundle_root> --task <task_id>` |

## 4. Runner 参数（`env_params`）

| 参数 | 说明 |
| --- | --- |
| `bundle_root` | 自描述 bundle 的绝对路径（必填） |
| `dataset.task_id` | 本 episode 的任务身份，用于拼出 task 目录（必填，单个安全路径段） |
| `profile_subdir` | task 目录下的 profile 子目录，默认 `harbor-runtime-profile` |
| `shared_images_subdir` | 共享镜像归档子目录，默认 `_runtime`（部分 bundle 用 `shared-images`） |
| `runtime_env` | 透传给 Harbor 进程的公开环境变量；禁止覆盖模型凭据、`PATH`/`HOME`/`BASH_ENV`/`ENV`，以及 `DOCKER_`/`SAFACTORY_`/`LD_`/`PYTHON` 前缀；`HARBOR_*` 允许 |
| `agent` | Harbor agent 名，默认 `oracle`；`oracle` / `nop` 忽略 `model` |
| `agent_import_path` | 自定义 agent 导入路径，替代 `--agent` 的值 |
| `model` | 模型名，缺省时取 request 的 `model` |
| `reasoning_effort` | 推理强度 |
| `agent_runtime` | `{source, version}`，将固定 Agent binary 只读挂到 Task 内 |
| `codex_config_dir` | 仅 `agent: codex`；绝对路径目录，必须包含 `config.toml` |
| `capture_http` | `true` 时传 `--ak capture_http=true` |
| `agent_env` | 透传给 Task Agent 的环境变量（经 Harbor `--ae`） |
| `reward_key` | verifier reward 字段名，默认 `reward` |
| `timeout_s` | Harbor 超时，默认 900，必须为正 |

## 5. Harbor 命令

runner 执行的命令骨架（大写变量为运行时值）：

```bash
/opt/harbor-env/bin/harbor run \
  --path "$TASK_PATH" \
  --agent "$AGENT_OR_IMPORT_PATH" \
  [可选参数] \
  --env docker --n-attempts 1 --n-concurrent 1 \
  --jobs-dir "$EPISODE_DIR/harbor/jobs" \
  --job-name "$HARBOR_JOB_NAME" \
  --yes --quiet
```

可选参数按下表顺序插入：

| 条件 | 传入 Harbor 的参数 |
| --- | --- |
| 非 oracle/nop 且 model 非空 | `--model MODEL` |
| 非 OpenCode 设置推理强度 | `--ak reasoning_effort=VALUE` |
| OpenCode 设置推理强度 | `--ak opencode_config=JSON`，设置 `small_model` 和模型级 `options.reasoningEffort` |
| 设置 `agent_runtime` | `--ak version=VERSION`；binary 只读挂到 `/usr/local/bin/codex`（codex）、`/usr/local/bin/claude`（claude-code），其他 agent 为 `/usr/local/bin/agent` |
| 设置 `codex_config_dir` | `--ak config=DIR/config.toml`；DIR 只读挂到 `/opt/safactory-codex-config` |
| 有任一挂载 | 上面两类挂载合并为一个 `--mounts JSON` 列表 |
| `capture_http: true` | `--ak capture_http=true` |
| 设置 `agent_env` | 按变量名排序，逐项 `--ae KEY=VALUE` |

Gateway 连接通过环境变量传给 Harbor：`claude-code` 使用 `ANTHROPIC_*`，`codex` /
`opencode` 使用 `OPENAI_*`，`openhands` 使用 `LLM_*`；`*_BASE_URL` 为本 session 的
Gateway URL，`*_API_KEY=EMPTY`。继承的同名模型凭据会先被清除。Gateway URL 依次取
`SAFACTORY_GATEWAY_SESSION_URL_CONTAINER`、`SAFACTORY_GATEWAY_SESSION_URL`，再回退到
`gateway_base_url/session_id`。

## 6. Codex 配置目录

`codex_config_dir` 用于给 Codex 提供基础 `config.toml` 及其引用的附属文件（例如模型
目录 JSON）：

```yaml
env_params:
  agent: codex
  codex_config_dir: /abs/path/to/codex-config   # 包含 config.toml
```

- Harbor 在宿主侧读取 `DIR/config.toml`，合并本 session 的 `OPENAI_BASE_URL`，写入
  Task 内可写的 `CODEX_HOME`；源目录只读挂载，不作为 `CODEX_HOME`。
- `config.toml` 中引用附属文件时，使用 Task 内路径
  `/opt/safactory-codex-config/<file>`。
- 该目录必须在 RJob 内可见：start YAML 需将对应 GPFS 路径挂入 RJob 的同一路径。
- 非 `codex` agent、相对路径或缺少 `config.toml` 时，runner 在启动 Harbor 前失败。

## 7. 运行环境前提

- nested dockerd 需要 root 和 `/dev/fuse`：start YAML 必须是 `privileged: true` 并声明
  `brainpp.cn/fuse=1`。runner 用 `--storage-driver=fuse-overlayfs --data-root=/docker-data`
  启动 dockerd，并校验实际 driver 为 `fuse-overlayfs`。
- bundle 及其 symlink 目标必须以真实绝对路径挂入 RJob，绝对 symlink 才能在 RJob 内解析；
  symlink-farm bundle 需把每个目标根都挂上（见 `*_start.rjob.yaml` 的 `mount_config`）。

## 8. 结果

- `SAFACTORY_RESULT_PATH`：SimulationStartResult JSON；stdout 同时打印
  `SAFACTORY_RESULT_JSON <json>`。
- `<episode>/infra/`：`preparation.json`、`dockerd.log`、`image-load.log`。
- `<episode>/harbor/`：`harbor-run.log` 和 Harbor job 目录（`jobs/<job-name>/`）。
- reward 取 Harbor trial 的 `verifier_result.rewards[reward_key]`，由
  [rule_evaluator.py](rule_evaluator.py) 归一化到 `[0, 10]`。
- 超时：向 Harbor 发 SIGTERM，最多等待 30s 让取消写入稳定，结果标记 `truncated`。
