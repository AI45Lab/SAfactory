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

挂载约束（依据 Harbor 上游 `ServiceVolumeConfig` 与 adapter 安装逻辑）：

- `--mounts` 中每项的 `read_only` 只能省略或为 `true`：schema 类型是
  `Literal[True]`，写 `false` 会校验失败。target 为文件的 bind 必须为 `true`，
  否则 Harbor 在 `compose up` 后对可写挂载执行 `mkdir -p`，破坏文件路径。
  runner 生成的挂载固定为 `read_only: true`。
- `agent_runtime.version` 必须与挂载 binary 自报的版本（`claude --version` /
  `codex --version` 中的 `x.y.z`）完全一致。一致时 adapter 跳过安装；不一致时会
  联网安装 CLI，RJob 无公网时安装失败；安装成功的 `$HOME/.local/bin/claude` 还会因
  PATH 前置而遮蔽挂载的 `/usr/local/bin/claude`。版本号写成带引号的三段式字符串，
  例如 `"2.1.228"`。

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

## 9. runner.py workflow

runner 以 `/opt/harbor-env/bin/python runner.py` 在 RJob 容器内作为 episode 入口运行，
`main()` 按固定顺序把一条请求跑成一个 SimulationStartResult：

1. **注册信号处理** —— 给 `SIGTERM`/`SIGINT` 装 handler；收到即 `nested.cleanup()`
   停掉 Harbor 与 dockerd 进程组后退出，保证被 launcher 取消时不留孤儿 daemon。
2. **读取请求** —— `read_request()` 从 `SAFACTORY_START_REQUEST_JSON` 解析 JSON；
   先单独取出 `session_id` 和 `env_params.dataset.task_id`，使后续 resolve 阶段的
   失败也能带上身份信息。
3. **解析 RunSpec** —— `resolve_run_spec()` 把 `env_params` 规整成一个 `RunSpec`：
   `resolve_task_record()` 按约定定位 task 目录（见 §3），解析 Gateway URL（§5）、
   `agent`/`model`/`reasoning_effort`、挂载（`agent_runtime`/`codex_config_dir`）、
   `agent_env`、`runtime_env`、`reward_key`、`timeout_s`，并算出 episode 目录、
   Harbor `jobs` 根和 job 名。
4. **校验 task** —— 建 `episode_dir`，`validate_task_record()` 检查 task 目录存在且
   含 `task.toml`、`instruction.md`、`tests/test.sh` 和 `environment/`，缺任一项即失败，
   不启动 dockerd。
5. **起 nested dockerd** —— `nested.start()` 要求 root 和 `/dev/fuse`，以
   `--storage-driver=fuse-overlayfs --data-root=/docker-data` 启动 dockerd，最多重试
   120 次（每轮 `sleep 0.5s`、每次 `docker info` 超时 5s）等 daemon 就绪，并校验实际
   driver 为 `fuse-overlayfs`。daemon 连接只通过显式的 Unix `DOCKER_HOST` 暴露给后续步骤。
6. **加载镜像** —— `load_task_images()` 以隔离模式（`python -I`）运行 bundle 自带的
   `bin/load_shared_images.py` 和 `bin/load_image.py`，经 `DOCKER_HOST` 接到本 episode
   的 daemon；每个 loader 超时 `1800s`，输出写 `infra/image-load.log`。runner 不解析
   归档格式，镜像加载由 bundle 脚本负责（见 §3）。
7. **写准备回执** —— `infra/preparation.json` 记 `status=prepared` 及 task 身份、路径，
   作为“已进入 Harbor 前”的落点。
8. **运行 Harbor** —— `run_harbor()` 用 daemon env 叠加 `model_connection_env()`
   （`*_BASE_URL`=Gateway、`*_API_KEY=EMPTY`）和 `runtime_env`，`Popen` 执行
   `harbor_command()`（§5），`wait(timeout_s)`。超时则对进程组发 SIGTERM，返回码记
   `124` 并标记 `timed_out`。Harbor 日志写 `harbor/harbor-run.log`。
9. **取消收尾（仅超时）** —— `wait_for_cancellation_cleanup()` 最多轮询 `30s`，等
   `result.json`、trajectory 和 `manifest.json` 的大小/mtime 连续两次不变，确保 Harbor
   的异步取消写入稳定后再解析。
10. **解析结果** —— `parse_harbor_result()` 读 job `result.json` 和唯一 trial
    `result.json`，取 `verifier_result.rewards[reward_key]` 作 reward，收集 trial 与各
    step 的 `exception_info` 为 `fatal_errors`，据此定 `status`：超时为 `truncated`、
    无致命错误且有数值 reward 为 `succeeded`、否则 `failed`（失败/超时 reward 记 0）。
11. **异常兜底与清理** —— 任一步抛异常走 `_failure()`，带上已知 `task_id` 和 `timed_out`；
    `finally` 里 `nested.cleanup()` 停 Harbor 与 dockerd。
12. **落盘与打印** —— 把结果写到 `SAFACTORY_RESULT_PATH`（原子替换），并在 stdout 打印
    `SAFACTORY_RESULT_JSON <json>` 供 launcher 回退解析。`main()` 恒返回 0，成败由结果
    JSON 的 `status` 表达。

## 10. Harbor 镜像内部 workflow

runner 调用的 `harbor run` 在同一个运行时镜像内执行（`harbor_config.rjob.yaml` 的
`env_image`，含 `/opt/harbor-env` 下的 Harbor CLI 与 Python）。Harbor 把 `--path`
指向的 task 跑成**一个 trial**，阶段划分与各阶段起止时间可在 trial `result.json` 中看到
（`environment_setup` / `agent_setup` / `agent_execution` / `verifier`），各阶段预算由
task 自身 `task.toml` 声明：

1. **读取 task** —— 解析 `task.toml`：`[environment]` 的 `docker_image` 与
   `build_timeout_sec`、`[agent].timeout_sec`、`[verifier]` 的 `topology` /
   `environment_mode` / `timeout_sec`，以及 task 目录内的 `docker-compose.yaml`、
   `instruction.md`、`tests/test.sh`、`environment/`。
2. **environment_setup** —— 以 `--env docker` 在 nested daemon 上拉起 Compose 服务、
   网络与容器，挂载 task 工作目录。镜像引用的 tag 必须已由 runner 调用的 bundle loader（§3、§9 第 6 步）加载进 daemon，
   Harbor 自身不加载归档；
   `environment_mode=shared` 表示 Agent 与 verifier 共用同一套环境容器。
3. **agent_setup** —— 把 Agent runtime 以只读 bind 挂进 Agent 容器
   （`--mounts`，如 claude-code 的 binary 挂到 `/usr/local/bin/claude`），注入经
   `--ae` 传入的 `agent_env`，并准备 Agent 自身的配置目录（例如 `CLAUDE_CONFIG_DIR`、
   codex 的 `CODEX_HOME`）。
4. **agent_execution** —— 在 Agent 容器内启动 Agent 适配器（claude-code 形如
   `claude --output-format=stream-json --print …`），模型请求经 `*_BASE_URL` 指向的
   Gateway 转发；Agent 在环境容器里按 `instruction.md` 执行任务。过程写
   `agent/claude-code.txt`、`agent/trajectory.json`、`agent/sessions`。
5. **verifier** —— 运行 task 的 `tests/test.sh`，产出 `verifier_result.rewards`
   （落 `verifier/reward.txt`、`verifier/test-stdout.txt`）；`environment_mode` 决定校验
   是复用 shared 环境还是起 isolated 容器。runner 从此处的 `rewards[reward_key]` 取分。
6. **汇总产物** —— Harbor 写 job 级 `result.json`（`stats`、`evals`、`reward_stats`）、
   trial 级 `result.json`（含各阶段起止、`agent_result` 的 token/cost、
   `verifier_result`）、`artifacts/manifest.json` 及各容器日志，供 runner 回读（§8）。

单个 trial 目录（`harbor/jobs/<job-name>/<trial-name>/`）布局：

| 路径 | 内容 |
| --- | --- |
| `result.json` / `config.json` / `lock.json` | trial 结果、展开后的运行配置、并发锁 |
| `trial.log` | Harbor 对本 trial 的编排日志 |
| `agent/` | `claude-code.txt`、`trajectory.json`、`sessions/`、`setup/` 等 Agent 产物 |
| `verifier/` | `reward.txt`、`test-stdout.txt` 等 verifier 产物 |
| `artifacts/` | `manifest.json` 与收集到的容器 `logs/` |

Harbor 只按 task 定义创建环境、Agent 与 verifier 并产出 reward；它不从 dataset 推导
task、不猜错误镜像名、也不补全缺失归档——这些在进入 `harbor run` 前已由 runner 和
bundle loader 处理完毕。
