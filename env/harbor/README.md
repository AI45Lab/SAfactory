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
| [runner.py](runner.py) | 解析 request、定位 bundle 任务、准备 nested Docker、委派镜像加载、渲染 job 模板并运行 Harbor、按证据定状态与 reward |
| [rule_evaluator.py](rule_evaluator.py) | 校验 Harbor 原生 reward 并归一化到 SAfactory 0–10，透传 `[GRADE]` 判分项 |
| [templates/](templates/) | Harbor job 模板：agent、model、`kwargs`、`env`、`mounts`、超时倍数（静态部分）。每个批次一份 |
| [templates/verify_harbor_run.sh](templates/verify_harbor_run.sh) | 不含 SAfactory 框架的复现脚本，执行同一条 `harbor run -c` |
| `*_config.rjob.yaml` | dataset、bundle、`harbor_template` 和 runner 参数（`env_params`） |
| `*_start.rjob.yaml` | RJob 资源、挂载、容器环境、runner 注入和模板投递（`embedded_files`） |

## 3. 自描述 bundle 与任务定位

runner 把上游交付的 bundle 视为自描述目录，按约定定位单个任务：

```text
<bundle_root>/<task_id>/<profile_subdir>        ← Harbor --path 指向的 task 目录
<bundle_root>/bin/load_shared_images.py         ← legacy 模式必需，加载共享镜像归档
<bundle_root>/bin/load_image.py                 ← 两种模式都必需，加载本任务镜像
<bundle_root>/<shared_images_subdir>/           ← 共享镜像归档目录（仅 legacy 模式使用）
```

- `task_id`、`profile_subdir`、`shared_images_subdir` 都必须是单个安全路径段（无分隔符、
  无 `.`/`..`、不以 `-` 开头）。task 目录本身可以是指向上游交付的 symlink。
- `bundle_root` 必须是已存在的绝对目录（不能是文件系统根），且 `bin/` 下存在
  `image_load_mode` 所需的 loader（legacy 两个，task 只需 `load_image.py`）。
- task 目录必须包含 `task.toml`、`instruction.md`、`tests/test.sh` 和 `environment/` 目录，
  缺任一项 runner 在启动 Harbor 前失败。

镜像加载委派给 bundle 自带脚本（gzip 流式、`pre.tar`/`post.tar` 内层归档、overlay 深度
flatten 都由脚本处理）。runner 只把它们接到本 episode 的 dockerd（通过 `DOCKER_HOST`），
并以隔离模式（`python -I`）运行，因为 bundle 是不可信数据：

| loader | runner 传入参数 | 运行于 |
| --- | --- | --- |
| `bin/load_shared_images.py` | `--runtime-dir <bundle_root>/<shared_images_subdir>` | 仅 legacy |
| `bin/load_image.py` | `--root <bundle_root> --task <task_id>` | legacy、task |

`image_load_mode` 选择调用哪些 loader：

- `legacy`（默认）：先加载整个共享镜像目录，再调用 `load_image.py`。旧 bundle 和现有
  配置不需要改动。
- `task`：只调用 `bin/load_image.py --task <task_id> --root <bundle_root>`，由该脚本
  负责本任务需要的全部镜像。用于合并后的 PJ 任务集合（如
  `20261019_deliver/exploit_gym/pj-exploit-400`）：其 `bin/load_image.py` 按任务的
  `harbor-runtime-profile/.pj_source_bundle` 找到来源批次，再调用来源批次的 loader 或
  直接 `docker load` 声明的归档。task 模式不加载整个共享镜像目录（该集合约 40G），
  不传 `--load-shared`（该选项走 `sudo -n`，运行镜像内没有 sudo）。

## 4. Runner 参数（`env_params`）

| 参数 | 说明 |
| --- | --- |
| `bundle_root` | 自描述 bundle 的绝对路径（必填） |
| `dataset.task_id` | 本 episode 的任务身份，用于拼出 task 目录（必填，单个安全路径段） |
| `profile_subdir` | task 目录下的 profile 子目录，默认 `harbor-runtime-profile` |
| `shared_images_subdir` | 共享镜像归档子目录，默认 `_runtime`（部分 bundle 用 `shared-images`）；仅 legacy 模式使用 |
| `image_load_mode` | `legacy`（默认，两个 loader）或 `task`（只调用 `bin/load_image.py`），见 §3 |
| `runtime_env` | 透传给 Harbor 进程的公开环境变量；禁止覆盖模型凭据、`PATH`/`HOME`/`BASH_ENV`/`ENV`，以及 `DOCKER_`/`SAFACTORY_`/`LD_`/`PYTHON` 前缀；`HARBOR_*` 允许 |
| `harbor_template` | Harbor job 模板的绝对路径；缺省时按 `SAFACTORY_HARBOR_TEMPLATE` → `runner.py` 同级 `templates/harbor_config.yaml` → `/opt/safactory-harbor/templates/harbor_config.yaml` 查找 |
| `reward_key` | verifier reward 字段名，默认 `reward` |
| `timeout_s` | Harbor 超时，默认 900，必须为正 |

Agent 相关参数（`agent`、`agent_import_path`、`model`、`reasoning_effort`、
`capture_http`、`agent_runtime`、`codex_config_dir`、`agent_env`、
`agent_timeout_multiplier`）已从 `env_params` 移除，改由 job 模板声明。runner 仍收到
这些键时会直接报错，而不是静默忽略。

## 5. Harbor 命令

Runner 只做两件事：渲染模板、执行 `-c`。

```bash
/opt/harbor-env/bin/harbor run \
  -c "$EPISODE_DIR/harbor/harbor-config.yaml" \
  --yes
```

- 模板见 [templates/](templates/)，只有三个会话相关的占位符会被替换：
  `{{task_path}}`、`{{job_name}}`、`{{gateway_url}}`。其余内容（agent、model、
  `kwargs`、`env`、`mounts`、超时倍数）全部静态写在模板里，因此新增透传项只需改
  模板，不需要改 Runner。
- 模板由启动配置投递到 `/opt/safactory-harbor/templates/`（RJob 用
  `rjob.embedded_files`，Sandbox 用 `sandbox.embedded_files`），并用
  `SAFACTORY_HARBOR_TEMPLATE` 指明文件名。
- `jobs_dir` 在模板里写成相对路径 `harbor/jobs`：runner 把 Harbor 的 cwd 设为
  episode 目录，所以 job 仍然落在 `<episode>/harbor/jobs/`。
- `--yes` 只用于跳过 Harbor 的 host environment access 确认，与 `-c` 不冲突。
- **不能**再传 `--agent`：Harbor 一旦收到 `--agent` 会清空配置里的整个 `agents`
  列表（`harbor/cli/jobs.py` 中 `config["agents"] = []`），模板内容随之失效。
- `agents[].override_timeout_sec` 没有对应的 `harbor run` flag（只有 `trials` /
  `exec` 子命令有 `--agent-timeout`），这也是必须走配置的原因之一。
- 已用运行镜像的 Harbor 0.23.0 对全部模板跑过 `harbor run --print-config -c`，
  确认 `override_timeout_sec`、`environment.mounts[].read_only`、`tasks[].path`
  均能通过校验。

单独验证同一条命令：`templates/verify_harbor_run.sh` 不含任何 SAfactory 框架代码，
只做与 Runner 相同的渲染与执行，可直接交给沙箱团队复现。

挂载约束（依据 Harbor 上游 `ServiceVolumeConfig` 与 adapter 安装逻辑）：

- `environment.mounts` 每项的 `read_only` 只能省略或为 `true`：schema 类型是
  `Literal[True]`，写 `false` 会校验失败。target 为文件的 bind 必须为 `true`，
  否则 Harbor 在 `compose up` 后对可写挂载执行 `mkdir -p`，破坏文件路径。
- `agents[].kwargs.version` 必须与挂载 binary 自报的版本（`claude --version` /
  `codex --version` 中的 `x.y.z`）完全一致。一致时 adapter 跳过安装；不一致时会
  联网安装 CLI，RJob 无公网时安装失败；安装成功的 `$HOME/.local/bin/claude` 还会因
  PATH 前置而遮蔽挂载的 `/usr/local/bin/claude`。版本号写成带引号的三段式字符串，
  例如 `"2.1.228"`。

Gateway 连接：模板的 `agents[].env` 里 `ANTHROPIC_BASE_URL` 由 `{{gateway_url}}`
渲染；`ANTHROPIC_API_KEY` 写成 `"${ANTHROPIC_API_KEY:-EMPTY}"`，由 Harbor 从进程环境
解析，默认 `EMPTY`。runner 同时把 `*_BASE_URL` / `*_API_KEY` 放进 Harbor 进程环境
（`model_connection_env`），前缀按模板声明的 agent 选择：`claude-code` 用
`ANTHROPIC_*`，`codex` / `opencode` 用 `OPENAI_*`，`openhands` 用 `LLM_*`。继承的
同名模型凭据会先被清除。Gateway URL 依次取
`SAFACTORY_GATEWAY_SESSION_URL_CONTAINER`、`SAFACTORY_GATEWAY_SESSION_URL`，再回退到
`gateway_base_url/session_id`。

## 6. Codex 配置目录

Codex 的基础 `config.toml` 及其附属文件（例如模型目录 JSON）在模板里声明：

```yaml
agents:
  - name: codex
    kwargs:
      config: /opt/safactory-codex-config/config.toml
```

模板同时需要把源目录只读挂到该 target：

```yaml
environment:
  mounts:
    - type: bind
      source: /abs/path/to/codex-config
      target: /opt/safactory-codex-config
      read_only: true
```

- Harbor 在宿主侧读取该 `config.toml`，合并本 session 的 `OPENAI_BASE_URL`，写入
  Task 内可写的 `CODEX_HOME`；源目录只读挂载，不作为 `CODEX_HOME`。
- `config.toml` 中引用附属文件时，使用 Task 内路径
  `/opt/safactory-codex-config/<file>`。
- 该目录必须在 RJob 内可见：start YAML 需将对应 GPFS 路径挂入 RJob 的同一路径。

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
- `<episode>/harbor/`：`harbor-config.yaml`（本次渲染的 job 配置）、`harbor-run.log`
  和 Harbor job 目录（`jobs/<job-name>/`）。

### 8.1 状态与 reward

`status` 只取三个值，含义固定（判定表见下方）：

| status | 含义 | `total_reward` |
| --- | --- | --- |
| `succeeded` | 沙箱启动、agent 运行、verifier 判分、reward 写出四步全部正常完成 | verifier reward（[0,1]） |
| `truncated` | agent 已开始运行，之后被时限或轮次上限截断 | `null` |
| `failed` | 其他所有情况（含判分未产出有效 reward、agent 因 API 错误中断） | `null` |

要点：

- **任何一层的返回码都不是状态依据**：`harbor run` 对 trial 失败恒返回 0，test.sh
  的退出码 Harbor 不读取，runner 进程恒返回 0。
- **有效 reward 由 runner 直接读文件判断**，不只看 trial 的 `verifier_result`：
  要求 `verifier.started_at` 非空、`reward.json`（优先）或 `reward.txt` 存在、文件
  mtime 不早于 `verifier.started_at`（排除 agent 预先写入的旧文件）、内容按 Harbor
  规则可解析、值在 `[0,1]` 内。任一不满足即 `harbor_reward=null`、
  `termination=verifier_failed`、`failure_stage=verifier`。**不写 0 分顶替。**
- `0` 分只表示「agent 做了但没做对」；判分无法完成记 `failed`。
- 判定顺序（命中第一条即停）：

  | 序号 | 条件 | status | termination | failure_stage |
  | --- | --- | --- | --- | --- |
  | R1 | runner 准备阶段异常 | failed | `prepare_failed` | `prepare` |
  | R2 | runner 时限触发，agent 已开始 | truncated | `runner_timeout` | — |
  | R3 | runner 时限触发，agent 未开始 | failed | `runner_timeout` | 停留的阶段 |
  | R4 | 无 trial 目录或无 trial result | failed | `harbor_error` | `harbor_start` |
  | R5 | `environment_setup` 未结束 | failed | `env_failed` | `environment` |
  | R6 | `agent_setup` 未结束或 agent 未开始 | failed | `agent_setup_failed` | `agent_setup` |
  | R7 | agent 结束记录为 `completed` | 进 R11 | — | — |
  | R8 | `max_turns` | truncated | `max_turns` | — |
  | R9 | `exception_type=AgentTimeoutError` | truncated | `agent_timeout` | — |
  | R10a | 有解析器的 agent 结束原因是 `api_error` | failed | `api_error` | `agent` |
  | R10b | 无解析器且无 agent 阶段异常 | 进 R11 | — | — |
  | R10c | 其余 agent 阶段异常 | failed | `agent_error` | `agent` |
  | R11 | reward 有效 | succeeded | `completed` | — |
  | R12 | 其余 | failed | `verifier_failed` | `verifier` |

- **R7 先于 R9**：agent 已正常结束、只是进程晚于时限退出，按 succeeded 处理，
  `AgentTimeoutError` 记入 `harbor_warnings`。
- **API 错误只在未截断时生效**（R8/R9 先于 R10a）。运行中报错但重试后正常完成的任务
  由 R7 先命中，尾部重试痕迹只记 `agent_exit.retrying_at_end` 供统计。
- `harbor_fatal_errors` **非空当且仅当 `status=failed`**（`rule_evaluator.py` 以它为空
  作为计分前提）；succeeded / truncated 中的 Harbor 异常进 `harbor_warnings`。

### 8.2 agent 结束原因

`<trial>/agent/claude-code.txt`（或 `codex.txt`）是 stream-json 全量输出，长任务可达
数十 MB，因此只读末尾 64 KiB 内的最后 20 个非空行：

- **claude-code**：窗口内最后一条 `type=result` 记录。以 `terminal_reason` 为准
  （`completed` / `max_turns` / `api_error` / `other:<值>`），`subtype=error_max_turns`
  作后备；API 错误时 `subtype` 仍是 `success`，只看它必然误判。错误文本同时查
  `result` 字段与 `message.model="<synthetic>"` 的 assistant 记录，按
  `API Error:` 归类为 `model_unavailable` / `connection` / `rate_limit` /
  `session_closed` / `quota` / `server_error` / `other`，仅用于统计。
- **codex**：`turn.completed` → `completed`；`turn.failed` / `error` →
  `api_error`（如 `exceeded retry limit, last status: 429` 归为 `rate_limit`）。
- 没有解析器的 agent 记 `reason=unknown, parser=null`，按 R10b/R10c 回退到 Harbor
  的 `exception_info`；解析器内部任何异常都捕获为 `unknown` 并记入
  `harbor_warnings`，不影响 episode 正常写出。
- Harbor 0.23.0 的 `exception_type` 是对 agent 全量 stdout 做正则首命中，业务文本会被
  误分类为 `ApiRateLimitError`/`UnknownApiError`，所以 `exception_type` 只记录，不参与
  判定（例外是 R9 的 `AgentTimeoutError`，它来自 Harbor 自己的计时）。

### 8.3 细粒度判分项 `[GRADE]`

任务侧按沙箱规范逐项打印 `[GRADE] check=... status=... score=... msg=...` 与
`[GRADE] summary reward=...`。runner 解析 `<trial>/verifier/test-stdout.txt`，写入
`metrics.grade = {checks: [{check,status,score,msg}], summary, errors, source}`；没有
`[GRADE]` 行时为 `null`。`rule_evaluator.py` 把它放进 `EvalResult.artifacts.grade`，
`evaluator/reward_committer.py` 再写成 `meta_json.eval.grade`（与 `eval.status`、
`normalized_score_10`、`result` 同级）。

`metrics` 还带 `termination`、`failure_stage`、`agent_exit`、`harbor_exception`、
`stage_timing`、`evidence{trial_result,agent_log,verifier_stdout,reward_file}`，便于从
结果库回查文件。

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
   `harbor_template`、`runtime_env`、`reward_key`、`timeout_s`，并算出 episode 目录、
   Harbor `jobs` 根和 job 名。请求里若还带着已迁移到模板的 agent 参数，直接报错。
4. **校验 task** —— 建 `episode_dir`，`validate_task_record()` 检查 task 目录存在且
   含 `task.toml`、`instruction.md`、`tests/test.sh` 和 `environment/`，缺任一项即失败，
   不启动 dockerd。
5. **起 nested dockerd** —— `nested.start()` 要求 root 和 `/dev/fuse`，以
   `--storage-driver=fuse-overlayfs --data-root=/docker-data` 启动 dockerd，最多重试
   120 次（每轮 `sleep 0.5s`、每次 `docker info` 超时 5s）等 daemon 就绪，并校验实际
   driver 为 `fuse-overlayfs`。daemon 连接只通过显式的 Unix `DOCKER_HOST` 暴露给后续步骤。
6. **加载镜像** —— `load_task_images()` 以隔离模式（`python -I`）运行 bundle 自带的
   loader（legacy：`bin/load_shared_images.py` 和 `bin/load_image.py`；task：只运行
   `bin/load_image.py`），经 `DOCKER_HOST` 接到本 episode
   的 daemon；每个 loader 超时 `1800s`，输出写 `infra/image-load.log`。runner 不解析
   归档格式，镜像加载由 bundle 脚本负责（见 §3）。
7. **写准备回执** —— `infra/preparation.json` 记 `status=prepared` 及 task 身份、路径，
   作为“已进入 Harbor 前”的落点。
8. **运行 Harbor** —— `run_harbor()` 先 `render_job_template()` 把模板渲染成
   `<episode>/harbor/harbor-config.yaml`，再用 `rendered_template_agent()` 回读
   `agents[0].name` 决定凭据前缀；随后用 daemon env 叠加 `model_connection_env()`
   （`*_BASE_URL`=Gateway、`*_API_KEY=EMPTY`）和 `runtime_env`，`Popen` 执行
   `harbor_command(config_path)`（§5），`wait(timeout_s)`。超时则对进程组发 SIGTERM，
   返回码记 `124` 并标记 `timed_out`。Harbor 日志写 `harbor/harbor-run.log`。
9. **取消收尾（仅超时）** —— `wait_for_cancellation_cleanup()` 最多轮询 `30s`，等
   `result.json`、trajectory 和 `manifest.json` 的大小/mtime 连续两次不变，确保 Harbor
   的异步取消写入稳定后再解析。
10. **解析结果** —— `parse_harbor_result()` 按 §8.1 的 R1–R12 判定表定 `status`：
    trial 目录按目录发现（超时被杀时 `result.json` 可能没落盘），agent 结束原因由
    §8.2 的解析器从尾部窗口得出，reward 由 §8.1 的五项检查从 reward 文件本身得出，
    `[GRADE]` 由 §8.3 从 `test-stdout.txt` 解析。`total_reward` 仅 succeeded 有值。
11. **异常兜底与清理** —— 任一步抛异常走 `_failure()`（`termination=prepare_failed`、
    `total_reward=null`），带上已知 `task_id` 和 `timed_out`；`finally` 里
    `nested.cleanup()` 停 Harbor 与 dockerd。
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
