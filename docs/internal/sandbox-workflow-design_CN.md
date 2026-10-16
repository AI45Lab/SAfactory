# Sandbox 工作流与环境改造方案

状态：非 ENV 框架已实现；本轮未修改 `env/`。本文件保留环境迁移设计，环境专用 YAML、runner 的 v2 发布和 Sandbox 内评分仍待接入。本文中的 Harbor 配置是迁移示例，不代表当前 Harbor 已可直接运行；尚未执行真实集群端到端验收。当前接口见 [非 ENV 实现说明](sandbox-implementation-plan_CN.md)。

Sandbox mode 使用独立的三份配置。H 集群保留 launcher、Gateway 和数据存储；Sandbox 运行 runner、原生任务、验证与评分。多个 runner 和辅助文件通过 API 上传，输入数据经 WebDAV 发布至 AFS，结果目录以 `readOnly: false` 挂载。最终评分只从 SDK 下载到 H 的标准结果 JSON 读取。

本方案替代先前“主控从 stdout 解析评分”和“WebDAV 回收评分文件”的方式。环境原生评分政策保持一致，Docker/RJob 的既有调用方式保持兼容。

## 配置文件与启动参数

文件命名固定为：

```text
sandbox-config.yaml
env/<env>/<env>.config.sandbox.yaml
env/<env>/<env>.start.sandbox.yaml
```

例如 Harbor 使用 `env/harbor/harbor.config.sandbox.yaml` 和 `env/harbor/harbor.start.sandbox.yaml`。不再依赖名称为 `.rjob.yaml` 的配置，也不从 Docker bind mount 推断 AFS 挂载。

### 平台配置

使用既有模板，不新增第二份平台模板：

```bash
cp config.sandbox.example.yaml sandbox-config.yaml
```

编辑 project、Environment 默认值（或在 env 启动文件指定）、Gateway 地址以及 AFS 设置。以下为长任务的 `sandbox-config.yaml` 示例；既有模板中的短任务超时默认值保持不变：

```yaml
sandbox:
  domain: deepenv.sandbox.pjlab.org.cn
  protocol: http
  project: your-project
  api_key_env: OPEN_SANDBOX_API_KEY
  gateway_base_url: http://your-gateway:8080/v1/sessions

  command_port: 44772
  use_server_proxy: true
  request_timeout_s: 60
  create_timeout_s: 600
  command_timeout_s: 39600
  lifecycle_minutes: 720
  cleanup_on_finish: true

  afs:
    webdav_url: http://10.140.95.76:28081

  results:
    local_root: ./results
    download_timeout_s: 60
    download_retries: 3
```

`local_root` 相对 launcher 工作目录解析为绝对路径，所有 session 都使用同一个根目录。WebDAV 仅用于 H 发布输入和准备 AFS 子目录，复用项目与 AK/SK；评分文件必须经 SDK 下载。AFS 发布逻辑直接实现在 `clusters/sandbox_cluster.py`，H 端需安装外部 rclone；不调用 Bash 迁移脚本，也不导入 `sandbox/` 中的 Python。镜像须提供 execd、`sh`、`base64`、`sha256sum` 和业务依赖。框架检查 H 到 Gateway 的 readiness/模型路由；Sandbox 到 Gateway 的网络、镜像依赖和原生验证须通过真实 env smoke 确认。

SDK 连接存续期间自动续期；实例因失败或调试而保留、连接关闭后不再自动续期，仍受平台 TTL 限制。输入发布、文件物化/manifest、WebDAV 锁、完整挂载列表和 AFS volume 兼容类型均实现在 `clusters/sandbox_cluster.py`，没有单独的 AFS 传输服务。`sandbox/` 目录只作开发参考，可以从部署包中完全省略。

Gateway 地址只解析一次：显式 `--gateway-base-url` 优先，其次使用本文件的 `sandbox.gateway_base_url`。解析结果统一写入主控的 `cfg.gateway_base_url`，同时注入 Sandbox 请求；per-env 配置不再另行覆盖。CLI 未显式传值时不能让旧的 localhost 默认值盖过平台 YAML；未配置可达地址则启动失败。

既有 Gateway 配置必须使用严格同步写入，并与 launcher 指向同一实际存储。合并以下设置到现有 Gateway YAML，不要写进 Sandbox 平台 YAML：

```yaml
telemetry:
  mode: strict
  async_cloud_writes: false
storage_config:
  enable_buffer: false
  enable_upsert_batching: false
  # SQLite 场景还需与 launcher --db-path 使用同一 db_url
```

`/readyz` 返回实际 `durability` 和 `storage_identity` 供主控校验；仅修改文件而未重启/应用 Gateway 配置不算满足条件。历史 telemetry 写入失败会留在 session 标记中，close 不能报告 flushed。当前实现不增加 durable outbox。

### 环境业务配置

以下 `env/harbor/harbor.config.sandbox.yaml` 待环境接入时创建；本轮没有写入 `env/`：

```yaml
environments:
  - env_name: harbor
    env_image: "<harbor-sandbox-runtime-image>"
    env_num: 1
    dataset: ./datasets/tasks.jsonl
    dataset_load_mode: eager

    env_params:
      task_family: harbor
      agent: claude-code
      model: harbor_model
      bundle_type: vulhub-local
      bundle_package_dir: /opt/harbor-data
      reward_key: reward
      timeout_s: 36000
```

`dataset` 相对本业务配置文件解析。`env_image` 必须是 Sandbox 可拉取的镜像 URI，并与 Environment 一致，不能使用本地 Docker image ID。`env_params` 中的文件路径均为 Sandbox 路径；必要时将 AFS 挂到原来 runner 使用的绝对路径，以保留原业务参数。

### 环境启动配置

以下 `env/harbor/harbor.start.sandbox.yaml` 同属待迁移示例，须先完成文末的 env 评分接口改造：

```yaml
agent_name: harbor

sandbox:
  environment_id: env-harbor
  workdir: /work

  runner_entrypoint:
    source: ./runner.py
    target: /tmp/safactory-harbor/runner.py
    command: >-
      /opt/harbor-env/bin/python
      /tmp/safactory-harbor/runner.py

  embedded_files:
    - source: ./bundle_materializer.py
      target: /tmp/safactory-harbor/bundle_materializer.py
      permissions: "0644"
    - source: ./image_archive_loader.py
      target: /tmp/safactory-harbor/image_archive_loader.py
      permissions: "0644"
    # 复用原评分模块，增加无主控依赖的 score_result 入口
    - source: ./rule_evaluator.py
      target: /tmp/safactory-harbor/rule_evaluator.py
      permissions: "0644"
    # 各 env 共用的标准库结果协议文件
    - source: ../../core/result_artifact.py
      target: /tmp/safactory-harbor/result_artifact.py
      permissions: "0644"

  env:
    PYTHONUNBUFFERED: "1"
    PYTHONDONTWRITEBYTECODE: "1"

  resource:
    cpu: "4"
    memory: "16Gi"

  extensions:
    runtimeClassName: sysbox-runc
    fuse: "true"

  storage:
    inputs:
      - host_path: /mnt/shared-storage-user/your-user/harbor-data
        afs_subpath: safactory/harbor/inputs/v1
        mount_path: /opt/harbor-data
        read_only: true
    outputs:
      - afs_subpath: safactory/harbor/results
        mount_path: /app/results
        read_only: false
```

启动配置完全使用 `sandbox:` 语义，不再使用 `container:`、`rjob:` 或 `mount_config`。每个 `source` 必须是文件，路径相对启动配置解析；目录不作为上传 source。入口文件自动加入上传列表，不需要在 `embedded_files` 重复声明。启动前检查声明文件，上传后回读校验并设置权限；运行与验证依赖由 env 准备在镜像中，需要分发的本地模块必须列全。

`storage.inputs` 的 `host_path` 表示 H 源目录；相对路径以启动 YAML 所在目录为基准。`afs_subpath` 是项目 AFS 相对目录，`mount_path` 是 Sandbox 路径。输入版本首次上传并校验，之后复用；同一版本内容不一致时停止启动，更新版本目录后再发布，运行期间不覆盖输入。

`storage.outputs` 不设置 H `host_path`：H 结果目录统一由平台配置的 `results.local_root` 决定，并通过 SDK 下载。框架将 `read_only` 转成 API 的 `readOnly`；输出必须为 `false`，且包含 `/app/results`。创建请求传完整 `volumes`，覆盖 Environment 的旧挂载，不再依赖自动 `/sandbox-data`。确实需要该路径的 env 须显式声明。

Environment 提供运行镜像、execd 服务及 `44772` 端口。创建实例时应用启动配置中的资源、扩展和完整挂载列表，不修改共享 Environment。

### 启动命令

完成上述 env 迁移、镜像和 Gateway 准备后，使用同一个 launcher：

```bash
export OPEN_SANDBOX_API_KEY='<ak>:<sk>'

python launcher.py \
  --mode sandbox \
  --sandbox-config sandbox-config.yaml \
  --agent-config env/harbor/harbor.config.sandbox.yaml \
  --agent-start-config env/harbor/harbor.start.sandbox.yaml \
  --job-id harbor-sandbox-001 \
  --llm-model harbor_model \
  --agent-start-timeout-s 39600 \
  --pool-size 1 \
  --multiplier 1 \
  --max-workers 1
```

`harbor_model` 必须替换为已有 Gateway 路由。任务、模型、并发、storage 参数沿用 launcher 契约；Gateway 和 launcher 使用同一实际存储后端。Sandbox mode 自动启用必需的文件评分，不依赖额外的 `--enable-evaluation`；传入该参数也不会在 H 重跑 env evaluator。

RL 使用 `AIEVOBOX_MODE=sandbox`、`AIEVOBOX_AGENT_CONFIG`、`AIEVOBOX_AGENT_START_CONFIG`；`AIEVOBOX_SANDBOX_CONFIG` 已透传平台配置路径。未指定 start 文件时支持 `.config.sandbox.yaml` 到 `.start.sandbox.yaml` 推导。框架要求显式选择业务/start 配置，避免扫描并同时加载多模式配置。

实际预热实例数为 `ceil(pool_size × multiplier)`；示例显式设置 `--multiplier 1`，避免 pool-size=1 时默认预热两个实例。job/session 路径身份只接受与框架 safe_path_part 一致的字符，避免标准结果目录和 runtime.json 目录分裂。

## Workflow

1. H 主控读取并检查三份配置、输入文件、上传文件、镜像、Gateway 路由和存储身份。
2. 每个 job 统一准备输入 AFS 版本目录，校验上传字节；创建可写的 AFS 输出目录。
3. 创建 Sandbox 并等待 execd 可用；输入按配置挂载，输出明确 `readOnly: false`。实际验证输入读取和结果目录写入。
4. 通过 API 上传入口和所有辅助文件，校验内容、设置权限；然后创建 session，运行 runner。
5. runner 在 Sandbox 内完成原生任务、必要验证和 env 评分，原子发布 `SAFACTORY_RESULT_PATH` 指定的标准 JSON。成功、失败和正常捕获的超时均写出对应状态。
6. runner 结束后，主控使用同一实例的 SDK 文件服务下载结果；即使命令非零退出，也先尝试下载。stdout/stderr 只用于日志诊断。
7. 本地临时文件通过 JSON、版本、任务身份及评分字段校验后，原子更名至标准位置。Gateway 完成 drain、flush 与底层持久化确认。
8. 通用文件 evaluator 重新读取本地标准 JSON，只取 `evaluation`；按下文状态规则处理，只有可提交的评分才调用 RewardCommitter。确认成功提交后才完成对应 session；失败状态不标记为成功。
9. 释放可回收的 Sandbox。下载按 download_retries 重试；仍失败或 reward 提交中断时，使用同一 job 的 `--resume` 恢复对应阶段。AFS 产物保留，不因下载失败重跑已完成的原生任务。

实例被强制终止而无法生成最终文件时，记录执行或回收失败，保留已有原生文件；不伪造标准结果文件和成功零分。resume 先阻断旧实例继续写入，再处理未完成任务。

## SDK 下载和本地目录

标准文件名沿用现有 `safactory_result.json`，不新增 `safactory-json` 文件名。文件路径固定为：

```text
Sandbox /app/results/<job_id>/<session_id>/safactory_result.json
    ↓ OpenSandbox SDK files.read_bytes
H <local_root>/<job_id>/<session_id>/safactory_result.json
```

用户所说的 `results/jobid/sessionid/json` 对应最后一级 JSON 文件，不额外创建 `json/` 子目录。例如：

```text
results/
└── harbor-sandbox-001/
    └── session-001/
        ├── safactory_result.json
        └── artifacts/             # 可选的原生日志与诊断文件
```

使用已锁定的 OpenSandbox 0.1.9 接口：在 runtime handle 增加 `files`，通过 `AdapterFactory.create_filesystem_service(endpoint)` 创建文件服务；`await files.read_bytes(remote_path)` 得到字节后写入 H。实际代码位于 `manager/sandbox_episode_runner.py`，不依赖参考目录中的下载示例。JSON 不通过 `cat`、stdout 或原始 HTTP proxy fallback 获取。

下载先写同目录临时文件，核验远端文件的大小和 SHA-256，并严格验证 JSON；成功后原子替换正式文件。SDK 文件信息提供大小；远端 SHA-256 在 writer 已确认停止后，通过 SDK 命令服务执行固定的 `sha256sum -- <result_path>` 获取，路径校验并正确转义。该命令输出仅作完整性校验，JSON 内容和评分仍只通过 SDK 文件服务取得。新执行的下载重试不得读取本地旧文件兜底；显式 resume 仅在可信 runtime.json 与身份匹配时恢复合法本地文件。当前自动回收只有标准 JSON，附属文件 SDK 回收属于后续可选扩展，不能成为 H 评分依赖。

框架统一注入以下内部环境变量，用户无需逐项填写：

| 变量 | 含义 |
| --- | --- |
| `SAFACTORY_RUNTIME` | 固定为 `sandbox` |
| `SAFACTORY_JOB_ID` / `SAFACTORY_SESSION_ID` | 本次任务身份 |
| `SAFACTORY_ATTEMPT_ID` | 本次业务执行标识；重新执行业务才变化，下载与提交重试沿用旧值 |
| `SAFACTORY_RESULT_PATH` | `/app/results/<job_id>/<session_id>/safactory_result.json` |
| `SAFACTORY_START_REQUEST_JSON` | 完整的现有请求契约 |
| `SAFACTORY_GATEWAY_SESSION_URL` | 本次任务的 Gateway session 地址 |

## 结果 JSON 与评分契约

新增 `evaluation` 承载已经完成的 env 评分。保留现有顶层 `SimulationStartResult` 字段；其历史 `total_reward` 尺度不统一，Sandbox 不使用该字段提交最终评分。例如 Harbor 原始得分 0.8：

```json
{
  "schema_version": "safactory.result.v2",
  "job_id": "harbor-sandbox-001",
  "session_id": "session-001",
  "attempt_id": "attempt-001",
  "status": "succeeded",
  "total_reward": 0.8,
  "step_count": 12,
  "terminated": true,
  "truncated": false,
  "error_text": null,
  "metrics": {
    "bench": "harbor",
    "harbor_reward": 0.8
  },
  "evaluation": {
    "session_id": "session-001",
    "status": "succeeded",
    "eval_id": "harbor_rule",
    "method": "rule_evaluator",
    "raw_score": 0.8,
    "normalized_score_10": 8.0,
    "normalizer_id": "harbor_reward_v1",
    "reason": "Harbor verifier reward converted to 0-10",
    "artifacts": {},
    "error_text": null
  }
}
```

唯一成功评分来源是本地文件中的 `evaluation.normalized_score_10`。H 不再读取 stdout、顶层 `total_reward` 或远端原生 metrics 推断分数，也不再次乘 10 或自动 clamp。

通用文件 evaluator 必须检查：

- schema、job、session、attempt 与当前执行完全一致；嵌套 evaluation 的 session 也一致。
- 顶层运行状态、`truncated` 和 evaluation 状态相容；缺失字段不能默认成功。
- 最终状态满足 `terminated or truncated`；允许 `terminated=False, truncated=True` 的 Gym 截断语义。
- 所有拟提交的分数，包括 truncated 分数，必须是 `[0,10]` 内的有限数字，排除 bool、NaN 和 Infinity。
- 成功零分必须明确为 `evaluation.status=succeeded`，不能由缺失评分、基础设施失败或下载失败推断。
- failed/timeout/truncated 保留各自状态。只有文件明确提供的截断评分才按既有截断政策提交；不再走主控无文件固定补零的路径。

`evaluation` 尽量复用 `EvalResult` 字段；`normalizer_id` 记录 env 的评分政策版本。原始评分证据放入 JSON 的 metrics/artifacts，使 H 单文件读取即可完成提交。附属日志可用于人工诊断，但不影响权威评分来源。

| 文件中的 evaluation 状态 | H 侧处理 |
| --- | --- |
| `succeeded` | 校验分数、提交 reward，然后标记成功完成；0 分同样需要成功状态。 |
| `truncated` | 仅提交文件中明确提供且校验通过的分数，保留截断状态；通常由 env 按原政策明确写 0，主控不补分。 |
| `failed` / `timeout` | 文件保留具体原因；主控运行摘要按失败处理，不调用 RewardCommitter、不标记成功。失败占位 0 不作为成功得分。 |
| 文件缺失、下载失败、格式或身份不合法 | 记录执行或回收失败，等待恢复；不合成成功结果或评分。 |

## Env 侧待办（本轮未修改 env/）

下面均为后续迁移任务，本轮没有修改这些 runner、evaluator 或新增 env 专用 YAML。各 env 的原生评分含义保持不变：在原 `rule_evaluator.py` 内抽出可在 Sandbox 使用的 `score_result()`，旧入口延迟导入主控类型并调用同一函数。Sandbox runner 将 scorer 交给共用 `finalize_result()` 调用一次，再原子发布；不先自行调用造成重复验证。不为每个 env 新建 `result_adapter.py`。需要模型调用或环境访问的验证必须在 Sandbox 完成，不能只搬运分数换算公式。

| 范围 | 需要修改的内容 |
| --- | --- |
| 所有接入 env | 新建 `<env>.config.sandbox.yaml`、`<env>.start.sandbox.yaml`；runner 强制写 `SAFACTORY_RESULT_PATH`，补版本、job/session/attempt 和 evaluation；先临时写再原子 rename。文件发布失败须显式失败，不吞掉错误。 |
| runner 和上传依赖 | 入口、辅助模块、评分模块全部列全；Python import 目录保持一致；可执行文件恢复权限。Sandbox 分支可通过注入的 `SAFACTORY_RUNTIME` 选择，避免改变旧 mode 的输出政策。 |
| Harbor | 在 `runner.py` 封装结果前复用 `rule_evaluator.py` 的 rollout/fatal errors 检查与 `harbor_reward` 归一化；0–1 原始分只转换一次；保留 warnings/fatal errors 区分和原始 rewards。 |
| OpenRT | 在 runner 内读取并校验本次原生 metrics，按现有 `insucess_rate` 聚合政策生成 evaluation。现有 `total_reward` 已可能是十分制，不能再乘 10；指标缺失或不可解析应失败。 |
| CyberGym | 修改 `runner_lib/result_writer.py` 的最终写入阶段，在验证完成后于 Sandbox 本地读取 verified `poc.db`，沿用 vulnerable/fixed 验证退出码与最终 PoC 的判分政策；将必要证据写入 JSON，H 不再打开远端数据库。 |
| CyberRange | 修改 `runner.sh` 的最终结果阶段及评分辅助模块，在 Sandbox 内读取 native result 和 milestones，沿用 E2E/里程碑、平台健康及证据状态政策；H 不再打开远端 artifact 路径重算。 |
| Geo3K | 原生 grader 后封装现有 score/passed 的十分制 evaluation，保留 ground truth、final answer 和 reward source，H 不再次判题。 |
| AgentCompass | 保留各 benchmark 的 binary/fractional/rubric 归一化策略；已经是 0–10 的研究评分不再缩放。 |
| PRMEval / DataElf Finance / DeepSafe Cyber | 分别复用 MSE 非线性转换、已有 normalized_score_10、terminal code 政策；不能套用统一乘 10。 |
| ExploitGym / LiveCVEBench | 在 runner 尾阶段完成既有基础设施检查与原生评分归一化，写入 evaluation。 |
| PatchEval | 原 evaluator 会执行官方验证；必须将完整验证步骤及依赖安排到 Sandbox，验证结束再发布评分文件。 |
| 其他 env | 同样审查当前 runner、native verifier 和 evaluator，完成评分依赖迁移后才声明支持 Sandbox，不能仅生成 YAML 就认定可运行。 |

输入文件的绝对路径如果需要保持不变，在 `mount_path` 中配置原路径；需要本地 staging 的 env 继续使用本地工作目录，最终权威 JSON 必须落到指定的 AFS 结果挂载。

## 当前非 ENV 实现

| 模块 | 修改 |
| --- | --- |
| `args.py` / `manager/simulation_config.py` | Sandbox 默认平台配置名改为 `sandbox-config.yaml`；新增独立 sandbox 启动配置解析；支持 afs/results、runner_entrypoint、permissions、storage；不经 Docker 配置解释 Sandbox 语义。统一解析主控与 Sandbox 的 Gateway URL，区分显式 CLI 与旧默认值。 |
| `clusters/sandbox_cluster.py` | AFS prepare_storage、ProjectAfsVolume、完整 volumes、SDK files 上传/回读、续期和命令停止确认；runtime.json 和基于旧 attempt 的恢复。AFS 不拆出独立模块。 |
| `manager/sandbox_episode_runner.py` | stdout/stderr 仅诊断；命令结束或失败后通过 SDK 下载标准结果文件；移除 stdout/exit_code/cat 评分 fallback；返回严格校验的运行结果与本地文件路径。 |
| `manager/types.py` / `core/result_artifact.py` | StartResult 仅增加可选 result_artifact_path；共用严格 reader 校验真实身份后才 normalize_result，不复制 evaluation 到第二个 metadata 结构。 |
| `manager/simulation_worker.py` | Sandbox 选择通用文件 evaluator，不在 H 发现并运行 env evaluator；必需的文件评分自动启用；截断也必须有文件依据；确认 Gateway 和存储持久化后才提交并 finished。 |
| `evaluator` | 增加通用标准结果文件 evaluator，严格读取 evaluation 并复用 RewardCommitter；不新增一套 env 评分公式。 |
| `rl/buffer_server.py` / `rl/gateway_autostart.py` | 透传平台配置路径并支持新的 sandbox 文件名；自动启动的 Gateway 使用同步写入、关闭缓冲；不同时加载多模式配置。 |
| Gateway / resume | readyz 暴露实际写入能力、存储身份和进程 instance_id；close 校验真实 drain/flush/失败标记与进程身份，并把确认结果写回 runtime.json。缺少历史确认且原进程已消失时拒绝恢复评分。只对确认重跑的行删除轨迹，回收和待提交行保留轨迹。旧 writer 状态不明时不 clean；closing/closed 绑定不随 TTL 清除，须显式 clean。不提供跨 Gateway 重启的持久封禁。 |

## 环境接入后的验收要求

非 ENV 测试使用 fake SDK/rclone，验证框架行为；下列真实业务与集群验收仍待 env 迁移后执行，不能由框架单测替代。

另用 `tests/test_sandbox_checkout.py` 检查 Git index 包含必要运行时模块，并导出完全没有 `sandbox/` 的干净副本验证 launcher 导入，避免本地未跟踪文件掩盖部署缺失。

1. 多个上传文件能按原 import/执行路径工作，输入与输出挂载的 readOnly 值正确。
2. 即使 stdout 为空、截断或存在其他 JSON，仍能通过 SDK 下载标准文件并正确评分。
3. H 与 Sandbox 均具有 `job_id/session_id/safactory_result.json` 层级，评分仅读取本次下载并校验的文件。
4. Harbor、OpenRT 至少覆盖有效零分、非零分、失败、超时；最终分与原政策一致，且只归一化一次。
5. 缺文件、损坏 JSON、错 job/session/attempt、非有限分数、越界分数和 SDK 下载中断均不能产生成功评分，也不能使用旧本地文件兜底。
6. 下载或 reward 提交重试不重复原生 rollout；恢复前阻断旧实例写入。完成结果在回收 Sandbox 后仍可从 H 本地文件读取。
7. Docker/RJob 原有配置、结果读取与评分路径通过回归检查；原生证据只在 Sandbox 内读取完成评分，不引入 H 侧远端路径依赖。
