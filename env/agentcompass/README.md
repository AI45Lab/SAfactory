# AgentCompass 环境

这个 adapter 用于在 SAfactory 中运行 AgentCompass benchmark。每行 JSONL 选择一个 benchmark、harness、environment 和 sample，并作为一个独立 episode 运行。

## 数据格式

`sample_id` 必须是对应 benchmark 中真实存在的样本 ID。数据位置和各组件参数放入对应的参数对象。

```json
{
  "task_id": "unique-task-id",
  "benchmark": "benchmark-id",
  "harness": "harness-id",
  "environment": "host_process",
  "sample_id": "exact-sample-id",
  "data_dir": "/app/data/agentcompass",
  "benchmark_params": {},
  "harness_params": {},
  "environment_params": {},
  "model_params": {},
  "timeout_seconds": 1800
}
```

`data_dir` 是原生 CLI 的数据根目录，与 Launcher 读取的任务清单不同；仅在 benchmark
需要时填写。diagnostic 保持现有 `benchmark_params.dataset_dir`，不需要新增这个字段。

## 已验证结果契约

下列 ID 均直接来自固定 AgentCompass revision 的 registry 和 benchmark/scorer 源码；该 revision
没有为这些 ID 注册 alias。名称相近不代表共享结果 schema。

| Benchmark ID | Source-proven result family | SAfactory normalization |
| --- | --- | --- |
| `browsecomp` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `browsecomp_zh` | `LLMJudgeScorer`（与 browsecomp 相同的 evaluate） | native boolean `correct` → 10/0 |
| `deepsearchqa` | `DeepSearchQAScorer` | native boolean `correct` → 10/0 |
| `frontierscience` | olympiad judge or research rubric | olympiad 10/0; research native 0–10 total |
| `gaia` | `LLMJudgeScorer`（仅 text 模态） | native boolean `correct` → 10/0 |
| `hle` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `hle_verified` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `scicode` | SciCode evaluation | native subproblem fraction 0–1 → 0–10 |
| `sealqa` | official SEALQA A/B/C scorer | native boolean `correct` → 10/0 |
| `sgi_deep_research` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `special_pattern_check` | analyzer-derived boolean `correct` | 10/0 |
| `swebench_verified` | evaluator `completed`/`resolved` | native `resolved` → 10/0 |
| `xbench_deepsearch` | exact-match 短路或 `xbench_llm_judge` | native boolean `correct` → 10/0 |

其余 15 个 benchmark 尚无结果转换，runner 对其 fail closed。

所有 detail 必须包含固定 revision 的顶层 `task_id` 和恰好一个 attempt。只有 `completed`、空
`error` 且 benchmark-specific 字段一致的结果才能评分。未知 ID、alias、schema、冲突字段和
`run_error`/`eval_error`/`skipped` 均 fail closed。runner 只会保留源码明确提供且不改变语义的字段，
不会从 `correct`、grade、label 或 summary 推导 verdict/reason，也不会过滤正常答案文本。

AgentCompass 原生 detail/log 的路径不进入 adapter 的 stdout 或 evaluator 输出；文件保存在
`<results_root>/<job_id>/<session_id>/<task_id>/`。公共 RJob 层仍可能在结果 metadata 中加入
artifact 路径和日志摘要。使用持久挂载后，可从 launcher 可见的同一 storage source 读取这些文件。
当前共享接口不能由 env-only 代码写入 WT `ground_truth_answer` 或 meta_json 中 eval 下的
context；因此 evaluator 不把 ground truth、model answer 或 judge 语义字段放入 EvalResult 的
artifacts 冒充 context。真实 WT 契约验证必须等待共享层支持。

默认 RJob 的结果回传使用 `/app/results/<job_id>/<session_id>/safactory_result.json`，由公共
`manager/episode_common.py` 生成并通过 `SAFACTORY_RESULT_PATH` 注入；公共配置允许覆盖路径。
父 launcher
与子 RJob 必须挂载同一个持久 storage source：父 launcher 将其挂载到工作区的 `./results`，子
RJob 的 target 固定为 `/app/results`。真实 source URI 由部署方在 private/global `--rjob-config`
的 `mount_config` 中提供，公共 per-agent start config 不定义 `mount_config`，也不保存真实 GPFS URI。
RJob runner 会在启动 AgentCompass benchmark 前验证结果路径和 `/app/results` 挂载点，缺失时非零退出。
该检查只验证子容器的挂载和可写性，不能证明父子使用同一个 storage source。公共 RJob backend
只校验 Docker mounts 是否有 RJob 对应配置，不验证远程持久挂载，因此保留这项 adapter 检查。
如覆盖结果路径，应保持 artifact 和原生 results 都在这个持久挂载下。

公开 diagnostic 仅验证 `special_pattern_check` + `openai_chat` + `host_process` 的框架链路，
不证明其他 benchmark 的数据、镜像依赖、harness 组合或 judge 配置已经可 rollout。

## Runtime image

```text
registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches:agentcompass-02
registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches@sha256:61e6f2b7c3e92d51981bcfd48b159964c5a40519c5d0c6b06166a2897ed324f2
```

该镜像为 Linux/AMD64，使用 Python 3.12，并固定 AgentCompass commit：

```text
d2c3e148902e948db3270fa34b2198fb1b10beb7
```

[Dockerfile.host-extras](Dockerfile.host-extras) 在其上叠加原生宿主侧代码路径所需依赖，不改
revision、不升降级基础包：wget/unzip/git-lfs（原生 loader 调用）、`frontier-engineering`、
`wildclawbench`、`taubench` extras、matplotlib（SciCode problem 80），以及按固定版本文档
`--no-deps` 安装的 tau2 v1.0.1。已发布：

```text
registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches:agentcompass-d2c3e148-host-extras-01
registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches@sha256:832a3ee0606e19c59360a6c9abbf63909d0d3dfc357f039e27eae01e5ac29d60
```

原生 `dependency_spec`（gdpval_ac、scicode、swebench_multilingual、swebench_verified、
wildclawbench）在该镜像内全部通过。Agent CLI（openclaw/claude/codex/openhands）与 Docker
不在此镜像内，属于原生 recipe 选择的各任务 sandbox 镜像。

默认只允许 `host_process`。如需启用其他 environment，必须通过 `AGENTCOMPASS_ALLOWED_ENVIRONMENTS` 显式配置。

主模型通过当前 episode 的 Gateway session 路由。

**原生 LLM judge 暂不启用（待框架确认）。** 上表中使用 LLM judge 的 benchmark 结果转换已实现，
但 SAfactory 目前没有约定 episode 内辅助调用（judge、用户模拟器、embedding、搜索摘要）应如何
路由和记录。固定版本的 judge 配置缺少 `base_url`/`api_key` 时会回落到主模型地址，也就是本
episode 的 Gateway session，会把含参考答案的 judge 调用混入策略轨迹和步数预算。因此 runner
在任何模型调用前拒绝这些 benchmark（`error_type=judge_routing_undefined`），不自行选择路由，
也不静默混入主模型轨迹。接入方式由框架维护者确认后再启用。上面的评分归一化表不是可运行组合矩阵。

## 标准开发和启动

本节对应公共层 `v2`（验收基线 `edd99ca`），cloud 依赖按仓库 cloud requirements 固定
（`wt-data-platform-sdk` v0.7.3 / `dldb-v1.1.9`），test profile 的 Landing 表为 `v2_landing_test`。

从仓库根目录运行，参照 Geo3K 的 request/result/evaluator 契约和 OpenRT 的原生 harness、
RJob 文件分发及结果挂载。`runner_entrypoint` 的 `source` 相对 start YAML 解析；Docker
`mounts` 中的 `source` 相对 launcher 当前工作目录解析。Docker 的 diagnostic 数据因此
挂载 `./env/agentcompass/datasets/special_pattern_check`，结果挂载 `./results`。

SQLite/cloud 都由公共 Gateway、Launcher、DataManager 和 RewardCommitter 管理。此目录不需要
专用 cloud 启动器或存储代码。Evaluator 按 `env_name` 自动发现，不需要在 `env_params` 注册；
它只读取已归一化的 metrics，不重新运行 benchmark，也不自行写数据库。session 关闭、轨迹读取、
reward/step_reward 回写及资源释放均由公共框架执行。

Cloud 使用仓库 cloud requirements 固定的依赖，见[存储说明](../../docs/guides/S3+LanceDB-storage_CN.md)。
先确认部署解释器版本和已安装 SDK 与依赖文件一致；不能仅凭 import 成功判断兼容。
尤其 `EnvConfigManager` 必须接受 `profile` 参数。
Gateway 和 Launcher 必须加载同一份私有环境配置。下面的 `.env` 是部署示例，仓库不提供该
文件；如配置保存在仓库外，应将 `source` 参数替换为实际私有文件路径，不要将凭证提交到 Git：

```bash
set -a
source .env
set +a
export WT_SDK_PROFILE=test
```

从 `configs/config.gateway.example.yaml` 派生私有 Gateway 配置，设置：

```yaml
storage_type: cloud
storage_config: {}
```

删除 SQLite 的 `db_url`，不显式指定 landing/env-config 表名，也不要保留旧的表名环境变量覆盖。
启动后 Gateway 日志应显示 `v2_landing_test`。这里只保存轨迹和环境行，原生 detail/log
文件仍依赖持久挂载，不会自动上传。

配置真实 `llm_routes`；当前 Gateway 不提供上游 model 别名映射，route key 应使用上游接受的
model 名。无需 monkey patch Gateway。启动入口始终为：

```bash
python -m gateway --config "$GATEWAY_CONFIG"
```

准备私有 `config.local.yaml`，沿用全局 RJob 配置，补充鉴权、配额/idle 标签及 results 的
`mount_config`。子 RJob 的挂载 target 为 `/app/results`，父 launcher 必须能读到相同 source。
全局 `rjob` 下的 `gateway_base_url` 若非空会覆盖 CLI URL，必须是子 RJob 可达地址。镜像和额外 benchmark
资产应在部署侧核实；公共 diagnostic 通过 `embedded_files` 分发其单条原生数据。

完成配置后，下面命令才会实际提交 episode 和写数据库。仅做审计/启动准备时不要执行；
RJob 的 `dry_run` 也不能保证 Launcher 不写数据库。

```bash
: "${GATEWAY_BASE_URL:?设置集群可达的 Gateway session root}"
: "${ROUTE:?设置与 Gateway llm_routes 一致的上游 model 名}"
python launcher.py \
  --mode rjob --rjob-config config.local.yaml \
  --agent-config env/agentcompass/agentcompass_config.rjob.yaml \
  --agent-start-config env/agentcompass/agentcompass_start.rjob.yaml \
  --storage-type cloud \
  --gateway-base-url "$GATEWAY_BASE_URL" --llm-model "$ROUTE" \
  --enable-evaluation --pool-size 1 --max-workers 1 --multiplier 1 \
  --max-steps 1 --agent-start-timeout-s 1800
```

省略 job ID 时公共 Launcher 会生成新 ID。Docker 模式改用两个不带 `.rjob` 的 YAML 和
`--mode docker`；镜像按 Dockerfile 构建。SQLite 模式还需将 Gateway 切换到相同 SQLite URI，
并向 Launcher 传入 `--storage-type sqlite --db-path`。

离线验证（不创建 Gateway、RJob 或数据库）：

```bash
python -B -m unittest discover -s tests/env/agentcompass -p 'test_*.py' -v
```

测试中的外部执行使用替身；缺少 AgentCompass 时原生 registry/loader 测试会跳过。这不替代
文档要求的真实 Geo3K 基线和 AgentCompass diagnostic smoke test。当前 runner 的成功结果
`step_count=1` 表示一次原生 case，不是多轮 harness 的实际模型请求数；实际调用需读取 Gateway
轨迹。步数预算只对已映射的 harness 注入，其他 harness 需要另行验证，不能据此宣称全部可用。

## 真实 benchmark 共享数据与批量运行

运行入口仍是 `launcher.py`，复用现有 config/start 和 `runner.py`。
[prepare_shared_data.py](prepare_shared_data.py) 是唯一的数据准备与任务选择工具：不启动
任务、不创建 session、不写数据库，也不新增启动入口。

每行保持一个原生 sample；`env_num: 1`，runner 强制 `sample_ids=[sample_id]`、`k=1`、
原生 `task-concurrency=1`。SciCode 一个 sample 内可以有多个子步骤，共用一个 session；
不能把整个 benchmark 放进一个 session，也不能把每个子步骤冒充独立 sample。

28 个 benchmark 的数据、评分资产、依赖与运行状态见 [共享数据准备状态](DATA_READINESS.md)。
数据和 catalog 已发布不代表对应运行环境已就绪，也不代表 benchmark 已验收；代码接入、
运行依赖和真实验收分别以该文件对应列为准。

### 部署者首次准备（集群内执行一次）

在固定镜像（下方 host-extras）内、共享数据根以读写方式挂载时执行，可多次执行：

```bash
/opt/venv/bin/python -B prepare_shared_data.py prepare \
  --benchmark <id> [--benchmark <id> ...] --data-dir <SHARED_ROOT>
```

- 每个 benchmark 从固定模块指定的来源获取**完整原生数据树**（含附件、图片、评分资产），
  在 `<SHARED_ROOT>/.staging/<id>/` 中依次完成：下载/解压 → 原生 loader 校验 → 逐文件 SHA256
  并写出待发布记录 → 将数据目录逐个 rename 到原生 `data_dir` 位置 → 最后写入
  `<SHARED_ROOT>/.published/<id>.json`。只有正式记录存在时数据才算已发布。
- staging 不会因普通异常被删除；`state.json` 记录已完成阶段。无论是普通异常还是进程被强制
  终止，重新执行都会从最后完成的阶段继续：已完成的下载和哈希不重做，"数据已移动、记录未写完"
  也能补完。下载中途中断的部分文件不可信，会重新下载。
- 正式记录包含来源、实际获取版本（模块未固定版本时为本次获取的 HF sha / git commit）、归档
  SHA256、每个文件的 SHA256（内容完整性依据），以及每个文件的大小和 mtime（仅用于快速复用
  检查），并记录各阶段实际耗时。datasets/HF 留下的 `*.lock` 运行锁文件不计入资产。
- 重复执行默认只比对文件列表、大小和 mtime：能发现缺失、多出、大小变化或重写的文件，但**不
  证明内容完整**（保持大小和 mtime 的原位修改无法发现）。需要内容级校验时加
  `--verify-content`，重新计算所有文件的 SHA256；输出的 `verification` 字段标明本次实际
  做的是哪一级。没有大小/mtime 元数据的旧记录，必须先通过完整内容校验，才会补写元数据并在
  `amendments` 留痕。
- 校验不通过时报错，不覆盖、不修补；更新版本须发布到新目录。`--reconcile` 只在已记录资产
  内容全部不变时，补记记录后新增的文件，或删除早期记录中误收的运行锁文件条目，并在
  `amendments` 留痕。
- 原生任务目录写入 `<SHARED_ROOT>/manifests/v2/`；更早版本的 `manifests/*.jsonl` 保留不动。
  目录行只含数据与原生参数，不含 harness 或 judge 等运行配置。
- SciCode 的官方 HDF5 首次获取与结构校验仍由 [generate_dataset.py](generate_dataset.py)
  `--prepare` 完成；统一工具在其基础上接管、记录并生成 validation/test 目录。
- skillsbench 与 special_pattern_check 在固定版本没有下载源，需部署方用 `--source <目录>`
  提供原生格式数据，同样经原生 loader 校验后发布；`status` 子命令会给出所需目录结构。
- 失败时输出结构化错误与 traceback 并以非零码退出；批量提交按 benchmark 汇总失败项。
- 准备只在需要网络的部署环境（经集群代理）运行；使用者的 runner 关闭 HF 在线访问，
  缺数据时直接失败，不会向共享目录下载。

部署映射。`<DATA_OWNER>` 是**发布共享数据的 fileset 所有者**，由部署者告知；使用者必须
挂载这份已发布的数据，不能把它换成自己的用户名（那会指向不存在的目录）。results 是每个
使用者自己的可写目录，与数据源相互独立：

| 用途 | Launcher 可见路径 | RJob 挂载源 | 容器路径 |
| --- | --- | --- | --- |
| 共享数据（只读使用） | `/mnt/shared-storage-user/<DATA_OWNER>/agentcompass-data/d2c3e148` | `gpfs://gpfs1/<DATA_OWNER>/agentcompass-data/d2c3e148` | `/app/data/agentcompass`（目录行 `data_dir`，路径参数已指向该挂载） |
| 个人 results（可写） | Launcher 工作目录下的 `./results` | 使用者自己的 results URI | `/app/results` |

在私有全局 `rjob.mount_config` 中同时列出这两项。公共 per-agent start
不设置该列表，避免覆盖全局挂载。runner 把原生工作区、HF/XDG 缓存及 SealQA 的 datasets
缓存放到各 session 的 results 下，共享数据只读使用。

### 使用者日常运行（复用共享数据）

使用者取得共享数据 fileset 的读权限后，由部署者提供 `SHARED_ROOT`（即上表共享数据的
Launcher 可见路径）、标准配置和私有连接配置。
Launcher 必须运行在能看到该共享存储的 worker 上，并具有项目代码、Launcher 依赖和私有环境
文件；Gateway 主机与 Launcher worker 可以不同，但子 RJob 必须能访问 Gateway。工作目录的
`./results` 必须是与子 RJob `/app/results` 相同的持久 source。日常操作只选择 benchmark、
harness、模型和任务范围，无需准备原生数据。

用 Launcher 的普通 Python 从已发布目录选任务（只读小型 JSONL，不需要 AgentCompass）：

```bash
: "${LAUNCHER_PY:?设置 Launcher Python 路径}"
: "${SHARED_ROOT:?设置部署者提供的共享版本目录绝对路径}"
: "${TASK_MANIFEST:?设置本次任务清单输出路径，不能覆盖已有文件}"
"$LAUNCHER_PY" -B env/agentcompass/prepare_shared_data.py select \
  --catalog "$SHARED_ROOT/manifests/v2/scicode-validation.jsonl" \
  --harness scicode_tool_use --harness-params '{"mode": "naive"}' \
  --sample-id 1 --sample-id 19 --output "$TASK_MANIFEST"
```

`--judge-model` 仅为将来的 judge 接入保留；judge benchmark 当前被 runner 拒绝（见上文）。目录经版本/校验和检查；样本 ID 必须来自原生 loader 生成的目录，未知 ID 报错；
`--limit N` 或 `--all` 可替代 `--sample-id`。

在标准 env 配置的私有副本中只把 `dataset` 换成 `TASK_MANIFEST` 的绝对路径，并把
`env_image` 换成 host-extras 镜像；保持 `env_num: 1`、`dataset_load_mode: eager`。

下面的标准运行命令会提交 RJob、调用模型并写入部署指定的 test Landing：

```bash
: "${PRIVATE_ENV:?设置部署者提供的私有环境文件路径}"
: "${RJOB_CONFIG:?设置含共享数据与 results 挂载的私有 RJob 配置路径}"
: "${AGENT_CONFIG:?设置已选择本次 dataset 的标准 env 配置路径}"
: "${GATEWAY_BASE_URL:?设置子 RJob 可达的 Gateway session root}"
: "${GATEWAY_HOST:?设置 Gateway 实际主机名或 IP，用于 NO_PROXY}"
: "${ROUTE:?选择部署者已配置的模型 ID}"
: "${JOB_ID:?设置本次独立 job ID，不能复用旧 ID}"
set -a
source "$PRIVATE_ENV"
set +a
export WT_SDK_PROFILE=test
export NO_PROXY="${NO_PROXY:-${no_proxy:-}},$GATEWAY_HOST,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
"$LAUNCHER_PY" launcher.py --job-id "$JOB_ID" \
  --mode rjob --rjob-config "$RJOB_CONFIG" \
  --agent-config "$AGENT_CONFIG" \
  --agent-start-config env/agentcompass/agentcompass_start.rjob.yaml \
  --storage-type cloud --gateway-base-url "$GATEWAY_BASE_URL" --llm-model "$ROUTE" \
  --enable-evaluation --pool-size 1 --max-workers 1 --multiplier 1 \
  --max-steps 12 --agent-start-timeout-s 3600 --disable-buffer
```

示例的 `--pool-size 1 --max-workers 1` 按模型路由并发上限 1 串行运行；提高并发前由部署者
核实服务容量及路由限制。`max_steps=12` 覆盖整个 validation 的最大子步骤数。Gateway
的 `max_steps` 也必须足够（或设置为 `-1`），当前公共层不会自动从 Launcher 同步它。预算只
对应 naive 模式的生成调用数，不限制 token；模型客户端及原生任务自动重试均为 0。

默认整条 episode 超时 3600 秒、每个子步骤评测超时 1800 秒，沿用现有 start 的 2 CPU/4 GiB。
SciCode test sample `80` 需要 matplotlib，仅 host-extras 镜像提供。

原生 detail 与 Landing 中的模型请求/响应必须按准确 job/session/model 身份核对，再读取
reward/step_reward。SciCode reward 为有效子题通过比例乘 10，可以为部分分；episode reward
的简单平均不等于按子题加权的官方总体分。close、summary 或等待函数返回不代替持久化证据。

## 文档与当前公共实现的差异

- 自定义环境指南中的旧示例仍将 evaluator 路径写入 `env_params`；当前
  `evaluator/rule_evaluator.py` 只按环境目录发现 evaluator。AgentCompass 不采用旧示例。
- 评测指南写到 evaluator 缺失时跳过、异常或超时按零分回写；当前
  `manager/simulation_worker.py` 在开启 evaluation 后会把缺失或失败的 evaluator 标记为失败，
  只有 evaluator 成功分支调用 reward commit。不要将 evaluator 返回的零分失败对象误认为已落库。
- RJob 指南仍描述 Geo3K RJob 配置可能不存在；当前仓库已经包含两个 Geo3K RJob YAML。
- Gateway 的存储一致性检查对 SQLite 比较 DB 路径，对 cloud 仅比较后端类型，见
  `manager/simulation_flow.py`。Cloud 的 profile、URI 和表名仍需部署者核对，不能只看 ready 状态。

## 已知限制与待验证项

以下为公共层现状，均不属于本 env 的修复：

- **写入完成等待**：`evaluator/gateway_client.py` 的 `wait_telemetry_flush` 在 session 已关闭、
  查询不到或超时时都会返回，不能作为 Landing 已持久化的证据；验收以按 job/session 回查为准。
- **步数终态标记**：`gateway/telemetry.py` 在 `llm_step_index == max_steps` 时把 inference
  记录同时标为 `is_truncated` 和 `is_session_completed`。命中预算的 episode 应结合原生结果解释，
  不能把该标志当作调用成功。
- **Cloud 周期 flush**：`_periodic_flush` 只处理取消，一般写入异常会终止后台 flush 任务。
  验收命令使用 `--disable-buffer` 规避，这不是修复。
- **分区创建/打开**：旧 `landing_test` 曾出现分区“创建报已存在、打开报不存在”的异常
  （dldb `open_table` 的并发创建回退）。在 v2 基线上用映射到新分区的 job 复测通过一次，
  不足以证明任意新分区并发安全。

旧基线上第二步顶层 `messages` 被第一步覆盖的问题来自已被上游删除的
`core/data_manager/image_processing.py` 前缀缓存；v2 基线复测见下文。

## 验收记录

| 范围 | 状态 |
| --- | --- |
| `special_pattern_check` + `openai_chat` + `host_process` 单条 diagnostic | 旧基线真实通过；v2 基线未复测 |
| `scicode` + `scicode_tool_use(naive)` + `host_process`，validation `1`、`19` | v2 基线真实通过 |
| 其他 benchmark 与组合 | 未接入或未验收，见 [共享数据准备状态](DATA_READINESS.md) |

SciCode 复测（v2 基线 `edd99ca`，GLM，`v2_landing_test`）：标准 Launcher 报告 2/2 成功，
共 3 次模型调用。在挂载共享 results 的 worker 上读取原生 detail，与按 job 回查的 Landing
逐条比对：每条 inference 记录的顶层 `messages`、`meta_json` 中请求快照的 messages 都等于原生
该步输入，`response` 的回答和 reasoning 等于原生输出，HTTP 均为 200，request ID 无重复，
无 synthetic stop。两个终态 reward/step_reward 均为 10，与原生 1/1、2/2 子题一致；多步任务
的非终态记录为 reward=null、step_reward=0。job 映射到此前不存在的 Landing 分区，运行中创建
成功。原生结果、Landing 快照和比对记录保留在部署侧，不进入 PR。

## 相关文档

- [仓库 README](../../README_CN.md)
- [配置说明](../../docs/reference/configuration_CN.md)
- [自定义环境](../../docs/guides/custom-environment_CN.md)
- [评测说明](../../docs/guides/evaluation_CN.md)
- [RJob 模式](../../docs/internal/rjob-mode_CN.md)
- [Gateway](../../docs/reference/gateway_CN.md)
- [数据管理](../../docs/guides/data-manager_CN.md)
- [Cloud 存储](../../docs/guides/S3+LanceDB-storage_CN.md)
- [标准环境说明](../../docs/reference/environments_CN.md)
- 参考实现：`env/geo3k/` 和 `env/openrt/` 下的 runner、rule evaluator、config/start YAML。
