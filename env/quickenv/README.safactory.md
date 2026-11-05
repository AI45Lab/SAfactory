# Quickenv 的 SAfactory 单任务接入

本接入封装现有 `run_mining_400.sh TASK_NAME`：一个 JSONL 行对应一个
SAfactory episode、Gateway session 和原生 Mining case。外层由 SAfactory
在 RJob 模式下调度并发；原生脚本每次只接收一个任务名，内部 `WORKERS=1`。
范围是已提供脚本支持的 **ARVO lv0 Mining**。Exploit 批次的单题入口没有
提供，未把 README 中的 Exploit 批量入口当作单题命令接入。

原交付指南保留在 [README.md](README.md)，原启动脚本也保持原样。
本目录是接入代码和部署配置；完整交付框架、离线运行时、真实清单、镜像包
仍需从交付服务器提供。当前未构建或推送镜像，也未提交真实 RJob。

## 文件与调度边界

| 文件 | 职责 |
| --- | --- |
| `runner.py` | 来自 SAfactory skill 的固定协议壳，读取请求，只输出一个结果 JSON |
| `adapter.py` | 选定一个 case，将模型调用绑定到当前 session，执行原生入口并收集产物 |
| `runtime.py` | 管理本 episode 的 nested Docker daemon 与子进程生命周期 |
| `quickenv_config.rjob.yaml` | RJob 镜像、dataset 和每题参数 |
| `quickenv_start.rjob.yaml` | runner、embedded files、资源、共享盘挂载及 RJob 清理策略 |
| `quickenv_config.yaml` / `quickenv_start.yaml` | skill 要求保留的 Docker 基础配置，用于 Linux 上的本地部署验证 |
| `datasets/quickenv_tasks.jsonl` | 明确标记 `example: true` 的行格式示例，adapter 会拒绝实际运行 |
| `build_dataset.py` | 从完整交付分组生成真实的一题一行 JSONL |
| `request.smoke.json` | 同样标记为 example 的协议请求示例 |
| `Dockerfile` | 从准备好的基础镜像和交付目录封装运行时的配方 |

`quickenv.start.rjob.yaml` 和 `quickenv.config.rjob.yaml` 是上述标准命名文件
的软链接，便于使用点号命名。配置格式是 YAML；不创建误拼的 `.raml` 文件。

每题流程：

```text
SAfactory 读取一个 dataset 行，创建独立 session
  → RJob 内启动 runner / adapter
  → 启动该 episode 专用 dockerd
  → bash <delivery_root>/run_mining_400.sh <task_name>
  → 原入口检查分组、加载镜像、启动 controller / LLM proxy / firewall
  → Harbor 执行 Claude agent，原生 verifier 评测
  → 收集原生 result.json、日志和 Harbor 产物，输出 SAfactory result
  → 停止本 episode 子进程与 dockerd，由 SAfactory 清理 RJob
```

各 RJob 使用自己的网络命名空间、Docker socket、Docker 数据目录和原生
controller/proxy/firewall。配置显式设置 `replicas: 1` 和 `host_network: false`，
覆盖全局 RJob 默认值，防止同一 session 被多副本重复执行或共用节点网络。
不要改成 `hostNetwork`、共享宿主机 Docker socket，
或让多个 RJob 写同一个 Docker data-root。原脚本对输出目录使用 `flock`；
adapter 给每个 episode 单独的输出目录，避免不同 session 争用锁或覆盖结果。
原生 `RESUME` 被关闭，SAfactory session 重试不会误读原脚本默认跳过的旧结果。

## 需要补齐的交付依赖

镜像内保持以下布局，运行框架和离线环境放在容器私有可写层；不要让所有
RJob 共同修改共享盘上的 `.venv`、离线 `relocate.py` 目标或服务状态文件。

```text
/opt/quickenv-delivery/
├── claude-code-2.1.228-linux-amd64/
│   ├── claude
│   └── manifest.json
└── exploit_gym/
    ├── run_mining_400.sh
    ├── env_cn.sh
    ├── scripts/run_pj_exploit.sh
    ├── scripts/setup/pre_run.py
    ├── examples/run_agent.py
    └── ...框架其余模块、项目元数据和配置
/opt/exploitgym-offline/
├── activate.sh
├── relocate.py
├── firewall-image.tar
└── ...原离线包的 Python、uv、Harbor、LLM proxy 等依赖
```

| 依赖 | 来源与要求 |
| --- | --- |
| 完整 ExploitGym 框架 | 当前仓库没有 `env_cn.sh`、`scripts/run_pj_exploit.sh`、`scripts/setup/pre_run.py`、`examples/run_agent.py` 等下层文件。需要完整交付目录，不是只复制三个入口文件 |
| 离线运行时 | README 中的 `exploitgym-offline-linux-x86_64.tar.gz`，解压为镜像内 `/opt/exploitgym-offline`；保留 `activate.sh`、`relocate.py` 和全部包内内容 |
| Claude agent | 有执行权限的 Linux amd64 实体二进制及随附文件，与框架目录同级；原脚本固定按此路径查找 |
| Docker 运行能力 | Linux amd64、root、Python 3.10+、bash、Docker CLI/Engine、Compose v2、`flock`、`fuse-overlayfs`、`/dev/fuse`；RJob 需 privileged 及相应 fuse 资源权限 |
| 整个任务分组 | `selected_tasks.txt`、`tasks.json`、`bin/load_image.py`、`shared-images/`、各题的 `harbor-runtime-profile/task.toml`、`image-lock.json`、`images/*.tar*`，以及 profile 引用的其他文件 |
| RJob SDK 与配置 | Launcher 运行环境可导入 `brainpp.rjob`，现有全局 RJob 配置含有效 endpoint、namespace、权限及配额；凭据不写入本 ENV |
| Gateway | RJob 和原生 LLM proxy 可访问的 SAfactory Gateway，模型 route 支持 Anthropic Messages 接口 |

**只运行一题也需要完整分组。** 原脚本先验证分组所有任务目录和镜像包是否
存在，再处理选中的 `TASK_NAME`。因此生成单行 JSONL 不代表只挂载这一题就
能通过原脚本检查。原 README 记录三个 Mining 分组为 31、184、185 题；本仓库
没有对应清单，不能据此虚构 400 个任务 ID。

任务数据仍从共享盘读取。示例挂载保留
`/mnt/shared-storage-gpfs2/trustcyberdata`，便于绝对软链接继续有效；若迁移数据，
必须核对软链接、task profile、Compose 文件中的路径。`MINING_BATCH_DIR`
由 dataset 行的 `batch_dir` 传入，不依赖镜像内默认 `pj-mining-400` 路径。

原生下层代码尚缺失，首次部署必须确认：它尊重传入的 `MODEL`、
`ANTHROPIC_BASE_URL`、`ANTHROPIC_API_KEY`、`OUT_DIR` 和 `DOCKER_HOST`；
不会重新覆盖为固定上游网关或共享输出目录；单题产物确实落在
`OUT_DIR/<task_name>/result.json`。原始 `env_cn.sh` 和 profile 若写入其他
固定共享路径，应在交付运行时中修正或隔离后再运行并发。

## 准备运行时镜像

Dockerfile 是打包配方，不会替代缺失依赖的安装。先准备带 Docker、Compose、
fuse-overlayfs、Python 等工具的 Linux amd64 基础镜像，以及独立构建上下文：

```text
quickenv-image-context/
├── exploitgym-offline/                 # 解压后的完整离线包
├── exploit_gym/                       # 完整框架，不包含庞大任务镜像包
└── claude-code-2.1.228-linux-amd64/     # agent 实体目录
```

`exploit_gym/` 中保留完整执行框架及原 `run_mining_400.sh`，去掉 `pj-*` 任务
数据、历史 `out/`、`results/` 和 `.git/`。这些数据在运行时挂载，不必复制进
每个镜像。不要把整个共享盘直接作为 Docker build context。

在仓库根目录执行，替换下面的镜像与目录占位值：

```bash
cp env/quickenv/.dockerignore /path/to/quickenv-image-context/.dockerignore
docker build --platform linux/amd64 \
  --build-arg QUICKENV_BASE_IMAGE=YOUR_PREPARED_LINUX_AMD64_BASE \
  -f env/quickenv/Dockerfile \
  -t YOUR_REGISTRY/quickenv-runtime:delivery-v1 \
  /path/to/quickenv-image-context

docker push YOUR_REGISTRY/quickenv-runtime:delivery-v1
```

然后将 `quickenv_config.rjob.yaml` 中的 `env_image` 改为该真实镜像。
`registry.example.com/...` 只是显式占位符。离线包的真实 Python/uv 布局、
`env_cn.sh` 与 relocation 行为需要在拿到完整交付物后做首次镜像验证。

## 从交付清单生成 JSONL

一行字段如下，所有路径必须是容器实际可见路径：

| 字段 | 含义 |
| --- | --- |
| `task_id` | SAfactory 行标识；生成器使用清单任务名 |
| `task_name` | 原脚本位置参数，必须来自分组 `selected_tasks.txt` |
| `batch_dir` | 包含该 case 的完整分组目录，作为 `MINING_BATCH_DIR` |
| `example` | 仅格式示例为 `true`；不要把删除此字段当成补齐真实任务的方法 |

在可读取交付共享盘的机器上，从仓库根目录运行：

```bash
QUICKENV_DELIVERY=/mnt/shared-storage-gpfs2/trustcyberdata/private/CVEDockerGen/wuxuansheng/20261019_deliver
python3 env/quickenv/build_dataset.py \
  --batch-dir "$QUICKENV_DELIVERY/exploit_gym/pj-mining-400" \
  --batch-dir "$QUICKENV_DELIVERY/pj-mining-part2" \
  --batch-dir "$QUICKENV_DELIVERY/pj-mining-part3" \
  --output env/quickenv/datasets/mining.jsonl
```

生成器逐分组核对清单、数量、profile 集合、image-lock、镜像包、共享目录与
loader，再合并为一题一行；跨分组重复任务会报错，不会静默重复调度。
它不验证归档 SHA256、不调用模型、不加载 Docker 镜像；SHA256 仍由原生
运行入口处理。已有输出文件会被拒绝覆盖。

如果本地读取路径和容器路径不同，为每个 `--batch-dir` 按相同顺序提供一个
`--runtime-batch-dir`。使用 `--task <清单中的任务名>` 选指定 case，或
`--limit 2` 生成两题 smoke 清单；这仍会验证所有传入分组。只需要第一个
分组时，只传一个 `--batch-dir` 即可。

生成后将两份 `quickenv_config*.yaml` 的 `dataset` 改成
`./datasets/mining.jsonl`（或生成的 smoke 文件），再开始部署。默认
`datasets/quickenv_tasks.jsonl` 和 `request.smoke.json` 只验证接入结构，
adapter 会明确拒绝 example 行，不会伪装成成功运行的真实 case。

## RJob 并发运行

提交前替换 RJob task config 中的镜像、start config 的 `CLUSTER_STORAGE`
结果盘，并准备真实 dataset。数据盘挂载应覆盖每行 `batch_dir` 及其软链接
目标。全局 `config.yaml` 继续使用现有集群配置；不要把 AK/SK 写入 ENV。

下面命令从仓库根目录启动最多 20 个并行 episode。Gateway 地址必须从
RJob 可达，不能用 localhost；模型凭据由 Gateway 配置管理。Launcher 和
Gateway 必须使用同一个 storage backend/DB URI。

```bash
python launcher.py \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/quickenv/quickenv_config.rjob.yaml \
  --agent-start-config env/quickenv/quickenv_start.rjob.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model YOUR_ANTHROPIC_ROUTE \
  --db-path sqlite:////path/to/the-same-gateway.db \
  --pool-size 20 --max-workers 20 \
  --max-steps 1 \
  --agent-start-timeout-s 3900 \
  --agent-start-timeout-grace-s 180 \
  --job-id quickenv-mining-smoke
```

`env_num: 1` 保持每行一次调度；并发由 `pool-size`、`max-workers`、集群配额
及提交速率共同决定。只设 `pool-size=1` 不会限制 dataset 长度，首次真实
验证请使用生成的 1–2 行文件。全局 RJob 配置的 `submit_concurrency` 控制
提交速率；若包含 `gateway_base_url`，须与命令一致，避免覆盖正确地址。

默认每题 6 CPU、24 GiB 内存只是初始配置，尚未用真实任务验证；20 并发
需要相应配额。Docker 镜像解包空间是 RJob 内的本地空间，不能只按原归档
大小估算；针对最大的 case 检查磁盘、内存及共享盘读带宽后再提高并发。

`env_params.quickenv.timeout=2700` 是原生任务超时，`launch_timeout=3600`
覆盖镜像加载与服务准备等整体调用，Launcher 的 3900 秒超时留出清理余量。
增加 `max_turns` 和原生 timeout 时，要同步增加整体超时及 Launcher 超时。
请求中的正数 `max_steps` 也会限制 `max_turns`，长评测须一起提高
Launcher 的 `--max-steps`。adapter 固定 `SETUP=0`，不在每个 RJob 内重新安装依赖。
模型走当前 session 的 Anthropic endpoint：
`<gateway_base_url>/<session_id>/v1/messages`，通过 Gateway 绑定轨迹，
不复用原脚本中的固定上游配置。网络与防火墙策略必须允许这条链路。

本次仅接入调度，不启用 SAfactory 评分，因此命令不带 `--enable-evaluation`。
原生 verifier 仍由原框架执行，原生结果保留为产物；`total_reward: 0.0`
是未评分的协议值，不代表该题失败或成功。`MAX_TURNS=1` 是启动交互验证，
达到轮数上限与成功解题须按原 README 分开判断。

## 验证和结果

本地静态检查：

```bash
python3 skills/safactory-workflows/scripts/validate_environment.py env/quickenv
```

缺少原生依赖时，按 skill 用 fixture 验证固定协议壳和 mock model 路由：

```bash
python3 skills/safactory-workflows/scripts/check_environment.py \
  --env env/quickenv \
  --fixture-adapter tests/fixtures/quickenv_protocol_adapter.py
```

运行 adapter、数据集生成器和进程生命周期测试：

```bash
python3 -m unittest discover -s tests -p 'test_quickenv_*.py' -v
```

其中原生调用测试使用临时交付目录、假原生脚本及本地 Anthropic HTTP
fixture，保留真实 adapter 和子进程执行器；Docker daemon 被明确替换为
fixture。本地 HTTP 测试需要允许监听 localhost。

当前工作站缺少交付物时，可以验证固定 runner 的失败协议；example 请求
应返回 `status: failed`、进程退出码 0，stdout 只有一个 JSON：

```bash
python3 env/quickenv/runner.py < env/quickenv/request.smoke.json
```

协议 fixture 检查与 adapter 测试仅证明请求映射、单题执行约束、session
路由、输出隔离及异常处理。它们不证明真实镜像拉取、共享盘挂载、nested
Docker、原生交付框架、真实模型调用或 RJob 并发已经可用。补齐上述依赖后，
先跑真实 1–2 case，再扩大并发。

原生输出位于
`results_root/quickenv/<job-safe-hash>/<session-safe-hash>-<random>/native/`，
包含 `<task_name>/result.json`、
`<task_name>/harbor-jobs/`、镜像加载日志以及文档约定的 Docker 清理日志；
`pre_run/` 下有 controller 和 LLM proxy 日志。SAfactory 的最终协议结果
另写到 `SAFACTORY_RESULT_PATH`，不会替换原生 `result.json`。
单题分支不承诺生成 `pj_user_batch_summary.json`；它在原指南中定义为批次汇总。
