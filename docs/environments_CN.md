# 支持的环境

Safactory 将每个环境视为外部 agent runtime。一个 runtime 由两份文件描述：

- Agent config：任务行、dataset、`env_params` 和镜像。
- Agent start config：Docker 或 RJob 启动细节。

当前 checkout 包含以下 v2 adapter。

## 总览

| Adapter | `env_name` / `agent_name` | Config | Start config | Runtime 说明 |
|---------|----------------------------|--------|--------------|--------------|
| OpenClaw | `openclaw` | `env/openclaw/openclaw_config.yaml` | `env/openclaw/openclaw_start.yaml` | 通用 OpenClaw CLI runtime，适合 smoke test 和工具使用任务。 |
| OpenRT | `openrt` | `env/openrt/openrt_config.yaml` | `env/openrt/openrt_start.yaml` | 通过 gateway session URL 运行 OpenRT `eval.py`。 |
| OpenRT RJob | `openrt` | `env/openrt/openrt_config.rjob.yaml` | `env/openrt/openrt_start.rjob.yaml` | 远程 RJob 版本，包含镜像、资源、嵌入 runner 和 GPFS 挂载。 |
| ExploitGym RJob | `exploitgym` | `env/exploitgym/exploitgym_config.rjob.yaml` | `env/exploitgym/exploitgym_start.rjob.yaml` | 每个 episode 运行一个 user、V8 或 kernel 任务的 privileged 嵌套 Docker benchmark。 |
| WildClawBench | `wildclawbench` | `env/wildclawbench/wildclawbench_config.yaml` | `env/wildclawbench/wildclawbench_start.yaml` | 需要 WildClawBench checkout 和匹配镜像。 |
| DTAP | `dtap` | `env/dtap/dtap_config.yaml` | `env/dtap/dtap_start.yaml` | 运行 DecodingTrust-Agent workload，并挂载 Docker socket。 |
| ClawEnvKit | `clawenvkit` | `env/clawenvkit/clawenvkit_config.yaml` | `env/clawenvkit/clawenvkit_start.yaml` | 运行 ClawEnvKit / Auto-ClawEval 任务。 |

仓库中的 YAML 有些包含本地路径和内部镜像名。请把它们视为工作示例，按你的机器或集群调整路径、镜像、挂载和模型 route key。

## 运行一个 Adapter

先启动 gateway，然后运行：

```bash
python launcher.py \
  --agent-config env/openclaw/openclaw_config.yaml \
  --agent-start-config env/openclaw/openclaw_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --pool-size 1 \
  --max-workers 1
```

可以用 `--agent-root env` 加载 `env/` 下所有 agent config。发现过程中，`*_start.yaml` 这类不是 agent config 的 YAML 会 warning 后跳过。

## OpenClaw

文件：

- `env/openclaw/openclaw_config.yaml`
- `env/openclaw/openclaw_start.yaml`
- `env/openclaw/runner.mjs`

Runner 收到 `SimulationStartRequest` 后，会写入指向 gateway session URL 的 OpenClaw config，然后运行：

```bash
openclaw agent --local --json --session-id <session_id> --message <task> --model safactory/<route>
```

常用设置：

| 字段 | 含义 |
|------|------|
| `env_params.task_family` | 放入任务 prompt。 |
| `env_params.dataset` | YAML loader 合并后的 dataset 行。 |
| `container.workdir` | 挂载到 OpenClaw 镜像中的 workspace。 |
| `container.extra_args` | 本地 Docker 访问 gateway 时应包含 `--add-host=host.docker.internal:host-gateway`。 |

OpenClaw runner 比较自包含，是最适合 smoke test 的 adapter。

## OpenRT

文件：

- `env/openrt/openrt_config.yaml`
- `env/openrt/openrt_start.yaml`
- `env/openrt/runner.py`
- `env/openrt/rule_evaluator.py`

Runner 会在容器内执行 OpenRT `eval.py`，并将 gateway session URL 当作 OpenAI 兼容 base URL。每条 dataset row 需要定义一个 attack，例如：

```json
{"task_id": "case-001", "attack": "PAIR"}
```

重要 `env_params`：

| 字段 | 含义 |
|------|------|
| `default_openrt_dataset` | 传给 OpenRT 的 dataset 名，默认 `harmbench`。 |
| `default_attacker_model` | OpenRT attacker model。 |
| `default_judge_model` | OpenRT judge model。 |
| `default_target_models` | Target model 列表。这些模型名应能通过 gateway 路由。 |
| `results_root` | 可选输出根目录。默认 `/app/results`。 |
| `max_workers`, `evaluator_workers` | 设置时会传给 OpenRT CLI。 |

OpenRT 也提供 RJob 版本：

```bash
python launcher.py \
  --mode rjob \
  --rjob-config config.yaml \
  --agent-config env/openrt/openrt_config.rjob.yaml \
  --agent-start-config env/openrt/openrt_start.rjob.yaml \
  --gateway-base-url http://YOUR_GATEWAY_HOST:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --storage-type cloud \
  --pool-size 8
```

## ExploitGym RJob

相关文件：

- `env/exploitgym/exploitgym_config.rjob.yaml`
- `env/exploitgym/exploitgym_start.rjob.yaml`
- `env/exploitgym/runner.py`
- `env/exploitgym/rule_evaluator.py`

本次接入仅支持 RJob。每条 dataset row 必须包含一个以 `user:`、`v8:` 或
`kernel:` 开头的 `task_id`。runner 只为该任务启动一次嵌套 Docker
ExploitGym，并将模型请求转发到当前 SAfactory Gateway session。

默认数据集只包含一个 ARVO 任务，适合首次运行。相邻的
`datasets/exploitgym_tasks_full.jsonl` 包含全部 869 个 v1 标准任务
（502 个 user、181 个 V8、186 个 kernel）；全量运行时需显式选用该文件。

默认 RJob 资源为 8 CPU、16 GiB 内存、0 GPU、privileged、1 个
`brainpp.cn/fuse` 资源和 100 GiB 本地存储。GPFS2 只读提供目标镜像
cache，GPFS1 持久化结果；kernel 任务还要求 `/dev/kvm` 可读写。

仓库配置已用不可变 digest 固定共享镜像。运行前只需替换结果挂载占位符，
并确保 Gateway 地址能从 RJob Worker 访问。真实 provider key 只保存在
Gateway，不会挂载进 ExploitGym RJob。完整配置、运行命令和结果解释见
`env/exploitgym/README.md`。

## WildClawBench

文件：

- `env/wildclawbench/wildclawbench_config.yaml`
- `env/wildclawbench/wildclawbench_start.yaml`
- `env/wildclawbench/runner.py`

该 adapter 需要 WildClawBench checkout 和能运行其任务的镜像。运行前请更新：

| 字段 | 更新内容 |
|------|----------|
| `env_params.wildclawbench_root` | WildClawBench checkout 的绝对路径。 |
| `container.workdir` | 容器内相同 root 路径。 |
| `container.mounts` | 挂载 WildClawBench checkout 和 runner 文件。 |
| `env.DEFAULT_MODEL`, `env_params.model_ref` | WildClawBench/OpenClaw 使用的模型引用。 |
| `env.JUDGE_MODEL`, `env_params.judge_model` | Judge route key。 |

## DTAP

文件：

- `env/dtap/dtap_config.yaml`
- `env/dtap/dtap_start.yaml`
- `env/dtap/runner.py`

DTAP 运行 DecodingTrust-Agent 任务。Start config 会挂载：

- `DecodingTrust-Agent` checkout。
- Safactory `results`。
- DTAP runner 脚本。
- `/var/run/docker.sock`，因为 DTAP 可能启动嵌套 Docker workload。

运行前重点检查：

| 字段 | 含义 |
|------|------|
| `env_params.dtap_root` | Runtime 内 DecodingTrust-Agent 路径。 |
| `env_params.dataset_root` | DTAP dataset 路径。 |
| `env_params.route_model` / `model_ref` | 任务 runtime 使用的 route/model reference。 |
| `container.network` | 示例默认为 `host`。 |
| `container.mounts` | 必须指向真实本地路径。 |

## ClawEnvKit

文件：

- `env/clawenvkit/clawenvkit_config.yaml`
- `env/clawenvkit/clawenvkit_start.yaml`
- `env/clawenvkit/runner.py`

ClawEnvKit 通过 OpenClaw harness 运行 Auto-ClawEval 风格任务。运行前请更新 dataset 和结果挂载。

重要字段：

| 字段 | 含义 |
|------|------|
| `env_params.dataset_root` | Runtime 内 dataset 路径。 |
| `env_params.clawenvkit_root` | ClawEnvKit 安装路径。 |
| `env_params.harness_entrypoint` | Harness entrypoint 脚本。 |
| `env_params.route_model` / `model_ref` | Gateway route/model reference。 |
| `container.extra_args` | 示例会重置 entrypoint，以用户 `0` 运行，并添加 host-gateway。 |

## Dataset 加载

Agent config 的 `dataset` 支持 JSON、JSONL、YAML list 和 parquet。

- JSONL 每行必须是合法 JSON object。
- 相对路径从 agent config 所在目录解析。
- `dataset_load_mode: eager` 会物化数据行。
- `dataset_load_mode: parquet_row_ref` 会保存轻量 parquet row reference。

每条 dataset row 会成为 runtime request 中的 `env_params.dataset`。

## Runtime Result Contract

所有 adapter 都应该向 stdout 输出一条 JSON result：

```json
{
  "session_id": "same-session-id",
  "status": "succeeded",
  "total_reward": 0.0,
  "step_count": 1,
  "terminated": true,
  "truncated": false,
  "error_text": null,
  "metrics": {}
}
```

在 JSON mode 下，如果 runtime 以非零状态退出，Docker/RJob runner 会视为失败，即使 stdout 中有部分输出。
