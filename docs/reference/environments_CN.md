# 支持的环境

SAfactory v2 将每个环境视为外部 agent runtime。一个 runtime 由以下部分描述：

- agent config：任务行、dataset、`env_params` 和镜像；
- agent start config：Docker、RJob 或 Sandbox 的启动细节；
- 可选的 `rule_evaluator.py`：rollout 后的 reward 转换。

用于新用户上手、smoke test、评测和 RL 示例的标准环境是 **Geo3K**。

## 环境矩阵

| 环境 | `env_name` / `agent_name` | 领域 | Config | Start config | Runtime 模式 | Evaluator |
|------|----------------------------|------|--------|--------------|--------------|-----------|
| Geo3K | `geo3k` | 几何 / VLM QA | `env/geo3k/geo3k_config.yaml` | `env/geo3k/geo3k_start.yaml` | Docker；RL 模板 | `env/geo3k/rule_evaluator.py` |
| OpenClaw | `openclaw` | 通用 OpenClaw CLI 任务 | `env/openclaw/openclaw_config.yaml` | `env/openclaw/openclaw_start.yaml` | Docker | 可选 |
| OpenRT | `openrt` | 安全 / red-team benchmark | `env/openrt/openrt_config.yaml` | `env/openrt/openrt_start.yaml` | Docker | `env/openrt/rule_evaluator.py` |
| OpenRT RJob | `openrt` | 远程 OpenRT benchmark | `env/openrt/openrt_config.rjob.yaml` | `env/openrt/openrt_start.rjob.yaml` | RJob | `env/openrt/rule_evaluator.py` |
| WildClawBench | `wildclawbench` | 社区 benchmark harness | `env/wildclawbench/wildclawbench_config.yaml` | `env/wildclawbench/wildclawbench_start.yaml` | Docker | 可选 |
| DTAP | `dtap` | DecodingTrust-Agent workload | `env/dtap/dtap_config.yaml` | `env/dtap/dtap_start.yaml` | Docker | 可选 |
| ClawEnvKit | `clawenvkit` | Auto-ClawEval 风格任务 | `env/clawenvkit/clawenvkit_config.yaml` | `env/clawenvkit/clawenvkit_start.yaml` | Docker | 可选 |

部分仓库内 YAML 包含本地路径或内部镜像名。它们应被视为工作示例，正式运行前需要按你的机器或集群调整路径、镜像、挂载和 Gateway route key。

## <a id="standard-environment-geo3k"></a>标准环境：Geo3K

Geo3K 是一个完整 SAfactory v2 adapter 的参考实现：

- `env/geo3k/runner.py` 实现外部 runtime 契约。
- `env/geo3k/rule_evaluator.py` 将 Geo3K 正确性转换为 SAfactory reward。
- `env/geo3k/Dockerfile` 用于构建本地 Docker 镜像。
- `env/geo3k/datasets/geo3k_sample.jsonl` 提供极小 smoke-test 数据集。
- `rl/examples/geo3k_vl/env.sh` 是标准 RL 模板。

构建 Docker 镜像：

```bash
docker build -t safactory-geo3k:py311 env/geo3k
```

完整 Geo3K 数据集请下载 [chenhegu/geo3k_imgurl](https://huggingface.co/datasets/chenhegu/geo3k_imgurl)，然后修改 `env/geo3k/geo3k_config.yaml` 指向本地 parquet 文件：

```yaml
dataset: chenhegu/geo3k_imgurl/train.parquet
dataset_load_mode: parquet_row_ref
dataset_columns: [problem, answer, images]
```

如果数据集保存在本地绝对路径，请将上面的 Hugging Face 风格示例替换为实际路径。

首次 smoke test 如果不使用完整数据集，建议复制 `env/geo3k/geo3k_config.yaml`，并在本地副本中指向仓库自带样例数据：

```bash
cp env/geo3k/geo3k_config.yaml env/geo3k/geo3k_config.local.yaml
```

```yaml
dataset: ./datasets/geo3k_sample.jsonl
dataset_load_mode: eager
```

同时从该 smoke-test 副本中移除 `dataset_columns`。只有使用 parquet 数据集和列选择时，才保留 `dataset_load_mode: parquet_row_ref`。

Gateway ready 后运行 Geo3K 评测：

```bash
python launcher.py \
  --mode docker \
  --agent-config env/geo3k/geo3k_config.yaml \
  --agent-start-config env/geo3k/geo3k_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model geo3k_model \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --job-id geo3k-docker-smoke \
  --pool-size 1 \
  --max-workers 1 \
  --max-steps 10
```

如果创建了 smoke-test 副本，请把 `--agent-config env/geo3k/geo3k_config.yaml` 替换为 `--agent-config env/geo3k/geo3k_config.local.yaml`。

Geo3K runner 行为：

- 从 `env_params.dataset` 读取当前 case；
- 通过当前 session Gateway URL 发送几何题和可选图片；
- episode 内支持 `calc_score` / `calc_geo3k_reward` 工具调用；
- 抽取 boxed final answer，并用 `math_utils.grade_answer_verl` 判分；
- 将 `metrics.score` 写成 `0.0` 或 `1.0`；
- 由 `rule_evaluator.py` 归一化为 `0` 或 `10` 分。

验证新模型 route、Docker 镜像、Gateway 存储、评测或 RL Buffer Server 链路时，优先用 Geo3K 作为基线。

## OpenClaw

文件：

- `env/openclaw/openclaw_config.yaml`
- `env/openclaw/openclaw_start.yaml`
- `env/openclaw/runner.mjs`

Runner 收到 `SimulationStartRequest` 后，会写入指向 Gateway session URL 的 OpenClaw config，然后运行：

```bash
openclaw agent --local --json --session-id <session_id> --message <task> --model safactory/<route>
```

OpenClaw 适合轻量 CLI 和工具调用 smoke test。运行前需要更新 workspace 挂载和 route key。

## OpenRT

文件：

- `env/openrt/openrt_config.yaml`
- `env/openrt/openrt_start.yaml`
- `env/openrt/runner.py`
- `env/openrt/rule_evaluator.py`

OpenRT 会在容器内运行 `eval.py`，并将 Gateway session URL 作为 OpenAI 兼容 base URL。每条 dataset row 应定义一个 attack 或 benchmark case。

仓库也包含 RJob 示例：

- `env/openrt/openrt_config.rjob.yaml`
- `env/openrt/openrt_start.rjob.yaml`

创建 Geo3K 或自定义 RJob 配置时，可以参考这两个文件。

## WildClawBench

文件：

- `env/wildclawbench/wildclawbench_config.yaml`
- `env/wildclawbench/wildclawbench_start.yaml`
- `env/wildclawbench/runner.py`

该 adapter 需要 WildClawBench checkout 和能运行其任务的 runtime 镜像。运行前请更新 checkout 路径、结果挂载、route key 和 judge route key。

## DTAP

文件：

- `env/dtap/dtap_config.yaml`
- `env/dtap/dtap_start.yaml`
- `env/dtap/runner.py`

DTAP 运行 DecodingTrust-Agent workload。示例会挂载 DecodingTrust-Agent checkout、SAfactory results、DTAP runner 和 `/var/run/docker.sock`，因为 DTAP 可能启动嵌套 Docker workload。

## ClawEnvKit

文件：

- `env/clawenvkit/clawenvkit_config.yaml`
- `env/clawenvkit/clawenvkit_start.yaml`
- `env/clawenvkit/runner.py`

ClawEnvKit 通过 OpenClaw harness 运行 Auto-ClawEval 风格任务。运行前请更新 dataset root、结果挂载、harness entrypoint 和 Gateway route key。

## Dataset 加载

Agent config 的 `dataset` 支持 JSON、JSONL、YAML list 和 parquet。

- JSONL 每行必须是合法 JSON object。
- 相对路径从 agent config 所在目录解析。
- `dataset_load_mode: eager` 会物化数据行。
- `dataset_load_mode: parquet_row_ref` 会保存轻量 parquet row reference。
- `dataset_columns` 只用于 parquet 列选择。

每条 dataset row 会成为 runtime request 中的 `env_params.dataset`。一行 dataset 应代表一个被调度的 episode。

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

在 JSON mode 下，如果 runtime 以非零状态退出，Docker/RJob/Sandbox runner 会视为运行时失败，即使 stdout 中有部分输出。
