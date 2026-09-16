# Harbor RL

一个 Harbor (vulhub-exploit) 环境的 RL 训练样例，setting 对齐 `rl/examples/patcheval`。

## 与 PatchEval 的区别

| 维度 | PatchEval | Harbor |
|---|---|---|
| 环境 | 77 个 CVE，每个一个 env_name，env_num=300 | 单个 env_name `harbor`，dataset 474 个 vulhub 任务 |
| 配置生成 | 需先跑 `env/patcheval/generate_full_config.py` 生成 generated dir | 无生成步骤，rjob 配置直接提交在 `env/harbor/` |
| Agent | openhands | claude-code（`env_params.agent`，runner.py 把 `ANTHROPIC_BASE_URL` 指向 RL gateway） |
| 镜像 | 每个 CVE 一个 tarball，从 archive dir 加载 | 单一 runtime 镜像 `safactory-harbor-runtime-v0.21.0`，RJob pod 从 registry 拉 |
| Reward | patcheval rule 校验 | Harbor verifier 0/1，`rule_evaluator.py` 归一化到 0-10 |
| 单 episode 时长 | ~40 LLM 步 | timeout_s=9000（2.5h），更长 |

## 硬件要求

- 2 台 8 卡 H200 机器（训练机 + 推理机，共 16 卡）
- 训练机：Megatron TP=4，跑 8 卡
- 推理机：SGLang 8 引擎，每引擎 1 卡

## 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | Qwen3.8-27B | GQA，TP 必须整除 num_query_groups=4 |
| TP_SIZE | 4 | 张量并行 |
| POOL_SIZE | 16 | 并发环境数 |
| RL_GROUP_SIZE | 8 | 每个 vulhub 任务采样 8 条轨迹 |
| RL_ROLLOUT_GROUP_BATCH_SIZE | 8 | 每批 8 个任务 |
| RL_GLOBAL_BATCH_SIZE | 64 | 每步训练 64 条轨迹 |
| RL_EPOCH | 100 | 训练轮数（474 任务 / 8 ≈ 59 batch/epoch） |
| MAX_TOKENS_PER_GPU | 2048 | 微批 token 上限 |
| TRAJ_TRUNCATION_MAX_SEQ_LEN | 8192 | 训练时截断长轨迹，防 OOM |
| OPTIMIZER_CPU_OFFLOAD | true | 优化器卸到 CPU，省 ~40GB 显存 |
| SGLANG_MEM_FRACTION_STATIC | 0.7 | KV cache 池占比 |
| GATEWAY_MAX_STEPS | 60 | 每条轨迹最大 LLM 步数（vulhub 比 CVE 补丁长） |
| AGENT_START_TIMEOUT_S | 3600 | RJob pod 启动超时（嵌套 dockerd + 镜像拉取） |
| Vulhub 任务 | 474 个 | `env/harbor/datasets/harbor_vulhub_all.jsonl` |

## 启动

```bash
# 推理机
ray start --address="<训练机IP>:6379" --num-gpus=8 --disable-usage-stats

# 训练机
ray start --head --node-ip-address="<训练机IP>" --port=6379 --num-gpus=8 --disable-usage-stats

# 训练机 - 窗口1：buffer
export HARBOR_VARIANT=vulhub_all
export RL_ENV_SH=$PWD/rl/examples/harbor/env.rjob.sh
export CLEANUP_BEFORE_RUN=false
bash rl/run_buffer_server.sh

# 训练机 - 窗口2：训练（等 gateway 起来后）
export SKIP_RAY_START=true
export MASTER_ADDR="<训练机IP>"
bash rl/run_slime_generator.sh
```

### 变体

`HARBOR_VARIANT` 选择不同的 harbor rjob 配置（都已在 `env/harbor/` 提交）：

| 变体 | config / start | 任务数 | 用途 |
|---|---|---|---|
| `vulhub_all`（默认） | `harbor_vulhub_all_*` | 474 | 正式训练 |
| `cvebench` | `harbor_cvebench_*` | 1 | cvebench smoke，端到端联调 |
| `smoke` | `harbor_*`（无后缀） | 1 | oracle smoke，验证 rjob 管线 |

## 注意事项

- TP 不能设 8（GQA 约束：num_query_groups=4 必须被 TP 整除）
- 启动前确认 8000 端口空闲，否则 gateway 起不来导致 0 轨迹
- Harbor episode 比 PatchEval 长得多（timeout_s=9000 vs 900），单步训练时间显著更长，先小 POOL_SIZE / 小 NUM_ROLLOUT 跑通再放量
- reward 全 0 是正常的（基座模型难解 vulhub），有解出才有学习信号
- `AIEVOBOX_DOCKER_IMAGE_ARCHIVE_DIR` 留空：harbor runtime 镜像由 RJob pod 从 registry 拉，不需要本地 tarball
- RJob pod 是 privileged（嵌套 Docker 跑 vulnerable 服务），已在 `*_start.rjob.yaml` 里设 `privileged: true`

## TODO

- GATEWAY_MAX_STEPS=60 是初值，需根据实际 vulhub 解出率 / 吞吐调优
- LLM_MAX_LENGTH=131072：vulhub 多轮 exploit 轨迹长，需确认 SGLang 显存够用，必要时降 POOL_SIZE
