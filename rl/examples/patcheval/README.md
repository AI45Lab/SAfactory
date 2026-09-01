# PatchEval RL

一个cyber env的RL训练样例

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
| RL_GROUP_SIZE | 8 | 每个 CVE 采样 8 条轨迹 |
| RL_ROLLOUT_GROUP_BATCH_SIZE | 8 | 每批 8 个 CVE |
| RL_GLOBAL_BATCH_SIZE | 64 | 每步训练 64 条轨迹 |
| RL_EPOCH | 100 | 训练轮数 |
| MAX_TOKENS_PER_GPU | 2048 | 微批 token 上限 |
| TRAJ_TRUNCATION_MAX_SEQ_LEN | 8192 | 训练时截断长轨迹，防 OOM |
| OPTIMIZER_CPU_OFFLOAD | true | 优化器卸到 CPU，省 ~40GB 显存 |
| SGLANG_MEM_FRACTION_STATIC | 0.7 | KV cache 池占比 |
| CVE 任务 | 77 个 JS | 每个 300 副本 |
| max_steps | 40 | 每条轨迹最大 LLM 步数 |

## 启动

```bash
# 推理机
ray start --address="<训练机IP>:6379" --num-gpus=8 --disable-usage-stats

# 训练机
ray start --head --node-ip-address="<训练机IP>" --port=6379 --num-gpus=8 --disable-usage-stats

# 训练机 - 窗口1：buffer
export PATCH_EVAL_GENERATED_DIR=$PWD/env/patcheval/generated_openhands_exp1_js77
export RL_ENV_SH=$PWD/rl/examples/patcheval/env.rjob.sh
export CLEANUP_BEFORE_RUN=false
bash rl/run_buffer_server.sh

# 训练机 - 窗口2：训练（等 gateway 起来后）
export SKIP_RAY_START=true
export MASTER_ADDR="<训练机IP>"
bash rl/run_slime_generator.sh
```

## 注意事项

- TP 不能设 8（GQA 约束：num_query_groups=4 必须被 TP 整除）
- 启动前确认 8000 端口空闲，否则 gateway 起不来导致 0 轨迹
- 每步约 40-60 分钟，100 epoch 约 3-4 天
- reward 全 0 是正常的（基座模型难解 CVE），有解出才有学习信号

## TODO

- MAX_TOKENS_PER_GPU：当前 2048 为临时值，需根据模型规模和显存调优
- TRAJ_TRUNCATION_MAX_SEQ_LEN：当前通过 monkey-patch 截断长轨迹，应改为框架原生支持
