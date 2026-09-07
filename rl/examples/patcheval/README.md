# PatchEval RL

一个 cyber env 的 RL 训练样例。

## 硬件要求

- 4 台 8 卡 H200 机器（1 训练 head + 3 推理 worker，共 32 卡）
- 训练 head：`10.102.242.51`，Megatron TP=4 × PP=2 × CP=4，跑 8 卡
- 推理 worker：`10.102.217.14`、`10.102.217.27`、`10.102.217.42`，SGLang colocate
- colocate 模式：训练 + 推理共享全部 32 卡

## 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 模型 | Qwen3.8-27B | GQA，TP 必须整除 num_query_groups=4 |
| TP_SIZE | 4 | 张量并行 |
| PP_SIZE | **2** | 流水线并行，64 层拆 2×32 层 |
| CP_SIZE | **4** | 上下文并行，处理长序列 |
| MEGATRON_TO_HF_MODE | **raw** | 对齐官方，CP 需要 raw 模式 |
| SLIME_COLOCATE | **true** | 训练+推理共享 GPU |
| POOL_SIZE | 16 | 并发环境数 |
| RL_GROUP_SIZE | 8 | 每个 CVE 采样 8 条轨迹 |
| RL_ROLLOUT_GROUP_BATCH_SIZE | 8 | 每批 8 个 CVE |
| RL_GLOBAL_BATCH_SIZE | 64 | = 8×8 |
| RL_OFF_BY_N | 1 | 允许 1 版本偏差（rollout_id 从 1 开始） |
| MAX_TOKENS_PER_GPU | 8192 | 对齐官方 |
| RECOMPUTE_NUM_LAYERS | 1 | full recompute，对齐官方 |
| TRAJ_TRUNCATION_MAX_SEQ_LEN | **0**（禁用） | 不截断，完整保留长轨迹 |
| OPTIMIZER_CPU_OFFLOAD | true | 优化器卸到 CPU，省 ~40GB 显存 |
| SGLANG_MEM_FRACTION_STATIC | 0.75 | KV cache 池占比 |
| SGLANG_MAMBA_SCHEDULER_STRATEGY | extra_buffer | 修复 mamba pool 问题 |
| SGLANG_SPECULATIVE_ALGORITHM | **EAGLE** | 投机解码，对齐官方 |
| ROLLOUT_MAX_RESPONSE_LEN | 32768 | 单轮生成上限，与 LLM_MAX_LENGTH 分离 |
| LLM_MAX_LENGTH | 65536 | 轨迹上限（实际~40K） |
| LOAD_DIR | Megatron 格式 | raw 模式加载 Megatron checkpoint |
| CVE 任务 | 77 个 JS | 每个 300 副本 |
| max_steps | 40 | 每条轨迹最大 LLM 步数 |

## 启动

### 第 1 步：启动 Ray 集群

```bash
# 训练机（10.102.242.51）先起 head
ray stop --force
ray start --head --port=6379 --num-gpus=8 --num-cpus=100 --disable-usage-stats

# 3 台推理机分别执行（不需要 --node-ip-address）
ray stop --force
ray start --address=10.102.242.51:6379 --num-gpus=8 --num-cpus=100 --disable-usage-stats
```

### 第 2 步：验证集群

```bash
# 训练机上跑，必须看到 4 个节点各 8 GPU
python3 -c "
import ray
ray.init(address='auto')
for n in ray.nodes():
    r = n['Resources']
    print(f\"Node: {n['NodeManagerAddress']}  Alive: {n['Alive']}  GPU: {r.get('GPU', 0)}\")
ray.shutdown()
"
```

### 第 3 步：训练机窗口 1 — 启动 buffer server

```bash
cd /mnt/shared-storage-user/leishanzhe/repo/SAfactory
export PATCH_EVAL_GENERATED_DIR=$PWD/rl/examples/patcheval/generated_openhands_exp1_js77
export RL_ENV_SH=$PWD/rl/examples/patcheval/env.rjob.sh
export CLEANUP_BEFORE_RUN=false
bash rl/run_buffer_server.sh
```

### 第 4 步：训练机窗口 2 — 启动训练（等 gateway 起来后）

```bash
cd /mnt/shared-storage-user/leishanzhe/repo/SAfactory
export PATCH_EVAL_GENERATED_DIR=$PWD/rl/examples/patcheval/generated_openhands_exp1_js77
export RL_ENV_SH=$PWD/rl/examples/patcheval/env.rjob.sh
export CLEANUP_BEFORE_RUN=false
export SKIP_RAY_START=true
bash rl/run_slime_generator.sh
```

> **关键**：窗口 2 必须设 `SKIP_RAY_START=true`，否则脚本会 `ray stop` 把已建好的集群杀掉。

## 清理

```bash
# 训练机和推理机都跑一遍
bash rl/cleanup_rl.sh
```

## 注意事项

- TP 不能设 8（GQA 约束：num_query_groups=4 必须被 TP 整除）
- 启动前确认 8000 端口空闲，否则 gateway 起不来导致 0 轨迹
- reward 全 0 是正常的（基座模型难解 CVE），有解出才有学习信号
- **启动前必须验证 Ray 集群**：两个节点不同 IP + 各 8 GPU，否则 SPREAD 会把所有 bundle 放到一台机器导致 Duplicate GPU

## PP vs CP 对比

| | PP=2 | CP=2 |
|---|------|------|
| 切分方式 | 按层切（Stage 0: 0-31层, Stage 1: 32-63层） | 按序列长度切（各处理一半 token） |
| 权重显存 | **减半** ✅ | 不变 ❌ |
| 激活显存 | **减半** ✅ | **减半** ✅ |
| 长序列支持 | ✅ | ✅ |
| 效率 | 有流水线 bubble ⚠️ | 无 bubble ✅ |
| GDN 兼容 | ✅ 原生支持 | ❌ 需要 raw 模式 + hack |
| 实现难度 | 简单 | 复杂 |

**选择 PP=2 的原因**：Qwen3.8-27B 有 64 层 GDN，单卡 TP=4 装不下全部权重 + 长序列激活。PP=2 权重和激活都减半，GDN 原生兼容，不需要 hack。

## Monkey-Patch 补丁说明

所有补丁在 `rl/patches/` 目录，通过 `sitecustomize.py` 在 Python 启动时自动加载，无需修改 slime/Megatron 源码。

| 补丁文件 | 作用 | 加载条件 |
|---|---|---|
| `gdn_packed_seq.py` | 让 GDN 支持 packed sequence（thd 格式） | 总是加载 |
| `traj_truncation.py` | 轨迹截断（TRAJ_TRUNCATION_MAX_SEQ_LEN>0 时生效） | 总是加载 |
| `raw_hf_checkpoint.py` | 允许 raw 模式直接加载 HF checkpoint | 仅 `MEGATRON_TO_HF_MODE=raw` |
| `spread_placement.py` | Ray placement group 策略 PACK→SPREAD（多机分配） | 总是加载 |
| `flush_cache_fix.py` | flush_cache 用 SGLang 内置 `?timeout=60`（v5） | 总是加载 |

> **注意**：`attention_mask_fix.py` 已删除（2026-09-05），仅 bridge 模式需要，raw 模式不需要。

## 2026-09-03 更新日志

### 配置变更

| 参数 | 之前 | 现在 | 原因 |
|------|------|------|------|
| PP_SIZE | 1 | **2** | 流水线并行，权重+激活减半 |
| CP_SIZE | 2 | **1** | PP 替代 CP，GDN 原生兼容 |
| MEGATRON_TO_HF_MODE | raw | **bridge** | PP 需要 bridge 模式 |
| MAX_TOKENS_PER_GPU | 1024 | **2048** | PP 省显存，可增大 |
| TRAJ_TRUNCATION_MAX_SEQ_LEN | 8192 | **0** | 不截断，完整保留长轨迹 |

### 新建文件

1. `rl/patches/attention_mask_fix.py` — 修复 bridge 模式 `preprocess_packed_seqs` 收到 `attention_mask=None` 的 bug（import hook）
2. `rl/examples/patcheval/generated_openhands_exp1_js77/` — 重新生成 77 个 JS CVE 的 rjob 配置（env_num=300）

### 修改文件

1. `rl/patches/sitecustomize.py` — `raw_hf_checkpoint` 改为仅 raw 模式加载；新增 `attention_mask_fix` 注册
2. `rl/examples/patcheval/env.rjob.sh` — PP=2 / CP=1 / bridge / MAX_TOKENS=2048 / TRAJ_TRUNCATION=0

### 解决的问题

| 问题 | 根因 | 修复 |
|------|------|------|
| Duplicate GPU (PP=2) | 推理机没加入 Ray 集群，16 bundle 全挤训练机 | 在推理机启动 `ray start --address` |
| KeyError: 0 (PP=2) | `raw_hf_checkpoint` 补丁在 bridge 模式也生效，层映射不兼容 PP | 改为仅 raw 模式加载 |
| AttributeError: 'NoneType' (PP=2) | slime 传 `attention_mask=None`，bridge 模型需要有效 tensor | `attention_mask_fix.py` 补丁（import hook） |

## 2026-09-04 更新日志

### 配置变更（对齐官方 Qwen3.5-27B 脚本）

| 参数 | 之前 | 现在 | 原因 |
|------|------|------|------|
| TP_SIZE | 4 | 4 | 不变（GQA 约束） |
| PP_SIZE | 2 | **2** | 不变 |
| CP_SIZE | 1 | **4** | 对齐官方，CP 处理长序列 |
| MEGATRON_TO_HF_MODE | bridge | **raw** | CP 需要 raw 模式 |
| RECOMPUTE_NUM_LAYERS | 32 | **1** | 对齐官方，full recompute |
| MAX_TOKENS_PER_GPU | 2048 | **8192** | 对齐官方 |
| SLIME_COLOCATE | false | **true** | 训练+推理共享 GPU |
| SGLANG_MEM_FRACTION_STATIC | 0.7 | **0.75** | 对齐官方 |
| ROLLOUT_NUM_GPUS_PER_ENGINE | 1 | **2** | 对齐官方 |
| DECODER_LAST_PIPELINE_NUM_LAYERS | - | **30** | 对齐官方，PP 不均匀切分 |
| SGLANG_MAMBA_SCHEDULER_STRATEGY | auto | **extra_buffer** | 修复 mamba pool illegal memory access |
| SGLANG_SPECULATIVE_ALGORITHM | - | **关闭** | raw 模式下 EAGLE 导致 mamba_pool 崩溃（详见下方） |
| ROLLOUT_NUM_GPUS | 16 | **不传**（colocate 自动=32） | 对齐官方，用满 32 卡 |
| RL_GROUP_SIZE | 8 | **2** | 减小长尾影响 |
| RL_ROLLOUT_GROUP_BATCH_SIZE | 8 | **1** | 减小长尾影响 |
| RL_GLOBAL_BATCH_SIZE | 64 | **2** | = group × batch |
| ROLLOUT_MAX_RESPONSE_LEN | =LLM_MAX_LENGTH(131072) | **32768** | 限制单轮生成，避免长尾卡死 offload |
| ROLLOUT_NUM_PROCESS | 100（默认） | **2**（=global_batch_size） | 避免 offload 时大量 pending 请求卡死 flush |
| LOAD_DIR | HF 格式 | **Megatron 格式** | 对齐官方，raw 模式加载 Megatron checkpoint |

### 新增参数

- `ROLLOUT_MAX_RESPONSE_LEN=32768` — 单轮生成 token 上限，与 `LLM_MAX_LENGTH`（轨迹上限）分离
- `ROLLOUT_NUM_PROCESS` — 并发 env 进程数，默认 = `RL_GLOBAL_BATCH_SIZE`，避免 flush_cache 超时

### 新增补丁

- `rl/patches/flush_cache_fix.py` — flush_cache 前先 abort 所有 pending 请求，避免长尾 env 卡死 offload（colocate 模式必需）

### 修改文件

1. `rl/examples/patcheval/env.rjob.sh` — CP=4 / raw / colocate / recompute=1 / MAX_TOKENS=8192 / EAGLE 关闭 / mamba extra_buffer / group=2 / ROLLOUT_MAX_RESPONSE_LEN=32768 / LOAD_DIR=Megatron格式
2. `rl/run_slime_generator.sh` — colocate 时不传 `--rollout-num-gpus`（自动=actor GPUs）；`--rollout-max-response-len` 改用 `ROLLOUT_MAX_RESPONSE_LEN`；新增 `--rollout-num-process`；EAGLE 条件传递；mamba/decoder-last-pipeline 参数传递；GRPO args 对齐官方
3. `rl/patches/sitecustomize.py` — 新增 `flush_cache_fix` 注册
4. `rl/patches/flush_cache_fix.py` — flush_cache 前 abort pending 请求
5. HF checkpoint 已转换为 Megatron 格式（`/mnt/shared-storage-user/evobox-share-gpfs2/leishanzhe/model/qwen3_8_27b_megatron`），raw 模式直接加载

### 解决的问题

| 问题 | 根因 | 修复 |
|------|------|------|
| `TimeoutError: Timeout while flushing cache`（第一次） | 单轮 `max_tokens=131072`，长尾轨迹跑满 128K token（~9分钟），offload 时 flush_cache 60秒超时 | 新增 `ROLLOUT_MAX_RESPONSE_LEN=32768`，单轮最多 32K token（~2分钟） |
| `TimeoutError: Timeout while flushing cache`（第二次） | `num_process=100`（默认），收齐 2 条轨迹后还有 98 个 env 在跑，SGLang 有 pending 请求，flush_cache 60秒等不完 | 新增 `--rollout-num-process`，默认 = `RL_GLOBAL_BATCH_SIZE`（当前=2）；新增 `flush_cache_fix.py` 补丁，flush 前 abort 所有 pending 请求 |
| `CUDA error: illegal memory access` (mamba_pool, rollout 阶段) | SGLang mamba cache pool 默认 `auto` 策略不兼容 | 设 `SGLANG_MAMBA_SCHEDULER_STRATEGY=extra_buffer` |
| `CUDA error: illegal memory access` (mamba_pool, log_probs 阶段) | EAGLE 投机解码 + mamba + raw 模式 + colocate flush/wake_up 循环不兼容：flush_cache 清了主模型 mamba state，但 EAGLE draft model 的 mamba state 不一致 → log_probs 时非法内存访问 | **关闭 EAGLE**（`SGLANG_SPECULATIVE_ALGORITHM` 留空）。官方脚本用 bridge 模式 EAGLE 正常，但 raw 模式下不兼容 |
| `RuntimeError: TorchMemorySaver is disabled` | colocate 时 `expandable_segments:True` 与 `torch_memory_saver` 冲突 | `run_slime_generator.sh` 的 `RUNTIME_ENV_JSON` 改为不强制传 `expandable_segments` |
| `IndexError: index 32 is out of range` (recompute) | PP=2 每 stage 32 层，`RECOMPUTE_NUM_LAYERS=64` 越界 | 改为 `RECOMPUTE_NUM_LAYERS=1`（对齐官方） |
| `KeyError: 'model.language_model.layers.0...'` (转换) | mbridge 转换缺 model spec | 转换命令加 `--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec` |
| `ModuleNotFoundError: megatron.training` (转换) | PYTHONPATH 缺 Megatron | `PYTHONPATH=/root/Megatron-LM:$PYTHONPATH` |

### EAGLE + raw 模式不兼容问题详解

**现象**：rollout 阶段正常，但训练的 log_probs 计算阶段 `mamba_pool.alloc` 报 `CUDA error: illegal memory access`。

**根因**：
1. EAGLE 投机解码有独立的 draft model，也有自己的 mamba state
2. colocate 模式下，offload 时 `flush_cache` 清了主模型的 KV cache + mamba state
3. `wake_up` 重新加载 Megatron 模型到 GPU
4. log_probs 阶段重新调用 SGLang，但 EAGLE draft model 的 mamba state 在 flush 后**状态不一致** → 非法内存访问

**官方 vs 我们**：
- 官方用 `bridge` 模式 + EAGLE，正常工作
- 我们用 `raw` 模式（因 GDN+CP 在 bridge 模式下不兼容）+ EAGLE，崩溃
- **结论**：raw 模式下 EAGLE 不兼容，必须关闭

**代价**：推理速度变慢（无投机解码加速），但训练能跑通。待 SGLang 修复后可重新启用。

### `max_tokens` vs `LLM_MAX_LENGTH` 说明

| 参数 | 限制对象 | 值 | 代码位置 |
|------|----------|-----|----------|
| `LLM_MAX_LENGTH` | **整个轨迹**（多轮对话 input+output 累计） | 131072 | `llm_proxy.py` `state.max_length` |
| `ROLLOUT_MAX_RESPONSE_LEN` | **单轮生成**（每次 LLM 调用） | 32768 | `slime_generator.py` → SGLang `sampling_params.max_tokens` |

`llm_proxy.py` 实际生效逻辑：`max_new_tokens = min(ROLLOUT_MAX_RESPONSE_LEN, LLM_MAX_LENGTH - 当前input_ids长度)`

**之前的问题**：两个值都是 131072，单轮就能跑满 128K，长尾轨迹 ~9 分钟不结束 → flush_cache 超时 → job 崩溃。

**修复后**：单轮最多 32K（~2 分钟），多轮累计可达 131K，长尾可控。

## 2026-09-05 更新日志

### 发现：补丁可能是 CUDA 崩溃的根因

**现象**：`20260904-115405` run 中，rollout 成功收集 64 条样本（`RL_OFF_BY_N=1` 修复生效），但训练 wake_up 后 SGLang 引擎 `CUDA error: illegal memory access`，所有引擎崩溃，job 失败。

**根因分析**：
- 官方 Qwen3.5-27B 脚本就是 `raw 模式 + EAGLE + colocate + CP=4`，**不需要任何补丁**就能跑通
- 我们之前加了两个补丁，反而可能破坏了稳定性：
  1. `flush_cache_fix.py` — flush_cache 前发 `/abort_request` 强制终止所有 pending 请求，可能破坏 mamba state pool 的一致性 → resume_memory_occupation 时非法内存访问
  2. `attention_mask_fix.py` — bridge 模式的补丁，raw 模式不需要，但仍在加载，可能干扰 raw 模式的 preprocess 流程

**官方 vs 我们对比**：

| | 官方 Qwen3.5-27B | 我们 |
|---|---|---|
| megatron-to-hf-mode | raw（默认） | raw |
| EAGLE | ✅ 开 | ❌ 关了（误判） |
| colocate | ✅ | ✅ |
| CP/TP/PP | 4/4/2 | 4/4/2 |
| 补丁 | 无 | flush_cache_fix + attention_mask_fix |
| 数据 | dapo-math-17k（单轮） | PatchEval（多轮 agent） |

**结论**：应该去掉补丁、开 EAGLE，完全对齐官方脚本。flush_cache 超时问题用其他方式解决（见下方）。

### 补丁处理

| 补丁 | 处置 | 原因 |
|---|---|---|
| `attention_mask_fix.py` | ❌ **删除** | bridge 模式补丁，raw 模式不需要 |
| `flush_cache_fix.py` | ✅ **保留（v5）** | flush_cache 用 SGLang 内置 `?timeout=60`；abort 移到 slime_generator rollout 阶段 |
| `gdn_packed_seq.py` | ✅ 保留 | GDN packed sequence 支持，官方也需要 |
| `traj_truncation.py` | ✅ 保留 | 轨迹截断（当前禁用，TRAJ_TRUNCATION_MAX_SEQ_LEN=0） |
| `raw_hf_checkpoint.py` | ✅ 保留 | raw 模式加载 checkpoint（已转 Megatron 格式，可能不需要） |
| `spread_placement.py` | ✅ 保留 | 多机 placement group SPREAD 策略 |

### `flush_cache_fix.py` v5 — 用 SGLang 内置 timeout，abort 移到 rollout 阶段

**v1-v4 的问题**：在 flush_cache 阶段发 `/abort_request`，但 abort 在 scheduler stream 操作 mamba state，forward stream 可能还在写 → **race condition** → `CUDA error: illegal memory access`（SGLang issue #24221, #24954）。固定 sleep（2s/10s/60s）不能可靠避免这个 race。

**v5 的改进**（基于 SGLang 源码调研）：
1. **abort 移到 rollout 阶段**（`slime_generator.py`）：收齐数据后先 `/stop_rollout` kill envs，再向所有 SGLang worker 发 `/abort_request`（和 slime 原版 `abort()` 一致）。abort 在 rollout 阶段发生，forward stream 有充足时间在 offload/flush 前稳定。
2. **flush_cache 用 SGLang 内置 `?timeout=60`**（SGLang PR #21413）：scheduler 在 event loop 里轮询 `is_fully_idle()`，继续跑 forward stream 排空 in-flight batch，idle 后自动 flush。不再在 flush 阶段 abort，避免 scheduler/forward stream race。
3. **不需要固定 sleep**：让 SGLang 自己的 idleness check 决定时机。

**三步流程**：
```
1. /stop_rollout → kill envs              ← 断新请求来源
2. /abort_request → 清残余请求（rollout 阶段）  ← 强制终止 SGLang 队列长生成
3. /flush_cache?timeout=60 → scheduler 轮询 idle  ← SGLang 自己确认 idle 后 flush
```

### 配置变更

| 参数 | 之前 | 现在 | 原因 |
|------|------|------|------|
| SGLANG_SPECULATIVE_ALGORITHM | 空（关闭） | **EAGLE** | 对齐官方，raw 模式 + EAGLE 官方支持 |
| RL_OFF_BY_N | 0 | **1** | 修复首条 rollout 版本过滤问题（rollout_id 从1开始，current_version=rollout_id+1=2，但 weight_version=1） |
| LLM_MAX_LENGTH | 131072 | **65536** | 轨迹实际~40K，64K够用，降低长尾风险 |
| RL_GROUP_SIZE | 2 | **8** | 恢复大 batch 实验 |
| RL_ROLLOUT_GROUP_BATCH_SIZE | 1 | **8** | 恢复大 batch 实验 |
| RL_GLOBAL_BATCH_SIZE | 2 | **64** | = 8×8 |

### 修改文件

1. `rl/patches/sitecustomize.py` — 移除 `attention_mask_fix`；`flush_cache_fix` 注释更新为 v5
2. `rl/examples/patcheval/env.rjob.sh` — EAGLE 重新开启；LLM_MAX_LENGTH=65536；RL_OFF_BY_N=1；group=8/group_size=8/global_batch=64
3. `rl/patches/flush_cache_fix.py` — v5：去掉 abort，改用 `/flush_cache?timeout=60`
4. `rl/slime_generator.py` — 收齐数据后先 `/stop_rollout`（kill envs），再向所有 SGLang worker 发 `/abort_request`（清残余），然后 sleep(3)
5. `rl/buffer_server.py` — 新增 `/stop_rollout` 端点（kill `aievobox_process` 子进程树，不杀进程组）

### 解决的问题

| 问题 | 根因 | 修复 |
|------|------|------|
| 所有轨迹被版本过滤丢弃 | `current_version = rollout_id + 1 = 2`，但 `weight_version = 1`，`off_by_n=0` 导致全过滤 | `RL_OFF_BY_N=1`，允许 1 版本偏差 |
| CUDA illegal memory access (wake_up 后) | `flush_cache_fix.py` v1-v4 在 flush 阶段 abort，scheduler stream 清 mamba state 与 forward stream 写竞争（SGLang issue #24221，PR #24954 修复但 0.5.9 未包含） | v5：abort 移到 rollout 阶段，flush 用 `?timeout=60` 不 abort |
| EAGLE 误判为不兼容 | 之前在 raw 模式崩溃，误判为 EAGLE+raw 不兼容；实际是 flush_cache_fix 的 abort 破坏了 mamba state | v5 修复后重新开启 EAGLE |
| flush_cache 超时 | 长尾 env 持续发新请求，SGLang 永不 idle | `/stop_rollout` kill envs + abort 残余 + `?timeout=60` |
| step 2 连不上 buffer server | `/stop_rollout` 用 `os.killpg(os.getpgid(pid))` 杀整个进程组，buffer server 和 launcher.py 同组 → 一起被杀 | 改用 `ps --ppid` 逐层找子进程单独 kill，不杀进程组 |

### step 1 验证结果（20260905-164831）

step 1 完整跑通，v5 方案验证成功：

| 阶段 | 耗时 | 状态 |
|------|------|------|
| rollout（64 samples） | 1932s (~32min) | ✅ |
| flush_cache | <1s | ✅ 无超时 |
| wake_up | 7.4s | ✅ 无 CUDA error |
| log_probs | 1249.9s (~21min) | ✅ |
| actor_train | 2370.6s (~39min) | ✅ loss=0.0（reward 全 0） |
| update_weights | 37.5s | ✅ |
| step 2 rollout | — | ❌ buffer server 被 `/stop_rollout` 误杀（已修复） |

### SGLang 版本与 mamba race 说明

SGLang 0.5.9（当前训练机版本）**不包含** PR #24954（把 mamba state 操作移到 forward stream，消除 scheduler/forward stream race）。该 PR 合并于 2026-05-19，首个包含的正式版本是 **v0.5.13**（2026-06-13）。

v5 方案通过将 abort 提前到 rollout 阶段（flush 阶段不 abort），规避了这个 race。如未来升级 SGLang 到 v0.5.13+，abort 将天然安全，可考虑简化补丁。

## TODO

- **验证**：`/stop_rollout` 修复后（不杀进程组），step 2+ 能否连续跑通多个 step
- 如 flush_cache 仍超时：可升级 SGLang 到 v0.5.13+（包含 PR #24954，abort 天然安全），或加 `SGLANG_DISABLE_OVERLAP_SCHEDULE=true` 彻底消除 race
- MAX_TOKENS_PER_GPU：当前 8192，可根据显存余量继续调优
- reward 全 0 问题：考虑 partial-credit reward（参考 SWE-RL / SecureCodeRL）或混合简单任务
- step 1 耗时分析：rollout 32min + log_probs 21min + actor_train 39min = ~93min/step，100 epoch 需要 ~155 小时（6.5 天）
