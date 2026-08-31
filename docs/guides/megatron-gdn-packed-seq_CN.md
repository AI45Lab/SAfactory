# Megatron GDN 不支持 Packed Sequence 的修复

## 问题现象

Qwen3.8-27B（混合架构：48 层 Linear Attention / GDN + 16 层 Full Attention）在 slime RL
训练时，第一步 `compute_log_prob` 即崩溃：

```
NotImplementedError: GDN does not support packed sequence for now.
  File "/root/Megatron-LM/megatron/core/ssm/gated_delta_net.py", line 302, in forward
    raise NotImplementedError("GDN does not support packed sequence for now.")
```

## 根因

### 1. slime 默认用 packed sequence（thd 格式）

slime 的 `slime/backends/megatron_utils/data.py::get_batch` 根据 `--qkv-format` 参数决定数据布局：

| qkv_format | 布局 | packed_seq_params | 说明 |
|------------|------|-------------------|------|
| `thd`（默认） | T-H-D，多条序列拼接成一条长流 | 非 None（含 cu_seqlens） | packing，省算力 |
| `bshd` | B-S-H-D，多条序列堆叠成 batch（padding 到等长） | None | padding，无 packing |

`thd` 模式下，micro-batch 里的多条变长 trajectory 被 **concat 成一条长序列**，
用 `cu_seqlens` 标记边界，通过 `PackedSeqParams` 传入 Megatron 各层。

### 2. Megatron GDN 显式拒绝 packed sequence

`/root/Megatron-LM/megatron/core/ssm/gated_delta_net.py` 第 300-302 行：

```python
if packed_seq_params is not None:
    # TODO: support packed sequence
    raise NotImplementedError("GDN does not support packed sequence for now.")
```

GDN（Gated DeltaNet）的递推状态在序列边界会"泄漏"到下一条序列，当前实现没有用
`cu_seqlens` 做边界隔离，所以直接 raise。

### 3. slime 的 qwen3_5 spec 没有生效

slime 自带的 `slime_plugins/models/qwen3_5.py` 里有 `Qwen3_5GatedDeltaNet`，
它用 fla 的 `chunk_gated_delta_rule(cu_seqlens=...)` 支持 packed sequence。
但 megatron-bridge 的 Qwen3 VL 模型（`megatron.bridge.models.qwen_vl.modelling_qwen3_vl`）
构建自己的 transformer block spec，**忽略了 slime 的 spec 替换**，
线性注意力层用的是 Megatron 原生 GDN，而非 slime 的实现。

调用链（从 traceback 提取）：

```
actor.train_actor → compute_log_prob → forward_only
  → forward_backward_no_pipelining → forward_step
    → megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.forward
      → text_model.forward → decoder
        → megatron.bridge...transformer_block.forward
          → transformer_layer.forward → _forward_attention
            → self.self_attention(...)
              → megatron.core.ssm.gated_delta_net.forward  ← raise NotImplementedError
```

## 修复方案：禁用 packed sequence（改用 bshd / padding）

### 原理

把 `--qkv-format` 从 `thd` 改成 `bshd`：
- micro-batch 里的多条序列 **堆叠成 batch 维度**（padding 到等长）
- `packed_seq_params = None`
- GDN 的 `forward` 不会进入 `if packed_seq_params is not None` 分支，不触发 raise

### 约束

slime `arguments.py` 第 1764-1768 行的断言：

```python
if args.qkv_format == "bshd":
    assert args.train_backend == "megatron"
    assert args.use_dynamic_batch_size is False, \
        "Dynamic batch size is not supported for bshd format. Please specify --micro-batch-size instead."
```

即 `bshd` 模式：
- 必须是 megatron backend（当前已是）
- **不能用 dynamic batch size**，必须指定 `--micro-batch-size`

### 改动

#### `rl/examples/patcheval/env.rjob.sh`

```bash
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-5000}"
# bshd (padding) instead of thd (packing): Megatron GDN does not support packed
# sequences (NotImplementedError). bshd pads sequences in a micro-batch to equal
# length instead of packing them into one stream, so packed_seq_params is None
# and GDN's forward never hits the raise. Requires fixed micro-batch-size (no
# dynamic batch size). Costs some compute on padding tokens.
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-false}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export QKV_FORMAT="${QKV_FORMAT:-bshd}"
```

#### `rl/run_slime_generator.sh`

```bash
TRAIN_ARGS=(
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --qkv-format "${QKV_FORMAT:-thd}"
)
if is_true "${USE_DYNAMIC_BATCH_SIZE}"; then
  TRAIN_ARGS+=(--use-dynamic-batch-size)
else
  TRAIN_ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE:-1}")
fi
```

### 代价

| 项目 | thd（packing） | bshd（padding） |
|------|----------------|-----------------|
| 算力浪费 | 无（无 padding） | 有（短序列被 pad 到等长） |
| batch 调度 | dynamic（自动平衡） | 固定 micro-batch-size |
| GDN 兼容 | ❌ 崩 | ✅ 不崩 |

- `MICRO_BATCH_SIZE=1`：无 padding，但每步只处理 1 条序列，吞吐最低
- `MICRO_BATCH_SIZE=2~4`：吞吐提高，但 padding 浪费增加
- 建议先用 `1` 验证训练能跑通，再调大找效率甜点

### 回退

如果以后 Megatron GDN 实现了 packed sequence 支持，或 megatron-bridge 修复了
spec 替换问题，可以改回 thd 模式恢复 packing 效率：

```bash
export QKV_FORMAT=thd
export USE_DYNAMIC_BATCH_SIZE=true
```

## 其他可选方案（未采用）

### 方案 A：改 Megatron GDN forward 支持 packed sequence

在 `gated_delta_net.py` 的 `forward` 里，用 `packed_seq_params.cu_seqlens_q` 拆分
`hidden_states` 为独立序列，逐段跑 GDN 递推（每段重置状态），再拼回去。
工作量大，需要理解 GDN 内部递推逻辑，且改的是 Megatron 核心代码。

### 方案 B：让 megatron-bridge 用 slime 的 qwen3_5 spec

slime 的 `qwen3_5.py` 已有支持 `cu_seqlens` 的 `Qwen3_5GatedDeltaNet`（用 fla 的
`chunk_gated_delta_rule`）。需要查 megatron-bridge 的 spec 构建逻辑，让它用 slime
的 `Attention` 类替代原生 GDN。这是最正确的长期修复，但需要深入 bridge 模型代码。

## 部署注意

训练容器（`registry.h.pjlab.org.cn/.../szsz:slime-0.3.1-safactory-v2-docker-20260819112130`）
里的 `/root/Megatron-LM` 是**打进镜像的**，rjob 不挂载该路径。本方案改的是
slime 的 `env.rjob.sh` 和 `run_slime_generator.sh`（在 GPFS 共享存储上），
训练容器通过 `--mount=gpfs://gpfs1/leishanzhe:...` 挂载，所以改完即可生效，
**不需要重新打镜像**。
