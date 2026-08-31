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

## 修复方案：运行时 monkey-patch（已采用）

### 原理

Megatron GDN 的 `forward` 调用的 `chunk_gated_delta_rule`（来自 fla）**本身已支持
`cu_seqlens` 参数**——slime 的 `qwen3_5.py` 就是这么用的。GDN forward 只是在入口处
`raise NotImplementedError` 拦住了 packed_seq_params，没有把 cu_seqlens 传进去。

修复方法：在 GPFS 上放一个 Python 文件，monkey-patch `GatedDeltaNet.forward`，
删掉 raise，把 `packed_seq_params.cu_seqlens_q` 提取出来传给 `chunk_gated_delta_rule`
和 `causal_conv1d_fn`。通过 `sitecustomize.py` + `PYTHONPATH` 在 Python 启动时自动加载。

**不需要重打镜像**，不需要改 Megatron 核心代码，不需要改 slime 代码。

### 为什么不用 bshd（padding）

bshd 模式下 `packed_seq_params=None`，GDN 不崩，但 padding 导致激活内存增大，
27B 模型 TP=4 在 140GB 卡上 OOM（差 822 MiB）。
thd（packing）模式内存更省，是正确选择。

### 为什么 bridge 模式忽略了 --spec

slime `model_provider.py` 第 82-119 行：bridge 模式下直接返回
`bridge.to_megatron_provider().provide`，用 bridge 自带的 spec 构建模型，
`--spec` 参数（slime 的 `qwen3_5.py`，有 cu_seqlens GDN）被完全忽略。
所以不能靠 `--spec` 解决，只能 patch GDN 本身。

### 文件

| 文件 | 作用 |
|------|------|
| `rl/patches/gdn_packed_seq.py` | monkey-patch GatedDeltaNet.forward，支持 cu_seqlens |
| `rl/patches/sitecustomize.py` | Python 启动时自动加载 gdn_packed_seq |

### env.rjob.sh 改动

```bash
# GDN packed-seq monkey-patch: patches Megatron GDN forward to pass cu_seqlens
# to chunk_gated_delta_rule, enabling thd (packing) mode without NotImplementedError.
# See rl/patches/gdn_packed_seq.py for details.
export PYTHONPATH="${REPO_ROOT}/rl/patches${PYTHONPATH:+:${PYTHONPATH}}"
export USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
```

`USE_DYNAMIC_BATCH_SIZE` 保持 `true`（thd packing + dynamic batch size），
`--qkv-format` 用默认 `thd`。

### patch 做了什么

1. 删掉 `raise NotImplementedError("GDN does not support packed sequence for now.")`
2. 从 `packed_seq_params.cu_seqlens_q` 提取 `cu_seqlens`
3. 把 `cu_seqlens` 传给 `chunk_gated_delta_rule(cu_seqlens=...)`（fla 已支持）
4. 把 `cu_seqlens` 转成 `seq_idx` 传给 `causal_conv1d_fn(seq_idx=...)`（避免卷积跨序列边界）

### 回退

删掉 `PYTHONPATH` 那行即可禁用 patch，回到原始 GDN（会 raise）。

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
