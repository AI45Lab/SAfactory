"""Runtime monkey-patch: make Megatron GDN support packed sequences (thd).

Megatron's `GatedDeltaNet.forward` raises `NotImplementedError` when
`packed_seq_params` is provided.  However, the core recurrence op
`chunk_gated_delta_rule` (from fla) already accepts a `cu_seqlens` argument
for variable-length packed sequences — slime's own `qwen3_5.py` uses exactly
this.  This patch removes the raise and forwards `cu_seqlens` through, so
slime can use `--qkv-format thd` (packing) without OOM and without hitting
the GDN limitation.

Loaded automatically when this directory is on PYTHONPATH (set in
env.rjob.sh).  No Megatron source modification, no image rebuild.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.gated_delta_net import (
    chunk_gated_delta_rule,
    torch_chunk_gated_delta_rule,
    causal_conv1d_fn,
    l2norm,
)
from megatron.core.transformer.utils import deprecate_inference_params
from megatron.core.utils import nvtx_range_push, nvtx_range_pop

_original_forward = GatedDeltaNet.forward


def _patched_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_context=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    rotary_pos_cos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    *,
    inference_params=None,
):
    inference_context = deprecate_inference_params(inference_context, inference_params)

    seq_len, batch, _ = hidden_states.shape
    seq_len = seq_len * self.sp_size

    if inference_context is not None:
        raise NotImplementedError("GDN does not support inference for now.")

    # --- packed sequence support: extract cu_seqlens ---
    cu_seqlens = None
    if packed_seq_params is not None:
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)

    # Input projection
    nvtx_range_push(suffix="in_proj")
    qkvzba, _ = self.in_proj(hidden_states)
    nvtx_range_pop(suffix="in_proj")

    qkvzba = qkvzba.transpose(0, 1)  # sbhd -> bshd

    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            (self.qk_dim * 2 + self.v_dim) // self.tp_size,
            self.v_dim // self.tp_size,
            self.num_value_heads // self.tp_size,
            self.num_value_heads // self.tp_size,
        ],
        dim=-1,
    )
    gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
    beta = beta.reshape(batch, seq_len, -1)
    alpha = alpha.reshape(batch, seq_len, -1)

    # Convolution on qkv
    qkv = qkv.transpose(1, 2).contiguous()  # b,s,d -> b,d,s
    nvtx_range_push(suffix="conv1d")
    if (causal_conv1d_fn is None) or self.config.deterministic_mode:
        qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
    else:
        # causal_conv1d_fn supports seq_idx for varlen; convert cu_seqlens
        # to seq_idx [batch, seq_len] when available.
        seq_idx = None
        if cu_seqlens is not None:
            seq_idx = torch.zeros(batch, seq_len, dtype=torch.int32, device=qkv.device)
            for i in range(len(cu_seqlens) - 1):
                start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
                if end > start:
                    seq_idx[:, start:end] = i
        qkv = causal_conv1d_fn(
            x=qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
    nvtx_range_pop(suffix="conv1d")

    qkv = qkv.transpose(1, 2)  # b,d,s -> b,s,d
    query, key, value = torch.split(
        qkv,
        [self.qk_dim // self.tp_size, self.qk_dim // self.tp_size, self.v_dim // self.tp_size],
        dim=-1,
    )
    query = query.reshape(batch, seq_len, -1, self.key_head_dim)
    key = key.reshape(batch, seq_len, -1, self.key_head_dim)
    value = value.reshape(batch, seq_len, -1, self.value_head_dim)

    if self.use_qk_l2norm:
        query = l2norm(query.contiguous())
        key = l2norm(key.contiguous())
    if self.num_value_heads // self.num_key_heads > 1:
        query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
        key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    alpha = alpha.contiguous()

    nvtx_range_push(suffix="g_and_beta")
    g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)
    beta = beta.sigmoid()
    nvtx_range_pop(suffix="g_and_beta")

    nvtx_range_push(suffix="gated_delta_rule")
    if self.config.deterministic_mode:
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
        )
    else:
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
        )
    nvtx_range_pop(suffix="gated_delta_rule")

    nvtx_range_push(suffix="gated_norm")
    norm_out = self._apply_gated_norm(core_attn_out, gate)
    nvtx_range_pop(suffix="gated_norm")

    norm_out = norm_out.reshape(batch, seq_len, -1)
    norm_out = norm_out.transpose(0, 1).contiguous()

    nvtx_range_push(suffix="out_proj")
    out, out_bias = self.out_proj(norm_out)
    nvtx_range_pop(suffix="out_proj")

    return out, out_bias


GatedDeltaNet.forward = _patched_forward
