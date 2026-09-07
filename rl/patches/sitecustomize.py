"""Auto-load GDN packed-seq monkey-patch at Python startup.

Python imports `sitecustomize` automatically during interpreter startup
from any directory on sys.path.  This file lives in rl/patches/ which
is added to PYTHONPATH by env.rjob.sh, so the patch is applied before
slime/train.py runs — no code change to slime or Megatron needed.
"""
# Import any pre-existing sitecustomize first (chained sitecustomize).
try:
    import _orig_sitecustomize  # noqa: F401
except Exception:
    pass

try:
    import gdn_packed_seq  # noqa: F401  — applies the monkey-patch
except Exception as _e:
    import sys
    print(f"[sitecustomize] WARNING: gdn_packed_seq failed to load: {_e}", file=sys.stderr)

try:
    import spread_placement  # noqa: F401  — SPREAD strategy for multi-node placement
except Exception as _e:
    import sys
    print(f"[sitecustomize] WARNING: spread_placement failed to load: {_e}", file=sys.stderr)

# REMOVED patches (2026-09-08, non-colocate + raw + Megatron-ckpt config):
# - traj_truncation: disabled via TRAJ_TRUNCATION_MAX_SEQ_LEN=0 (PP=2 显存充裕)
# - raw_hf_checkpoint: now loading Megatron-format checkpoints, not HF
# - flush_cache_fix: only needed in colocate mode; non-colocate has no flush issue
# - attention_mask_fix: only needed in bridge mode; using raw mode now
