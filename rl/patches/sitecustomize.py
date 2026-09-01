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
    import traj_truncation  # noqa: F401  — truncates long trajectories for training
except Exception as _e:
    import sys
    print(f"[sitecustomize] WARNING: traj_truncation failed to load: {_e}", file=sys.stderr)
