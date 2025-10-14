# Agent Environments - SQLGym

## Setup

``` sh
cd env/sqlgym
pip install -e .
```

## Launch

``` sh
SQLGYM_BIRD_PATH=/mnt/shared-storage-user/chenxinquan/ai_sandbox/env/data/bird sqlgymlaunch --host 0.0.0.0 --port 36001
```

## Item ID

| Item ID      | Description        |
| ------------ | ------------------ |
| 0 ~ 9427     | Train set for BIRD |
| 9428 ~ 10961 | Dev set for BIRD   |
