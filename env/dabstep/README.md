# DABStepEnv

DABStep benchmark 的 Gymnasium 环境封装，支持自动数据下载、多进程分片、逐步渲染。

---

## 快速开始

### 1. 前置依赖
```bash
# 1. 克隆 AIEvoBox 仓库
git clone https://gitee.pjlab.org.cn/L2/safeai/kilab/AIEvoBox.git

# 2. 安装依赖
cd AIEvoBox
pip install -r requirements.txt

# 3. 安装官方评测库（可选，不安装会回退到近似评分）
pip install git+https://huggingface.co/spaces/adyen/DABstep.git@main

# 4. 进入 dabstep 目录
cd env/dabstep
```

数据会在首次初始化时**自动从 HuggingFace 下载**，无需手动准备。

---

## 数据说明

数据来源：[adyen/DABstep](https://huggingface.co/datasets/adyen/DABstep)，CC-BY-4.0 协议。

本项目**不包含任何数据文件**，初始化时自动下载到 `data_dir` 指定目录：

```
<data_dir>/
├── context/          # CSV/JSON 数据文件（via snapshot_download）
└── tasks/
    ├── default_tasks.jsonl   # 450 个任务（无答案，用于提交 leaderboard）
    └── dev_tasks.jsonl       # 10 个任务（有答案，用于本地评分）
```

---

## Split 说明

| split | 任务数 | 有答案 | 用途 |
|-------|--------|--------|------|
| `default` | 450 | 否 | 正式评测，提交 leaderboard |
| `dev` | 10 | 是 | 本地调试与评分 |

---

## YAML 配置说明

```yaml
env_params:
  data_dir: "env/dabstep/data"            # 数据存储路径（相对于进程工作目录）
  artifacts_dir: "env/dabstep/artifacts"  # 产物输出路径
  split: "default"                        # "default" 或 "dev"
  limit: 0                                # 0 = 不限制，跑完该 shard 的全部任务
  shard_index: 0                          # 当前分片索引（从 0 开始）
  num_shards: 8                           # 总分片数
  max_steps: 10                           # 每题最大交互步数
  timeout: 60                             # 代码执行超时（秒）
```

### `limit` 的行为

- `limit: 1`：每个 shard 只取 1 个任务，**会反复跑同一题**，仅用于快速冒烟测试。
- `limit: 0`：跑完该 shard 分到的所有任务后停止（推荐）。

### 分片任务分布示例

`dev` split（10 个任务）按 `num_shards: 8` 分片：

| shard_index | 任务数 |
|-------------|--------|
| 0, 1 | 2 个 |
| 2 ~ 7 | 1 个 |

`default` split（450 个任务）按 `num_shards: 8` 分片：每个 shard 约 **56 个任务**。

---

## 多分片并行配置示例

```yaml
environments:
  - env_name: dabstepgym
    env_num: 1
    env_params:
      data_dir: "env/dabstep/data"
      artifacts_dir: "env/dabstep/artifacts"
      split: "default"
      limit: 0
      shard_index: 0
      num_shards: 8

  - env_name: dabstepgym
    env_num: 1
    env_params:
      data_dir: "env/dabstep/data"
      artifacts_dir: "env/dabstep/artifacts"
      split: "default"
      limit: 0
      shard_index: 1
      num_shards: 8
  # ... shard 2~7 同理
```

---

## 产物目录结构

每题运行后在 `artifacts_dir` 下生成独立子目录：

```
artifacts/
└── dabstep_20260309_193140_<task_id>/
    ├── env.log           # 完整运行日志
    ├── trace.jsonl       # 每步 thought/code/output 记录
    ├── dev_metrics.json  # 评分结果（仅 split=dev）
    └── render_step_*.png # 可视化图（调用 render() 时生成）
```

---

## 注意事项

- `data_dir` 和 `artifacts_dir` 均为**相对于进程工作目录**的路径，Ray worker 启动目录决定实际存储位置。
- `dev` split 答案字段为 `answer`，`default` split 答案为空（不公开）。
- 任务字段名为 `guidelines`（非 `answer_format`），环境内部已做兼容处理。

## 相关链接

- [DABStep HuggingFace Dataset](https://huggingface.co/datasets/adyen/DABstep)
- [DABStep 官方评测库](https://huggingface.co/spaces/adyen/DABstep)

## 许可证

本适配器遵循 AIEvoBox 的许可证。DABStep 数据集基于 CC-BY-4.0 协议，请参考其官方文档。