# DABStepEnv

DABStep benchmark 的 Gymnasium 环境封装，支持自动数据下载、多进程分片、逐步渲染。

> 本文档假设你已完成 AIEvoBox 的安装，当前工作目录为 `AIEvoBox/`。

---

## 快速开始

### 第一步：安装官方评测库（可选）
```bash
pip install git+https://huggingface.co/spaces/adyen/DABstep.git@main
```

> 不安装时会自动回退到近似评分，本地调试不影响使用。

---

### 第二步：配置环境参数

编辑 `env/dabstep/dabstep_config.yaml`：
```yaml
environments:
  - env_name: dabstepgym
    env_num: 1
    env_params:
      data_dir: "env/dabstep/data"
      artifacts_dir: "env/dabstep/artifacts"
      split: "dev"
      limit: 1
      shard_index: 0
      num_shards: 8
      max_steps: 10
      timeout: 60
```

**数据会在首次运行时自动从 HuggingFace 下载**，无需手动准备。下载后的目录结构：
```
env/dabstep/data/
├── context/          # CSV/JSON 背景数据
└── tasks/
    ├── default_tasks.jsonl   # 450 题（无答案，用于提交 leaderboard）
    └── dev_tasks.jsonl       # 10 题（有答案，用于本地评分）
```

---

### 第三步：运行
```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/dabstep/dabstep_config.yaml \
  --llm-base-url <your-base-url> \
  --llm-api-key  <your-api-key> \
  --llm-model    <model-name> \
  --pool-size 1
```

---

### 第四步：查看结果

每道题运行完毕后，产物写入 `env/dabstep/artifacts/`：
```
artifacts/
└── dabstep_20260309_193140_<task_id>/
    ├── env.log             # 完整运行日志
    ├── trace.jsonl         # 每步 thought / code / output 记录
    ├── dev_metrics.json    # 评分结果（仅 split=dev 时生成）
    └── render_step_*.png   # 可视化截图（调用 render() 时生成）
```

用 `dev` split 时，打开 `dev_metrics.json` 即可查看本地得分。

---

## Split 说明

| split | 任务数 | 有答案 | 用途 |
|-------|--------|--------|------|
| `dev` | 10 | ✅ | 本地调试与评分 |
| `default` | 450 | ❌ | 正式评测，提交 leaderboard |

---

## 扩展：多分片并行

确认单 shard 跑通后，将 `split` 改为 `default`，并在配置中展开多个 shard：
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
  # shard 2~7 同理
```

`default` split 按 8 分片后每个 shard 约 **56 题**；`dev` split 的分布：

| shard_index | 任务数 |
|-------------|--------|
| 0, 1 | 2 题 |
| 2 ~ 7 | 1 题 |

---

## Docker 部署（环境tag待补充）

- 构建镜像
```bash
docker build -f env/dabstep/Dockerfile -t dabstep:latest .
```

- 如需预构建镜像，请替换为你自己的镜像仓库地址，例如：
```bash
docker pull <your-registry>/dabstep:latest
```

---

## 相关链接

- [DABStep HuggingFace Dataset](https://huggingface.co/datasets/adyen/DABstep)
- [DABStep 官方评测库](https://huggingface.co/spaces/adyen/DABstep)

## 许可证

本适配器遵循 AIEvoBox 的许可证。DABStep 数据集基于 CC-BY-4.0 协议。
