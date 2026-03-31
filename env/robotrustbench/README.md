# RoboTrustBench Habitat Adapter

将 RoboTrustBench Habitat 环境按 `env/embodiedgym` 的单入口风格整理到 AIEvoBox 中。

## 📁 文件结构

```text
robotrustbench/
├── __init__.py                          # 统一导出 RoboTrustBenchEnv
├── robotrustbench_env.py               # 唯一环境入口，内部按 env_name / dataset_name 分流
├── test_robotrustbench_env.py          # 顶层 smoke test
├── robotrustbench_config.yaml          # 统一任务配置
├── README.md
└── rt_habitat/                         # Habitat 运行时资源与 RoboTrustBench 扩展
    ├── actions.py
    ├── measures.py
    ├── sensors.py
    ├── predicate_task.py
    ├── runtime_support.py
    ├── config/                         # Habitat 任务与扰动配置
    ├── dataset/                        # 自定义 dataset 代码与 yaml 配置
    └── datasets/                       # *.pickle 数据集文件
```

## 🚀 快速开始

### 1. 代码入口

- 统一环境类：`env.robotrustbench.RoboTrustBenchEnv`
- 不再保留 `RTHabEnv_*` 薄包装文件
- `env.app` / `env.registry` 已通过 `env_name` 注入同一个环境类

### 2. 支持并已验证的任务

当前只关注并验证以下 4 个变体：

- `truthfulness`
- `robust`
- `robustd`
- `safety`

`fairness` 和 `privacy` 暂不纳入本次结构重构验证范围。

### 3. GPU 机器测试环境

README 中涉及 Habitat reset 的验证，使用的是 GPU 机器上的 `embench_mnt` 环境。

登录方式示例：

```bash
ssh -CAXY j-1774411416-872848-16874022-cfcd2.huangchao+root.ailab-evobox.pod@h.pjlab.org.cn
```

进入仓库并激活环境：

```bash
cd /mnt/shared-storage-user/huangchao/project/AIEvoBox
conda activate embench_mnt
```

## 🧪 测试

### 顶层统一 smoke test

```bash
cd /mnt/shared-storage-user/huangchao/project/AIEvoBox
python env/robotrustbench/test_robotrustbench_env.py
```

只测部分变体：

```bash
python env/robotrustbench/test_robotrustbench_env.py \
  --variants truthfulness,robust,robustd,safety
```

脚本会直接实例化统一的 `RoboTrustBenchEnv`，并对每个变体执行：

- 环境初始化
- `reset()`
- 打印 `obs keys`
- 打印任务指令
- 打印动作数

### 已确认通过的 smoke variants

在 `embench_mnt` 环境下，以下 reset smoke test 已确认通过：

- `truthfulness`
- `robust`
- `robustd`
- `safety`

## ⚙️ 配置说明

顶层现在只保留一个统一配置文件：

- `robotrustbench_config.yaml`

通过配置文件顶层字段控制要运行的环境族：

```yaml
enabled_envs:
  - truthfulness
```

支持的值：

- `truthfulness`
- `robust`
- `robustd`
- `safety`

所有任务最终都会落到同一个 `RoboTrustBenchEnv`。

如果需要临时覆盖配置文件中的选择，也可以在启动前设置：

```bash
export AIEVOBOX_RT_ENABLED_ENVS=robustd
```

统一环境类的主要分流依据：

- `env_name`
- `dataset_name`
- `eval_set`
- `dynamic_perturbation`

其中：

- `truthfulness` 默认使用 `dataset_truthfulness.yaml`
- `robust` 默认使用 `dataset_robust.yaml`
- `robustd` 默认使用 `dataset.yaml`
- `safety` 默认使用 `dataset_safety.yaml`

示例：直接通过统一配置运行 `robustd`

```bash
AIEVOBOX_RT_ENABLED_ENVS=robustd python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/robotrustbench/robotrustbench_config.yaml
```

## 🐳 外部 Docker Upstream

当前使用的镜像：

```text
registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260319080534-v2s2
```

若按外部容器方式运行 upstream，请保持：

- `--no-start-local-upstream`
- `--local-upstream-port 36663`
- `--local-upstream-url http://127.0.0.1:36663`

示例：

```bash
docker pull registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260319080534-v2s2

docker run -d \
  --name robotrustbench_upstream \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 8g \
  registry.h.pjlab.org.cn/ailab-evobox-evobox_gpu/hc_ray:robotrustbench-20260319080534-v2s2
```
