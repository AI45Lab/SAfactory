# EmbodiedGym - Alfred Environment Adapter

将 EmbodiedBench 的 Alfred 环境集成到 AIEvoBox 框架中。

## 📁 文件结构

```
embodiedgym/
├── __init__.py                 # 模块初始化
├── embodied_env.py            # 核心适配器类
├── embodied_config.yaml       # 环境配置文件
├── test_embodied_env.py       # 测试脚本
└── README.md                  # 本文档
```

## 🚀 快速开始

### 1. 前置依赖

请按照以下步骤安装和配置环境：

```bash
# 1. 克隆 AIEvoBox 仓库
git clone https://gitee.pjlab.org.cn/L2/safeai/kilab/AIEvoBox.git

# 2. 进入 embodiedgym 目录
# 注意：根据实际克隆后的目录名称可能为 AIEvoBox 或 AIEvoBox-new
cd AIEvoBox/env/embodiedgym

# 3. 克隆 EmbodiedBench
git clone git@github.com:EmbodiedBench/EmbodiedBench.git

# 4. 创建并激活 Conda 环境
# 进入 EmbodiedBench 源码目录 (如果克隆下来的目录名是 EmbodiedBench-master 请相应修改)
cd EmbodiedBench-master
conda env create -f conda_envs/envir.yaml --prefix /root/envs/embench1
conda activate /root/envs/embench1
pip install -e .

# 5. 安装系统依赖
sudo apt-get update
sudo apt-get install -y xvfb
# 启动 Xvfb
Xvfb :1 -screen 0 1024x768x24 &

# 安装字体
apt-get update
apt-get install -y fonts-ubuntu

# 6. 配置 AI2THOR 资源
# 复制预下载的 AI2THOR 数据
cp -r /mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/ai2thor/.ai2thor ~/.ai2thor
```

### 2. 数据准备

下载并配置 EB-ALFRED 数据集：

```bash
# 确保在环境中
conda activate embench1

# 下载数据集
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED

# 移动数据到指定目录
# 注意：以下操作假设当前目录为 EmbodiedBench-master
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```

### 3. 启动 VLLM 服务

确保 VLLM 服务已启动并运行 Qwen2.5-VL-7B-Instruct 模型：

```bash
vllm serve /mnt/shared-storage-user/steai-share/hf-hub/Qwen2.5-VL-7B-Instruct \
    --dtype half \
    --port 8001 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max_model_len 30000
```

> **注意**：启动服务后，请修改 `examples/run_8_trading_envs.sh` 中的 `baseurl` 和 `port` 为实际服务地址。

### 4. 运行测试

测试环境是否正常工作：

```bash
cd /AIEvoBox/env/embodiedgym
python test_embodied_env.py
```

### 5. 运行示例

#### 方式 1：使用 Python 脚本

```bash
cd /AIEvoBox
python examples/multi_env_example.py
```

#### 方式 2：使用 Shell 脚本

```bash
cd AIEvoBox
bash examples/run_8_embodied_envs1.sh
```

## ⚙️ 配置说明

### 环境参数

在 `embodied_config.yaml` 中可以配置以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `eval_set` | str | 'base' | 评测集名称<br>可选值：'base', 'common_sense', 'complex_instruction', 'spatial', 'visual_appearance', 'long_horizon' |
| `down_sample_ratio` | float | 1.0 | 数据采样比例<br>范围：0.0-1.0，1.0 表示使用全部数据 |
| `resolution` | int | 500 | 图像分辨率（像素）<br>建议范围：300-800 |
| `detection_box` | bool | false | 是否在图像上显示物体检测框 |
| `max_episode_steps` | int | 30 | 每个 episode 的最大步数 |
| `max_invalid_actions` | int | 10 | 最大连续无效动作数 |
| `exp_name` | str | 'aievobox_alfred' | 实验名称，用于日志和结果保存 |

### 示例配置

```yaml
environments:
  - env_name: embodied_alfred
    env_num: 2
    env_params:
      eval_set: "base"
      down_sample_ratio: 0.1  # 使用 10% 数据快速测试
      resolution: 500
      max_episode_steps: 30
      exp_name: "test_base"
```



## 🔗 相关链接

- [EmbodiedBench GitHub](https://github.com/embodied-generalist/embodiedbench)
- [ALFRED Dataset](https://askforalfred.com/)
- [AI2THOR](https://ai2thor.allenai.org/)
- [VLLM](https://github.com/vllm-project/vllm)

## 📄 许可证

本适配器遵循 AIEvoBox 的许可证。EmbodiedBench 和 ALFRED 有各自的许可证，请参考其官方文档。

