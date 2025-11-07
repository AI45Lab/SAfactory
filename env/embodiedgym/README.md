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

确保已安装以下依赖：

```bash
# EmbodiedBench 依赖
cd /Users/gaozhenkun/study/eaieval/EmbodiedBench-master
pip install -r requirements.txt

# AI2THOR 模拟器
pip install ai2thor

# AIEvoBox 依赖
cd /Users/gaozhenkun/study/eval/AIEvoBox
pip install -r requirements.txt
```

### 2. 启动 VLLM 服务

确保 VLLM 服务已启动并运行 Qwen2.5-VL-7B-Instruct 模型：

```bash
vllm serve /mnt/shared-storage-user/steai-share/hf-hub/Qwen2.5-VL-7B-Instruct \
    --dtype half \
    --port 8001 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max_model_len 20000
```

### 3. 运行测试

测试环境是否正常工作：

```bash
cd /Users/gaozhenkun/study/eval/AIEvoBox/env/embodiedgym
python test_embodied_env.py
```

### 4. 运行示例

#### 方式 1：使用 Python 脚本

```bash
cd /Users/gaozhenkun/study/eval/AIEvoBox
python examples/multi_env_example.py
```

#### 方式 2：使用 Shell 脚本

```bash
cd /Users/gaozhenkun/study/eval/AIEvoBox
bash examples/run_8_trading_envs.sh
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

## 🔧 核心类：EmbodiedAlfredGym

### 初始化

```python
from env.embodiedgym.embodied_env import EmbodiedAlfredGym

env = EmbodiedAlfredGym(
    eval_set='base',
    down_sample_ratio=1.0,
    resolution=500,
    max_episode_steps=30
)
```

### 主要方法

#### reset()
重置环境到初始状态

```python
reset_output = env.reset()
# reset_output.observation: 包含图像、指令、可用动作
# reset_output.info: 包含 episode 信息
```

#### step(action: str)
执行动作（LLM 输出的 JSON 字符串）

```python
llm_output = '''
{
    "reasoning": "需要先找到苹果",
    "executable_plan": [
        {"action_id": 0, "description": "find a apple"}
    ]
}
'''

step_output = env.step(llm_output)
# step_output.observation: 新观测
# step_output.reward: 奖励值
# step_output.terminated: 是否终止
# step_output.info: 额外信息
```

#### get_task_prompt()
生成包含图像的多模态 prompt

```python
prompt_output = env.get_task_prompt()
# prompt_output.system_message: System 消息
# prompt_output.user_message: User 消息（包含图像）
```

#### render()
渲染当前环境状态

```python
render_output = env.render()
# render_output.image_data: 图像二进制数据
# render_output.image_base64: Base64 编码图像
# render_output.step: 当前步骤
```

#### close()
关闭环境释放资源

```python
env.close()
```

## 🎯 LLM 输出格式

LLM 必须输出以下 JSON 格式：

```json
{
  "reasoning": "你的推理过程",
  "executable_plan": [
    {
      "action_id": 123,
      "description": "find a apple"
    },
    {
      "action_id": 456,
      "description": "pick up the apple"
    }
  ]
}
```

**关键要求**：
- `executable_plan` 必须是列表，至少包含一个动作
- 每个动作必须有 `action_id`（整数，范围：0 到动作空间大小-1）
- 可以规划多步，但建议从 1-3 个动作开始

## 📊 评测集说明

| 评测集 | 说明 | Episode 数量（约） |
|--------|------|-------------------|
| `base` | 基础评测集 | 140 |
| `common_sense` | 常识推理 | 30 |
| `complex_instruction` | 复杂指令 | 30 |
| `spatial` | 空间推理 | 30 |
| `visual_appearance` | 视觉外观 | 30 |
| `long_horizon` | 长期规划 | 30 |

## 🐛 故障排查

### 1. AI2THOR 启动失败

**症状**：`Failed to start AI2THOR`

**解决方案**：
- 确保系统有 X Display（Linux）或 GUI 环境
- 检查 GPU 是否可用
- 尝试设置环境变量：`export DISPLAY=:0`

### 2. EmbodiedBench 模块未找到

**症状**：`ModuleNotFoundError: No module named 'embodiedbench'`

**解决方案**：
- 检查 EmbodiedBench 路径是否正确
- 修改 `embodied_env.py` 中的 `embodied_bench_path` 变量

### 3. JSON 解析失败

**症状**：步骤显示 "LLM 输出解析失败"

**解决方案**：
- 检查 LLM 输出格式是否正确
- 确保 `executable_plan` 存在且为列表
- 确保 `action_id` 在有效范围内

### 4. 图像分辨率过大导致慢

**症状**：环境运行缓慢

**解决方案**：
- 降低 `resolution` 参数（如 300）
- 减少 `down_sample_ratio`（如 0.1）

## 📝 日志和结果

运行结果会保存在以下位置：

```
running/eb_alfred/{exp_name}/
├── images/          # 每步的截图
│   └── episode_X/
│       └── step_Y.png
├── results/         # 评测结果
│   └── episode_X_final_res.json
└── episode_X_step_Y.json  # 交互日志
```

## 🔗 相关链接

- [EmbodiedBench GitHub](https://github.com/embodied-generalist/embodiedbench)
- [ALFRED Dataset](https://askforalfred.com/)
- [AI2THOR](https://ai2thor.allenai.org/)
- [VLLM](https://github.com/vllm-project/vllm)

## 📄 许可证

本适配器遵循 AIEvoBox 的许可证。EmbodiedBench 和 ALFRED 有各自的许可证，请参考其官方文档。

