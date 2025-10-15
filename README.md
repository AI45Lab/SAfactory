# AI Sandbox Environment

一个通用 AI 沙箱开发套件，提供标准化接口，兼容任意任务、环境和评估手段，实现对多个真实世界环境的逼真建模。

本项目旨在为大模型和智能体开发者提供统一平台：
- 大模型开发者可测试不同任务与环境下基座模型的Agentic表现
- 业务智能体开发者可在统一Pipeline上测试和选型不同模型
- 所有测试结果可固化为模型训练数据，形成能力提升闭环

## 🚀 Quick Start

<!-- ### 垂直领域大模型业务开发者（一个环境评测多个模型）

（这类开发者设计环境，但不懂模型，准备好自己的环境后，评测大模型）

```python
import Gym
from simulator import Simulator
from models import ModelAPI
from envs import EnvAPI
from tasks import Task

llm_list = ModelAPI(['gpt-4o-mini', 'Qwen-2.5-VL-72B'])
user_env = Gym.make('user_env_name') # 自定义的环境
env_api = EnvAPI(user_env) # 环境调用包装成API

sim = Simulator(env_api, llm_list)

task = Task('user_env_task.json')

logger, report, dataset = sim.run(task)  
```

### 基座模型开发者（一个模型评测多个环境）

（这类开发者需要测试和提升模型，但不懂环境，准备好自己的模型后，选择一些适配的环境开始模拟）

```python
from vllm import LLM
from simulator import Simulator
from models import ModelAPI
from envs import EnvAPI
from tasks import Task

llm = LLM('user_model_name') # 自定义的LLM
llm_api = ModelAPI(llm) # 模型调用包装成API
env_list = EnvAPI(['minecraft-v0', 'android_world-v0'])

sim = Simulator(env_list, llm_api)

task_list = Task(['mincraft-v0-default-task.json', 'android_world-v0-default-task.json'])

loggers, reports, datasets = sim.run(task_list)
``` -->

### 1. 示例环境Trading的部署指南

Trading环境模拟真实股票交易场景，部署步骤如下：

```bash
# 安装环境依赖
cd env/agentenv-trading
pip install -e .

# 启动环境服务
tradinglaunch --host 0.0.0.0 --port 36002
```
服务启动后将在`36002`端口提供 API 接口，可通过`http://localhost:36002`访问。

### 2. 新环境的集成指南

#### 接口规范

一个标准的`Env`环境需要实现以下核心功能接口

| 组件 | 说明 |
| :---------- | :---------------------------------------- |
| **观察空间** | 定义智能体可以从环境中获取的信息的格式、范围和类型 |
| **动作空间** | 定义智能体可以执行的动作的类型和范围            |
| `reset()`   | 重置环境到初始状态，返回初始观测和辅助信息       |
| `step(action)` | 执行动作并返回：下一观测、奖励值、终止状态、截断状态、环境信息 |
| `render()`（可选） | 可视化环境状态            |
| `close()`（可选） | 清理环境资源            |

#### 集成步骤
1. 实现上述标准接口，确保环境输出格式统一
2. 在`clients`目录下创建环境客户端类，继承`BaseEnvClient`
3. 实现客户端与环境服务的交互方法
4. 编写环境启动脚本，遵循`[env]launch`命名规范

### 3. 交互模拟的说明

交互模拟是本项目的核心功能，实现了LLM与环境的多步交互过程，具体说明如下：

#### 模拟流程

  - 初始化LLM代理和环境客户端
  - 创建指定数量的环境实例
  - 对于每个环境，创建一下交互会话记录
  - 在每个环境中，循环执行以下步骤直到环境完成：
    - 从环境获取状态，提取Prompt
    - 调用LLM生成Response
    - 记录当前步骤的Prompt、Response、Reward等信息
    - 将Response发送给环境，更新状态

#### 运行模拟

在启动环境服务和 LLM 推理服务后，执行以下命令：

```bash
python base_eval.py \
  --model_name "Qwen3-30B-Instruct" \
  --base_url "http://localhost:8001/v1" \
  --env_server_base "http://127.0.0.1:36002" \
  --num_envs 2
```

### 4. 可视化部分的说明

项目提供多维度可视化能力，直观展示交互过程：

1. 环境状态可视化

  环境会在每一步保存状态图片，最终合成GIF展示完整过程

2. 性能指标可视化

  支持生成奖励曲线、步骤分布等统计图表，便于分析智能体表现

3. 交互流程可视化

  可展示“观察 - 决策 - 执行 - 反馈” 的完整闭环过程

![trading模拟可视化](fig/visualize.gif "trading模拟可视化")

### 5. 交互数据 log 记录的说明

项目使用 SQLite 数据库记录 LLM 与环境的交互数据，便于后续分析和查询：

#### 数据存储结构

- **`interaction_sessions`表**：存储会话级元数据
  - 环境名称、模型名称、开始时间、结束时间、总奖励、完成状态等。

- **`interaction_steps`表**：存储步骤级详细，
  - 会话 ID、步骤编号、时间戳、环境状态
  - Prompt、Response、奖励值、完成状态等。

#### 日志记录流程

  - 每个环境的模拟开始时，创建一条会话记录
  - 每一步交互都会创建一条步骤记录，包含当前步的 Prompt、Response 和 Reward 等信息
  - 模拟完成后，更新会话记录的总奖励、结束时间和完成状态

#### 日志查询方法

```bash
# 列出最近的会话
python scripts/query_interactions.py --list

# 查看指定会话的详细信息
python scripts/query_interactions.py --session-id 123
```

数据文件位于项目根目录：`llm_env_interactions.db`。


## 📦 模块说明

![模块关系图](fig/agentic_sandbox.PNG "模块关系图")

<!-- ![模块关系图（简化）](fig/simple_arch.png "模块关系图（简化）") -->

### Environment 服务
环境服务是沙箱仿真引擎的基础支撑模块，其核心目标是为大模型智能体提供多样化、可扩展且高保真的交互环境，以满足不同场景下的评测需求。为了实现这一目标，环境服务在设计上采用了模块化、标准化的架构，同时具备灵活的扩展能力与高效的部署方式。

### 模型Agent服务
模型 Agent 是沙箱仿真引擎中执行任务的核心主体，其功能是将大语言模型转化为具备自主决策能力、能够与环境交互的智能体。

集成主流的推理后端框架：vLLM 与 SGLang 等。

### 交互器

交互器是连接模型 Agent 与环境服务的关键桥梁，其核心功能是实现 Agent 与环境之间的动态交互，驱动仿真过程的循环进行。交互器的交互逻辑遵循 “观察 - 决策 - 执行 - 反馈” 的循环机制，具体流程如下：

1. 观察状态获取：交互器首先向环境服务发送状态获取请求，环境服务通过step()方法（初始阶段通过reset()方法）生成当前环境的观察状态，并将其按照预设格式返回给交互器。

2. Prompt构建与发送：交互器将获取到的环境观察状态与当前任务目标进行整合，构建符合大模型理解习惯的 Prompt。清晰描述环境状态、任务要求以及可执行的动作范围，确保 Agent 能够准确理解当前场景与任务目标。随后，交互器将构建好的 Prompt 发送至模型 Agent。

3. 模型Agent决策：模型 Agent 基于接收到的 Prompt 进行推理，生成下一步的动作指令，并将其返回给交互器。

4. 动作执行与状态更新：交互器将动作指令发送至环境服务，环境服务调用step()方法执行该动作，并根据动作的执行结果更新环境状态，同时计算对应的奖励值，判断是否达到仿真终止条件。

5. 反馈与循环：环境服务将更新后的环境状态、奖励值以及终止标志返回给交互器，交互器将这些信息反馈给模型 Agent，并根据终止标志判断是否结束仿真循环。若未达到终止条件，交互器将再次获取新的环境观察状态，进入下一轮交互循环；若达到终止条件，则结束当前仿真任务，并记录仿真结果。

### 任务发布器

任务发布器是沙箱仿真引擎的仿真流程入口，其核心功能是根据用户的评测需求，定义仿真任务的目标、规则与参数，驱动环境服务、模型 Agent 与交互器协同工作，完成整个仿真评测过程。

任务发布器支持用户通过 API 接口进行任务参数配置，主要参数包括：任务目标定义、最大迭代步数、终止条件定义、环境与Agent指定和奖励机制配置