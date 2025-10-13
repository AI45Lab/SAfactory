# AI Sandbox Environment

一个通用 AI 沙箱开发套件，拥有一套通用接口，可兼容任意任务、环境和评估手段的技术平台，实现对多个真实世界环境进行逼真的建模。

本项目旨在为大模型和智能体开发者提供一个统一的平台。对于大模型开发者来说，开发人员可以在该平台上测试不同任务不同环境基座模型Agentic表现，对于具体业务的智能体开发者来说可以在一套统一的Pipeline上测试和选型不同的模型，最终所有的测试结果可以被固化为模型训练数据，不断提升模型能力。

## Quick Start

### 垂直领域大模型业务开发者（一个环境评测多个模型）

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
```


## 模块说明

![模块关系图](fig/agentic_sandbox.PNG "模块关系图")

![模块关系图（简化）](fig/simple_arch.png "模块关系图（简化）")

### Environment 服务
一个标准的Env通常需要实现以下功能：
- 观察空间（Observation Space）
  - 定义智能体可以从环境中获取的信息的格式、范围和类型
- 动作空间（Action Space）
  - 定义智能体可以执行的动作的类型和范围
- reset()方法
  - 在每个训练/评估回合开始时调用
  - 将环境重置到初始状态，并返回初始观测
  - 标准返回：initial_observation, info（Info为可选的辅助信息字典）
- step(action)方法
  - 在智能体执行一个动作后调用
  - 根据智能体的动作更新环境状态，计算奖励，并判断回合是否结束
  - 标准返回：
    - next_observation：执行动作后的新
    - reward：智能体因执行该动作获得的奖励
    - terminated：表示回合是否因达到终止条件而结束
    - truncated：表示回合是否因时间限制等非自然终止条件而结束
    - info：字典，环境状态
- render()方法（可选）
  - 用于可视化环境状态
- close()方法（可选）
  - 用于清理环境资源


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