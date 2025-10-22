# AI Sandbox Environment

一个通用 AI 沙箱开发套件，提供标准化接口，兼容任意任务、环境和评估手段，实现对多个真实世界环境的逼真建模。

本项目旨在为大模型和智能体开发者提供统一平台：
- 大模型开发者可测试不同任务与环境下基座模型的Agentic表现
- 业务智能体开发者可在统一Pipeline上测试和选型不同模型
- 所有测试结果可固化为模型训练数据，形成能力提升闭环

## 🚀 Quick Start

### 安装依赖

```bash
# 克隆仓库
git clone https://gitee.pjlab.org.cn/L2/safeai/kilab/AISandbox.git
cd AISandbox

# 安装核心依赖
pip install -r requirements.txt
```

### 运行交易环境示例

运行脚本前先使用推理框架（例如`vLLM`，`SGLang`）部署`LLM`并在`examples/run_8_trading_envs.sh`中配置`agent-api-key` `agent-base-url` `agent-model` `agent-temperature`。

```bash
# 脚本运行
bash examples/run_8_trading_envs.sh
```

## 🔥 自定义环境开发

### 1. 核心概念

- **环境（Environment）**：模拟真实世界场景的交互系统，提供观察状态和接收动作
- **智能体（Agent）**：基于LLM的决策主体，通过环境观察做出决策
- **交互器（Interactor）**：协调环境与智能体交互的核心控制器

### 2. 自定义环境与环境注册

#### 2.1 接口规范

一个标准的`Env`环境需要继承`core.env.base_env`中的`BaseEnv`并实现以下核心功能接口

<table>
    <tr>
        <td>类型</td>
        <td>组件</td>
        <td>说明</td>
    </tr>
    <tr>
        <td rowspan="3">Prompt</td>
        <td>observation_space</td>
        <td>定义智能体可以从环境中获取信息的格式、范围和类型，例如当日股价，股市情绪推文</td>
    </tr>
    <tr>
        <td>action_space</td>
        <td>定义智能体可以执行的动作的类型和范围，例如 买入 和 卖出</td>
    </tr>
    <tr>
        <td>get_task_prompt</td>
        <td>生成指导LLM决策的自然语言提示，并告知LLM的环境状态以及可用动作，例如 最大化股市收益</td>
    </tr>
    <tr>
        <td rowspan="4">Function</td>
        <td>reset()</td>
        <td>重置环境到初始状态，返回初始状态</td>
    </tr>
    <tr>
        <td>step(action)</td>
        <td>接收LLM回复解析动作并执行，更新环境状态，返回下一观测、奖励值、终止状态、截断状态、环境信息</td>
    </tr>
    <tr>
        <td>render()</td>
        <td>可视化环境状态，返回当前步骤环境可视化渲染图</td>
    </tr>
    <tr>
        <td>close()（可选）</td>
        <td>清理环境并释放资源</td>
    </tr>
</table>

#### 2.2 环境注册

环境类创建完成后导入`core.env.env_register`中的`register_env`方法并修饰新建的环境类，参数为环境的注册名，例如，

```python
# 导入register_env方法并修饰新建环境类，"trading_gym"为注册名
@register_env("trading_gym")
class TradingGym(BaseEnv):

    def __init__():
      pass
```

随后在`examples/base_eval.py`或`入口函数`中导入新类完成注册，例如，

```python
# 导入环境来注册
from env.tradinggym.trading_env import TradingGym  
```

可通过`core.env.env_register`中的 `list_registered_envs`方法来查看环境是否被注册

### 3. 交互模拟

`examples/base_eval.py`提供了基础的环境测试脚本，注册新环境后可实现全自动交互模拟，下面为示例以及参数解释，其中`env-config-yaml` 环境配置文件中每个环境应包含两个参数`env_name`和`env_params`，`env_params`中包含新创建的环境类中的所有参数配置。

```bash
# 环境测试脚本
python examples/base_eval.py \
  # 环境配置yaml文件
  --env-config-yaml "/mnt/shared-storage-user/chenxinquan/ai_sandbox/examples/configs/trading_env_configs.yaml" \
  # 环境并行数量
  --max-workers 8 \
  # 环境最大运行步长
  --max-steps 1000 \
  # Agent相关配置
  --agent-api-key "EMPTY" \
  --agent-base-url "http://localhost:8001/v1" \
  --agent-model "Qwen3-30B-Instruct" \
  --agent-temperature 0.3
```

## 📺 交互与指标可视化

项目提供多维度可视化能力，直观展示交互过程：

1. 环境状态可视化

  若环境实现了`render()`方法，环境会在每一步保存状态图片，最终合成GIF展示完整过程

2. 交互流程可视化

  可展示智能体每一步的决策

3. 性能指标可视化

  支持生成奖励曲线等统计图表，便于分析智能体表现

![trading模拟可视化](fig/visualize.gif "trading模拟可视化")

## 📰 交互数据 log 记录

项目使用 SQLite 数据库记录 LLM 与环境的交互数据，便于后续分析和查询：

### 数据存储结构

- **`interaction_sessions`表**：存储会话级元数据
  - 环境名称、模型名称、开始时间、结束时间、总奖励、完成状态等。

- **`interaction_steps`表**：存储步骤级详细，
  - 会话 ID、步骤编号、时间戳、环境状态
  - Prompt、Response、奖励值、完成状态等。

### 日志记录流程

  - 每个环境的模拟开始时，创建一条会话记录
  - 每一步交互都会创建一条步骤记录，包含当前步的 Prompt、Response 和 Reward 等信息
  - 模拟完成后，更新会话记录的总奖励、结束时间和完成状态

### 日志查询方法

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