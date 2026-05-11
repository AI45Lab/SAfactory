# 多 LLM Policy 代码设计

本文档只说明代码层面的整改方向，具体字段和实现细节后续再补。

## 目标

支持多个 LLM policy 在同一个 env/world 里协作、竞争或分工。一个 env/world 只启动一份 shared rollout；多个可训练 policy 分别由多个 Slime trainer 训练，并从同一个 buffer 中按 `policy_id` 取自己的样本。

## 总体设计

Env 决定当前应该调用哪些 policy。框架只负责三件事：

1. 根据 env 的请求调用对应 policy。
2. 把 action 一次性提交回 env。
3. 根据 env 返回的 reward，把样本写到对应 policy 的训练数据里。

单智能体 env 保持兼容；多智能体 env 才需要返回多个 agent/policy 的调用请求。

Interactor 不理解具体任务流程，也不写死任何角色。所有顺序、并行、依赖关系都由 env 决定。框架只执行 env 当前返回的调用请求。

Interactor 的“一轮”定义为一次 env 调度周期：

```text
env 给出当前要行动的一批 agent/policy
-> Interactor 调用这些 policy
-> Interactor 收齐 action
-> Interactor 一次性提交给 env
-> env 返回状态更新和 reward
```

如果这一批里只有一个 agent，就是回合制一轮；如果有多个 agent，就是并行 joint action 一轮。

因此：

- 顺序流程：env 每轮只返回当前 agent。
- 并行流程：env 同一轮返回多个 agent。
- 混合流程：env 可以先返回一个 agent，后续某轮再返回多个 agent。
- 延迟奖励：env 可以在任意后续轮次给之前行动过的 agent 返回 reward。

## Env 需要表达什么

多智能体 env 不需要暴露复杂调度器，只需要在每轮明确三件事：

1. 当前哪些 agent 需要行动。
2. 这些 agent 分别对应哪个 `policy_id`。
3. 本次状态转移后，哪些 agent/policy 获得 reward。

reward 的语义由 env 定义。框架不判断 reward 是否合理，只负责把 reward 绑定到对应 generation 和 policy 训练数据里。

## 需要整改的模块

### Interactor

Interactor 从“单 LLM 调用器”变成“多 policy 调度器”：

- 识别 env 当前要求哪些 agent 行动。
- 根据 `agent_id -> policy_id` 找到对应 LLM endpoint。
- 如果当前只有一个 agent，就只调用这个 agent 的 policy。
- 如果当前有多个 agent，才并发调用多个 policy。
- 收齐当前轮次的生成结果后，由 Interactor 组装 action 并提交给 env。
- 维护未结算 reward 的 generation，等 reward 回来后再写样本。

Interactor 不负责判断“谁先谁后”，也不负责解释 reward 语义。这些都属于 env 的职责。

### Buffer Server

Buffer 需要支持按 `policy_id` 分流数据。

原因是 shared rollout 会同时产生多个 agent/policy 的样本，但每个 Slime trainer 只能训练自己的 policy。Buffer 必须能保证：

- 某个 policy 的 trainer 只拿这个 policy 的样本。
- 其他 policy、不可训练 agent、工具 agent 的样本不会混进来。
- reward 未结算的 pending generation 不进入训练。

MVP 仍然使用一个全局 DB cursor 顺序读取数据。Buffer server 读到记录后解析 metadata 里的 `policy_id`，再放入不同 policy bucket。各 trainer 请求数据时，只从自己的 bucket 取样本。

也就是说：

```text
DB 顺序读取一次
-> 按 policy_id 分桶
-> get_rollout_data(policy_id) 返回对应 bucket
```

第一版不为每个 policy 维护独立 DB cursor，避免重复扫描和同步复杂度。

### Slime Generator

Generator 需要拆成两个职责：

- `rollout_owner`：只有一个 owner 负责启动 shared rollout。
- `policy_trainer`：每个 trainer 只拉取并训练自己的 `policy_id` 数据。

也就是说，多 agent 不是启动多份 rollout，而是：

```text
一份 shared rollout -> 产生所有 policy 的样本
多个 Slime trainer -> 各自消费对应 policy 的样本
```

这意味着 Slime trainer 不再天然等于 rollout 启动者。只有 `rollout_owner` 负责启动 env；其他 trainer 只等待 buffer 里出现自己的 policy 数据。

### Trajectory Mask

多 agent 后，trajectory mask 最大风险是不同 agent 的生成轨迹混在一起。

整改方向：

- 每次 generation 都要能唯一定位到 `agent_id`、`policy_id` 和 session。
- reward 延迟返回时，要能把 reward 绑定回正确 generation。
- 训练样本中的 token、loss mask、reward、policy_id 必须来自同一次 generation。

Mask 逻辑本身不一定重写，但 session 和 generation 的隔离必须做。

### Data / Metadata

MVP 可以先不做 DB migration，把多 agent 信息写入 metadata：

- `agent_id`
- `policy_id`
- `policy_version` 或 `weight_version`
- `session_id`
- generation id 或等价标识

后续如果跑通，再考虑把这些字段正式迁移到表结构。

## MVP 范围

第一版只解决必须问题：

- 一个 env/world 只有一份 shared rollout。
- 多个 Slime trainer 按 `policy_id` 各自训练。
- Buffer 能按 `policy_id` 分流。
- Generation 和 reward 能正确绑定。
- Trajectory mask 不串 agent/policy。

暂时不做：

- 复杂 transaction 或 `state_version`。
- episode-level version freeze。
- centralized critic。
- league training。
- 多 policy proxy runtime。
- 强制 DB migration。
