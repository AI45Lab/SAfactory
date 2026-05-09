# 多 LLM Policy 代码设计

本文档承接 `multi-agent-llm-rollout.md`，说明按新架构落到代码时需要改哪些地方。核心原则保持不变：env 决定当前调用哪些 policy，框架负责调用并记录训练样本。

## 1. 基础数据语义

引入三个逻辑对象，不一定需要新建复杂类，MVP 可以先用 dict 表达：

- `prompt_dict`: `{agent_id: messages}`，表示当前哪些 agent 需要行动。
- `action_dict`: `{agent_id: action}`，表示本轮提交给 env 的 action。
- `reward_dict`: `{agent_id: reward}`，表示本次状态转移后给哪些 agent 结算 reward。

兼容规则：

- 旧的 `messages` 自动视为 `{"default": messages}`。
- 旧的 `action` 自动视为 `{"default": action}`。
- 旧的 `float reward` 自动视为 `{"default": reward}`。

## 2. Env 侧改动

`BaseEnv` 先不强制改抽象签名，避免影响现有环境。多智能体 env 只需要遵守约定：

- `get_task_prompt()` 可以返回单个 messages，也可以返回 `prompt_dict`。
- `step()` 可以接收单个 action，也可以接收 `action_dict`。
- `StepOutput.reward` 长期应支持 `float | dict[str, float]`。

MVP 可先不改所有 env。单智能体 env 保持原样；只有多智能体 env 返回 dict。

需要注意：当前 `StepOutput.reward` 是 `float`。实现时有两个选择：

1. 短期：reward dict 放进 `StepOutput.info["reward_dict"]`，`reward` 字段保留聚合值。
2. 长期：将 `StepOutput.reward` 类型扩展为 `float | dict[str, float]`。

建议 MVP 先用方案 1，减少对已有环境和 HTTP 序列化的影响。

## 3. Interactor 改动

`Interactor._run_one_environment()` 是主改动点。

新增几个内部 helper：

- `normalize_prompts(raw_prompt) -> dict[agent_id, messages]`
- `normalize_rewards(step_output) -> dict[agent_id, reward]`
- `resolve_policy(agent_id) -> policy_id`
- `create_or_get_llm(policy_id, session) -> LLM`

主流程从“单 prompt -> 单 LLM -> 单 action -> 单 reward”改为：

1. 从 env 获取 prompt。
2. 规范化为 `prompt_dict`。
3. 对每个 `agent_id` 找到 `policy_id`。
4. 并发调用对应 LLM policy。
5. 组装 `action_dict`，一次提交给 env。
6. 规范化 `reward_dict`。
7. 对有 reward 的 agent 写训练样本；没有 reward 的 generation 先暂存。

## 4. Pending Generation

延迟奖励需要在 `Interactor` 里维护 pending generation。

建议结构：

```text
pending_generations[agent_id] = [
  {
    messages,
    response,
    policy_id,
    policy_version,
    finish_reason,
    env_state,
    step_id
  }
]
```

当 `reward_dict` 返回某个 `agent_id` 的 reward：

1. 取出该 agent 最早或最近一条 pending generation。
2. 将 reward 绑定到这条 generation。
3. 调用 `DataManager.record_step()` 写入训练数据。

默认策略建议用 FIFO，适合 attacker -> defender -> judge 这类顺序流程。

## 5. Session 和 Policy

当前 session 主要绑定 `env_id`。多 policy 后，需要在 metadata 中至少记录：

- `agent_id`
- `policy_id`
- `policy_version`
- `episode_id`

MVP 不要求立刻改变 DB schema，可以先写入 `env_state` JSON。

建议 `session_id` 保持唯一，避免不同 agent 的 `llm_proxy` 轨迹混在一起：

```text
session_id = env_id + ":" + agent_id + ":" + policy_id
```

如果短期不改 `DataManager.create_session()`，也至少要保证传给 LLM proxy 的 session 后缀包含 agent/policy 信息。

## 6. LLM 选择

新增 policy 配置概念：

```text
agent_id -> policy_id -> model/base_url/api_key/temperature
```

MVP 可以先支持两种模式：

- 未配置多 policy：所有 agent 使用现有 `llm_model` 和 `llm_base_url`。
- 配置多 policy：Interactor 根据 `policy_id` 创建不同 LLM client。

如果多个 trainable policy 都走 Slime 训练，后续再把 `llm_proxy` 扩展成按 `policy_id` 路由。MVP 可以先假设 policy endpoint 已经可访问。

## 7. DataManager 改动

短期不迁移 DB，先把多 agent 信息写进 `env_state`：

```text
{
  "weight_version": ...,
  "agent_id": ...,
  "policy_id": ...,
  "policy_version": ...,
  "episode_id": ...
}
```

`record_step()` 调用时仍写 `messages`、`response`、`reward`，但 reward 来自对应 agent 的 `reward_dict`。

长期可以给 `session_steps` 增加列：

- `agent_id`
- `policy_id`
- `episode_id`

## 8. Buffer Server 改动

`buffer_server` 组装训练样本时，需要从 `env_state` 或将来的字段中取出：

- `agent_id`
- `policy_id`
- `policy_version`

新增按 policy 过滤：

```text
get_rollout_data(policy_id)
```

如果没有传 `policy_id`，保持当前行为，返回所有可训练样本。

返回给 Slime 的 `extra_info` 需要包含：

- `agent_id`
- `policy_id`
- `policy_version`
- `session_id`
- `weight_version`

## 9. Slime Generator 改动

每个 trainer 需要知道自己训练哪个 `policy_id`。

启动 rollout 时传入当前 `policy_id`，拉数据时也带上该 `policy_id`：

```text
trainer(policy_a) -> buffer_server.get_rollout_data(policy_a)
trainer(policy_b) -> buffer_server.get_rollout_data(policy_b)
```

版本过滤也应按 policy 做：

```text
current_version[policy_id] - sample.policy_version <= off_by_n
```

MVP 可以先复用现有 `weight_version` 字段，但 metadata 中要带 `policy_id`。

## 10. 推荐实施顺序

1. 在 `Interactor` 增加 prompt/reward normalization，保证旧 env 不受影响。
2. 增加 `agent_id -> policy_id` 配置解析和 LLM 选择。
3. 增加 pending generation，支持延迟奖励。
4. 将 `agent_id`、`policy_id`、`policy_version` 写入 `env_state`。
5. 修改 `buffer_server` 支持按 `policy_id` 过滤。
6. 修改 `slime_generator` 拉取指定 `policy_id` 的数据。
7. 最后再考虑是否扩展 `StepOutput.reward` 类型和 DB schema。

## 11. MVP 不做

- 不改所有已有 env。
- 不引入复杂 transaction 或 `state_version`。
- 不做 centralized critic。
- 不做 league training。
- 不强制多 policy proxy runtime。
- 不要求第一版完成 DB migration。
