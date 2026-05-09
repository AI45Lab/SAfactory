# 多 LLM Policy 在同一个 Env 中协作/对抗

目标：支持多个 LLM policy 在同一个 env episode 里协作或对抗，同时保证 env 状态、policy 调用、reward 归属和训练样本能对齐。

## 需要解决的问题

1. 并行问题：同一个 obs 下，agent A 和 agent B 同时给出 action 时，如何保证 reward 是基于同一个 env 状态计算的？
2. 回合制问题：如果 agent B 必须等 agent A 先和 env 交互并改变状态，如何表达这个先后约束？
3. 奖励延迟问题：例如 agent A 先生成攻击数据，agent B 生成防御数据，再由 agent C 评估后，才得到攻击 reward 和防御 reward。
4. 异步更新问题：policy update 阶段，如何避免影响正在进行的 env-agent 交互？

## 设计原则

1. Env 决定当前应该调用哪些 LLM policy。框架不猜测调度顺序，只执行 env 给出的调用请求。
2. 框架负责维护 policy 的可访问性，并调用对应 LLM policy 生成 action。
3. Env 返回的 reward 不一定属于当前刚交互的 policy，也可以属于之前交互过的 policy，或一次性返回多个 policy 的 reward。

## 实现语义

1. 并行：如果几个 agent 面对同一个 env 状态，env 在同一轮返回这些 agent。框架并发生成 action 后，以 joint action 一次性提交给 env，reward 由 env 基于 joint action 计算。
2. 回合制：如果 B 必须等 A 改变 env，env 先只返回 A；A 的 action 提交后 env 更新状态，再返回 B。
3. 延迟奖励：如果 reward 暂时未知，框架先暂存 generation；后续 env 返回对应 reward 后，再把 generation 写成训练样本。
4. 异步更新：每条 generation 记录 `policy_id` 和 `policy_version`。一个 episode 内建议固定 policy version，新权重只影响后续 episode。
