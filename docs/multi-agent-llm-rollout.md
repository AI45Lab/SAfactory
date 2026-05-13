# 多 LLM Policy 在同一个 Env 中训练

目标：支持多个 LLM policy 在同一个 env episode 里训练，同时保证 env 状态、policy 调用、reward 归属和训练样本能对齐。

## 需要解决的问题

1. 并行问题：同一个 obs 下，agent A 和 agent B 同时给出 action 时，如何保证 reward 是基于同一个 env 状态计算的？
2. 回合制问题：如果 agent B 必须等 agent A 先和 env 交互并改变状态，如何表达这个先后约束？
3. 奖励延迟问题：例如 agent A 先生成攻击数据，agent B 生成防御数据，再由 agent C 评估后，才得到攻击 reward 和防御 reward。
4. 异步更新问题：policy update 阶段，rollout 如何处理服务短暂不可用和样本版本混杂？
5. Rollout 归属问题：多个 agent 是否各自启动 rollout，还是共享同一个 env rollout？
6. 训练归属问题：多个 policy 如何分别调用 Slime 训练？

## 设计原则

1. Env 决定当前应该调用哪些 LLM policy。框架不猜测调度顺序，只执行 env 给出的调用请求。
2. 框架负责维护 policy 的可访问性，并调用对应 LLM policy 生成 action。
3. Env 返回的 reward 不一定属于当前刚交互的 policy，也可以属于之前交互过的 policy，或一次性返回多个 policy 的 reward。
4. 同一个 env episode 只启动一份 shared rollout。多个 agent 都生活在这一个 env/world 里，不能各自启动独立 rollout。
5. 训练按 `policy_id` 拆分。shared rollout 负责产出所有 agent 的数据，多个 Slime trainer 分别消费属于自己 policy 的样本。
6. Env 是被动的世界状态和调度规则持有者；Interactor 是主动驱动循环的一方，负责询问 env、调用 policy、再把 action 提交回 env。

## 实现语义

1. 并行：如果几个 agent 面对同一个 env 状态，env 在同一轮返回这些 agent。框架并发生成 action 后，以 joint action 一次性提交给 env，reward 由 env 基于 joint action 计算。
2. 回合制：如果 B 必须等 A 改变 env，env 先只返回 A；A 的 action 提交后 env 更新状态，再返回 B。
3. 延迟奖励：如果 reward 暂时未知，框架先暂存 generation；后续 env 返回对应 reward 后，再把 generation 写成训练样本。
4. 异步更新：MVP 允许同一个 episode 内出现版本混杂；每条 generation 只记录 `policy_id` 和可获得的 `policy_version`，用于后续分析或过滤。
5. Rollout：一个 env group 只启动一份 shared rollout，由它调度所有 agent 和 policy。可以用 `rollout_owner` 避免多个 trainer 重复启动 env。
6. Slime：每个可训练 policy 对应一个 Slime trainer；trainer 按 `policy_id` 从 buffer 拉取样本，只训练自己的 policy。

## 代码框架设计

可以把整个系统想成四个角色：

1. Env 是世界。它知道现在轮到谁说话，也知道说完以后世界怎么变化。
2. Interactor 是执行者。它问 Env：“现在谁要说话？”然后去调用对应模型，把模型回复交回给 Env。
3. Buffer Server 是仓库。它把每个模型产生的数据存起来，并标记这条数据属于哪个 policy。
4. Slime Trainer 是训练器。每个 policy 启一个 trainer，只从仓库里拿自己的数据训练。

一次交互大概是这样：

1. Env 说：现在需要 `attacker` 生成攻击 prompt。
2. Interactor 找到 `attacker` 对应的 `attacker_policy`，调用这个模型生成回复。
3. Interactor 把回复交给 Env。
4. Env 更新状态，然后说：现在需要 `defender` 回复。
5. Interactor 再调用 `defender_policy`。
6. Env 根据攻击和防御结果计算 reward。
7. Buffer Server 把 attacker 的数据放到 `attacker_policy` 下面，把 defender 的数据放到 `defender_policy` 下面。
8. 两个 Slime Trainer 分别拉自己的数据，各自训练自己的模型。

这里最重要的是两个名字：

1. `agent_id`：Env 里的角色名，比如 `attacker`、`defender`、`judge`。
2. `policy_id`：真正要训练的模型名，比如 `attacker_policy`、`defender_policy`。

Env 只关心角色怎么互动；训练系统只关心每条数据属于哪个 policy。这样一个 Env 里可以有多个 agent，也可以让多个 agent 共用同一个 policy。

如果多个 agent 要同时行动，Env 一次性告诉 Interactor：“现在 A 和 B 都要说话。”Interactor 等 A 和 B 都生成完，再一起交给 Env。这样 reward 就是基于同一个世界状态算出来的。

如果 reward 不是马上出来，例如 attacker 先说，defender 后说，最后 judge 才打分，那么 Interactor 先暂存 attacker 和 defender 的回复。等 Env 返回 reward 后，再把 reward 补到对应的数据上，然后写入 Buffer Server。
