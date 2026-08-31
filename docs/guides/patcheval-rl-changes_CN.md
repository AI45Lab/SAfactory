# Patcheval RL 调通：改动总结

本文汇总 patcheval RL（RJob 模式，Qwen3.8-27B）从跑不起来到能正常产出训练数据期间的所有改动，包括修复的 Bug、新增的 Feature/配置、以及相关文档。

---

## 一、修复的 Bug

### B1. Jinja2 `TemplateError: System message must be at the beginning.` / `No user query found in messages.`

- **现象**：RolloutManager 在 `apply_chat_template` 时崩溃。Qwen3.5/3.6/3.8 的 chat template 有两条严格 guard：system message 必须在最前、必须有 user query。mask builder 逐条渲染 message delta 时，单独渲染一个 system message 会同时违反这两条。
- **根因**：`TrajectoryMaskBuilder._render_message_delta_str` 对 system message 用 `[msg] + BASE_CHAT_HISTORY` 渲染再剥离，但 Qwen 模板会注入合成 system message，导致剥离失败。
- **修复**（`rl/mask/trajectory_mask_builder.py`）：新增 `_USER_ONLY_BASE` 常量与 `_get_user_suffix_str()` 懒加载 helper（用 `render(BASE + user_msg) - render(BASE)` 得到干净的 user 后缀），system message 改为渲染 `[system_msg] + _USER_ONLY_BASE` 再剥掉 `_USER_ONLY_BASE` 部分，同时满足两条 guard。

### B2. `IndexError: list index out of range` in `_init_suffix_tokens`

- **现象**：`test_tokens[idx] == eos_id` 越界。
- **根因**：`tokenizer.apply_chat_template(..., tokenize=True)` 返回 `BatchEncoding` 而非纯 list，直接按下标迭代走的是 `_encodings`，长度/索引不对。
- **修复**（`rl/mask/trajectory_mask_builder.py`）：在 `_init_suffix_tokens` 里把 `BatchEncoding` 解包成纯 token list 再处理。（此修复一度因 `trajectory_mask_builder.py` 被意外删除、从 git HEAD 恢复时丢失，后重新补回。）

### B3. `TypeError: TrajectoryMaskBuilder.prepare_generate_input() takes 3 positional arguments but 4 were given`

- **现象**：`llm_proxy.py` 调 `prepare_generate_input(session_id, messages, tools)` 传了 4 个参数，但 builder 只收 3 个。
- **根因**：`tools` 支持是一次未提交的工作区改动，文件被从 HEAD 恢复后丢失了签名。
- **修复**（`rl/mask/trajectory_mask_builder.py`）：给 `prepare_generate_input` / `_ensure_path` / `_add_prompt_message` 加 `tools: Optional[List[Dict]]=None` 参数；`_ensure_path` 在 session 第一条 system message 时把 `tools` 传下去；`_add_prompt_message` 在 `tools is not None` 时改用新 helper `_render_first_system_delta_str` 渲染（带 `<tools>` 块，匹配 sglang 渲染），否则走原 `_render_message_delta_str`。

### B4. `TypeError: Can only get item pairs from a mapping.`

- **现象**：Jinja 模板里 `tool_call.arguments|items` 报错。
- **根因**：OpenHands 发的是 OpenAI 格式 `tool_calls`，`tool_call.function.arguments` 是 JSON 字符串；Qwen 模板对它用 `|items` 过滤器要求 dict/mapping。
- **修复**（`rl/llm_proxy.py`）：新增 `_normalize_messages_for_qwen_template`，把 `tool_call.function.arguments` 从 JSON 字符串解析成 dict，并把非字符串 `content` 强制成字符串；在 `proxy_chat_completions` 里 `prepare_generate_input` 之前调用。

### B5. Buffer 游标死锁：`rollout data is not ready` 永远不就绪

- **现象**：buffer server 持续 `new_items=0, ready_groups=0`，pending 组永远凑不齐 `group_size=8`，训练拿不到数据。DB 里其实已有满组（8+8），但 buffer 看不到。
- **根因**：`fetch_done_steps_with_context` 用 `id` 自增主键做游标（`id__gt=after_id`），但 terminal step 是 `reward_committer` **UPDATE 现有行**翻转 `is_terminal`（不是 INSERT），行 `id` 在创建时就定了。eval 晚翻转的行 id 已被游标越过 → 永远漏捞 → 组凑不齐 → 死锁。
- **修复**（滑动窗口游标，不动表，内存有界）：
  - `sqlite_strategy_impl.py`：`fetch_done_steps_with_context` 加 `lookback` 参数，查询改为 `id > after_id - lookback`（回看窗口 + 新行），`lookback>0` 时不加 `limit`。
  - `cloud_strategy_impl.py`：同签名加 `lookback`（暂不应用，云游标机制不同，留 follow-up）。
  - `manager.py`：转发 `lookback`。
  - `buffer_server.py`：新增 `served_pks` 集合 + `FETCH_LOOKBACK`（env `BUFFER_FETCH_LOOKBACK`，默认 100000），对回看重复行去重，`last_served_id` 仍按高水位推进，定期剪枝 `pk <= last_served_id - lookback`；`init_data_manager` 重启时清 `served_pks`。
- **详见**：`docs/guides/buffer-cursor-deadlock_CN.md`

### B6. Episode 全 eval 失败 + 熔断 → pool 停 → 二次死锁（`max_output_tokens` 截断）

- **现象**：buffer 又卡在 `new_items=0, ready_groups=0, pending={b377236b...: 5}`。DB 里该 job terminal 数停在 23 不涨，launcher 也不再起新 episode。launcher 日志显示 `lease pool exhausted` + `circuit_breaker_reason: "failure_rate=1.000 threshold=0.800 samples=20"`，且**每个 episode 的 eval 都失败**：`EVAL RULE complete: status=failed score=0.0000`，reason=`PatchEval runner did not provide cve_id, patch, and programming language`。
- **根因**（两层，但只有一层是真 bug）：
  1. **真 bug：OpenHands 生成被 `max_tokens` 截断**。存库的 response `finish_reason=length`，content 在 `...Let's first explore the repository.\n\n<tool` 处被切断——工具调用刚开始就被切掉。请求体里**没有 `max_tokens` 字段**（keys 只有 `messages/model/tools`）。OpenHands 默认 `max_output_tokens=0`（"自动检测"），对自定义 gateway 路由的 model 自动检测失败 → 请求不带 `max_tokens` → sglang 用极小默认值（~128）→ 生成在工具调用中途被截断 → OpenHands 拿到不完整的 tool call 无法执行 → agent 1 步就结束 → 没改文件 → `git diff` 空 → `patch=""`。
  2. rule_evaluator 要求 `cve_id`、`patch`、`language` 三者齐全才进官方评估；`patch` 为空时返回 `status=failed`（而非 `SUCCEEDED, score=0`）。于是**每个 episode 都 failed** → 连续 20 个失败触发 `simulation_lease_pool` 熔断器（`failure_rate≥0.8`）→ lease pool 耗尽 → launcher 不再起 episode → 只跑 29 个 → 凑不齐 `global_batch_size=64` → 训练死锁。
  - 注：`AIEVOBOX_MAX_STEPS=1` **不是** agent 步数限制（gateway 的 `AIEVOBOX_GATEWAY_MAX_STEPS=30` 才是，admission_control 按 session 计步），agent 只跑 1 步是截断导致没产出有效 tool call，不是被 step 限制卡住。`cve_id`/`programming_language` 在 dataset 里都有，唯一缺的就是 `patch`。
- **修复**（两层，gateway 注入是保底）：
  - `env/patcheval/openhands_runner.py`：在 `_run_openhands` 的 env 里显式设 `LLM_MAX_OUTPUT_TOKENS`（OpenHands 标准环境变量），默认 `8192`，可用 `PATCHEVAL_OPENHANDS_MAX_OUTPUT_TOKENS` 覆盖。新增 `DEFAULT_MAX_OUTPUT_TOKENS=8192` 常量与 `_positive_int` helper。（best-effort：实测当前 rjob 镜像里的 OpenHands 版本并未把该 env 翻译进请求的 `max_tokens`，请求里仍无 `max_tokens`，故还需 gateway 兜底。）
  - `gateway/app.py`：新增 `_ensure_default_max_tokens(payload)`，在两个请求 handler 解析完 payload 后调用——若请求里**既没有 `max_tokens` 也没有 `max_completion_tokens`**，则注入默认 `max_tokens`（env `GATEWAY_DEFAULT_MAX_TOKENS`，默认 `8192`，设 0 关闭）。gateway 是所有 LLM 调用的必经之路，在此注入**保证生效**，不依赖 OpenHands 版本/env 翻译。这样 OpenHands 请求被补上 `max_tokens=8192` → sglang 不再截断 → agent 能生成完整 tool call → 多步执行/改文件/产 patch → eval 进官方评估（`status=SUCCEEDED`，patch 错也只 `score=0`，不算 failed）→ 熔断器不跳 → pool 持续跑 → 凑齐 batch。
- **重启要求**：runner 跑在 rjob pod 里（从 gpfs 挂载），gateway 由 buffer server 经 `gateway_autostart` 拉起。改动要随新 episode 生效——**必须重启 buffer server**（会重启 launcher + gateway），让后续 episode 用上新 runner 代码 + 新 gateway 代码。

### B7. 封盘超时不匹配 → 52% episode 孤儿（`is_terminal=0`）→ group 凑不满 → 死锁

- **现象**：DB 里 44 个 session `is_terminal=0`（孤儿）vs 40 个 `is_terminal=1`（已封盘），孤儿率 52%。孤儿所在的 group 永远凑不满 `group_size` 个 terminal → buffer `pending={group_id: N}`（`N < group_size`）→ `ready_groups=0` → trainer 拿不到数据 → 死锁。
- **根因**：封盘时两边超时不匹配。
  - runner 侧（`simulation_worker` 调 `close_session`）：`gateway_close_timeout_s` 默认 **15s**（`args.py` / `types.py`）。
  - gateway 侧（close 端点）：`close_mode=soft_close` + `drain_timeout_s=30`（`gateway/config.py`），close 端点会**同步阻塞最多 30s** 等在途 LLM 请求 drain 完才回执。
  - sglang 单步 65s（27B 单卡解码 ~56 tok/s + 模型过度思考生成长），封盘时几乎总有 1 个在途请求 → gateway 必然 drain 满 30s → runner 15s 到点抛 `httpx.TimeoutException` 放弃 → **runner 以为失败走了**。
  - 关键：gateway 在 drain 超时后**仍然强封并写 `is_terminal=1`**（`app.py` drain 返回 False 后继续 enqueue session_close），但 runner 已经不等回执了 → 两边对不上 → 孤儿。
  - 注：每步推理的 65s 请求**不受 15s 管**（成功，`status=200 latency 95s`），15s 只卡最后的 `close_session`。
- **修复**（三层，治本 + 治源头）：
  - `args.py` / `manager/types.py`：`gateway_close_timeout_s` 默认 **15 → 45**（>gateway 的 30s drain），runner 能等到 gateway 封完回执 → 孤儿消失。**这是治本。**
  - `rl/buffer_server.py`：把 `--gateway-close-timeout-s` 注入 launcher cmd，可用 `AIEVOBOX_GATEWAY_CLOSE_TIMEOUT_S` 覆盖（默认 45）。
  - `rl/examples/patcheval/env.rjob.sh`：`AIEVOBOX_GATEWAY_MAX_STEPS` **30 → 12**，缩短 episode → 减少封盘时在途请求概率 + 降低 drain 压力。
  - `gateway/app.py`：`GATEWAY_DEFAULT_MAX_TOKENS` **16384 → 6144**，单步生成上限收紧 → 单步延迟从 ~290s 降到 ~110s 内 → drain 更容易在 30s 内自然完成（不用走到强封）。代价：模型"过度思考"长独白（~7-9k token）会更频繁撞 6144 上限被截断（`finish_reason=length`）——这是**有意的权衡**：宁可截断但封盘，不要完整但孤儿；长独白本就是低价值动作。
- **重启要求**：改 `args.py`/`types.py`/`buffer_server.py` 需重启 buffer server（会重启 launcher）；改 `env.rjob.sh` 需重启 buffer server 让新 env 生效；改 `gateway/app.py` 需重启 buffer server（gateway 是其子进程）。**总之重启 buffer server 即可全部生效。**

---

## 二、新增的 Feature / 配置

### B11. `get_training_info` matched=0 → 0 trainable groups → weight_version 永远 1（真正的训练阻断）

- **现象**：slime.log 大量 `get_training_info failed: session=..., has_data=True, matched=0, expected=N`，且 `Trainable groups added this round: 0`。weight_version 一直停在 1（从未发生权重更新）。12 步轮（64 个失败）和 40 步轮（56 个失败）都有——**长期 bug，非 40 步引入**。
- **根因**：生成与训练取数之间的消息格式不一致。
  - **生成时**（`llm_proxy.proxy_chat_completions`）：先调 `_normalize_messages_for_qwen_template(messages)`（`tool_call.arguments` JSON string→dict、`content` None→""），再 `prepare_generate_input`。所以 mask builder 内存树里的 `raw_message` 是**归一化后**的消息（arguments 是 dict）。
  - **训练取数时**（`slime_generator._get_record_training_info`）：直接读 `record["messages"]`（DB 存的是**原始 OpenAI 格式**，arguments 是 JSON string、content 可能 None），不归一化就传给 `get_training_info`。
  - `_message_matches` 比较：树的 dict-arguments vs DB 的 JSON-string-arguments → `left_meta != right_meta` → 不匹配 → `matched=0`。`has_data=True` 说明树**有**该 session 的子节点，只是消息对不上。
  - 后果：所有 session matched=0 → 返回空 tokens/mask → 0 trainable groups → trainer 拿不到数据 → 永不更新权重 → weight_version 卡 1。**这比 reward=0 更根本**——即使 reward 非零，0 trainable groups 也训不动。
- **修复**：`slime_generator.py::_get_record_training_info` 在调 `get_training_info` 前，对 `oai_messages` 调 `_llm_proxy_module._normalize_messages_for_qwen_template` 做同样的归一化，使 DB 取出的消息与内存树里的格式一致。
- **重启要求**：改的是 slime_generator（RolloutManager Ray actor）。需重启 slime generator（`run_slime_generator.sh`）生效；buffer server 不用重启。

---

## 二、新增的 Feature / 配置

### F1. tools 渲染支持（首条 system message 带 `<tools>` 块）

- `trajectory_mask_builder.py` 新增 `_render_first_system_delta_str`，session 第一条 system message 在带 `tools` 时正确渲染出 `<tools>...</tools>` 块进 prompt token，与 sglang 推理时的渲染对齐，保证 mask 与生成一致。

### F2. DAPO filter 默认关闭

- `rl/examples/patcheval/env.rjob.sh`：`export DAPO_filter="${PATCHEVAL_DAPO_FILTER:-false}"`，默认不过滤全 0 group，避免 pipeline 在早期 reward 全 0 时卡死。

### F3. `env.rjob.sh` 自包含

- 移除 `source "${REPO_ROOT}/rl/examples/geo3k_vl/env.sh"`，把所需基础设施默认值内联，避免引入 VL 任务的无关默认值导致配置串味。

### F4. `RL_EPOCH` 默认 100

- `env.rjob.sh`：`export RL_EPOCH="${PATCHEVAL_EPOCH:-100}"`（从 2 改为 100）。

### F5. buffer lookback 机制（可调）

- 新增环境变量 `BUFFER_FETCH_LOOKBACK`（默认 100000 id 单位），控制回看窗口大小以捞 late-flip terminal step；窗口大于 eval 时延折算的 step 数即安全。

---

## 三、相关文档

- `docs/guides/buffer-cursor-deadlock_CN.md` — B5 游标死锁的完整诊断、证据、修复方案、复现命令、提交归属。
- （`docs/guides/qwen3.5-system-message-error_CN.md` 原计划记录 B1，但磁盘上已缺失，内容已并入本文第一节。）

---

## 四、改动文件清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `rl/mask/trajectory_mask_builder.py` | bugfix + feature | B1/B2/B3 + F1：Qwen 模板 system/tools 渲染、BatchEncoding 解包、`tools` 参数 |
| `rl/llm_proxy.py` | bugfix | B4：`_normalize_messages_for_qwen_template` 解析 tool_call.arguments；加 `rl/mask` 到 sys.path |
| `rl/slime_generator.py` | 配置 + bugfix | 加 `rl/mask` 到 sys.path；B11：`_get_record_training_info` 对 DB 消息做归一化（治 matched=0 / 0 trainable groups） |
| `rl/examples/patcheval/env.rjob.sh` | 配置 | F2/F3/F4 + B7：DAPO filter 默认 false、自包含、RL_EPOCH=100、`AIEVOBOX_GATEWAY_MAX_STEPS` 30→12 |
| `core/data_manager/strategy/sqlite_strategy_impl.py` | bugfix | B5：`fetch_done_steps_with_context` 加 `lookback` |
| `core/data_manager/strategy/cloud_strategy_impl.py` | 签名对齐 | B5：加 `lookback` 参数（暂不应用） |
| `core/data_manager/manager.py` | 转发 | B5：`fetch_done_steps_with_context` 透传 `lookback` |
| `rl/buffer_server.py` | bugfix | B5：`served_pks` + `FETCH_LOOKBACK` + 去重 + 剪枝；`init_data_manager` 清 `served_pks`；B7：注入 `--gateway-close-timeout-s`（env `AIEVOBOX_GATEWAY_CLOSE_TIMEOUT_S`，默认 45） |
| `env/patcheval/openhands_runner.py` | bugfix | B6：`_run_openhands` 显式设 `LLM_MAX_OUTPUT_TOKENS`（默认 8192，best-effort） |
| `gateway/app.py` | bugfix | B6：`_ensure_default_max_tokens` 在请求缺 `max_tokens` 时注入默认；B7：默认值 16384→6144（收紧单步生成，降低 drain 压力） |
| `args.py` / `manager/types.py` | bugfix | B7：`gateway_close_timeout_s` 默认 15→45（>gateway drain 30s，治孤儿根因） |
| `docs/guides/buffer-cursor-deadlock_CN.md` | 文档 | B5 诊断与修复记录 |

---

## 五、已知遗留 / Follow-up

- **buffer 跨 job 状态污染**：`last_served_id` / `pending_items_by_instance` / `served_pks` 只在 `restart_training=True` 时清，换 job 不重启 buffer 会残留旧状态（同 `group_id` 跨 run 重复 → pending 累积）。建议 `start_rollout` 检测 job_id 变化时自动清。
- **cloud 后端 late-flip**：`cloud_strategy_impl.py` 的 `fetch_done_steps_with_context` 游标是 created_at 时间戳，理论上同样有 late-flip 风险，`lookback` 暂未应用，需单独验证。
- **github 屏蔽**：`env/patcheval/openhands_runner.py::_block_github_cdn` 是 patcheval 故意的防作弊 + 快速失败优化，**不是 bug**，无需改；但部分 episode 会在 openhands 启动 clone 扩展时失败（非致命，agent 靠预挂仓库继续）。
