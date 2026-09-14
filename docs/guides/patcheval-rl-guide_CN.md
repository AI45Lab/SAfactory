# PatchEval RL 调通指南

汇总 patcheval RL（RJob 模式，Qwen3.5-9B / Qwen3.8-27B）从跑不起来到能产出训练数据的关键改动。按问题分类，每条给出现象、根因、修复。

---

## 一、生成 / 模板相关 Bug

### B1. `TemplateError: System message must be at the beginning` / `No user query found`

- **现象**：RolloutManager `apply_chat_template` 崩溃。Qwen3.x 模板要求 system 在最前、必须有 user query。
- **根因**：`TrajectoryMaskBuilder._render_message_delta_str` 单独渲染 system message 时，模板注入合成 system → guard 失败。
- **修复**（`rl/mask/trajectory_mask_builder.py`）：新增 `_USER_ONLY_BASE` + `_get_user_suffix_str()`，system message 改渲染 `[system_msg] + _USER_ONLY_BASE` 再剥离，同时满足两条 guard。

### B2. `IndexError` in `_init_suffix_tokens`

- **根因**：`apply_chat_template(tokenize=True)` 返回 `BatchEncoding` 而非 list，按下标迭代错位。
- **修复**：解包成纯 token list 再处理。

### B3. `prepare_generate_input() takes 3 positional arguments but 4 were given`

- **根因**：`tools` 支持是未提交改动，文件从 HEAD 恢复后丢签名。
- **修复**：给 `prepare_generate_input` / `_ensure_path` / `_add_prompt_message` 加 `tools` 参数；首条 system 带 `<tools>` 块时用 `_render_first_system_delta_str` 渲染（F1）。

### B4. `TypeError: Can only get item pairs from a mapping`

- **根因**：OpenHands 发 OpenAI 格式 `tool_calls`，`arguments` 是 JSON 字符串，Qwen 模板对它用 `|items` 要求 dict。
- **修复**（`rl/llm_proxy.py`）：`_normalize_messages_for_qwen_template` 把 `arguments` JSON string→dict、`content` None→""，在 `prepare_generate_input` 前调用。

---

## 二、Buffer / 取数相关 Bug

### B5. Buffer 凑不齐 group → `rollout data is not ready` 死锁

- **现象**：buffer 持续 `new_items=0, ready_groups=0`，pending 组永远凑不齐 `group_size`，训练拿不到数据。DB 里其实已有满组，buffer 看不到。
- **根因**：terminal step 是 `reward_committer` **UPDATE 现有行**翻转 `is_terminal`（不是 INSERT），行 `id` 在创建时就定了。用 `id` 自增游标增量捞 terminal step，eval 晚翻转的行 id 已被游标越过 → 永远漏捞 → 组凑不齐 → 死锁。
- **现行修复（finished-env 两阶段 fetch，已取代早期 lookback 方案）**：
  - 不再用 `id` 游标捞单步 terminal。改为：
    - Phase 1：`list_environment_rows(finished=True)` —— 先捞被 `mark_environment_finished` 标记完成的 env。
    - Phase 2：`list_terminal_steps_for_sessions(env_ids)` —— 再批量取这些 env 的 terminal step。
  - 不变量：`mark_environment_finished` 只在 env 全部 step 都 `is_terminal=True` 后才调用 → `finished=True` 保证训练 step 全到齐 → 无 late-flip → 不需要 lookback / 去重。
  - 早期 lookback + `served_pks` 方案（`fetch_done_steps_with_context`）已废弃删除（提交 `2ba46ef`）。
- **改动文件**：`rl/buffer_server.py`（`fetch_new_items_from_db` 两阶段逻辑）、`core/data_manager/manager.py` + `sqlite/cloud_strategy_impl.py`（`list_environment_rows` / `list_session_step_rows`）。

### B11. `get_training_info matched=0` → 0 trainable groups → weight_version 卡 1

- **现象**：slime.log 大量 `matched=0, expected=N`，`Trainable groups added this round: 0`，weight_version 永远 1。
- **根因**：生成时 `llm_proxy` 先 `_normalize_messages_for_qwen_template`（arguments→dict），训练取数时直接读 DB（arguments 是 JSON string）→ `_message_matches` 对不上 → matched=0 → 0 trainable groups → 永不更新权重。**比 reward=0 更根本**。
- **修复**（`rl/slime_generator.py::_get_record_training_info`）：调 `get_training_info` 前对 DB 消息做同样的 `_normalize_messages_for_qwen_template` 归一化。

---

## 三、Episode / 封盘相关 Bug

### B6. 每步生成被截断 → agent 1 步结束 → patch 空 → eval 全 failed → 熔断 → pool 停 → 死锁

- **根因**：OpenHands 请求不带 `max_tokens`（自定义 gateway 路由的 model 自动检测失败）→ sglang 用极小默认值 → 生成在工具调用中途被截断（`finish_reason=length`）→ agent 拿不到完整 tool call → 1 步结束 → 没改文件 → `patch=""` → rule_evaluator 返回 failed → 连续失败触发熔断 → pool 停。
- **修复**（gateway 兜底，保证生效）：
  - `gateway/app.py`：`_ensure_default_max_tokens`，请求缺 `max_tokens` 时注入默认（env `GATEWAY_DEFAULT_MAX_TOKENS`，默认 8192，设 0 关闭）。
  - `env/patcheval/openhands_runner.py`：`_run_openhands` 显式设 `LLM_MAX_OUTPUT_TOKENS`（best-effort，当前镜像版本未翻译进请求，靠 gateway 兜底）。

### B7. 封盘超时不匹配 → 孤儿 session（`is_terminal=0`）→ group 凑不满 → 死锁

- **根因**：runner `close_session` 超时 15s < gateway `drain_timeout=30s` → runner 15s 抛超时放弃，gateway 仍强封写 `is_terminal=1`，两边对不上 → 孤儿。
- **修复**（治本 + 治源头）：
  - `args.py` / `manager/types.py`：`gateway_close_timeout_s` 默认 15→45（>gateway 30s drain）。**治本。**
  - `rl/buffer_server.py`：注入 `--gateway-close-timeout-s`（env `AIEVOBOX_GATEWAY_CLOSE_TIMEOUT_S`，默认 45）。
  - `rl/examples/patcheval/env.rjob.sh`：`AIEVOBOX_GATEWAY_MAX_STEPS` 收紧，缩短 episode。
  - `gateway/app.py`：`GATEWAY_DEFAULT_MAX_TOKENS` 16384→6144，收紧单步生成，降低 drain 压力（有意权衡：宁可截断但封盘，不要完整但孤儿）。

---

## 四、Megatron GDN 不支持 Packed Sequence

### GDN packed-seq monkey-patch

- **现象**：Qwen3.8-27B（48 层 GDN + 16 层 Full Attention）第一步 `compute_log_prob` 崩溃：`NotImplementedError: GDN does not support packed sequence for now.`（`megatron/core/ssm/gated_delta_net.py:302`）。
- **根因**：slime 默认 `thd`（packed）布局，多条 trajectory concat 成长序列用 `cu_seqlens` 标边界。Megatron 原生 GDN forward 入口直接 raise，没把 `cu_seqlens` 传给底层 `chunk_gated_delta_rule`（fla 本身已支持 cu_seqlens）。bridge 模式忽略 `--spec`（slime 的 `qwen3_5.py` 有支持 cu_seqlens 的 GDN），用不上。
- **修复**（运行时 monkey-patch，不重打镜像、不改 Megatron/slime 核心）：
  - `rl/patches/gdn_packed_seq.py`：patch `GatedDeltaNet.forward`，删 raise，从 `packed_seq_params.cu_seqlens_q` 提取 cu_seqlens 传给 `chunk_gated_delta_rule` 和 `causal_conv1d_fn`。
  - `rl/patches/sitecustomize.py`：启动时自动加载。
  - `env.rjob.sh`：`export PYTHONPATH="${REPO_ROOT}/rl/patches${PYTHONPATH:+:${PYTHONPATH}}"`。
- **为什么不用 bshd（padding）**：27B TP=4 在 140GB 卡 OOM（差 822 MiB），thd packing 内存更省。
- **回退**：删 `PYTHONPATH` 那行即禁用。

---

## 五、配置 / Feature

- **F2 DAPO filter 默认关闭**：`env.rjob.sh` `DAPO_filter=false`，避免早期 reward 全 0 时 pipeline 卡死。
- **F3 env.rjob.sh 自包含**：移除 `source geo3k_vl/env.sh`，内联默认值，避免 VL 任务配置串味。
- **F4 RL_EPOCH 默认 100**（从 2）。
- **reward pre-gate 放宽**（`env/patcheval/rule_evaluator.py`）：缺 cve_id/patch/language 或容器起不来时返回 `SUCCEEDED+0.0`（对齐 bench 的 validation_fail=0 语义），不再返回 FAILED（FAILED 会让 RewardCommitter 拒写 → reward NULL → session 不 seal）。同步覆盖 162 个 per-CVE 副本。
- **LOSS_MASK_TYPE 注入**（`rl/run_slime_generator.sh`）：`RUNTIME_ENV_JSON.env_vars` 加 `LOSS_MASK_TYPE=qwen3_5`，修复 RolloutManager Ray actor 拿不到 → 用 base adapter → B1 报错的 bug。
- **gateway host**：env 脚本 `hostname -I` → `hostname -i`。

---

## 六、改动文件清单

| 文件 | 说明 |
|---|---|
| `rl/mask/trajectory_mask_builder.py` | B1/B2/B3 + F1：模板渲染、BatchEncoding 解包、tools |
| `rl/llm_proxy.py` | B4：`_normalize_messages_for_qwen_template` |
| `rl/slime_generator.py` | B11：训练取数归一化；加 `rl/mask` 到 sys.path |
| `rl/buffer_server.py` | B5：finished-env 两阶段 fetch；B7：`--gateway-close-timeout-s` |
| `core/data_manager/manager.py` / `*_strategy_impl.py` | B5：`list_environment_rows` / `list_session_step_rows` |
| `env/patcheval/rule_evaluator.py` | reward pre-gate 放宽（+ 162 per-CVE 副本） |
| `env/patcheval/openhands_runner.py` | B6：`LLM_MAX_OUTPUT_TOKENS` |
| `gateway/app.py` | B6：`_ensure_default_max_tokens`；B7：默认 6144 |
| `args.py` / `manager/types.py` | B7：`gateway_close_timeout_s` 15→45 |
| `rl/patches/gdn_packed_seq.py` + `sitecustomize.py` | GDN packed-seq monkey-patch |
| `rl/examples/patcheval/env.rjob*.sh` | F2/F3/F4 + B7 + gateway host + max_steps |
| `rl/run_slime_generator.sh` | LOSS_MASK_TYPE 注入 |

---

## 七、已知遗留 / Follow-up

- **cloud 后端 late-flip**：`cloud_strategy_impl.py` 的游标是 created_at 时间戳，理论上同样有 late-flip 风险，finished-env 方案需在云后端验证。
- **无 dockerd 时的学习信号**：launcher 侧无 dockerd 时 CVE 容器起不来，所有 reward=0.0 → 组内无方差 → GRPO 梯度≈0。管道能跑（`ready_groups>0`）但无真实学习。要 1.0 正样本需让 `openhands_runner` 在 pod 内判分并上报 `strict_success/poc_passed`，使 `rule_evaluator` 的 runner fallback 命中。
- **github 屏蔽**：`openhands_runner._block_github_cdn` 是 patcheval 故意的防作弊 + 快速失败，**不是 bug**。
