# Safactory 数据流转介绍

## Schemas

Safactory 在 Cloud 模式下使用两个逻辑 schema：`env_config` 保存环境实例，`session_steps` 保存运行轨迹。物理表由 `WT_SDK_PROFILE` 选择；默认 profile 是 `test`，`production` / `prod` 使用生产表。

### env_config

一行代表一个环境实例，保存所属 job、环境参数、镜像以及 `finished` 调度状态。

| 类型 | Test | Production | Serving |
| --- | --- | --- | --- |
| 对应表 | `env_config_test` | `evaluation_env_config` | 无；`env_config` 不设 Serving 层 |

#### env_config 字段 schema

| 字段 | Arrow 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `id` | `int64` | 否 | 应用层分配的逻辑行 ID，不是数据库自增主键。 |
| `env_id` | `string` | 否 | 环境实例 UUID；正常流程中也作为 `session_id`。 |
| `job_id` | `string` | 否 | 所属 evaluation job，用于任务级查询、resume 和 rebuild。 |
| `group_id` | `string` | 是 | RL / GRPO 的样本分组 ID。 |
| `finished` | `bool` | 否 | 调度完成状态；创建时为 `false`，运行完成后更新为 `true`。 |
| `env_name` | `string` | 否 | 环境或 adapter 名称。 |
| `env_params` | `string` | 是 | JSON 字符串，保存环境参数、dataset 及 Safactory 内部运行元数据。 |
| `image` | `string` | 是 | 环境使用的 Docker / runtime image。 |
| `created_at` | `int64` | 否 | 创建时间，使用 Unix timestamp。 |

`env_config` 不分区；常用索引字段为 `env_name`、`env_id`、`job_id`、`group_id` 和 `finished`。

### session_steps

Safactory 的 `session_steps` 在 Cloud 中写入 Landing 表；Serving 表使用相同 schema，承接数据平台处理后的发布数据。

| 数据层 | Test | Production | Safactory 使用方式 |
| --- | --- | --- | --- |
| Landing | `v2_landing_test` | `wind_tunnel_landing` | Safactory runtime 读写的原始 step / 轨迹表 |
| Serving | `serving_test` | `wind_tunnel_serving` | 数据平台 ETL 的发布表；Safactory 正常 runtime 不读写 |

#### session_steps 字段 schema

Landing 与 Serving 使用相同的 28 列 schema：

| 字段 | Arrow 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `dataset_type` | `string` | 否 | 数据集类型；Safactory 轨迹默认写入 `RL`。 |
| `dt` | `string` | 否 | 记录日期，通常为 `YYYY-MM-DD`。 |
| `id` | `string` | 否 | Step 的应用层记录 ID；Safactory 逻辑字段名为 `record_id`。 |
| `session_id` | `string` | 是 | Session / trajectory ID，正常情况下等于 `env_id`。 |
| `created_at` | `int64` | 否 | 原始记录创建时间，Safactory 通常使用 Unix 秒。 |
| `source_updated_at` | `int64` | 否 | Source 最近更新时间，由 SDK 管理，使用 Unix 毫秒。 |
| `serving_updated_at` | `int64` | 是 | 数据发布到 Serving 的时间；Safactory runtime 不使用。 |
| `step_id` | `int32` | 是 | Session 内的 step 序号。 |
| `is_terminal` | `bool` | 是 | 当前记录是否终止该 session。 |
| `env_id` | `string` | 是 | 产生该轨迹的环境实例 ID。 |
| `job_id` | `string` | 是 | 所属 job，也是 HASH 分区键；Safactory 正常写入时必填。 |
| `is_truncated` | `bool` | 是 | Session 是否因超时、步数限制等原因被截断。 |
| `step_reward` | `float32` | 是 | 当前 step 的评分。 |
| `reward` | `float32` | 是 | Session 的最终或累计评分。 |
| `messages` | `json<string>` | 是 | 请求侧对话历史，底层以 JSON 字符串保存。 |
| `response` | `json<string>` | 是 | 当前模型输出，底层以 JSON 字符串保存。 |
| `chosen_trace` | `json<string>` | 是 | Preference / RL 数据中的 chosen trajectory。 |
| `rejected_trace` | `json<string>` | 是 | Preference / RL 数据中的 rejected trajectory。 |
| `ground_truth_answer` | `string` | 是 | 标准答案。 |
| `reference_answer` | `string` | 是 | 参考答案。 |
| `search_text` | `string` | 是 | 数据平台 ETL 聚合生成的全文检索文本。 |
| `agent_model` | `string` | 是 | 生成响应的模型；对应 Safactory 的 `llm_model`。 |
| `env_name` | `string` | 是 | 环境或 adapter 名称。 |
| `is_session_completed` | `bool` | 是 | Session 是否已经完成。 |
| `is_trainable` | `bool` | 是 | 该记录是否具备训练资格。 |
| `meta_json` | `json<string>` | 是 | 扩展元数据，包括 `group_id`、request、dataset、telemetry 和评测结果。 |
| `tags` | `list<string>` | 是 | 记录标签列表。 |
| `blob_manifest` | `list<string>` | 是 | 多模态资产清单，通常保存需要预加载的 `s3://` 资源。 |

Landing 与 Serving 都按 `job_id` 做 128 bucket 的 HASH 分区；两者字段一致，但 Serving 建有更多面向查询和发布的索引。

## Workflow

下图只展示启用 Evaluator 且任务成功完成的主流程，不包含其他控制或异常分支。为了让图保持简洁，图中的 **EnvManager** 把 Launcher、SimulationFlow 以及调度和执行组件画在了一起；它不是当前代码中的实际类或独立进程，也不等同于 SDK 的 `EnvConfigManager`。

Draw.io 源文件：[safactory-workflow.drawio](./safactory-workflow.drawio)。在 diagrams.net 中选择 **File → Import From → Device** 即可继续编辑。

下面只说明正常主流程中的数据读写。该流程不会执行 **DELETE**，也不包含 rebuild 或 resume。

### 1. EnvManager 把环境信息写入 env_config

1. EnvManager 会先读取 YAML 配置，展开 dataset、`env_num` 和 RL group。每个环境实例都会得到一个独立的 `env_id`；同一环境、同一 `task_idx` 展开的多个实例会共用一个 `group_id`。这一步读取的是配置文件，还没有读取数据库。
2. 接着，EnvManager 会向当前 profile 对应的 `env_config` 表执行 **INSERT**。每个环境实例写入一行，字段包括 `env_id`、`job_id`、`group_id`、`env_name`、`env_params`、`image`、`created_at` 和 `finished=false`；`id` 由 SDK 分配。
3. 这一步不会读取或更新 Landing，也不会写入 Serving。

### 2. EnvManager 从 env_config 读取待运行环境

1. EnvManager 会按当前 `job_id` 和 `finished=false` **READ** `env_config`，找到还没有运行完成的环境。
2. 它会读取这一行的 `id`、`env_id`、`env_name`、`env_params`、`image` 和 `group_id`，并在后续运行中使用 `env_id` 作为 `session_id`。
3. 这一步只读数据，不更新任何字段；当前正在运行哪个环境只保存在 EnvManager 内存中，不会写回 `env_config`。`finished` 仍然是 `false`，Landing 和 Serving 也不会发生变化。

### 3. Gateway 把每次模型调用写入 Landing

1. Gateway 第一次处理某个 `session_id` 时，会按 `env_id=session_id` **READ** `env_config`，取得 `job_id`、`env_name`、`group_id`，并从 `env_params` 中读取 dataset。
2. 每完成一次模型调用，Gateway 就会向 Landing 执行一次 **INSERT**。新行会写入 `dataset_type=RL`、`id`、`session_id`、`env_id`、`step_id`、`job_id`、`env_name`、`agent_model`、`messages`、`response` 和 `meta_json`。其中 `meta_json` 保存 `group_id`、原始 request、dataset 以及本次调用的状态、耗时和 token 等信息。
3. 新行初始写入 `step_reward=0.0`、`reward=null`、`is_terminal=false`、`is_session_completed=false` 和 `is_trainable=false`；`is_truncated` 按实际情况写入。`dt`、`source_updated_at` 和 `blob_manifest` 由 SDK 补齐。strict、async 和 streaming 只会影响这次 INSERT 何时完成，不会改变写入内容；Gateway 不会在这里更新已有 Landing 行，也不会写入 Serving。

### 4. Gateway 把尚未写完的 Landing 记录写完

1. Environment / Agent 运行结束后，EnvManager 会通知 Gateway 关闭这个 session。Gateway 不会为“关闭 session”单独新增一行。
2. 如果还有正在处理或已经排队的模型调用，它们仍会按第 3 步的字段写入 Landing。Gateway 会一直等到这些 **INSERT** 全部完成；如果没有剩余记录，这一步就不会产生新的表写入。
3. Gateway 不会在这一步写入最终 reward，也不会更新 `is_terminal`、`is_session_completed`、`env_config.finished` 或任何 Serving 字段。

### 5. 评测结果写回 Landing

1. Evaluator 会根据本次运行结果计算分数、状态和原因。当前主流程中的 Evaluator 不读取 `env_config`、Landing 或 Serving，也不直接修改这三张表。
2. 评测成功后，RewardCommitter 会先取得 Gateway 记录的最新成功 `step_id`，再用 `job_id`、`session_id`、`step_id` 和 `agent_model` **READ** Landing，读取对应行的 `id`、`created_at` 和 `meta_json` 等字段。
3. RewardCommitter 随后按 `(job_id, id)` 更新这条 Landing 记录。写入时会带上原来的 `dataset_type` 和 `created_at`，实际更新的评测字段是 `step_reward`、`reward`、`meta_json.eval`、`is_terminal=true`、`is_truncated` 和 `is_session_completed=true`。`step_reward` 与 `reward` 都使用评测分数，`meta_json.eval` 保存评测状态、分数、原因和完整结果；原来的 `messages`、`response` 和 `is_trainable` 不会被改写。图中的正常主流程已经存在目标行，这次更新使用 `insert_missing=false`，不会补插一条新记录；更新后还会再次 **READ** 同一条记录，确认字段已经写入，SDK 同时刷新 `source_updated_at`。

### 6. EnvManager 更新 env_config 的完成状态

1. 确认 Landing 的评测结果已经写入后，EnvManager 会按当前 `job_id` 和 `env_id` **READ** `env_config`，确认对应的环境记录存在。
2. 然后它只把该行的 `finished` **UPDATE** 为 `true`，`env_name`、`env_params`、`image` 和 `group_id` 等其他字段保持不变。
3. Landing 的更新和 `env_config.finished` 的更新是前后两次独立提交；这一步不会再次修改 Landing，也不会写入 Serving。

### 7. Data Platform ETL 更新 Landing，再写入 Serving

1. Data Platform ETL 的任务服务会定时 **READ** `env_config` 的 `id`、`job_id`、`env_id` 和 `finished`。同一个 job 的环境全部变为 `finished=true`，并且 `env_id` 没有空值或重复值后，它才会把这个 job 放入异步处理队列；这不是 `env_config` 更新后立即触发的同步操作。
2. 任务开始后，ETL 会 **READ** Landing 中 `is_session_completed=true` 的 session。它会比较计算结果和当前值，只在内容确实变化时按 `(job_id, id)` **UPDATE** Landing：更新各行的 `is_trainable`，把最终 reward 写入选中的可训练行，并在需要补充模型的推理内容时更新 `messages`。发生实际更新的记录会同时刷新 `source_updated_at`。
3. 随后，ETL 会 **READ** Landing 中 `is_trainable=true` 的记录，根据 `messages` 和 `response` 生成 `chosen_trace`，根据 `job_id` 生成 `tags`，再生成 `search_text`，最后按 `(job_id, id)` 把完整记录 **UPSERT** 到 `serving_test` 或 `wind_tunnel_serving`。Serving 保留 Landing 的 `source_updated_at`，并在每次写入时刷新 `serving_updated_at`。当前流程不会删除 Landing 或 Serving 中的记录；一条已发布记录后来变为不可训练时，ETL v1 也不会自动从 Serving 删除它。
