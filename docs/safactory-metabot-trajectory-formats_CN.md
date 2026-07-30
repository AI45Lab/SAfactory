# SAfactory、MetaBot Rollout 与 Origin-CoT 轨迹格式对比

## 1. 文档范围

本文对比三类数据：

1. SAfactory 当前 PatchEval Claude Code 流程产生的轨迹；
2. MetaBot rollout 完成后产生的原始轨迹和标准化 conversation；
3. MetaBot `get_origin_cot` 完成后产生的派生轨迹。

三者不是同一种文件的三个版本。SAfactory 以训练 step 和评测生命周期为中心；
MetaBot rollout 同时保存 Harness、Backend 和标准化 conversation 三种边界；
Origin-CoT 则是在 MetaBot Backend Raw Trace 上仅替换已签名 thinking 后生成的派生数据。

```text
SAfactory Claude Code
  -> session_steps（训练/评测 step）
  -> response（每次 Provider 请求的完整 Raw Artifact JSON）
  -> provider raw artifacts（可选的外部审计副本）

MetaBot rollout
  -> conversation_epoch<N>.json（标准化对话）
  -> proxy_traces_epoch<N>.json（分析轨迹）
  -> proxy_traces_raw_epoch<N>.json（Harness Raw）
  -> proxy_traces_backend_raw_epoch<N>.json（Backend Raw）

MetaBot get_origin_cot
  -> 读取 Backend Raw + 原 conversation
  -> 按唯一 Signature 调用 CoT 提取服务
  -> 只替换已签名 content block 的 thinking
  -> 重建并校验 conversation
  -> 发布 *_origin_cot.json 和状态文件
```

## 2. SAfactory 当前轨迹

### 2.1 存储布局

SAfactory 当前把完整 Provider Raw Artifact 保存到 SQLite `response`，同时保留
外部 JSON 作为审计副本：

```text
<run>.db
  └── session_steps
        ├── messages
        ├── response（完整 Provider Raw Artifact）
        ├── reward / terminal / trainable
        └── env_state.provider_trace ─────────────┐
                                                  │ artifact_path
provider-traces/<session_id>/<request_id>.json <──┘
```

`response` 是 Provider 原始证据的数据库主副本，包含 Signature；外部 artifact
是相同内容的原子审计副本。`env_state.provider_trace` 只保存路径、SHA-256、
Signature 数量与采集完整性，不重复嵌入 Raw Artifact。

### 2.2 `session_steps` 行结构

每次 LLM 调用对应一行，核心字段如下：

```json
{
  "session_id": "session UUID",
  "step_id": 1,
  "env_name": "patcheval_cve_...",
  "llm_model": "claude-opus-4-6-thinking",
  "group_id": "...",
  "job_id": "...",
  "messages": "[JSON conversation history]",
  "response": "{\"schema_version\":1,\"boundary\":\"provider_raw\",...}",
  "step_reward": 0.0,
  "reward": 0.0,
  "env_state": "{JSON telemetry and provider trace reference}",
  "is_terminal": 0,
  "is_truncated": 0,
  "is_session_completed": 0,
  "is_trainable": 1
}
```

`messages` 是标准化后的累积对话历史。Anthropic assistant response 会被转换成：

```json
{
  "role": "assistant",
  "content": "visible text",
  "reasoning_content": "thinking text",
  "reasoning": "thinking text",
  "tool_calls": [
    {
      "id": "toolu_...",
      "type": "function",
      "function": {
        "name": "Bash",
        "arguments": "{\"command\":\"...\"}"
      }
    }
  ]
}
```

这个标准化消息保留 thinking 文本和工具语义，但不保留 Anthropic
`signature` 原文，也不保留 SSE 事件边界。

`env_state` 保存 Gateway telemetry，例如 endpoint、状态码、token、TTFT、
延迟、stream 状态、截断状态和 Provider Trace 引用：

```json
{
  "event_type": "gateway_inference",
  "request_id": "...",
  "endpoint": "messages",
  "status_code": 200,
  "is_stream": true,
  "prompt_tokens": 123,
  "completion_tokens": 45,
  "ttft_ms": 16829.62,
  "output_chunk_count": 61,
  "finish_reason": "tool_use",
  "llm_step_index": 1,
  "provider_trace": {
    "schema_version": 1,
    "boundary": "provider_raw",
    "capture_mode": "full",
    "artifact_path": ".../<session_id>/<request_id>.json",
    "artifact_sha256": "...",
    "signature_count": 1,
    "signature_total_bytes": 772,
    "capture_complete": true,
    "capture_error": null
  }
}
```

### 2.3 Provider Raw Artifact

每次 Anthropic Provider 调用产生一个下列 JSON 对象。它序列化到对应
`session_steps.response`，并写入一个内容相同的外部 JSON 审计副本：

```json
{
  "schema_version": 1,
  "boundary": "provider_raw",
  "session_id": "...",
  "request_id": "...",
  "llm_step_index": 1,
  "model": "claude-opus-4-6-thinking",
  "endpoint": "messages",
  "status_code": 200,
  "request": {
    "model": "claude-opus-4-6-thinking",
    "messages": [],
    "tools": [],
    "thinking": {
      "type": "enabled",
      "budget_tokens": 1024
    },
    "stream": true
  },
  "response": {
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "thinking",
        "thinking": "...",
        "signature": "..."
      },
      {
        "type": "tool_use",
        "id": "...",
        "name": "Bash",
        "input": {}
      }
    ]
  },
  "stream_text": "event: message_start\ndata: ...",
  "signatures": [
    {
      "field": "signature",
      "json_path": "$.response.content[0].signature",
      "signature": "...",
      "sha256": "...",
      "byte_length": 772
    }
  ],
  "capture": {
    "complete": true,
    "streamed": true,
    "error": null,
    "captured_at": "ISO-8601"
  }
}
```

它同时保存 Provider 请求、重建后的 Anthropic response、原始 SSE 文本、
Signature 索引和采集完整性。它不包含 SAfactory reward、terminal 或官方评测
结果，这些属于 DB 中的训练/环境生命周期。

## 3. MetaBot rollout 轨迹

MetaBot 对同一次 rollout 保存多个边界，而不是只保存一个“轨迹文件”。
文件名中的 `<N>` 是 epoch；重复写入时还可能产生 `.1`、`.2` 等版本快照。

### 3.1 `conversation_epoch<N>.json`

这是供分析、训练或交付使用的标准化 conversation：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "..."}
      ]
    },
    {
      "role": "assistant",
      "content": "visible answer",
      "reasoning_content": "provider thinking",
      "reasoning": "provider thinking",
      "tool_calls": [
        {
          "id": "toolu_...",
          "type": "function",
          "function": {
            "name": "Bash",
            "arguments": "{\"command\":\"...\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "toolu_...",
      "name": "Bash",
      "content": "tool output"
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "Bash",
      "description": "...",
      "parameters": {}
    }
  ]
}
```

它和 SAfactory `session_steps.messages` 最接近，但并不等价：

- MetaBot 是每个 epoch 一个完整 conversation 文件；
- SAfactory 是每个 LLM step 一行累积 history；
- SAfactory 额外绑定 reward、terminal、job、environment 和 evaluation；
- 两者都将工具调用标准化，但具体 system/user message 投影规则不同。

### 3.2 `proxy_traces_epoch<N>.json`

这是用于分析的 turn 级轨迹。它保留 session、epoch、模型、工具定义、成本、
时延和 turn 信息，但有意不保存 `raw_request` / `raw_response` 大对象。

典型顶层结构：

```json
{
  "session_id": "...",
  "epoch": 0,
  "process_id": null,
  "metadata": {
    "model": "claude-opus-4-6-thinking",
    "bot_name": "...",
    "task_status": "..."
  },
  "tools": [],
  "turns": [
    {
      "turn_number": 1,
      "model": "claude-opus-4-6-thinking",
      "cost_usd": 0.19,
      "duration_s": 5.1,
      "is_first_turn": true
    }
  ]
}
```

### 3.3 `proxy_traces_raw_epoch<N>.json`

这是 Claude Code/Harness 边界看到的原始协议，顶层带：

```json
{
  "trace_boundary": "harness",
  "session_id": "...",
  "epoch": 0,
  "metadata": {},
  "total_turns": 1,
  "total_cost_usd": 0.19,
  "created_at": "...",
  "completed_at": "...",
  "turns": [
    {
      "turn_number": 1,
      "raw_request": {},
      "raw_response": {}
    }
  ]
}
```

这个边界用于回答“Agent/Harness 实际发送和接收了什么”。当链路中存在协议转换
时，它不一定包含 Provider Backend 最终返回的字段。

### 3.4 `proxy_traces_backend_raw_epoch<N>.json`

这是 LiteLLM 与模型服务之间的 Backend Raw 边界，也是完整
`get_origin_cot` pipeline 的主要输入：

```json
{
  "trace_boundary": "litellm_backend",
  "session_id": "...",
  "epoch": 0,
  "metadata": {
    "model": "claude-opus-4-6-thinking"
  },
  "total_turns": 1,
  "total_cost_usd": 0.19,
  "turns": [
    {
      "turn_number": 1,
      "model": "claude-opus-4-6-thinking",
      "raw_request": {
        "messages": [],
        "tools": []
      },
      "raw_response": {
        "type": "message",
        "content": [
          {
            "type": "thinking",
            "thinking": "provider-visible thinking",
            "signature": "opaque signature"
          },
          {
            "type": "text",
            "text": "visible answer"
          }
        ]
      }
    }
  ]
}
```

Signature 必须在这个边界仍然存在，Origin-CoT 才能提取。若 Provider 前的协议桥
已删除 Signature，后处理无法恢复。

## 4. `get_origin_cot` 后的轨迹

### 4.1 输入

完整 pipeline 以同一 task 的两个文件为基线：

```text
proxy_traces_backend_raw_epoch0.json
conversation_epoch0.json
```

它递归收集 Backend Raw 中所有非空 `signature`，以 SHA-256 去重；相同
Signature 在历史 request 和最终 response 中出现多次时，只调用一次提取服务，
然后替换所有对应 occurrence。

### 4.2 替换规则

输入 block：

```json
{
  "type": "thinking",
  "thinking": "provider-visible thinking",
  "signature": "opaque signature"
}
```

输出 block：

```json
{
  "type": "thinking",
  "thinking": "extracted origin CoT",
  "signature": "opaque signature"
}
```

只允许 `thinking` 改变。`signature`、block 类型、文本回答、工具调用、工具结果、
消息顺序、tool ID、参数和 tool schema 必须保持不变。

### 4.3 最终文件

`proxy_traces_backend_raw_epoch0_origin_cot.json`

- 与原 Backend Raw envelope 相同；
- 所有成功提取的已签名 thinking 被 Origin-CoT 替换；
- 原 Signature 保留，用于 provenance 和复核；
- 未签名 thinking 不允许被修改。

`conversation_epoch0_origin_cot.json`

- 从替换后的 Backend Raw 重新投影；
- 可见回答和工具轨迹应与原 `conversation_epoch0.json` 等价；
- `reasoning` / `reasoning_content` 更新为提取后的 Origin-CoT。

`origin_cot_state.json`

```json
{
  "schema_version": 1,
  "status": "completed",
  "source_sha256": "...",
  "baseline_sha256": "...",
  "config": {
    "model": "claude-sonnet-4-6",
    "max_tokens": 128000
  },
  "signatures": {
    "<signature_sha256>": {
      "status": "success",
      "occurrence_paths": [],
      "attempts": 1,
      "response_sha256": "...",
      "last_error": null
    }
  },
  "reasoning_replacements": [
    {
      "assistant_path": "/messages/1",
      "reasoning_path": "/messages/1/reasoning",
      "reasoning_content_path": "/messages/1/reasoning_content",
      "source_block_paths": [
        "/turns/1/raw_response/content/0"
      ],
      "signature_digests": [
        "<signature_sha256>"
      ]
    }
  ],
  "final_artifacts": {
    "backend_raw": {
      "relative_path": "...",
      "sha256": "..."
    },
    "conversation": {
      "relative_path": "...",
      "sha256": "..."
    }
  }
}
```

处理中还会出现：

```text
proxy_traces_backend_raw_epoch0_origin_cot.partial.json
```

它与 `origin_cot_state.json` 配合支持逐 Signature checkpoint、失败重试和安全续跑。

### 4.4 等价性保证

发布前 pipeline 会执行两层检查：

1. Backend Raw 等价：屏蔽目标 `thinking` 后，Origin-CoT Backend Raw 必须与
   原 Backend Raw 完全相同；
2. Conversation 等价：除已声明 provenance 的 `reasoning` /
   `reasoning_content` 外，标准化 conversation 必须与 rollout baseline 等价。

因此 Origin-CoT 结果不是普通的“补一个 CoT 字段”，而是带来源路径、hash、
checkpoint 和不可见行为不变约束的派生数据。

## 5. 三类轨迹的核心区别

### SAfactory 当前轨迹

- 主索引：`session_id + step_id`；
- 主存储：SQLite `session_steps`；
- 原始证据：每个 request 的 Provider Raw Artifact 存入 `response`，并可保留外部副本；
- 强项：reward、terminal、trainable、environment、job、evaluation 生命周期；
- Signature：原文在 `response` artifact；`env_state` 存路径、hash 和统计；
- CoT：标准化 thinking 可进入 `messages.reasoning`，但没有 Origin-CoT 派生物；
- 版本能力：当前 artifact 为单 request 原子文件，没有 MetaBot epoch/version envelope。

### MetaBot rollout 轨迹

- 主索引：`session + epoch + turn`；
- 主存储：一组 JSON 文件；
- 原始证据：Harness Raw 与 Backend Raw 分离；
- 强项：协议边界、成本、turn、版本快照、conversation 投影；
- Signature：位于 Backend Raw Anthropic thinking block；
- CoT：rollout 时保存 Provider 返回的 thinking，不代表 Origin-CoT；
- 版本能力：支持 epoch 和同名文件 `.N` 历史版本。

### Origin-CoT 后轨迹

- 主索引：原 task/session，加 Signature digest 和 occurrence path；
- 主存储：独立派生输出目录中的 JSON；
- 原始证据：保留 rollout Backend Raw 的结构和 Signature；
- 强项：按 Signature 去重提取、checkpoint、provenance、等价性验证；
- Signature：保留原文，状态文件只使用 digest；
- CoT：已签名 thinking 被提取出的 Origin-CoT 替换；
- 版本能力：通过 input/output hash 和 completed artifact hash 检测漂移。

## 6. 字段映射关系

SAfactory Provider Artifact 与 MetaBot Backend Raw 的概念映射：

```text
SAfactory artifact.session_id
  -> MetaBot session_id

SAfactory artifact.llm_step_index
  -> MetaBot turns[].turn_number

SAfactory artifact.request
  -> MetaBot turns[].raw_request

SAfactory artifact.response
  -> MetaBot turns[].raw_response

SAfactory artifact.stream_text
  -> MetaBot Backend Raw 采集过程中的原始 SSE 证据

SAfactory artifact.response.content[].signature
  -> MetaBot Backend Raw thinking.signature

SAfactory session_steps.messages
  -> MetaBot conversation_epoch<N>.json 的概念对应物

SAfactory reward / terminal / trainable / evaluation
  -> MetaBot rollout 与 Origin-CoT 文件中没有直接对应字段
```

这只是语义映射，不表示文件可以直接互换。

## 7. SAfactory 接入 MetaBot `get_origin_cot` 的差距

当前 SAfactory Artifact 已经保存提取所需的 Signature。两种使用方式不同：

### 只提取 Signature 对应的 CoT

可以直接扫描：

```text
provider-traces/**/*.json
  -> signatures[].signature
  -> 调用 /v1/extract-claude-cot
```

这种方式不需要转换成 MetaBot Backend Raw，但只会得到 Signature 到文本的映射，
没有完整 conversation 重建和等价性证明。

### 使用完整 `get_origin_cot` pipeline

需要新增 exporter：

1. 按 SAfactory `session_id` 聚合所有 request artifact；
2. 按 `llm_step_index` 排序并构造 `turns[]`；
3. 输出 `trace_boundary: "litellm_backend"` envelope；
4. 将每个 `request` / `response` 映射为 `raw_request` / `raw_response`；
5. 从 SAfactory `messages` 生成符合 MetaBot validator 的
   `conversation_epoch0.json` baseline；
6. 保留 Signature occurrence path，处理 retry、失败 step 和重复 request；
7. 通过 MetaBot projector 与 equivalence validator 后再执行 Origin-CoT。

因此当前状态是：

```text
Signature 数据：已具备
单 Signature 提取：可直接实现
完整 MetaBot Origin-CoT pipeline 输入：尚需 exporter
```

## 8. 选择建议

- 只研究 Signature 能否还原 CoT：读取 `session_steps.response` artifact 即可；
- 训练 SAfactory RL：继续以 `session_steps` 为主，避免改成 MetaBot 文件布局；
- 需要可审计的 Origin-CoT 训练集：实现 SAfactory → MetaBot Backend Raw exporter，
  再复用完整 `get_origin_cot`；
- 需要长期证据链：校验 `response` 内容与 `env_state.provider_trace.artifact_sha256`，
  并对 DB 和外部审计副本统一设置访问控制与 retention；
- 完整 request、SSE 和 Signature 原文位于 `response`，不重复写入
  `session_steps.env_state`。

## 9. 代码与样例位置

SAfactory：

```text
gateway/storage.py
gateway/provider_trace.py
gateway/telemetry.py
rl/examples/patcheval/run_eval.sh
rl/examples/patcheval/patcheval_claude-opus-4-6-thinking_claudecode-exp1_anthropic-compat-smoke-20260729-194826.db
```

MetaBot：

```text
claude_utils/litellm_proxy/trace_capture/
tools/get_origin_cot/pipeline.py
tools/get_origin_cot/projector.py
tools/get_origin_cot/validator.py
tools/get_origin_cot/checkpoint.py
tests/get_origin_cot/fixtures/workers/shard_00/trace/task-alpha/
```
