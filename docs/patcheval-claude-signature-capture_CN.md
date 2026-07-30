# SAfactory 接入 Claude Code Signature 保存能力

## 1. 当前确认目标

当前目标不是在现有 OpenAI 转换链路末端补存一个字段，而是：

1. 让 **SAfactory Gateway 原生接收和转发 Anthropic Messages API/SSE**；
2. Claude Code 直接连接 SAfactory Gateway，不再依赖 `claude_adapter` 做 Anthropic → OpenAI → Anthropic 转换；
3. 在 Gateway 观察到的原始 Anthropic 响应中保存 `thinking.signature` 和流式 `signature_delta`；
4. 使用现有 `session_id`、`request_id`、`llm_step_index` 和 PatchEval 环境关联轨迹；
5. 完整 Provider Raw Trace 保存到现有 `session_steps.response`，并保留独立 artifact 审计副本；
6. **不新增数据库表或字段**，现有 `session_steps.env_state` 只保存 artifact 引用、hash 和采集状态；
7. 默认 trajectory、SFT 和 GRPO 导出不携带 Signature。

目标链路为：

```text
Claude Code
  -> SAfactory Gateway /v1/sessions/<session_id>/v1/messages
  -> Anthropic-compatible 上游 /v1/messages
  -> SAfactory Gateway（SSE 原样返回，同时旁路采集）
  -> Claude Code

Provider Raw Trace artifact
  <- Gateway 原始 request/response/SSE

session_steps.response
  <- 完整 Provider Raw Trace artifact JSON

session_steps.env_state.provider_trace
  <- artifact 路径、SHA-256、Signature 数量和完整性状态
```

Signature 是服务端生成的不透明数据，不应尝试在在线评测链路中解密、修改或伪造。

### 当前实现状态

已完成第一版代码：

- Gateway session 路由支持 `/v1/messages` 和 `/v1/messages/count_tokens`；
- Anthropic 非流式 JSON 与 SSE 流式响应直接转发；
- SSE 中的 thinking、Signature、文本和工具输入可聚合为 telemetry response；
- Provider Raw Trace 保存到 `session_steps.response`，并使用原子文件保存审计副本；
- `session_steps.env_state.provider_trace` 保存 artifact 引用，不修改 SQL schema；
- Claude Code runner 直接使用 Gateway，`run_eval.sh` 不再启动 Adapter；
- `DISABLE_INTERLEAVED_THINKING` 改为显式可选开关；
- 测试覆盖 SSE 原样返回、Signature 聚合、artifact 写入和数据库元数据引用；
- 已使用真实 Anthropic-compatible 中转站完成单 CVE smoke test：6 个 LLM step
  全部生成完整 Raw Trace，其中 1 个 step 捕获到 772 字节 Signature，官方规则
  评估得分为 10；
- 对不支持 adaptive thinking/context-management 的中转站，Gateway 提供显式
  `fixed_thinking` 兼容模式；默认 `native` 模式仍保持请求原样。

## 2. 关键结论

仅修改 SAfactory 的 SQLite 字段是不够的。

当前链路为：

```text
Claude Code
  -> env/patcheval/claude_adapter（Anthropic Messages -> OpenAI Chat）
  -> SAfactory Gateway
  -> OpenAI-compatible 上游
  -> Gateway
  -> claude_adapter（OpenAI Chat -> Anthropic SSE）
  -> Claude Code
```

现有 `claude_adapter` 会：

- 把 Claude Code 的 Anthropic 请求转成 OpenAI Chat；
- 强制上游使用 `stream: false`；
- 从 OpenAI 响应重新合成 Anthropic SSE；
- 只转换文本和工具调用。

此外，`claudecode_runner.py` 当前显式设置：

```text
DISABLE_INTERLEAVED_THINKING=true
```

这会在 Claude Code 客户端侧关闭交错 thinking。Signature 采集实验需要先删除该设置，或将其改成默认关闭、可通过环境变量启用的兼容开关。

因此以下数据会在进入 SAfactory 数据库之前丢失：

- Anthropic `thinking` block；
- `thinking.signature`；
- 流式 `signature_delta`；
- Backend 原始 Anthropic request/response。

如果 OpenAI-compatible 上游本身已经删除 Signature，SAfactory 后续任何存储逻辑都无法恢复它。实施前必须先确认上游原始响应确实包含 `signature` 或 `signature_delta`。

## 3. MetaBot 当前做法

MetaBot 不只保存转换后的统一对话，而是把不同边界的数据分开保存：

- `conversation_epoch<N>.json`：统一后的分析轨迹；
- `proxy_traces_raw_epoch<N>.json`：Agent/Harness 看到的原始协议；
- `proxy_traces_backend_raw_epoch<N>.json`：LiteLLM 与模型服务之间的 Backend 原始协议。

Claude Signature 位于 Backend Raw Trace 的 Anthropic content block 中：

```json
{
  "type": "thinking",
  "thinking": "...",
  "signature": "..."
}
```

当链路经过 OpenAI Responses 协议桥时，同一类数据可能使用：

```json
{
  "type": "reasoning",
  "encrypted_content": "..."
}
```

因此采集器需要同时识别 Anthropic `signature` 和 Responses
`encrypted_content`，并统一索引为 `generation_signature`；原始 artifact 中仍保留
Provider 的字段名，不能覆盖或改写。

流式响应中则表现为：

```json
{
  "type": "signature_delta",
  "signature": "..."
}
```

相关实现：

- `metabot/claude_utils/litellm_proxy/session_accumulator.py`
  - 按 session/epoch 累积请求和响应；
  - 分别生成 Harness Raw、Backend Raw 和 Analysis Trace。
- `metabot/claude_utils/litellm_proxy/trace_capture/`
  - adapter、projector、validator、writer 等采集组件。
- `metabot/tools/get_origin_cot/extract_query_traces.py`
  - 从 `proxy_traces_backend_raw_epoch0.json*` 递归提取所有非空 `signature`。

SAfactory 应复用“Backend Raw 是事实源、统一轨迹只是投影”的边界，而不是把 Signature 塞进普通 assistant 文本。

## 4. 可能实现

### 4.1 当前采用方案：Gateway 原生 Anthropic MVP

第一版实现以下能力：

```text
Claude Code
  -> SAfactory Gateway（Anthropic Messages/SSE）
  -> Anthropic-compatible 上游

Gateway Raw Trace Writer  构建 Backend Raw Trace 和 Signature，并保存审计副本
SAfactory SQLite          response 保存完整 artifact，env_state 保存引用和摘要
```

MVP 范围：

1. 新增 session-scoped `/v1/messages`；
2. 新增 `/v1/messages/count_tokens`；
3. 转发 `x-api-key`、`anthropic-version`、`anthropic-beta` 和必要请求头；
4. Anthropic request/response JSON 原样转发；
5. SSE chunk 原样返回，不把流式响应重建成 OpenAI Chat；
6. 旁路识别 `thinking`、`signature`、`thinking_delta` 和 `signature_delta`；
7. 将完整 Raw Trace JSON 写入 `session_steps.response`，并按 session/request 原子写入审计副本；
8. 在 `session_steps.env_state.provider_trace` 保存 artifact 引用；
9. 调整 Claude Code runner，使 `ANTHROPIC_BASE_URL` 直接指向 Gateway；
10. 将 `DISABLE_INTERLEAVED_THINKING` 改为可配置。

该方案不要求复制 MetaBot 的全部组件，也不要求新增数据库表。

### 4.2 后续完整方案：MetaBot 级轨迹能力

MVP 稳定后，可以按需求增加：

- Harness Raw 与 Backend Raw 分离；
- retry/fallback 的每次 Backend attempt 记录；
- epoch 和上下文 compact 检测；
- versioned artifact；
- 严格的 correlation、完整性验证和恢复机制；
- 跨进程 writer、容量治理和 retention；
- OpenAI Responses `encrypted_content` 的统一采集。

这些属于 MetaBot 级证据系统，不是当前 PatchEval Signature 保存 MVP 的前置条件。

## 5. SAfactory 需要修改的文件

### 5.1 `env/patcheval/claudecode_runner.py`

- 将 `DISABLE_INTERLEAVED_THINKING=true` 改成可配置项；
- Signature 采集运行默认允许 interleaved thinking；
- 非采集运行可继续通过兼容开关关闭；
- runner 只负责开关和启动 Claude Code，不把 Signature 打印到 stdout。

### 5.2 `gateway/app.py`

- 新增 `POST /v1/sessions/{session_id}/v1/messages`；
- 新增 `POST /v1/sessions/{session_id}/v1/messages/count_tokens`；
- 复用现有 session resolution、并发控制、step 计数和 telemetry 生命周期；
- 非流式响应保留 Anthropic JSON；
- 流式响应透传 Anthropic SSE，并在旁路累计完整性信息；
- 客户端取消、上游取消和 partial SSE 必须反映到 capture 状态；
- 不把 Signature 打印到访问日志或异常文本。

### 5.3 `gateway/inference_forwarder.py`

- 增加 `forward_anthropic_messages()`；
- 增加 `open_anthropic_stream()`；
- 上游 endpoint 使用 `/v1/messages`；
- 转发 Anthropic 必需的版本、beta 和鉴权头；
- 建议增加 `X-Safactory-Session-Id`、`X-Safactory-Request-Id` 和
  `X-Safactory-Step-Index`；
- 对 raw body/SSE 只做旁路采集，不改变返回给 Claude Code 的数据。

### 5.4 新增 Gateway Anthropic Raw Trace Writer

建议放在独立模块，例如：

```text
gateway/provider_trace.py
```

职责包括：

- 记录原始 request、response 或 ordered SSE events；
- 从 Anthropic block 和 SSE delta 中建立 Signature 索引；
- 使用临时文件 + rename 原子发布；
- 计算 artifact SHA-256；
- 返回 `capture_complete`、错误类型和统计信息；
- 写入失败时不阻断模型调用，但禁止把不完整 artifact 标成成功。

### 5.5 `gateway/models.py` 和 `gateway/telemetry.py`

完整 Provider Raw Artifact 序列化到现有 `response` 字符串；telemetry 另保留以下摘要：

```json
{
  "provider_trace": {
    "schema_version": 1,
    "boundary": "backend_raw",
    "relative_path": "signatures/<session_id>/<request_id>.json",
    "sha256": "...",
    "signature_count": 1,
    "signature_total_bytes": 2596,
    "capture_complete": true
  }
}
```

如果暂时不希望迁移 `session_steps` 表，可先放进 `env_state` 的独立命名空间：

```text
env_state.provider_trace
```

当前目标明确不新增表或字段。完整 Blob 进入 `session_steps.response` 和外部
Raw Trace 审计副本；`env_state` 只保存引用和摘要，避免同一行重复存储。

### 5.6 `gateway/storage.py` 及存储后端

不修改 `SessionStep` schema：

- `session_steps.messages` 继续保存标准化轨迹；
- `session_steps.response` 保存完整 Provider Raw Artifact JSON；
- `session_steps.env_state.provider_trace` 保存 artifact 路径、hash、Signature
  数量和完整性状态；
- 完整 Signature 位于 `response` 和受控 artifact 审计副本中；
- cloud storage 继续透传现有 JSON 字段，不需要增加字段映射。

只有未来出现跨 session SQL 检索、独立权限和 retention 的明确需求时，才重新评估独立表。

### 5.7 `rl/examples/patcheval/run_eval.sh`

增加可选配置：

```bash
PATCH_EVAL_SIGNATURE_CAPTURE=off|metadata|full
PATCH_EVAL_PROVIDER_TRACE_DIR=/shared/path/patcheval-provider-traces
PATCH_EVAL_SIGNATURE_CAPTURE_FAIL_POLICY=fail_open
```

默认建议为 `off`。研究运行显式设置为 `full`。

Claude Code 的 `ANTHROPIC_BASE_URL` 应设置为：

```text
http://<gateway-host>:<gateway-port>/v1/sessions/<session_id>
```

Claude Code 会在该 base URL 后请求 `/v1/messages`。

### 5.8 `env/patcheval/generate_full_config.py`

- 把 capture 模式和 trace 路径写入 agent 环境；
- 为每个环境绑定唯一的 SAfactory session；
- 确保远程 Docker 与 Gateway 都能访问共享 artifact 目录；
- 不再把 Claude Code 指向独立 `claude_adapter` 服务；
- 不把 Signature 写入任务 JSONL 或 prompt。

### 5.9 `env/patcheval/claude_adapter/`

当前目标实现后，`claude_adapter` 不再位于 Claude Code 主调用链路：

- 第一阶段保留代码和旧配置兼容性；
- 新配置默认直连 Gateway Anthropic endpoint；
- 完成回归验证后再决定是否删除 adapter；
- 不需要继续扩展其 OpenAI ↔ Anthropic Signature 转换逻辑。

## 6. Provider Raw Artifact 格式

推荐保存完整 Backend Raw Trace，而不是只保存一个 Signature 字符串：

```json
{
  "schema_version": 1,
  "boundary": "backend_raw",
  "session_id": "safactory-session-id",
  "request_id": "gateway-request-id",
  "llm_step_index": 3,
  "model": "claude-opus-4-6-thinking",
  "request": {},
  "response": {},
  "signatures": [
    {
      "content_block_index": 0,
      "signature": "...",
      "sha256": "...",
      "byte_length": 2596
    }
  ],
  "capture": {
    "complete": true,
    "streamed": true,
    "captured_at": "ISO-8601"
  }
}
```

该对象完整序列化到 `session_steps.response`；外部文件保存相同字节内容作为审计
副本。完整原始 response 用于证明 Signature 的来源和上下文；`signatures` 是便于
查询的索引。

如果担心重复存储，可只在 raw response 中保存 Blob，在索引中保存 JSONPath、hash 和长度。

## 7. 安全与数据治理

Signature 应按敏感模型数据处理：

- 文件权限至少为 `0600`；
- SQLite DB、备份和副本按与 artifact 文件相同的敏感级别保护；
- 不打印到 Gateway、Adapter、测试和异常日志；
- 不进入默认 trajectory/SFT/GRPO 导出；
- 用 SHA-256 去重和做完整性验证；
- 对 artifact 目录设置保留周期；
- 导出 Signature 必须使用显式参数；
- 请求失败时不把完整 response 拼进异常文本；
- `signature` 必须加入 SAfactory 的敏感字段策略。

采集故障建议采用：

```text
模型数据面 fail-open，证据完整性 fail-closed
```

也就是采集写盘失败不应让 PatchEval 模型调用失败，但该条记录必须标记 `capture_complete=false`，不能伪装成完整数据。

## 8. 测试要求

至少新增以下测试：

1. 非流式 Anthropic response 能保存 `thinking.signature`；
2. 流式 `signature_delta` 按顺序聚合且字节完全一致；
3. 同一 session 多轮调用能够关联不同 request/step；
4. 并发 session 不会交叉写入；
5. artifact 原子写入，中断后不会产生看似完整的文件；
6. Signature 不出现在普通日志和默认训练导出中；
7. 上游没有 Signature 时记录 `signature_count=0`，不伪造数据；
8. writer 失败时模型请求仍可完成，但 `capture_complete=false`；
9. SQLite `response`、记录的 SHA-256 与 artifact 文件一致；
10. 真实 Claude Code smoke test 能从 Backend Raw Trace 中递归找到 Signature。

## 9. 实施顺序

1. **能力预检**：直连当前 Anthropic-compatible 中转站，确认 `/v1/messages`
   响应或 SSE 中存在 `signature`/`signature_delta`；
2. **Gateway API**：实现 session-scoped `/v1/messages` 和 `/count_tokens`；
3. **原生转发**：实现 Anthropic JSON/SSE 透明转发、取消和错误处理；
4. **Raw Capture**：保存原始 request/response/SSE 和 Signature 索引；
5. **现有 DB 保存**：完整 artifact 写入 `session_steps.response`，摘要写入
   `env_state.provider_trace`，不新增表或字段；
6. **Runner 切换**：Claude Code 的 `ANTHROPIC_BASE_URL` 从 adapter 改为 Gateway；
7. **Thinking 开关**：将 `DISABLE_INTERLEAVED_THINKING` 改为可配置；
8. **导出工具**：按 session/CVE 从 `session_steps.response` 导出 MetaBot
   Backend Raw 与 Signature；
9. **测试和安全**：完成并发、流式、中断、原子写入和默认导出隔离测试；
10. **兼容清理**：稳定后再决定是否删除 `claude_adapter`。

## 10. 工作量估计

- 当前 Anthropic MVP、真实上游 smoke test 和运行数据核验已完成；
- 加入 MetaBot 级 retry/attempt、epoch、版本化 artifact 和严格证据校验：
  总体约 **1～2 周或更久**；
- 当前方案不新增数据库表，因此不包含 schema migration 和 cloud 字段映射工作。

当前验证的中转站可返回原生 Anthropic SSE 和 Signature，但需要启用
`fixed_thinking` 兼容模式；其他上游仍应先用最小请求确认协议能力。
