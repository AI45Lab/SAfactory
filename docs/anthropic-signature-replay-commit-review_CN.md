# `feat/claude-trace` 分支说明

## 目标

该分支解决 Claude Code 经过 OpenAI 协议转换后，Anthropic 原始 `thinking`、
`signature` 和 SSE 数据丢失的问题。目标是让 Claude Code 直接连接 SAfactory
Gateway，并将实际 Provider request/response 与标准训练轨迹分开保存。

## 三个问题与对应改动

### 1. Gateway 不支持 Anthropic 协议

原 Gateway 只支持 OpenAI 风格接口。Claude Code 必须经过 `claude_adapter` 做
Anthropic → OpenAI → Anthropic 转换，原始 thinking、Signature 和 SSE 会在转换中
丢失。为此，`gateway/app.py` 和 `inference_forwarder.py` 新增原生
`/v1/messages`、`count_tokens` 和 SSE 转发；`claudecode_runner.py` 改为让
Claude Code 直接连接 Gateway，不再以 Adapter 作为主链路。

### 2. 如何保存 Provider request/response

SAfactory 原有轨迹是统一格式，不保存 provider 原始 content block 顺序、完整
request/response 和 Signature 的对应关系。现在为 `session_steps` 新增 `request`
字段，保存 Gateway 转换后实际发往 Provider 的 JSON body；`response` 直接保存
Provider 响应，Anthropic SSE 聚合后仍保留 `thinking + signature` content blocks。
不再构造 Provider Artifact，也不保存原始 SSE 或外部 JSON 审计副本。

### 3. 如何回传 Signature，避免多轮链路中断

Gateway 虽然会把上游 Signature 返回给 Claude Code，但 Claude Code 构造下一轮请求时
可能删除历史 assistant 消息中的 `thinking + signature`。新增
`anthropic_thinking_history.py`，按 session 缓存上游返回的完整 signed-thinking
block；下一次请求到达时，根据公开 text/tool blocks 精确匹配对应历史消息，并把原始
block 恢复后再发给上游。无法唯一匹配时跳过，避免串错 Signature。

这里保证的是“已有 Signature 在后续请求中不断链”，不能保证上游每一步都生成新的
Signature。当前实测 51 个后续请求全部成功携带历史 Signature，但上游仍只在每个任务
的首轮响应签发新 Signature。

## 验证结果

6 个 PatchEval task 全部成功且官方评分均为 10；旧版验证库包含 57 个 Provider
Artifact；Gateway 成功为 51 个后续请求恢复 Signature。上游仍只在每个任务的首轮响应
产生 Signature，说明“历史 Signature 回传”已解决，但上游暂未持续生成新 Signature。
