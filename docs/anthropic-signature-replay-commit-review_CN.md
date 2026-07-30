# `feat/claude-trace` 分支说明

## 目标

该分支解决 Claude Code 经过 OpenAI 协议转换后，Anthropic 原始 `thinking`、
`signature` 和 SSE 数据丢失的问题。目标是让 Claude Code 直接连接 SAfactory
Gateway，并完整保存可审计的 provider-native 轨迹，不新增数据库字段。

## 三个问题与对应改动

### 1. Gateway 不支持 Anthropic 协议

原 Gateway 只支持 OpenAI 风格接口。Claude Code 必须经过 `claude_adapter` 做
Anthropic → OpenAI → Anthropic 转换，原始 thinking、Signature 和 SSE 会在转换中
丢失。为此，`gateway/app.py` 和 `inference_forwarder.py` 新增原生
`/v1/messages`、`count_tokens` 和 SSE 转发；`claudecode_runner.py` 改为让
Claude Code 直接连接 Gateway，不再以 Adapter 作为主链路。

### 2. 为什么 Provider Artifact 要保存到 `response`

SAfactory 原有轨迹是统一格式，不保存 provider 原始 content block 顺序、完整
request/response、SSE 和 Signature 的对应关系，只额外保存一个 Signature 无法完整
审计。因此新增 `provider_trace.py` 构造完整 Provider Artifact，并直接写入现有
`session_steps.response`。不再生成外部 JSON 审计副本，也不再写入
`env_state.provider_trace`，无需修改数据库 schema。

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

6 个 PatchEval task 全部成功且官方评分均为 10；57 个 Provider Artifact 已存入
数据库；Gateway 成功为 51 个后续请求恢复 Signature。上游仍只在每个任务的首轮响应
产生 Signature，说明“历史 Signature 回传”已解决，但上游暂未持续生成新 Signature。

## 需要核对

分支中的 `env/geo3k/wheels/*.whl` 与 Claude Trace 功能无直接关系，建议提交 PR 前
确认是否移除。MetaBot 的 DB 导出脚本属于另一个仓库，不包含在本分支中。
