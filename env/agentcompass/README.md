# AgentCompass 环境

这个 adapter 用于在 SAfactory 中运行 AgentCompass benchmark。每行 JSONL 选择一个 benchmark、harness、environment 和 sample，并作为一个独立 episode 运行。

## 数据格式

`sample_id` 必须是对应 benchmark 中真实存在的样本 ID。数据位置和各组件参数放入对应的参数对象。

```json
{
  "task_id": "unique-task-id",
  "benchmark": "benchmark-id",
  "harness": "harness-id",
  "environment": "host_process",
  "sample_id": "exact-sample-id",
  "benchmark_params": {},
  "harness_params": {},
  "environment_params": {},
  "model_params": {},
  "timeout_seconds": 1800
}
```

## 已验证示例

下表只是已经验证过的示例，不是 AgentCompass 的完整支持列表。

| Benchmark | Harness | Environment | Sample |
| --- | --- | --- | --- |
| `special_pattern_check` | `openai_chat` | `host_process` | `empty_content_gate4_0` |
| `scicode` | `scicode_tool_use` | `host_process` | `10`, `78` |
| `sgi_deep_research` | `mini_swe_agent` | `host_process` | `SGI_DeepResearch_0000` |

AgentCompass 中已注册组件并不表示任意组合都能直接运行；镜像依赖、数据、组件兼容性和评测依赖仍需准备并验证。遇到新的评分 schema 时，需要同步补充 `runner.py`、`rule_evaluator.py` 和相关测试。

## Runtime image

```text
registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches:agentcompass-02
```

该镜像为 Linux/AMD64，使用 Python 3.12，并固定 AgentCompass commit：

```text
d2c3e148902e948db3270fa34b2198fb1b10beb7
```

默认只允许 `host_process`。如需启用其他 environment，必须通过 `AGENTCOMPASS_ALLOWED_ENVIRONMENTS` 显式配置。

主模型通过当前 episode 的 Gateway session 路由。需要 judge 的 benchmark 必须为 judge 配置独立的 non-session Gateway `/v1` 地址，不能复用主模型的 session URL。

## 相关文档

- [配置说明](../../docs/reference/configuration_CN.md)
- [自定义环境](../../docs/guides/custom-environment_CN.md)
- [评测说明](../../docs/guides/evaluation_CN.md)
- [RJob 模式](../../docs/internal/rjob-mode_CN.md)
