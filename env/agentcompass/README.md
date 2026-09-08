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

## 已验证结果契约

下列 ID 均直接来自固定 AgentCompass revision 的 registry 和 benchmark/scorer 源码；该 revision
没有为这些 ID 注册 alias。名称相近不代表共享结果 schema。

| Benchmark ID | Source-proven result family | SAfactory normalization |
| --- | --- | --- |
| `browsecomp` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `deepsearchqa` | `DeepSearchQAScorer` | native boolean `correct` → 10/0 |
| `frontierscience` | olympiad judge or research rubric | olympiad 10/0; research native 0–10 total |
| `hle` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `hle_verified` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `scicode` | SciCode evaluation | native subproblem fraction 0–1 → 0–10 |
| `sealqa` | official SEALQA A/B/C scorer | native boolean `correct` → 10/0 |
| `sgi_deep_research` | `LLMJudgeScorer` | native boolean `correct` → 10/0 |
| `special_pattern_check` | analyzer-derived boolean `correct` | 10/0 |
| `swebench_verified` | evaluator `completed`/`resolved` | native `resolved` → 10/0 |

所有 detail 必须包含固定 revision 的顶层 `task_id` 和恰好一个 attempt。只有 `completed`、空
`error` 且 benchmark-specific 字段一致的结果才能评分。未知 ID、alias、schema、冲突字段和
`run_error`/`eval_error`/`skipped` 均 fail closed。runner 只会保留源码明确提供且不改变语义的字段，
不会从 `correct`、grade、label 或 summary 推导 verdict/reason，也不会过滤正常答案文本。

AgentCompass results/detail/log 路径只用于容器内诊断，不进入公开 stdout 或 evaluator 输出。
当前共享接口不能由 env-only 代码写入 WT `ground_truth_answer` 或 meta_json 中 eval 下的
context；因此 evaluator 不把 ground truth、model answer 或 judge 语义字段放入 EvalResult 的
artifacts 冒充 context。真实 WT 契约验证必须等待共享层支持。

RJob 的结果回传固定使用 `/app/results/<job_id>/<session_id>/safactory_result.json`。父 launcher
与子 RJob 必须挂载同一个持久 storage source：父 launcher 将其挂载到工作区的 `./results`，子
RJob 的 target 固定为 `/app/results`。真实 source URI 由部署方在 private/global `--rjob-config`
的 `mount_config` 中提供，公共 per-agent start config 不定义 `mount_config`，也不保存真实 GPFS URI。
RJob runner 会在启动 AgentCompass benchmark 前验证结果路径和 `/app/results` 挂载点，缺失时非零退出。

公开 diagnostic 仅验证 `special_pattern_check` + `openai_chat` + `host_process` 的框架链路，
不证明其他 benchmark 的数据、镜像依赖、harness 组合或 judge 配置已经可 rollout。

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
