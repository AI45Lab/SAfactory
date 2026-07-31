# PatchEval Environment

PatchEval 现在统一为 SAfactory 调用链：

- **正式评测**：SAfactory Launcher、Runner 和 Gateway 负责生成与轨迹记录；
  Launcher 按约定自动发现 `rule_evaluator.py`，每个 CVE 都由官方
  `evaluation/run_evaluation.py:Evaluation` 在运行资源释放前评分。
原版 PatchEval 提供两类 baseline：

- **LLM baseline**：将漏洞知识和相关代码包装进 prompt，由 LLM 直接生成补丁；
  LLM 不能使用仓库浏览或编辑工具。
- **Agent baseline**：SWE-agent、OpenHands 或 Claude Code 等 agent 在容器仓库
  中运行，可以在有限工具调用次数内搜索、读取和修改完整 codebase。

所有配置均由 `generate_full_config.py` 动态生成。LLM baseline 使用
`strict_runner.py`；当前已接入的 Claude Code Exp1 使用
`claudecode_runner.py`。
`run_eval.sh` 先将官方 helper 同步到远程 Docker 可见的
`patcheval-runtime` 共享目录，再只读挂载到容器的 `/opt/patcheval`。Runner
不再复制实现 `LLMClient.build_prompt`、`PatchParser`、`FuncReplacer`、
`CodeApplier` 和 `FeedbackHelper` 的逻辑。

## 1. Environment Data

运行时生成的 `patcheval_config.yaml` 为每个任务指定 Docker 镜像和 JSONL：

```yaml
- env_name: patcheval_<name>
  env_image: <docker-image>
  env_num: 1
  dataset: ./datasets/<task>.jsonl
```

严格模式 JSONL 每行包含：

```json
{
  "cve_id": "CVE-YYYY-NNNN",
  "work_dir": "/workspace/<repository>",
  "setting": "s1.1",
  "prompt_template": "<official template>",
  "official_record": {"cve_id": "...", "vul_func": []}
}
```

任务元数据由官方 `datasets/input.json` 提供；SWE-Agent `dataset.jsonl` 只
用于补充镜像名和容器内仓库路径。

### 能否直接使用原版 PatchEval 数据

`generate_full_config.py` 按 CVE ID 合并官方 `input.json` 与 Agent
`dataset.jsonl`，不会使用 `problem_statement` 重新构造 prompt。

## 2. Environment–LLM Interaction

### Environment 输入

Runner 接收 `SimulationStartRequest` JSON，包含 session、任务数据、Gateway
地址、模型名、temperature 和 timeout。

### LLM 输入

Runner 用官方 `LLMClient.build_prompt` 和对应 S1.x 模板生成 user message，
并发送官方 system message `You are a helpful assistant`，固定
`temperature=0`、`max_tokens=16384`。请求 URL 为当前 SAfactory session
的 `/v1/sessions/<session_id>/chat/completions`。

### LLM 输出

LLM 输出官方函数级 JSON：`[{"id": "vul_*", "patch": "..."}]`。Runner
直接调用官方 `PatchParser`、`FuncReplacer` 和 `CodeApplier` 解析输出、
按行范围替换函数并生成 unified diff。

### Environment 输出

Runner 返回补丁和生成阶段 metrics。启用 `--enable-evaluation` 后，
`rule_evaluator.py` 调用官方 evaluator：PoC 与单测均通过时
`raw_score=1`、SAfactory 标准化 reward 为 `10`，否则均为 `0`。该结果直接
写入当前 trajectory，不再导出补丁或执行批量回写。

## 3. How the Environment Validates a Patch

Runner 将官方函数级输出转换为 unified diff，并写入：

```text
/workspace/fix.patch
```

随后环境执行以下步骤。

### 3.1 验证漏洞是否修复

```bash
bash /workspace/fix-run.sh
```

`fix-run.sh` 由每个 PatchEval Docker 镜像提供。它会应用候选补丁和安全测试，
然后运行该 CVE 对应的 PoC 回归测试。例如 Gogs 的验证逻辑是：

```bash
cd /workspace/gogs
git apply /workspace/test.patch /workspace/fix.patch
go test -run Test_isRepositoryGitPath
```

- 返回码非 0：安全测试失败。
- 返回码为 0：漏洞攻击已被阻止，继续运行普通单元测试。

因此 `poc_passed=true` 表示安全验证通过，不表示攻击成功。

### 3.2 检查原有功能

如果镜像存在 `/workspace/unit_test.sh`，环境继续执行：

```bash
bash /workspace/unit_test.sh
```

- 安全测试通过、单元测试失败：strict success 为 `false`。
- 安全测试和单元测试都通过：strict success 为 `true`。
- 没有 `unit_test.sh`：安全测试通过后 strict success 为 `true`。

Gateway 保存 prompt、模型回答、token 和延迟；Evaluator 只提交官方二值
strict-success reward，不再提交 1/7/10 阶段 reward。

## 4. Running

### Claude Code Agent baseline（Exp1）

Claude Code Exp1 使用官方 `exp_agent/claudecode/dataset.jsonl` 和
`templates/default.md`，包含漏洞知识和位置，不向 Agent 提供 PoC 或单元测试
反馈。Agent 最多执行 100 次工具调用，并在任务容器内浏览和修改完整仓库。

`run_eval.sh` 使用 Gateway 原生 Anthropic Messages/SSE 接口。任务容器中的
Claude Code 直接请求
`/v1/sessions/<session_id>/v1/messages`，不再启动 Claude Adapter，也不再经过
Anthropic → OpenAI → Anthropic 转换。Gateway 转发原生流式事件，把实际发往
Provider 的 JSON body 写入 `session_steps.request`，并把聚合后的 Anthropic
响应写入 `session_steps.response`。不再构造或保存 Provider Artifact。

先运行一个样本：

```bash
export PATCH_EVAL_API_KEY="<API token>"
export DOCKER_HOST="tcp://<docker-host>:2376"
export PATCH_EVAL_MODEL="claude-opus-4-8"
export PATCH_EVAL_BASELINE="claudecode"
export PATCH_EVAL_AGENT_EXPERIMENT="exp1"
export PATCH_EVAL_TASK_LIMIT=1
export PATCH_EVAL_POOL_SIZE=1

./rl/examples/patcheval/run_eval.sh
```

`PATCH_EVAL_MODEL` 是底层模型路由，不是 Agent 名称。Claude Code baseline
要求显式设置它，避免误用 LLM baseline 默认的 DeepSeek 模型。

原生 Anthropic 或完整兼容的上游保持默认
`PATCH_EVAL_ANTHROPIC_COMPATIBILITY=native`。如果中转站只接受固定 thinking
budget，不接受 Claude Code 新版发送的 adaptive thinking 和 context-management
字段，可增加：

```bash
export PATCH_EVAL_ANTHROPIC_COMPATIBILITY="fixed_thinking"
export PATCH_EVAL_ANTHROPIC_THINKING_BUDGET_TOKENS=1024
export PATCH_EVAL_ANTHROPIC_MAX_TOKENS=8192
```

若上游支持 adaptive thinking，但不接受 Claude Code 的 `context_management`
字段，可使用 MetaBot 对齐模式：

```bash
export PATCH_EVAL_ANTHROPIC_COMPATIBILITY="adaptive_thinking"
export PATCH_EVAL_ANTHROPIC_INTERLEAVED_THINKING=true
export PATCH_EVAL_ANTHROPIC_MAX_TOKENS=64000
```

该模式将请求对齐为 `thinking.type=adaptive` 和
`output_config.effort=max`，删除 `context_management`，并只向上游发送实验
所需的 `interleaved-thinking` Beta Header，避免 Bedrock 拒绝 Claude Code
的内部 Beta 标志。

`fixed_thinking` 模式只在 Gateway 发往 Provider 的边界把 thinking 改为 `enabled`，删除
`context_management`/`output_config`，并限制 `max_tokens`；Claude Code 与
Gateway 之间仍使用原生 Anthropic Messages/SSE。

首次启动每个 CVE 容器时会安装 Node.js 和 `@anthropic-ai/claude-code`，因此
Agent baseline 的启动时间和网络开销明显高于 LLM baseline。确认单样本运行
正常后，将 `PATCH_EVAL_TASK_LIMIT` 改为 `0` 再运行全量。

### LLM baseline（S1.x）

一键启动 Gateway、生成配置并运行标准 Launcher：

```bash
export PATCH_EVAL_API_KEY="<API token>"
export DOCKER_HOST="tcp://<docker-host>:2376"
export PATCH_EVAL_MODEL="bailian/deepseek-v4-flash"
export PATCH_EVAL_BASELINE="llm"
export PATCH_EVAL_SETTING="s1.1"
export PATCH_EVAL_TASK_LIMIT=1  # smoke test；全量改为 0

./rl/examples/patcheval/run_eval.sh
```

脚本只负责进程编排，最终执行的仍是标准
`launcher.py --enable-evaluation`；没有额外的批量评测或结果回写阶段。每次运行
使用带时间戳的新数据库，路径会在启动时打印。

与 OpenRT 相同的标准 SAfactory 启动方式（Gateway 需已单独启动）：

```bash
GENERATED_DIR=/tmp/safactory-patcheval-s1.1
python env/patcheval/generate_full_config.py \
  --output-dir "${GENERATED_DIR}" \
  --archive-dir /mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images \
  --official-runtime-dir /mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-runtime \
  --setting s1.1 \
  --evaluation-timeout-s 3600 \
  --shared-tmp /mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-tmp

python launcher.py \
  --mode docker \
  --docker-pull-policy never \
  --docker-image-archive-dir /mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images \
  --cleanup-docker-image \
  --agent-config "${GENERATED_DIR}/patcheval_config.yaml" \
  --agent-start-config "${GENERATED_DIR}/patcheval_start.yaml" \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --enable-evaluation \
  --db-path sqlite://env_trajs.db \
  --pool-size 5 \
  --max-workers 5
```

`generate_full_config.py` 会将 `rule_evaluator.py` 放到生成配置目录，因此
Launcher 按标准约定自动发现它，不需要 evaluation YAML。

默认运行 Docker 支持的全部 230 个 CVE。镜像归档默认从以下目录按需加载：

```text
/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-images
```

每个任务启动前，Launcher 会在 Docker 中检查对应的
`ghcr.io/anonymous2578-data/cve-*:latest`。若镜像不存在，则加载匹配的
`cve-*-latest.tar`；任务容器结束后再删除本次加载的镜像。这样无需把约
503 GB 的镜像同时放进 Docker 数据目录。

Smoke test 时给 `generate_full_config.py` 增加 `--limit 1`；全量评测省略
`--limit` 或设置为 `--limit 0`。并发由 Launcher 的 `--pool-size` 和
`--max-workers` 控制。
