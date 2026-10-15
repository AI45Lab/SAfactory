# Sandbox 非 ENV 实现接口与环境迁移计划

状态：非 ENV 框架已实现；本轮没有修改 `env/`，没有创建环境专用 Sandbox YAML，也没有执行真实集群端到端验收。下文的配置解析、AFS、SDK、文件评分、Gateway 与恢复接口对应当前代码；Env 评分接口和环境验收仍是后续工作。配置和结果协议见 [Sandbox 工作流](sandbox-workflow-design_CN.md)。

## 修改范围

沿用 `SimulationFlow → AgentPoolManager → RuntimeLeaseAllocator → SimulationWorkerGroup`。以下公共接口不改：

```text
RuntimeLeaseAllocator.start(plan)
RuntimeLeaseAllocator.allocate(*, row_id, env_name, env_id, image, env_params, group_id)
RuntimeLeaseAllocator.release(entry, *, succeeded)
RuntimeLeaseAllocator.remove(entry)
RuntimeLeaseAllocator.close()
AgentStartClient.start(lease, request)
SandboxEpisodeRunner.start(lease, request)
RewardCommitter.commit(*, session_id, eval_result)
```

新增的公共实现文件是 `core/result_artifact.py` 和 `evaluator/result_file.py`。AFS 辅助逻辑保留在既有 `clusters/sandbox_cluster.py`，没有独立 AFS helper 模块。不新增 SandboxClient、结果收集服务、存储管理框架、scheduler、评分 registry 或 per-env result_adapter。

## 配置解析接口

`args.py` 和 `manager/simulation_config.py` 当前接口：

```python
DEFAULT_SANDBOX_CONFIG_PATH = "sandbox-config.yaml"

def load_agent_start_config(
    path: str | None, *, mode: str = "docker"
) -> dict[str, dict]: ...

def _normalize_agent_start_sandbox(
    agent_name, spec, cfg_path: Path, docker: dict | None = None
) -> dict: ...
```

`build_manager_runtime_config(cfg)` 调用 loader 时传 `mode=cfg.mode`。Sandbox 分支只解析 `sandbox:`，不调用 `_normalize_agent_start_docker()`，也不读取 RJob 挂载。Docker/RJob 保留旧默认和解析行为；必要的相对文件路径处理继续复用 `_normalize_runner_entrypoint()`、`_normalize_embedded_file()`。

Sandbox normalizer 负责：

- 将 `runner_entrypoint.command` 归入内部已有 `run_command`，入口文件去重加入 `embedded_files`。
- 校验本地 source 存在，目标路径为绝对路径且无冲突，保留文件 permissions。
- 将 inputs 的 host_path 转为 H 绝对路径；校验 AFS subpath、mount_path、只读属性和重复挂载。
- 要求 outputs 包含可写 `/app/results`；不增加 outputs.host_path。

平台配置的 `_normalize_sandbox_config()` 解析 results.local_root（相对 launcher 工作目录）及下载参数；`build_manager_runtime_config()` 向内部 Sandbox config 注入 job_id。上传 source 和相对 inputs.host_path 则相对启动 YAML 解析。API 凭据不传入 runner。

CLI 的 Gateway 默认值先保留为 `None`，在 `load_simulation_run_config()` 中按 mode 解析。Sandbox 使用“显式 CLI → sandbox.gateway_base_url”，写入唯一 `cfg.gateway_base_url`，主控、runner 和 close client 共用；没有可达配置则报错。其他 mode 维持原来的默认行为。

Sandbox 必需的文件评分由 `evaluation_enabled = (mode == "sandbox") or args.evaluation_enabled` 控制，不新增评分开关。要求显式选择本模式业务配置，避免扫描目录时重复加载多个 mode。

## AFS 输入发布接口

`clusters/sandbox_cluster.py` 内的函数型 helper：

```python
async def prepare_storage(
    *, inputs: list[dict], outputs: list[dict],
    project: str, webdav_url: str, api_key: str,
) -> list[ProjectAfsVolume]: ...
```

这是一个函数型 helper：输入发布、预建输出目录、返回 SDK 可接受的完整 volumes。由 backend.start() 对每个 env 调用一次，相同 source/subpath 去重；allocate() 不重复上传大目录。

复用方式：

1. `sandbox/migrate_gpfs_data_to_afs.sh` 已支持 `--source`、`--dest-subpath`、`--project`、`--webdav-url` 和 `--publish-once`；保留原用法的默认值。
2. 现有软链物化、旧 GPFS 前缀映射、断链/环检查和 `rclone check --download` 留在此脚本，Python 不复制一套遍历算法。
3. Python 通过 `asyncio.create_subprocess_exec()` 调用该脚本，输出目录用 `rclone mkdir` 准备；AK/SK 只通过子进程环境和 stdin 传递，不拼接 shell 字符串、不修改全局进程环境。
4. 使用带版本的输入目录。发布前以 WebDAV MKCOL 获取目的目录的排他锁；只有明确取得锁的进程才负责释放。已发布目录先比对物化后的完整文件清单、大小和 SHA-256，再逐字节验证后复用；新版本或未完成发布才执行 copy。确认源未改变且远端没有多余文件后，才写入 manifest 完成标记。未知锁状态、并发冲突或锁释放失败均报错；中断留下的锁须确认没有活跃发布者后人工处理，不能自动抢占。
5. backend 的 `ProjectAfsVolume` 同样定义于 `clusters/sandbox_cluster.py`，沿用 `create_sandbox.py` 的轻量兼容表示；CLI 脚本保持自己的既有实现。该表示支持字典与 SDK 所需属性，不伪造 host/PVC/OSSFS backend。

没有 AFS→H 的评分回收接口；标准结果文件统一通过 SDK 下载。

## Sandbox backend 与 SDK 接口

`clusters/sandbox_cluster.py` 的现有实现：

```python
async def start(self, plan: BindingPlan) -> None: ...

async def _create_sandbox(
    self, *, image: str, environment_id: str, cfg: dict,
    volumes: list[ProjectAfsVolume] | None = None,
    metadata: dict[str, str] | None = None,
) -> _SandboxRuntimeHandle: ...

async def _connect_instance(
    self, *, sandbox_id: str, cfg: dict,
    connection: Any, sandbox_service: Any,
) -> _SandboxRuntimeHandle: ...

async def _prepare_instance(self, sandbox, cfg: dict) -> None: ...
```

`start()` 完成 Environment 校验、输入准备和完整挂载缓存。`_validate_environment()` 检查模板镜像、execd 端口及实际生效的覆盖挂载，不再要求所有 mount_path 已存在于模板中。

`allocate()` 首先检查 recovery_by_session；恢复分支保持原 attempt 和 runtime.json，不重新分配身份或执行业务准备。只有新运行分支生成 attempt_id，写入本地 runtime.json，再带以下字符串 metadata 创建实例：job_id、session_id（沿用 env_id 契约）、attempt_id、env_name。创建成功后补记 sandbox_id；请求超时先按 metadata 找回，不能盲重建。实例 Ready 和上传检查通过后返回原 PoolEntry。

现有 `_create_sandbox()` 的 `volumes=None`、`metadata={}` 改为实际值；Environment 的 ports、image、entrypoint 继续继承，挂载列表整体覆盖。`_connect_instance()` 抽取已有 endpoint 规范化、commands 和 ping 逻辑，供新建及恢复共用，显式复用所属 connection/service，不新增一层 SDK client。

扩展原 handle：

```python
class _SandboxRuntimeHandle:
    id: str
    commands: Any                   # 已有
    files: Any                      # 新增 SDK service
    retain: bool = False             # 开始执行后置 True，处理完成后允许回收
    execution_stopped: bool = True   # 未确认主命令停止时拒绝权威结果回收
    command_id: str | None = None

    async def renew(self, ttl_seconds: float) -> None: ...
    # 原 get_endpoint / kill / close 接口不变
```

lease 的 runtime_config 是复制的 dict，不能用其中的可变 flag 通知 backend；retain 放在两端共同持有的原 handle 上。它只控制自动释放，不新增生命周期状态类。

以下接口已核对 OpenSandbox 0.1.9，直接使用：

```python
files = factory.create_filesystem_service(adapter_endpoint)

await files.create_directories([WriteEntry(path=parent, mode=755)])
await files.write_files([
    WriteEntry(path=target, data=source.read_bytes(), mode=644)
])
await files.set_permissions([SetPermissionEntry(path=target, mode=755)])
content = await files.read_bytes(target)

await sandbox_service.renew_sandbox_expiration(
    sandbox_id, new_expiration_time  # timezone-aware datetime
)
```

`WriteEntry` 和 `SetPermissionEntry` 来自 `opensandbox.models.filesystem`。SDK mode 是十进制外形的权限数字 644/755，配置 `"0755"` 转为整数 755，不能传 Python `0o755` 的数值 493。

`_prepare_instance()` 使用 SDK 批量上传入口和辅助文件，read_bytes 回读校验小文件 SHA-256；实际探测输入目录可读和输出文件可写、可回读。Python 包、原生工具、评分依赖、镜像能否访问 Gateway 仍需 env 镜像准备及真实 smoke 验证，框架不会自动安装或穷举检查这些依赖。续期 task 由 handle 持有，close 时取消并 await；保留实例关闭连接后不再自动续期，仍受平台 TTL 限制。

`release()` 仅负责释放，不下载结果、不执行评分。retain=True 或 cleanup_on_finish=False 时关闭 SDK 连接并保留实例；其他实例删除。`close()` 同样遵守 retain，避免失败后 finally 再无条件删除。显式 remove() 仍可用于已确认的旧 writer 清理。

### 命令超时与中断

保留现有前台 SSE 执行方式，扩展原 helper 的一个可选回调，不引入后台任务轮询框架：

```python
async def run_sandbox_command(
    sandbox: Any, command: str, *, timeout_s: float,
    on_init: Callable[[str], Awaitable[None]] | None = None,
) -> SandboxCommandOutput: ...
```

内部调用已核对的 SDK API：

```python
execution = await sandbox.commands.run(
    command,
    opts=RunCommandOpts(timeout=timedelta(seconds=timeout_s)),
    handlers=ExecutionHandlers(on_init=handle_init),
)
await sandbox.commands.interrupt(execution_id)
status = await sandbox.commands.get_command_status(execution_id)
```

`handle_init` 从事件 `id` 保存执行标识并调用可选 on_init；长任务的回调将 command_id 补入原 runtime.json，探测命令不需要持久记录。服务端 timeout 加上本地有余量的等待，防止仅取消 HTTP 等待却留下远端 writer。

本地超时或取消时，对 interrupt 和后续停止确认执行独立限时、受保护的清理；必须确认 `status.running is False` 才进入权威结果收集。拿不到 execution_id、中断失败或停止状态不明时，保留实例并报告运行状态不确定，不提交分数、不自动重跑。正常路径不轮询后台日志；只有停止确认需要短暂查询 status。env runner 仍需负责退出时终止其原生子任务，SDK interrupt 不能被当成所有脱离父进程子任务已清理的证明。

同时修正 `SandboxCommandOutput.from_sdk()`：Execution 有 `exit_code: int | None` 和 error，没有 code 字段；删除 `int(exit_code or 0)`。有明确 SDK error 或无可信终态不能默认退出码 0；可用已记录 id 查询一次终态，无法确认则失败。小型探测命令和长 runner 共用这一处修正。

## 标准结果协议接口

标准库模块 `core/result_artifact.py` 已实现；后续 env 迁移时将同一源码上传至 Sandbox：

```python
def finalize_result(
    result: Mapping[str, Any], *,
    env_params: Mapping[str, Any],
    scorer: Callable[..., Mapping[str, Any]],
    identity: Mapping[str, str],
) -> dict[str, Any]: ...

def write_result_atomic(path: str | Path, payload: Mapping[str, Any]) -> None: ...

def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None: ...

def read_result_file(
    path: str | Path, *, job_id: str, session_id: str, attempt_id: str
) -> dict[str, Any]: ...
```

`finalize_result()` 只调用传入 scorer 一次，补齐 v2 identity/evaluation；集中处理 scorer 异常为失败 evaluation，避免每个 env 重复异常模板。normalizer_id 和 eval_id 由 env scorer 返回，不能由用户 YAML 改写。

writer 与 reader 共用私有 `_validate_result()`。`write_result_atomic()` 验证协议后调用通用 `write_json_atomic()`；后者统一执行 `allow_nan=False`、同目录临时文件、文件/目录 fsync 和原子 replace。backend 导入别名 `_atomic_json`，runner 的 runtime.json 读取、身份核验、merge 后也复用同一 helper。通用 writer 不验证结果协议；SDK 原始字节下载仍保留独立的摘要校验与字节原子发布。发布失败必须抛出。reader 严格读取文件并验证身份、状态及所有可提交分数；不默认成功、不乘 10、不 clamp。JSON 非法常量和重复键也拒绝，避免同一文件有歧义。

最终文件必须满足 `terminated or truncated`，允许 Gym 风格的 `terminated=False, truncated=True`；truncated/timeout 顶层状态与 truncated 标志必须一致。

这个模块不导入 manager、evaluator、SDK 或 benchmark 包。H 用 `core.result_artifact`；Sandbox 在 start YAML 中将同一文件上传到 runner 目录，按 `result_artifact` 导入。不复制协议实现或打包整个 SAfactory。

`SimulationStartResult` 只追加一个可选字段：

```python
result_artifact_path: str | None = None
```

不再追加 evaluation 或第二个 metadata dict。完整权威数据保留在文件；attempt 来自既有 lease.runtime_config/request.metadata，下载与命令诊断使用既有 metrics。

## SDK 下载与 runner 接口

只扩展 `manager/sandbox_episode_runner.py`：

```python
async def _collect_result(
    self, lease: SimulationAgentLease, request: SimulationStartRequest
) -> SimulationStartResult: ...
```

`start()` 保留已有请求序列化、stdin 和环境变量注入，增加 runtime/attempt 标识。执行开始时 handle.retain=True。正常结束、非零退出或命令超时后均尝试 `_collect_result()`；取消时先阻断生产者并进行有界回收，不能提前释放实例。

`_collect_result()` 的顺序固定：

1. 复用 `episode_common.result_artifact_path(request)` 得到远端固定文件路径。
2. 使用 handle.files.read_bytes 下载，按平台配置重试；远端大小及受控 `sha256sum` 结果仅用于完整性检查。
3. 写入 `<local_root>/<job_id>/<session_id>/` 中唯一临时文件，fsync。
4. 调用共用 read_result_file() 检查完整协议。expected identity 来自 request 与 lease，不从待验证文件反向获取。
5. 校验通过后原子替换 `safactory_result.json`；不使用历史本地文件兜底。
6. 复用 normalize_result() 转换运行字段，前提是身份已经严格验证；设置 result_artifact_path，并将 total_reward 保持 None，直到正式 reward 提交完成。

删除此 runner 中 stdout JSON 解析、exit_code 成功评分、`cat` fallback 和命令超时直接补结果分数的路径。退出码与超时只作运行诊断：外层强制超时或传输/执行错误不能被某个有效 JSON 覆盖为运行成功，必要时记录传输失败；有效的业务失败文件仍应下载保存。

不新增独立 ResultCollector；大小、摘要、重试和落盘都在这个私有方法内。`SandboxSync.connect + files.read_bytes` 的参考为 `sandbox/get_first_sandbox.py`；现有 async handle 直接增加 files，不为下载再建立第二条连接。

## 文件 evaluator 和 worker 接口

`evaluator/result_file.py` 实现已有 RuleEvaluationBackend 协议：

```python
class ResultFileEvaluatorBackend:
    async def evaluate(
        self, *, request: EvalRequest, spec: EvalSpec, trajectory: Trajectory
    ) -> EvalResult: ...
```

读取 request.start_result.result_artifact_path，使用共用 reader 和可信 request/lease identity；将 evaluation 原值构造成 EvalResult。normalizer_id 等附加证据可放入 artifacts，不扩张 EvalResult。不得依赖 metrics 中的原生远端路径。这里不经过 RuleEvaluatorBackend._coerce_result()，避免隐式 clamp。

扩展现有 factory 的依赖注入接口：

```python
def build_evaluation_service(
    *, trajectory_reader=None, max_concurrency=64,
    backend: RuleEvaluationBackend | None = None,
) -> EvaluationService: ...
```

默认仍是 RuleEvaluatorBackend。SimulationFlow 在 mode=sandbox 时注入 ResultFileEvaluatorBackend，并选固定 `EvalSpec(eval_id="safactory_result_file")`；不新增 EvalMethod，也不将 runtime 判断塞进评分 service。

`SimulationWorkerGroup` 复用原流程，只调整以下分支：

- `_build_start_request()` 在已有 metadata 中增加 attempt_id，runtime、job/session 继续复用。
- 原 `result.total_reward=None` 保留，防止误用历史顶层分数。
- Sandbox 不进入旧的 truncated 固定补零路径；由文件 evaluator 返回 succeeded/truncated/failed/timeout。
- succeeded 或合法且明确提供分数的 truncated 复用 RewardCommitter.commit，确认后完成对应 session；failed/timeout/无文件不提交评分。
- env rule evaluator 的 discover 仅用于非 Sandbox，不再发生一次原生评分、一次 H 重评分。
- 结果本地持久化、Gateway finalization 和相应提交/失败处理均结束后，才将 handle.retain=False；异常路径保留以供恢复。仅恢复合法本地文件时可以没有 handle，操作 retain 前必须判空。

`_finalize_gateway_session()` 原签名不变，但检查 close 返回体：身份正确、drained=True、telemetry_status=flushed、client_timed_out 不为 True；不再以 await 无异常代表成功。

为保持本轮简洁，不新增 durable outbox 或 DAO flush 抽象。Sandbox 使用现有同步严格写入部署：Gateway telemetry.mode=strict、async_cloud_writes=false，storage_config.enable_buffer=false，并关闭 enable_upsert_batching。已有 Gateway 配置承载这些选项，不新增第四份 Sandbox YAML。

`GET /readyz` 增加脱敏的实际写入配置与存储身份摘要，供 SimulationFlow.check_gateway_ready() 验证上述条件和两端同一后端；旧客户端可忽略新增字段。这样 close 的 flushed 不会仅代表数据进入下一层内存队列。未满足同步确认条件时 Sandbox 不启动，异步持久化优化另行实现。

GatewaySessionBinding 已增加 `telemetry_write_failed: bool = False`。strict 写入异常时由 TelemetryRecorder 置位；close 完成时若置位，`telemetry_status` 返回 failed，不能报 flushed。本轮不自动清除此标记或丢弃失败事件后继续宣称完整；worker 按 close 返回体拒绝提交。这个字段补足“没有在排队”和“历史写入全部成功”的区别，不引入队列或 outbox。

Gateway 在每次启动时生成 `instance_id`，由 readyz、session status 和最终 close 响应返回；框架将启动时的值记入原 runtime.json。关闭确认必须来自同一进程；worker 再把 `gateway_finalized: true/false` 原子写回 runtime.json，持久化失败同样禁止评分提交。恢复时已有 True 确认可跨 Gateway 重启继续提交；False 明确失败，拒绝自动恢复；缺少确认时必须仍有原进程的会话绑定，新的空进程或重建绑定不能补造历史持久化证明。这些字段全部由框架维护，不增加用户参数或 YAML。它们不恢复已经丢失的写入，也不提供跨重启的持久封禁。

## Env 评分接口（待迁移，本轮未改 env/）

以下是后续改造约定，当前 env 尚未接入：在原 `rule_evaluator.py` 提供评分入口，不新增 scoring.py/result_adapter.py：

```python
def score_result(
    result: Mapping[str, Any], *, env_params: Mapping[str, Any]
) -> dict[str, Any]: ...

async def evaluate_rule(*, request, spec, trajectory):
    # 延迟导入 evaluator 类型；复用 score_result 后转成既有 EvalResult
    ...
```

`score_result()` 返回 status、raw_score、normalized_score_10、eval_id、normalizer_id、reason、artifacts、error_text。接收完整 result，以覆盖 Harbor error_text、AgentCompass 状态和 PatchEval official_record 等需求。保留本地证据读取和官方验证；不是统一的无 I/O 纯函数。

后续将 evaluator.* 导入移入旧入口或 TYPE_CHECKING。Sandbox 上传原 rule_evaluator.py、共用 result_artifact.py 和已声明的必要依赖；runner 将 `score_result` 作为 scorer 传给 `finalize_result()`，再调用 `write_result_atomic()`，不要先单独调用 score_result 造成重复验证。旧 mode 保留原输出调用方式。

| Env | 接口迁移位置与政策 |
| --- | --- |
| Harbor | 原 rule_evaluator 中 rollout/fatal 检查、harbor_reward 归一化移入 score_result；runner 封装原生结果后调用，0.8 只变成 8 一次。 |
| OpenRT | 原指标聚合和 insucess_rate 政策原样移入；runner 本地指标完整后评分，不再用已是十分制的 total_reward 乘 10。 |
| CyberGym | score_result 本地读 verified poc.db；runner_lib/result_writer.py 在 verify 后调用，验证凭据进入 JSON。 |
| CyberRange | score_result 本地读 native/milestone；runner.sh 的 Python 最终写入阶段调用，不在 H 打开 artifact。 |
| Geo3K | grader 后复用原 score/passed→十分制政策，H 不重新判题。 |
| AgentCompass 等 | 保留各 benchmark 原政策，不强制统一乘 10；已经十分制的原值直接封装。 |
| PatchEval | score_result 执行原官方验证一次；旧 async 入口可通过 asyncio.to_thread 调用同步验证，Sandbox 在任务尾阶段调用，依赖在镜像中准备。 |

## 恢复接口

不建立新调度器或数据库状态表。每个 session 在现有结果目录保存一份 runtime.json，包含 job/session/attempt、sandbox_id、配置摘要和远端路径；创建前记录 attempt，创建后补 sandbox_id。

既有 `manager/resume_cleanup.py` 的当前接口：

```python
async def prepare_sandbox_resume(
    *, cfg, manager_cfg: dict,
    environment_rows: Sequence[dict], data_manager,
) -> tuple[list[dict], dict[str, dict]]:
    # rerun_rows, recovery_by_session
    ...
```

在 SimulationFlow.prepare_storage() 删除 unfinished 轨迹之前调用，内部委托 `SandboxClusterBackend.prepare_resume(environment_rows)`。合法本地结果只恢复提交；远端结果仅待下载则保留旧 attempt、连接旧实例收集。原实例不在但 AFS 最终文件存在时，复用同 Environment 与挂载创建仅收集实例，SDK 下载旧 attempt 文件，不执行业务 runner。不能确认原 runner 正常退出时保留 execution_error；合法 JSON 不能抹掉非零退出、SDK 错误或状态不确定。

只有已阻断旧 writer、确认没有可恢复最终结果且需要重跑的 rerun_rows 才交给原 delete_resume_session_steps() 和 Gateway cache 清理。无法确认时停止恢复，不先删轨迹。

recovery_by_session 放入内部 Sandbox config。原 allocate() 返回恢复 lease，runner.start() 先检查恢复信息并跳过业务命令；公共 allocator/worker 签名不改。恢复失败保留原 runtime.json、AFS 与轨迹。只新增本地引用文件，不将 JSON 内自报的 attempt 当成可信预期值。

## 当前验证边界与后续工作

非 ENV 单元测试覆盖共用结果协议、文件 backend、SDK 结果下载、挂载覆盖、多文件上传、续期与命令停止、配置解析、Gateway 持久化确认和恢复分流；使用 fake SDK/rclone，不请求真实集群。测试不能证明某个业务镜像已具备完整评分依赖或远程网络可达。

后续优先迁移 Harbor/OpenRT：创建两份独立环境 YAML、接入 v2 文件发布、验证旧/新评分一致，再执行真实 smoke。其他 env 同样完成评分依赖与退出清理审查后才可声明支持；当前示例属于接入约定，不能直接把既有 `.rjob.yaml` 或未迁移 runner 当作完成支持。
