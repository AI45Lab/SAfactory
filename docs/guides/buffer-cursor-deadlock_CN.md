# Buffer 游标死锁：`fetch_done_steps_with_context` 漏捞 late-flip terminal step

## 现象

RL 训练（patcheval / RJob 模式，Qwen3.8-27B）启动后，`RolloutManager` 一直打印：

```
(RolloutManager pid=66186) rollout data is not ready, have been waiting for 30 seconds
```

buffer server 日志（`logs/buffer_server.log`）持续打印同一行，pending 永远不变：

```
new_items=0, ready_groups=0, pending={'c37a804e-...': 7, '473bb515-...': 5, 'b377236b-...': 2}
```

环境侧其实正常：29/29 episode 都 `exit_code=0`，openhands agent 多轮改代码（18 个 episode 跑满 30 步），DB 里也确实攒出了完整的组。但 buffer 永远凑不齐 `group_size=8`，`ready_groups` 恒为 0，训练永远拿不到数据 → **死锁**。

## 根因

`core/data_manager/strategy/sqlite_strategy_impl.py::fetch_done_steps_with_context` 用 **`id` 自增主键做游标** 增量捞 terminal step：

```python
steps = await SessionStep.filter(
    job_id=job_id,
    is_terminal=True,
    id__gt=after_id        # ← 游标
).order_by("id").limit(limit)
```

而 terminal step 的写法是 **UPDATE 现有行**，不是 INSERT 新行。`evaluator/reward_committer.py::_commit_data_manager` 在 eval 完成后：

```python
updated = await _update_persisted_row(
    self.data_manager, terminal,
    {
        "step_reward": ...,
        "reward": ...,
        "is_terminal": True,          # ← 把已有行从 0 翻成 1
        "is_session_completed": True,
    },
)
```

一个 step 行的 `id` 在 **step 创建时（is_terminal=0）** 就定了。之后 eval 才把它 UPDATE 成 `is_terminal=1`。这两件事在时间上错开，而游标只会单调递增：

1. step A 在 `id=213` 创建（`is_terminal=0`），此时不被 `is_terminal=True` 选中。
2. 别的组的 step B 在 `id=243` 先被 eval 翻成 terminal，buffer 把它捞走，游标推到 `243`。
3. 之后 step A（`id=213`）才被 eval UPDATE 成 `is_terminal=1`。
4. 但 `id__gt=243` 永远不会再选到 `id=213` → **buffer 永远漏掉这一行**。

代码注释其实已经埋了线索（`sqlite_strategy_impl.py:699-703`）：

```
# NOTE: is_trainable is never flipped to True by the sqlite
# reward-commit path (reward_committer only sets is_terminal /
# is_session_completed) ...
```

即 reward-commit 走的是 UPDATE 翻转，不是 INSERT。

## 证据（实跑数据）

同一 job `9f7ca7b0...`，对比 DB 实际 terminal step 数 vs buffer pending：

| group_id | DB 实际 usable terminal | buffer pending | 丢失 |
|---|---|---|---|
| `c37a804e-...` | 8 | 7 | 1（late flip，游标已过） |
| `473bb515-...` | 8 | 5 | 3（late flip） |
| `14922466-...` | 6 | 0 | 6（全部 late flip） |

- DB 在 08:34 就已经有 8+8 两个满组，但 buffer 在 09:16（40+ 分钟后）还把它们当成 7 和 5，`new_items=0` 不变。
- buffer pending 里还有 `b377236b: 2`，而 DB 快照里该组 0 个 terminal 行 —— 说明 buffer 捞过、DB 后续又被改写，两边视图已不一致。
- 所有 terminal step 的 `created_at` 都在 08:27–08:30，buffer 却在 09:16 还没捞全 → 不是"还没写"，是"写过了但游标越过了"。

## 为什么不是其他原因

- **不是 github 屏蔽**：`env/patcheval/openhands_runner.py::_block_github_cdn` 是 patcheval **故意**的防作弊 + 快速失败优化，29/29 每个 episode 都有，CVE 仓库是预挂的，agent 照常干活。与死锁无关。
- **不是 SQLite WAL 读快照过期**：buffer 早期确实捞到了 14 个 item（pending 非空），只是后续 late-flip 的行捞不到；WAL 过期会连早期行都丢，现象不符。
- **不是 pool 没起环境**：pool 停止起新环境是死锁的**结果**（buffer 不消费 → launcher 不再投新 episode），不是原因。

## 修复方向

不要用 `id` 游标来增量选 `is_terminal` 行。`id` 游标只对"只 INSERT、不 UPDATE 筛选列"的写法成立，而 terminal step 是 UPDATE 翻转。

### 已采用方案：滑动窗口游标（只改 buffer 侧，不动表，内存有界）

关键观察：late-flip 只在"行创建后不久"发生——eval 在 episode 结束后几秒~几分钟内 commit。一个行创建超过 T 仍未翻，基本不会再翻。所以不用全表扫、也不用记全部 served pk，用一个**滑动窗口**回看：

- 保留 `last_served_id` 高水位（正常游标）。
- 策略层 `fetch_done_steps_with_context` 多收一个 `lookback` 参数，查询改为
  `is_terminal=True AND id > (after_id - lookback)`（即回看窗口 `(after_id-lookback, after_id]` + 新行 `(after_id, +∞)`）。
  `lookback>0` 时不加 `limit`，避免窗口里已服务的旧行把新行挤掉。
- buffer 层维护 `served_pks: set`，对回看窗口重复返回的行去重；`last_served_id` 仍按已服务行的最大 id 推进。
- 剪枝：`served_pks` 只保留 `pk > last_served_id - lookback` 的行——更老的行不会再被回看扫到，可安全丢弃。

**内存 = O(窗口内 terminal 行数)**，有界（`lookback` 取 100000 id 单位，约几千个 terminal 行，几 MB）。`lookback` 必须大于"eval 时延折算成的 step 插入数"——eval 在 episode 后几分钟内完成，`lookback=100000` 远大于该量，安全。可用环境变量 `BUFFER_FETCH_LOOKBACK` 调整。

### 其他方案（未采用）

- **`reward_committed_at` 时间戳列**：加列 + reward_committer UPDATE 时写时间戳 + buffer 按时间戳游标。内存 O(1)、扫描可走索引、语义最干净，但要改表结构。
- **reward_committer 改 INSERT**：把 UPDATE 现有 terminal 行改成 INSERT 新 terminal 行（新 id），现有 `id` 游标天然能捞到。但改变"terminal = 最后一行 in-place"语义，trainer/advantage 读 `session_steps` 可能受影响，风险大。
- **全量 served set**：每次全表扫 terminal 行 + 全量 served pk 去重。内存 O(总历史) 会随训练增长，不推荐。

### 改动文件

- `core/data_manager/strategy/sqlite_strategy_impl.py` — `fetch_done_steps_with_context` 加 `lookback`，查询用 `id > after_id - lookback`，`lookback>0` 时不 limit
- `core/data_manager/strategy/cloud_strategy_impl.py` — 同签名加 `lookback`（暂不应用，云游标机制不同，留作 follow-up）
- `core/data_manager/manager.py` — 转发 `lookback`
- `rl/buffer_server.py` — `served_pks` 集合 + `FETCH_LOOKBACK`（env `BUFFER_FETCH_LOOKBACK`，默认 100000）+ 去重 + 剪枝；`init_data_manager` 重启时清 `served_pks`

## 复现 / 验证

```bash
DB=/mnt/shared-storage-user/leishanzhe/repo/SAfactory/rl/examples/patcheval/patcheval_qwen3_8_27b.db
# DB 实际 terminal 数（按组）
sqlite3 -header -column "$DB" "
SELECT group_id, count(*) AS usable_terminal
FROM session_steps
WHERE job_id='9f7ca7b038a44d2a8441dcbc5b055cc9' AND is_terminal=1
  AND messages IS NOT NULL AND messages NOT IN ('[]','null','')
GROUP BY group_id ORDER BY usable_terminal DESC;"
# buffer 看到的（日志）
grep 'new_items=' /mnt/shared-storage-user/leishanzhe/repo/SAfactory/logs/buffer_server.log | tail -5
```

DB 有满组、buffer pending 不满、且 `new_items` 长期为 0 → 即为本 bug。

## 相关文件

- `core/data_manager/strategy/sqlite_strategy_impl.py` — `fetch_done_steps_with_context`（游标逻辑，需改）
- `core/data_manager/strategy/cloud_strategy_impl.py` — 同名实现（需同步改）
- `rl/buffer_server.py` — `fetch_new_items_from_db` / `accumulate_and_pop_ready_groups`（调用方、组聚合）
- `evaluator/reward_committer.py` — `_commit_data_manager`（UPDATE 翻转 `is_terminal` 的源头）
