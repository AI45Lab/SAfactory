# 数据管理器

Safactory 会在每次运行期间记录环境配置和 step 级交互。默认 CLI 数据库 URI 是 `sqlite://env_trajs.db`；可以使用 `--db-path` 写入其他位置。

```bash
python launcher.py \
  --env-config env/osgym/os_config.yaml \
  --db-path sqlite://runs/os_eval.db \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL
```

默认启用带缓冲写入。只有在调试时需要立即同步写入，才使用 `--disable-buffer`。

## 表

每次运行由两张表记录：

| 表 | 用途 |
|----|------|
| `job_environments` | 每个环境实例一行。存储 job ID、环境名、配置参数、group ID、完成状态和时间戳。 |
| `session_steps` | 每个智能体 step 一行。存储消息、模型响应、奖励、环境状态、终止标志和时间戳。 |

核心关系：

```text
job_environments.id  ->  session_steps.session_id
```

重要的 `session_steps` 字段：

| 字段 | 说明 |
|------|------|
| `step_id` | session 内的 step 索引。 |
| `messages` | 发送给模型的完整或截断后的 OpenAI 风格聊天历史。 |
| `response` | 原始模型输出。 |
| `step_reward` | 每步奖励信号。 |
| `reward` | 累计奖励。 |
| `env_state` | 序列化后的环境状态或观察元数据。 |
| `is_terminal` | 该行是否结束 episode。 |
| `is_session_completed` | session 是否成功完成。 |

## 查询示例

列出最近的环境 session：

```bash
sqlite3 env_trajs.db "
  SELECT id, env_name, group_id, finished, created_at
  FROM job_environments
  ORDER BY created_at DESC
  LIMIT 20;"
```

查看一个 session：

```bash
sqlite3 env_trajs.db "
  SELECT step_id, step_reward, reward, is_terminal, created_at
  FROM session_steps
  WHERE session_id = '<session-id>'
  ORDER BY step_id;"
```

汇总已完成奖励：

```bash
sqlite3 env_trajs.db "
  SELECT env_name, COUNT(*) AS episodes, AVG(reward) AS avg_reward, MAX(reward) AS best_reward
  FROM session_steps
  WHERE is_session_completed = 1
  GROUP BY env_name;"
```

## 训练数据用途

记录的轨迹可以转换为 SFT 或 RL 数据：

- `messages` 存储对话上下文。
- `response` 存储模型动作或答案。
- `step_reward` 和 `reward` 提供奖励信号。
- `env_state` 保留环境侧上下文，便于调试和筛选。

在线 RL 请使用 [RL 训练](../rl/README_CN.md) 中介绍的 Buffer Server。
