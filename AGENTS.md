# SAfactory ExploitGym worktree instructions

- 先读 `README_CN.md`、`env/exploitgym/README.md` 和
  `../../docs/WORKSPACE.md`。
- 这是当前 ExploitGym → SAfactory 接入的活跃 worktree；开始前检查
  `git status` 和相关 diff，保护现有未提交实验。
- adapter 的直接事实来源是 `env/exploitgym/runner.py`、
  `rule_evaluator.py`、配置和 dataset；一条 dataset 记录只对应一个 episode。
- 本地 `env/exploitgym/MANUAL_RUN.md` 是未提交运行记录，使用其中地址和参数
  前必须与当前配置核对，不能视为长期事实。
- 修改要保持开发机 Docker 与 RJob 两条路径的任务语义、结果位置和评分一致。
- 结果、轨迹、session 与 task 必须可追溯；区分正常结束、空响应、循环、
  timeout、gateway 错误和环境失败。
- 不提交或输出模型凭据；Gateway 管理真实凭据，adapter 只接收 episode 会话。
- 镜像构建/推送、RJob 提交/停止和批量 rollout 默认只提供命令与验收点，
  除非用户明确要求执行。
- 不把改动手工复制到 `../../SAfactory/`；需要同步时先确认提交与分支关系。
