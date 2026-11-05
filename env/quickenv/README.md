# ExploitGym：Exploit 与 Mining 分批验证指南

本次交付包含两批任务，**分别检查、分别启动、分别查看结果**。默认每批并发 20 题，每题最多 1 轮 agent 交互，用于验证任务启动及模型交互。


| 批次           | 任务目录               | 数量             | 启动入口                       | 默认结果目录                 |
| ------------ | ------------------ | -------------- | -------------------------- | ---------------------- |
| Exploit      | `pj-exploit-400/`  | 400 条非 ARVO 任务 | 离线环境的 `start.sh`           | `out/exploit_400/`     |
| Mining 第 1 份 | `pj-mining-400/`   | 31 条 ARVO lv0  | `run_mining_400.sh`        | `out/mining_400/`      |
| Mining 第 2 份 | `pj-mining-part2/` | 184 条 ARVO lv0 | 同一脚本，指定 `MINING_BATCH_DIR` | `out/pj-mining-part2/` |
| Mining 第 3 份 | `pj-mining-part3/` | 185 条 ARVO lv0 | 同一脚本，指定 `MINING_BATCH_DIR` | `out/pj-mining-part3/` |


`start.sh` **只启动 Exploit，不会自动启动 Mining。** 建议先完成第一批，再单独启动第二批；两批共用 controller、LLM proxy 和防火墙服务。

## 1. 公共准备：只需做一次

目标服务器需要 Linux x86_64、Docker Engine、Docker Compose v2、bash 和 flock，并能访问交付共享盘及内网模型 API。本文使用 `sudo` 启动，避免普通用户无权访问 Docker socket。

```bash
cd /mnt/shared-storage-gpfs2/trustcyberdata/private/CVEDockerGen/wuxuansheng/20261019_deliver/exploit_gym
sudo tar -xzf exploitgym-offline-linux-x86_64.tar.gz -C "$HOME"
sudo docker info
sudo docker compose version
```

已解压过当前版本时，无需重复解压。**无需安装 conda、执行** `uv sync` **或手动激活 Python 环境。**

离线包包含 Python、Harbor、LLM proxy 依赖及防火墙镜像；不包含 Docker Engine、Compose 或全部题目数据。任务及镜像继续从交付共享盘读取，模型调用仍需访问内网 API。

两批统一使用交付目录旁的 Claude agent 实体副本：

```text
20261019_deliver/
├── claude-code-2.1.228-linux-amd64/
│   ├── claude
│   └── manifest.json
└── exploit_gym/
    ├── pj-exploit-400/
    ├── pj-mining-400/
    ├── run_exploit_400.sh
    ├── run_mining_400.sh
    └── scripts/run_pj_exploit.sh
```

脚本按相邻目录定位 agent，并只读挂载到任务容器的 `/usr/local/bin/claude`。迁移时应保持这两个交付目录同级；任务软链接及配置仍可能依赖原共享盘，不能仅复制框架就认为完成迁移。

## 2. 第一次验证：Exploit 400 条



### 2.1 检查环境和任务清单

```bash
sudo bash "$HOME/exploitgym-offline/start.sh" --check
```

检查 Python 模块导入、Harbor 版本、400 条任务清单及必要文件。该操作不启动服务、不加载镜像、不调用模型，也不验证模型 API 连通性。

### 2.2 启动 1 轮交互验证

```bash
set -o pipefail
sudo env WORKERS=20 MAX_TURNS=1 \
  bash "$HOME/exploitgym-offline/start.sh" 2>&1 \
  | tee "$HOME/exploit-400-start-$(date +%Y%m%d-%H%M%S).log"
```

执行链路：`start.sh → run_exploit_400.sh → scripts/run_pj_exploit.sh`。脚本加载共享镜像、准备服务，并逐题加载镜像和执行 Harbor。

### 2.3 查看本批结果

```bash
cd /mnt/shared-storage-gpfs2/trustcyberdata/private/CVEDockerGen/wuxuansheng/20261019_deliver/exploit_gym
sudo cat out/exploit_400/pj_user_batch_summary.json
```

汇总在批次结束后生成。运行中可查看终端输出、`$HOME/exploit-400-start-*.log`，以及 `out/exploit_400/launcher-*.log`。

本批任务清单为 `data/task_ids/pj.txt`。

## 3. 第二次验证：Mining 400 条

在 Exploit 验证完成后，单独运行 Mining。Mining 共 400 条，现按容量拆成 31、184、185 条三份，分别约 400、769、770 GB。下面的默认命令只运行第一份；第二、三份需显式选择目录。每份都有独立清单和输出目录。

### 3.1 检查任务清单

```bash
cd /mnt/shared-storage-gpfs2/trustcyberdata/private/CVEDockerGen/wuxuansheng/20261019_deliver/exploit_gym
sudo bash run_mining_400.sh --check
```

该检查核对当前分组的 lv0 任务、必要文件及镜像包是否存在，不运行 Docker 或模型。

### 3.2 启动 1 轮交互验证

```bash
set -o pipefail
sudo env WORKERS=20 MAX_TURNS=1 \
  bash run_mining_400.sh 2>&1 \
  | tee "$HOME/mining-400-start-$(date +%Y%m%d-%H%M%S).log"
```

另外两份分别执行（先完成一份，再运行下一份）：

```bash
sudo env MINING_BATCH_DIR="$PWD/../pj-mining-part2" bash run_mining_400.sh
```

```bash
sudo env MINING_BATCH_DIR="$PWD/../pj-mining-part3" bash run_mining_400.sh
```

加上 `--check` 可分别检查；它们的结果目录为 `out/pj-mining-part2/` 和 `out/pj-mining-part3/`。

脚本自动寻找当前用户的 `~/exploitgym-offline`；使用 sudo 时按原用户定位。若环境解压在其他位置，可用 `sudo env EXPGYM_OFFLINE_ROOT=/实际解压目录 ...` 显式指定。

执行链路：`run_mining_400.sh → scripts/run_pj_exploit.sh`。下层脚本使用传入的 Mining 目录，不会因此启动 Exploit 任务。

### 3.3 查看本批结果

```bash
sudo cat out/mining_400/pj_user_batch_summary.json
```

运行中可查看终端输出、`$HOME/mining-400-start-*.log`，以及 `out/mining_400/launcher-*.log`。

Mining 按 `selected_800_mining_400.jsonl` 精确匹配到源目录 `cybergym-arvo-1031-clean-level0-20261014`。完整 400 条清单仍保留在 `data/task_ids/pj-mining-400.txt`；各分组只使用自身的 `selected_tasks.txt` 与 `tasks.json`。分盘清单见 `mining-partition-plan.json`。

可选：只生成当前分组的执行计划，不加载镜像或调用模型：

```bash
sudo bash run_mining_400.sh --dry-run
```

计划写入 `out/<分组目录名>_dry_run/`，不计作实际交互验证结果。

## 4. 如何判断验证成功

两批都采用相同判断口径：

- **有交互**：agent 日志中出现实际模型文字回复或工具调用；仅有初始化信息不算。
- **达到轮数上限**：默认 `MAX_TURNS=1`，出现 `error_max_turns` 可能是正常触发上限，需结合前面的交互记录判断。
- **解题成功**：需要查看验证器结果或 reward，不能仅凭“有交互”或进程返回码判断。
- **启动异常**：检查镜像加载、服务初始化、Harbor 及 agent 日志，定位具体阶段。

每批结果目录结构相同：


| 文件或目录                        | 内容                      |
| ---------------------------- | ----------------------- |
| `launcher-日期时间.log`          | 启动器及批次日志                |
| `pre_run/controller.log`     | controller 日志           |
| `pre_run/llm_proxy.log`      | LLM proxy 日志            |
| `pj_user_batch_summary.json` | 批次返回状态汇总，不等同于 reward 汇总 |
| `<任务名>/result.json`          | 单题执行结果                  |
| `<任务名>/logs/load_image.log`  | 镜像加载日志，具体以适配器输出为准       |
| `<任务名>/harbor-jobs/`         | Harbor 日志、agent 交互和评测产物 |


`launcher` 日志在部分前置检查后才开始记录；上述 `tee` 命令还能保存最开始的环境检查及 Docker 权限报错。

## 5. 参数、增加轮数与重跑


| 参数          | 默认值       | 说明                                                    |
| ----------- | --------- | ----------------------------------------------------- |
| `MODEL`     | `glm-5.2` | 模型                                                    |
| `WORKERS`   | `20`      | 每批并发任务数                                               |
| `MAX_TURNS` | `1`       | 每题 agent 轮数上限                                         |
| `TIMEOUT`   | `2700`    | 每题超时，单位为秒                                             |
| `BUDGET`    | `20`      | 每题代理预算，单位为美元                                          |
| `RESUME`    | `1`       | 跳过已有 `result.json` 的任务，包括失败任务                         |
| `OUT_DIR`   | 按批次区分     | Exploit 为 `out/exploit_400`；Mining 为 `out/mining_400` |


通过 sudo 启动时，用 `sudo env 参数=值` 传参。增加轮数后应使用新输出目录，避免跳过已有的一轮验证结果。

以下命令在交付目录执行，分别用于两批的较长评测，不要作为一组同时启动：

```bash
# Exploit：每题最多 300 轮
sudo env WORKERS=20 MAX_TURNS=300 TIMEOUT=5400 \
  OUT_DIR="$PWD/out/exploit_400_full" \
  bash "$HOME/exploitgym-offline/start.sh"
```

```bash
# Mining：每题最多 300 轮
sudo env WORKERS=20 MAX_TURNS=300 TIMEOUT=5400 \
  OUT_DIR="$PWD/out/mining_400_full" \
  bash run_mining_400.sh
```

重新执行同一命令、使用同一输出目录时，默认跳过所有已有 `result.json` 的任务，包括失败任务。全部重跑建议使用新 `OUT_DIR`；也可设置 `RESUME=0`，但原目录中的旧结果可能被覆盖。

## 每条任务结束后的 Docker 清理

默认在每次 Harbor 调用结束后，只清理本次 trial 对应 Compose project 的容器、网络及卷。镜像按本任务加载记录或 task-scoped 清单处理，不使用全局 prune，也不强制删除镜像。其他运行任务、初始化/停止容器仍引用的镜像保留或延后处理，公共镜像、评测开始前已存在的镜像（包括预先加载的任务镜像），以及无法安全归属的镜像均保留；只清理本次新加载且没有其他消费者的任务镜像。原始 tar 包不删除。

每条任务的清理记录在输出目录 `任务名/logs/docker_cleanup.json`。共享镜像的延后删除可能记录在最后一个释放该镜像的任务日志里。旧版已启动的进程不会自动获得本次修改；新启动任务生效。