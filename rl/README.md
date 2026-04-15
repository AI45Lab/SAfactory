# RL 训练

## 运行方式

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入实际配置
```

### 2. 启动 Slime 训练（终端 1）
启动训练前必须按照官方教程安装好slime

方式1: install from scratch的[sh脚本](https://github.com/THUDM/slime/blob/main/build_conda.sh)

方式2: 使用slime官方[docker](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md#pull-and-start-docker-container)


启动训练进程：
```bash
cd /root/AIEvoBox/rl

# 文本模型
./run_slime_generator.sh

# Math500 文本模型（OPD + sglang teacher）
./examples/math500/run_slime_generator_opd_sglang.sh

# VL 模型
./run_slime_generator_vl.sh
```

### 3. 启动 Buffer Server（终端 2）
安装本项目的主要依赖即可，详见根目录的readme，不需要安装slime。

```bash
cd /root/AIEvoBox/rl
./run_buffer_server.sh
```

启动后，Buffer Server 会自动拉起 AIEvoBox Runner，开始rollout 数据。


### 注意终端1和终端2可以运行在不同的机器上，只要能够通信即可。

## 关键配置说明

### 环境配置（二选一）

- `AIEVOBOX_ENV_CONFIG` — 指定单个 yaml 文件，适合调试或运行单个环境
- `AIEVOBOX_ENV_ROOT` — 指定环境根目录，自动加载所有子目录下的 yaml

### 训练参数

| 变量 | 说明 |
|------|------|
| `RL_GROUP_SIZE` | 每个 prompt 的采样数（对应 Slime 的 n_samples_per_prompt） |
| `RL_EPOCH` | 训练轮数 |
| `RL_OFF_BY_N` | 允许的最大权重版本差（0=只用当前版本数据） |
| `SLIME_GLOBAL_BATCH_SIZE` | 全局 batch size |

### 服务端口

| 服务 | 默认端口 | 环境变量 |
|------|----------|----------|
| Buffer Server | 18889 | `BUFFER_SERVER_PORT` |
| LLM Proxy | 18890 | `LLM_PROXY_PORT` |

完整配置见 `.env.example`。Slime 训练脚本中的更多参数请参考 Slime 文档。
