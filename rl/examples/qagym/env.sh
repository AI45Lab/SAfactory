# -------------------------------------------
# AIEvobox (rollout) Settings
# -------------------------------------------
export RAY_BIN="${RAY_BIN:-/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin/ray}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin/python3.12}"
export SLIME_ENV_BIN="${SLIME_ENV_BIN:-/mnt/shared-storage-user/evobox-share/yinzhenyun/slime-env-0.2.3/bin}"
# Keep this env bin first so plain `python3`/`ray` resolve to the same Python runtime.
case ":${PATH}:" in
  *":${SLIME_ENV_BIN}:"*) ;;
  *) export PATH="${SLIME_ENV_BIN}:${PATH}" ;;
esac

# Some subprocesses call `python3`/`ray` directly. Point both to the slime runtime.
export AIEVOBOX_SHIM_BIN="${AIEVOBOX_SHIM_BIN:-/tmp/aievobox-bin-shim}"
mkdir -p "${AIEVOBOX_SHIM_BIN}"
ln -sf "${PYTHON_BIN}" "${AIEVOBOX_SHIM_BIN}/python3"
ln -sf "${RAY_BIN}" "${AIEVOBOX_SHIM_BIN}/ray"
case ":${PATH}:" in
  *":${AIEVOBOX_SHIM_BIN}:"*) ;;
  *) export PATH="${AIEVOBOX_SHIM_BIN}:${PATH}" ;;
esac

_QAGYM_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_QAGYM_DEFAULT_AIEVOBOX_ROOT="$(cd -- "${_QAGYM_ENV_SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
export AIEVOBOX_ROOT=${AIEVOBOX_ROOT:-${_QAGYM_DEFAULT_AIEVOBOX_ROOT}}
export STORAGE_TYPE=${STORAGE_TYPE:-sqlite}
export AIEVOBOX_DB_URL=${AIEVOBOX_DB_URL:-sqlite:///${AIEVOBOX_ROOT}/rl/examples/qagym/qagym.db}
export AIEVOBOX_MAX_STEPS=${AIEVOBOX_MAX_STEPS:-3}
export AIEVOBOX_MESSAGE_CUT=${AIEVOBOX_MESSAGE_CUT:-0}
# ENV_CONFIG 指定单个 yaml 文件
export AIEVOBOX_ENV_CONFIG=${AIEVOBOX_ENV_CONFIG:-${AIEVOBOX_ROOT}/env/qagym/qa_env.yaml}
# ENV_ROOT 指定读取目录下所有子目录的环境
# export AIEVOBOX_ENV_ROOT=${AIEVOBOX_ROOT}/env
export AIEVOBOX_POOL_SIZE=${AIEVOBOX_POOL_SIZE:-512}

# OpenRT imports some attack modules that expect these variables to exist at import time.
# The actual qagym models still use api_key/base_url from env/qagym/qa_env.yaml.
export OPENAI_API_KEY=${OPENAI_API_KEY:-XXX}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-XXX}


# -------------------------------------------
# RL Settings
# -------------------------------------------
export RL_GROUP_SIZE=${RL_GROUP_SIZE:-8}
export RL_EPOCH=${RL_EPOCH:-1000}
export RL_OFF_BY_N=${RL_OFF_BY_N:-0}

# no use, will be removed
export RL_MODEL=${RL_MODEL:-model}
export RL_API_KEY=${RL_API_KEY:-openai_api_key}

#
export DAPO_filter="${DAPO_filter:-false}"

# -------------------------------------------
# Buffer Server Settings (run_buffer_server.sh)
# -------------------------------------------
# Buffer Server 由 run_buffer_server.sh 启动，负责管理 rollout 数据并拉起 AIEvoBox launcher。
# HOST 是其他服务连接 Buffer Server 用的地址（服务本身始终监听 0.0.0.0）。
# Slime Generator 通过此地址调用 /get_rollout_data 和 /start_rollout。
# 如果 Buffer Server 和 Slime Generator 运行在不同机器上，改为 Buffer Server 所在机器的 IP。
export BUFFER_SERVER_HOST=${BUFFER_SERVER_HOST:-127.0.0.1}
export BUFFER_SERVER_PORT=${BUFFER_SERVER_PORT:-18889}


# -------------------------------------------
# LLM Proxy Settings (hosted in-process by Slime Generator)
# -------------------------------------------
# LLM Proxy 由 Slime Generator (run_slime_generator.sh) 在进程内启动，提供 /v1 chat completions 接口。
# HOST 是其他服务连接 LLM Proxy 用的地址（服务本身始终监听 0.0.0.0）。
# AIEvoBox launcher（由 Buffer Server 拉起）通过此地址调用 LLM。
# 如果 Buffer Server 和 Slime Generator 运行在不同机器上，改为 Slime Generator 所在机器的 IP。
export LLM_PROXY_HOST=${LLM_PROXY_HOST:-127.0.0.1}
export LLM_PROXY_PORT=${LLM_PROXY_PORT:-18890}
export LLM_MAX_LENGTH=${LLM_MAX_LENGTH:-4096}
export LLM_TEMPERATURE=${LLM_TEMPERATURE:-1.0}


# -------------------------------------------
# Slime Training Settings (reference RL values)
# -------------------------------------------
export SLIME_ROLLBUF_RESTART_TRAINING=${SLIME_ROLLBUF_RESTART_TRAINING:-True}
export SLIME_N_SAMPLES_PER_PROMPT=${SLIME_N_SAMPLES_PER_PROMPT:-$RL_GROUP_SIZE}
export SLIME_GLOBAL_BATCH_SIZE=${SLIME_GLOBAL_BATCH_SIZE:-512}
export SLIME_ROLLOUT_BATCH_SIZE=${SLIME_ROLLOUT_BATCH_SIZE:-$((SLIME_GLOBAL_BATCH_SIZE / RL_GROUP_SIZE))}
