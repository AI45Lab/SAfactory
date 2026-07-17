# -------------------------------------------
# AIEvobox (rollout) Settings
# -------------------------------------------
export AIEVOBOX_ROOT=/root/Safactory
export STORAGE_TYPE=sqlite
export AIEVOBOX_DB_URL=sqlite:///${AIEVOBOX_ROOT}/rl/examples/geo3k_vl/geo3k_vl.db
export AIEVOBOX_MAX_STEPS=10
export AIEVOBOX_MESSAGE_CUT=0
# v2: AGENT_CONFIG 指定单个 agent yaml；AGENT_START_CONFIG 指定容器启动 yaml。
# 若不设 AGENT_START_CONFIG，buffer_server 会从 AGENT_CONFIG 自动推导
# (<name>_config.yaml -> <name>_start.yaml)。
export AIEVOBOX_AGENT_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_config.yaml
export AIEVOBOX_AGENT_START_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_start.yaml
export AIEVOBOX_ENABLE_EVALUATION=1
export AIEVOBOX_EVALUATION_CONFIG=${AIEVOBOX_ROOT}/env/geo3k/geo3k_rule_eval.yaml
# AGENT_ROOT 指定读取目录下所有子目录的环境（与 AGENT_CONFIG 二选一）
# export AIEVOBOX_AGENT_ROOT=${AIEVOBOX_ROOT}/env
# v2 docker 模式下每个 pool slot 是一个容器，pool_size 不宜过大。
export AIEVOBOX_POOL_SIZE=16
export AIEVOBOX_LLM_MAX_CONCURRENCY=$AIEVOBOX_POOL_SIZE
export AIEVOBOX_LLM_PROXY_WORKERS=$AIEVOBOX_POOL_SIZE
export AIEVOBOX_LLM_STARTUP_JITTER_S=0
export AIEVOBOX_TRAININFO_WORKERS=$AIEVOBOX_POOL_SIZE
export STORAGE_TYPE=sqlite
export AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE=256
export AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S=0.01



# -------------------------------------------
# RL Settings
# -------------------------------------------
export RL_GROUP_SIZE=8
export RL_EPOCH=1000
export RL_OFF_BY_N=0

# no use, will be removed
export RL_MODEL=model
export RL_API_KEY=openai_api_key


# -------------------------------------------
# Buffer Server Settings (run_buffer_server.sh)
# -------------------------------------------
# Buffer Server 由 run_buffer_server.sh 启动，负责管理 rollout 数据并拉起 AIEvoBox launcher。
# HOST 是其他服务连接 Buffer Server 用的地址（服务本身始终监听 0.0.0.0）。
# Slime Generator 通过此地址调用 /get_rollout_data 和 /start_rollout。
# 如果 Buffer Server 和 Slime Generator 运行在不同机器上，改为 Buffer Server 所在机器的 IP。
export BUFFER_SERVER_HOST=127.0.0.1
export BUFFER_SERVER_PORT=18889

# -------------------------------------------
# LLM Proxy Settings (hosted in-process by Slime Generator)
# -------------------------------------------
# LLM Proxy 由 Slime Generator (run_slime_generator*.sh) 在进程内启动，提供 /v1/chat/completions。
# 链路：docker(env) -> gateway -> llm_proxy -> sglang。gateway 必须在前，
# 它把 session id 通过 X-Safactory-Session-Id header 转发给 llm_proxy。
# HOST 是 gateway 连接 llm_proxy 用的地址（llm_proxy 本身始终监听 0.0.0.0）。
export LLM_PROXY_HOST=127.0.0.1
export LLM_PROXY_PORT=18890
export LLM_MAX_LENGTH=5120
export LLM_TEMPERATURE=1.0
export LLM_PROXY_ENABLE_CONSOLE_LOG=0

# -------------------------------------------
# Gateway Settings (must front the llm_proxy)
# -------------------------------------------
# buffer_server 会自动拉起 gateway（生成配置：route 指向 llm_proxy、共用 RL DB、
# max_steps=-1），并等 /readyz。runner 只打 gateway 的 session 端点。
# 想用外部/手动 gateway 时设 AIEVOBOX_GATEWAY_AUTOSTART=0 关掉自动拉起。
export AIEVOBOX_GATEWAY_HOST=127.0.0.1
export AIEVOBOX_GATEWAY_PORT=8000
export AIEVOBOX_GATEWAY_BASE_URL=http://${AIEVOBOX_GATEWAY_HOST}:${AIEVOBOX_GATEWAY_PORT}/v1/sessions
# export AIEVOBOX_GATEWAY_AUTOSTART=0

# -------------------------------------------
# Slime Training Settings (reference RL values)
# -------------------------------------------
export SLIME_ROLLBUF_RESTART_TRAINING=True
export SLIME_N_SAMPLES_PER_PROMPT=$RL_GROUP_SIZE
export SLIME_GLOBAL_BATCH_SIZE=512
export SLIME_ROLLOUT_BATCH_SIZE=$((SLIME_GLOBAL_BATCH_SIZE / RL_GROUP_SIZE))
