#!/usr/bin/env bash
# =============================================================================
# 一键清理 RL 训练/推理残留进程 + Ray 集群
# 在训练机和推理机上都跑一遍即可。幂等，可重复执行。
# =============================================================================
set +e

echo "==================== RL 清理开始 ===================="
echo "主机: $(hostname)  IP: $(hostname -I | awk '{print $1}')"

# ---------- 1. 停 Ray 集群 ----------
echo "[1/5] 停 Ray ..."
ray stop --force 2>/dev/null
# 杀残留 Ray 守护进程
pkill -9 -f "ray::" 2>/dev/null
pkill -9 ray raylet gcs_server plasma_store monitor 2>/dev/null
sleep 1

# ---------- 2. 杀 SGLang 推理引擎 ----------
echo "[2/5] 杀 SGLang ..."
pkill -9 -f sglang 2>/dev/null
pkill -9 -f "sglang_router\|srt_server\|sglang.srt" 2>/dev/null

# ---------- 3. 杀训练/调度相关 Python ----------
echo "[3/5] 杀训练/调度进程 ..."

# 强制杀掉所有相关 Python 进程（防止残留占显存/端口）
pkill -9 -f "sglang" 2>/dev/null
pkill -9 -f "slime" 2>/dev/null
pkill -9 -f "ray" 2>/dev/null
pkill -9 -f "python" 2>/dev/null
sleep 3

# buffer server / simulation worker / launcher
pkill -9 -f "buffer_server.py" 2>/dev/null
pkill -9 -f "simulation_worker" 2>/dev/null
pkill -9 -f "launcher.py" 2>/dev/null
# slime generator / llm proxy / gateway (含 eval 用的 -m gateway)
pkill -9 -f "slime_generator.py" 2>/dev/null
pkill -9 -f "llm_proxy.py" 2>/dev/null
pkill -9 -f "gateway_autostart" 2>/dev/null
pkill -9 -f "gateway" 2>/dev/null
pkill -9 -f "python3 -m gateway" 2>/dev/null
pkill -9 -f "start_eval_gateway" 2>/dev/null
# eval 残留 (patcheval/harbor eval 用的 launcher.py --resume)
pkill -9 -f "patcheval" 2>/dev/null
pkill -9 -f "run_eval" 2>/dev/null
# 启动脚本本身
pkill -9 -f "run_buffer_server" 2>/dev/null
pkill -9 -f "run_slime_generator" 2>/dev/null
# Megatron 训练入口
pkill -9 -f "train.py" 2>/dev/null
pkill -9 -f "megatron" 2>/dev/null
# torch_memory_saver 残留
pkill -9 -f "torch_memory_saver" 2>/dev/null

# ---------- 4. 杀占用训练端口的进程 ----------
echo "[4/5] 释放端口 8000/18000/18889/18890/6379/8265 ..."
for port in 8000 18000 18889 18890 6379 8265; do
  fuser -k -9 ${port}/tcp 2>/dev/null
  # ss 拿 PID 兜底
  PIDS=$(ss -ltnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | sort -u)
  for p in $PIDS; do
    kill -9 "$p" 2>/dev/null
  done
done

sleep 2

# ---------- 5. 验证 ----------
echo "[5/5] 验证 ..."

echo "--- 残留 RL 相关 Python 进程 ---"
LEFT=$(ps -eo pid,cmd | grep -E 'patcheval|buffer_server|simulation_worker|slime_generator|llm_proxy|gateway_autostart|gateway|sglang|run_buffer_server|run_slime_generator|launcher.py|start_eval_gateway' | grep -v grep)
if [ -n "$LEFT" ]; then
  echo "$LEFT"
  echo "  ⚠ 仍有残留，手动 kill -9 上面列出的 PID"
else
  echo "  ✅ 无残留进程"
fi

echo "--- 端口占用 ---"
PORTS=$(ss -ltnp 2>/dev/null | grep -E ':8000 |:18000 |:18889 |:18890 |:6379 |:8265 ')
if [ -n "$PORTS" ]; then
  echo "$PORTS"
  echo "  ⚠ 端口仍被占用"
else
  echo "  ✅ 端口已全部释放"
fi

echo "--- GPU 进程 ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
else
  echo "  (无 nvidia-smi)"
fi

echo "--- Ray 状态 ---"
ray status 2>&1 | head -3 || echo "  (ray 已停)"

echo "==================== 清理完成 ===================="
