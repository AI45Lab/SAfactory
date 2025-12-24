#!/usr/bin/env bash
set -euo pipefail
# 如果你想调试，看清楚执行了什么命令，可以打开：
# set -x

echo "[run.sh] 启动 Xvfb..."

# 1. 建议用 Xvfb 的绝对路径（避免 PATH 问题）
# 可以在终端先跑 which Xvfb，比如是 /usr/bin/Xvfb
XVFB_BIN=/usr/bin/Xvfb

# 2. 启动虚拟显示，日志写到文件里方便调试
$XVFB_BIN :1 -screen 0 1024x768x24 > /tmp/xvfb.log 2>&1 &
XVFB_PID=$!

echo "[run.sh] Xvfb PID = $XVFB_PID"

# 3. 导出 DISPLAY 环境变量，后续程序才能找到这个虚拟屏幕
export DISPLAY=:1

# 可选：稍微等一下，确保 Xvfb 真正起来了
sleep 2

set +u

# 4. 初始化 conda（这一步在脚本里不会自动做，一定要手动）
echo "[run.sh] 初始化 conda..."
source /home/ray/anaconda3/etc/profile.d/conda.sh

echo "[run.sh] 激活环境 /root/envs/embench1..."
conda activate /root/envs/embench1

set -u

# 5. 启动你的 python 程序
echo "[run.sh] 启动 python app.py..."
python /workspace/AIEvoBox/app.py

# 如果你希望脚本退出前等 Xvfb 一起结束，可以：
# wait $XVFB_PID
