# 运行 OSGym 环境评测示例脚本

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置 PYTHONPATH 以便找到项目模块
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行评测
python "$SCRIPT_DIR/base_eval.py" \
  --env-config-yaml "$PROJECT_ROOT/env/osgym/os_config.yaml" \
  --max-workers 1 \
  --max-steps 10000 \
  --visual-save-path "$PROJECT_ROOT/visualize/os_env_test" \
  --agent-api-key "" \
  --agent-base-url "" \
  --agent-model "gpt-5.1" \
  --agent-temperature 0.0
