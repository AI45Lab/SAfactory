# OSGym 使用指南

OSGym 是将 [OSWorld](https://github.com/xlang-ai/OSWorld) / [RiOSWorld](https://github.com/yjyddq/RiOSWorld) 的运行环境和桌面任务封装进 AIEvoBox 的环境，便于训练和评测桌面代理/强化学习模型。

## 1. 目录结构

```
env/osgym/
├── __init__.py                      # 导出 OSGym 类
├── os_env.py                        # 主环境类 (协调器)
├── os_config.yaml                   # 环境配置文件
├── test_osgym.py                    # 测试脚本
│
├── core/                            # 核心功能模块
│   ├── task_manager.py              # 任务加载和管理
│   ├── action_parser.py             # 动作解析 (WAIT/DONE/FAIL/代码块)
│   ├── observation_processor.py     # 观察处理和攻击覆盖
│   ├── result_persistence.py        # 截图/轨迹/结果保存
│   └── prompt_builder.py            # 提示构建
│
├── evaluation/                      # 评估模块
│   └── evaluator.py                 # 任务评估 (OSWorld + RiOSWorld)
│
├── desktop_env/                     # 桌面环境核心 (来自 OSWorld)
│   ├── desktop_env.py               # DesktopEnv 类
│   ├── providers/                   # VM 提供商
│   ├── controllers/                 # 控制器 (python/setup)
│   ├── evaluators/                  # 评估器 (metrics/getters)
│   └── server/                      # X11 桌面服务器
│
├── mm_agents/                       # 多模态代理 (来自 OSWorld)
│   ├── agent.py                     # LLM 代理接口
│   ├── prompts.py                   # 系统提示
│   └── prompt_helper.py             # 提示工具
│
├── env_risk_utils/                  # 风险评估工具 (来自 RiOSWorld)
│   ├── attack.py                    # 攻击场景生成
│   └── ...                          # 钓鱼/弹窗相关
│
├── datasets/                       # JSONL 数据集
│   ├── osworld_cases.jsonl         # OSWorld 任务配置
│   └── riosworld_cases.jsonl       # RiOSWorld 任务配置
```

## 2. 依赖安装

```bash
# 在 AIEvoBox 根目录
pip install -r requirements.txt

# 并在 osgym 目录
cd env/osgym && pip install -r requirements.txt
```

## 3. VM 镜像

仓库不包含大文件 `docker_vm_data/Ubuntu.qcow2`。运行时会自动从 HuggingFace 下载：
[下载链接](https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip)

若自动下载失败，可手动下载并解压到 `docker_vm_data/` 目录。

## 4. 配置参数

通过 `os_config.yaml` 或构造函数传参配置：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | 任务数据集路径 | `datasets/osworld_cases.jsonl` |
| `benchmark_type` | 基准类型 | `osworld` / `riosworld` |
| `provider_name` | 后端提供商 | `docker` |
| `observation_type` | 观察类型 | `screenshot_a11y_tree` |
| `action_space` | 动作空间 | `pyautogui` |
| `screen_width/height` | 屏幕分辨率 | `1920x1080` |
| `max_steps` | 最大步数 | `15` |
| `result_dir` | 结果目录 | `null` |

## 5. 运行示例

**验证环境：**
```bash
cd AIEvoBox/env/osgym
python test_osgym.py
```

**运行评测：**
```bash
cd AIEvoBox
bash examples/run_os_env.sh
```

## 6. 评估逻辑

- **OSWorld**: 使用 `evaluate()` 进行任务完成度评估
- **RiOSWorld**: 使用 `evaluate()` + `evaluate_step()` 进行风险评估

评估配置来自任务 JSON 文件中的 `evaluator` 和 `risk_evaluator` 字段。
