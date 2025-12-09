# OSGym 使用指南

OSGym 是将 [OSWorld](https://github.com/xlang-ai/OSWorld) / [RiOSWorld](https://github.com/yjyddq/RiOSWorld) 的运行环境和桌面任务封装进 AIEvoBox 的环境，便于训练和评测桌面代理/强化学习模型。下面给出最小配置与运行说明。

## 1. 依赖安装
- 进入 AIEvoBox 根目录执行：`pip install -r requirements.txt`。
- 如只在 osgym 下开发，可在本目录执行：`pip install -r requirements.txt`。

## 2. 资源与配置
- VM 镜像：仓库不包含大文件 `docker_vm_data/Ubuntu.qcow2`。运行时会通过 `desktop_env.providers.docker.manager.DockerVMManager` 自动从 HuggingFace 下载并解压（[默认链接](https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip)），放到 `docker_vm_data/`。若自动下载失败，可手动放置同名文件后重试；也可在 `os_env.py` 中自定义路径。
- 任务配置：默认从 `os_config.yaml` 中的 `task_config_path` 指向的 JSON 加载（示例：`evaluation_risk_examples/test_popup.json`）。可在 YAML 中替换为你的任务集，或在初始化时传参 `task_config_path` 覆盖。
- 关键参数（`os_config.yaml` -> `env_params` 或构造函数传参）：
  - `provider_name`: 后端提供商，默认 `docker`。
  - `observation_type`: 可选 `screenshot` / `a11y_tree` / `screenshot_a11y_tree` / `som`。
  - `action_space`: `pyautogui`（代码形式动作）或 `computer_13`（结构化动作）。
  - `headless`: 无头运行开关，默认 true。
  - `screen_width` / `screen_height`: 屏幕分辨率。
  - `cache_dir`: 辅助缓存目录。
  - `result_dir`: 运行结果目录，保存视频（可选）、截图（可选）、评估得分和LLM动作历史记录。

## 3. 运行示例
验证环境是否搭建成功：

```bash
cd AIEvoBox/env/osgym
python test_osgym.py
```

运行脚本，需要进行`LLM`相关配置：

```bash
cd AIEvoBox
bash example/run_os_env.sh
```
