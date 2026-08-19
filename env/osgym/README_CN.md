# OSGym 使用指南

OSGym 是将 [OSWorld](https://github.com/xlang-ai/OSWorld) / [RiOSWorld](https://github.com/yjyddq/RiOSWorld) 的运行环境封装进 AIEvoBox 的环境，便于训练和评测桌面代理/强化学习模型。

## 1. 依赖安装

```bash
# 在 AIEvoBox 根目录
pip install -r requirements.txt

# 并在 osgym 目录
cd env/osgym && pip install -r requirements.txt
```

## 2. VM 镜像

仓库不包含大文件 `docker_vm_data/Ubuntu.qcow2`。运行时会自动从 HuggingFace 下载：
[下载链接](https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip)

若自动下载失败，可手动下载并解压到 `env/osgym/docker_vm_data/` 目录。

也可以在配置中显式指定镜像路径，例如 `vm_path: "/abs/path/Ubuntu.qcow2"`。
若填写相对路径，则相对于 `env/osgym` 目录解析。

## 3. 配置参数

通过 `os_config.yaml` 或构造函数传参配置：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset` | 任务数据集路径（支持相对路径，相对于 `env/osgym` 目录） | `datasets/cases.jsonl` |
| `eval_mode` | 评估模式 (`standard` / `safety`) | `standard` |
| `provider_name` | 后端提供商；仅支持 `docker` | `docker` |
| `vm_path` | 显式指定 VM 镜像路径；支持相对 `env/osgym` 的路径 | `None` |
| `prompt_observation_type` | 提示词模态 (`screenshot`, `a11y_tree`, `screenshot_a11y_tree`) | `screenshot` |
| `prompt_format` | 提示词协议格式 (`kimi`, `qwen3vl`, `qwen35`) | `qwen35` |
| `action_space` | 动作空间 | `pyautogui` |
| `screen_width/height` | 屏幕分辨率 | `1920x1080` |
| `download_cache_dir` | Pod 内任务初始化下载的共享缓存目录 | `/tmp/osgym-download-cache` |
| `reset_timeout_seconds` | OSGym reset/setup 总截止时间，应小于 Manager HTTP 超时 | `270.0` |
| `setup_connect_timeout_seconds` | setup HTTP 请求的连接超时 | `10.0` |
| `max_steps` | 每个任务的最大允许步数 | `30` |
| `repeated_click_distance_threshold` | 连续 click 坐标距离小于该像素阈值时视为同一点击 | `10.0` |
| `repeated_click_limit` | 连续重复 click 达到该次数后截断任务；设为 `0` 可关闭 | `2` |
| `repeated_action_signature_limit` | 连续相同低层 pyautogui 动作签名（`moveTo`, `scroll`, `dragTo`, `typewrite`, `press`, `hotkey`, `keyDown`, `keyUp`）达到该次数后触发 stuck 处理；设为 `0` 可关闭 | `3` |

OSGym 自动决定是否采集 accessibility tree：a11y 提示词模式会开启，safety 的 `popup` / `induced_text` 任务也会开启。

safety 任务使用的 `6002`–`6008` Web 服务会在同一 Pod 的 OSGym actor
之间共享。OSGym 会在使用前校验服务身份，并在最后一个 actor 结束后才关闭服务。

## 4. 运行示例

### 本地运行:

```bash
python launcher.py \
    --mode local \
    --env-config env/osgym/os_config.yaml \
    --llm-base-url your_url \
    --llm-api-key your_key \
    --llm-model model_name \
    --pool-size 2
```

### 容器运行

```bash
docker pull safactory/osworld:v0.1.0
sudo docker run --privileged -d --name os_env safactory/osworld:v0.1.0 tail -f /dev/null

# 启动第一个容器终端
docker exec -it os_env /bin/bash
dockerd

# 启动第二个容器终端
docker exec -it os_env /bin/bash
docker pull happysixd/osworld-docker:latest

python launcher.py \
  --mode local \
  --env-config env/osgym/os_config.yaml \
  --llm-base-url your_url \
  --llm-api-key your_key \
  --llm-model model_name \
  --pool-size 2
```
