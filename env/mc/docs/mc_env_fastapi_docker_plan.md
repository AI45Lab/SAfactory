## 目标
在官方 Ray 基础镜像之上构建一个 Docker 镜像，镜像启动后即可运行 FastAPI 服务并托管 `MCGym` 环境。整体需求包括：
- 完整安装 `env/mc` 环境（含 `MineStudio` 的 `pip install -e .`）。
- 解除 `malmo.py`、`entry.py`、`utils/temp.py` 中硬编码的本地绝对路径。
- 在镜像内妥善放置/挂载 `offline-mcprec-6.13.jar`。
- FastAPI 提供 `reset / step / render / close` 等接口。

以下文档分步骤说明设计与注意事项，方便逐步讨论和调整。

## 1. 运行与测试（先上手，后了解细节）
- 构建（离线 wheel 可选）：  
  ```
  docker build --no-cache -f env/mc/Dockerfile.mc-env -t mc-fastapi \
    --build-arg USE_LOCAL_WHEELS=true \
    --build-arg http_proxy="$http_proxy" \
    --build-arg https_proxy="$https_proxy" \
    --build-arg no_proxy="$no_proxy" \
    .
  ```
- 运行（CPU/headless 默认）：  
  `docker run --rm -p 8000:8000 mc-fastapi`。若 8000 被占用，可改 `-p 8010:8000` 或先停 `nginx`。  
- 服务测试（将 `localhost` 替换为实际 IP 可供他人访问；所有接口均为 JSON）：  
  1. 健康检查  
     ```bash
     curl http://localhost:8000/healthz
     ```  
     - 返回：`env_ready`（是否已 reset）、`active_config`、`jar_path` 等。  
  2. 初始化环境（参数可选）  
     ```bash
     curl -X POST http://localhost:8000/reset \
       -H 'Content-Type: application/json' \
       -d '{"env_config":"env/mc/config/collect/collect_wood.yaml","seed":123,"force_recreate":true}'
     ```  
     - `env_config`：具体任务 YAML，相对 `PROJECT_ROOT`。  
     - `seed`：Minecraft 世界种子。  
     - `force_recreate`：若 `true`，强制关闭并重建环境实例。缺省为 `false`。  
  3. 发送动作（支持字符串或字典；示例为字符串，便于 LLM 输出）  
     ```bash
     curl -X POST http://localhost:8000/step \
       -H 'Content-Type: application/json' \
       -d '{"action":"[{\"action\":\"forward\",\"camera\":{\"yaw\":5.0,\"pitch\":-3.0},\"attack\":1,\"jump\":0}]"}'
     ```  
     - 也可传递字典：`{"action":{"buttons":{"attack":1,"forward":1}, "camera":{"yaw":5,"pitch":-3}}}`。  
     - 返回：`observation`（包含 `image` 等）、`reward`、`terminated`、`truncated`、`info`。  
  4. 获取渲染帧  
     ```bash
     curl -X POST http://localhost:8000/render \
       | jq -r '.image_base64' | base64 -d > render_frame.png
     ```  
     - 若 `last_obs` 为空（未 reset），接口会返回 400。  
  5. 关闭环境  
     ```bash
     curl -X POST http://localhost:8000/close
     ```  
  6. 远程保存示例（Python）  
     ```python
     import base64, requests, pathlib
     resp = requests.post("http://100.101.28.243:8000/render")
     resp.raise_for_status()
     pathlib.Path("render_frame.png").write_bytes(
         base64.b64decode(resp.json()["image_base64"])
     )
     ```
- 烟雾测试（容器内）：`python -c "from env.mc.mc_env import MCGym; env=MCGym(); env.reset(); env.close()"`。

## 2. 基础镜像选择
- 已指定使用 `registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base`（约 10.3 GB，8 周前版本），该镜像自带 Ray 运行时且定位为 CPU 场景。  
- 仍保留 `ARG RAY_BASE_IMAGE` 以便未来替换，但默认值锁定为上述镜像。  
- 镜像内默认使用 `/workspace/AIEvoBox` 作为项目根（`PROJECT_ROOT` 可重写）。  
- 目前不考虑 GPU/CUDA 依赖，Dockerfile 与运行说明均以 CPU-only 角度编写。

## 2. 系统依赖
- 为 `opencv-python`、`pyglet`、`Pyro4`、渲染/视频组件等安装以下 APT 包：  
  `xvfb openjdk-8-jre-headless ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libvulkan1 libsdl2-dev mesa-utils`。  
- 另需 `build-essential cmake git curl python3-dev pkg-config` 支撑编译型依赖。  
- 设置 locale（如 `en_US.UTF-8`）和时区，避免 Ray 运行时警告。

## 3. Python 依赖与 MineStudio 安装
1. 将 `env/mc/requirements.txt` 拷贝到镜像（如 `/tmp/requirements_mc.txt`），执行 `pip install --no-cache-dir -r /tmp/requirements_mc.txt`。  
2. 使用 `ARG PROJECT_ROOT=/workspace/AIEvoBox`，在镜像内创建目录并 `COPY . ${PROJECT_ROOT}`。  
3. 在容器内执行 `pip install --no-cache-dir -e ${PROJECT_ROOT}/env/mc/MineStudio`，保证可编辑安装生效。  
4. 若 `requirements.txt` 后续增加依赖，优先更新该文件，避免在 Dockerfile 内重复 `pip install`。  
5. 通过 `ENV PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}` 让 FastAPI 服务能 import 项目模块。

## 4. 绝对路径问题处理
`minestudio/simulator/minerl/env/malmo.py`、`minestudio/simulator/entry.py`、`minestudio/utils/temp.py` 已改为优先读取 `MINECRAFT_JAR_PATH`/`MINESTUDIO_DIR`，并以 `PROJECT_ROOT`（默认 `/workspace/AIEvoBox`）为兜底，避免硬编码宿主机路径。  
- Jar 默认放在 `${PROJECT_ROOT}/env/mc/build/offline-mcprec-6.13.jar`，也可通过环境变量覆盖。  
- 保持 `ENV PYTHONPATH=${PROJECT_ROOT}` 以便 import 项目模块。

## 5. FastAPI 服务设计
- 位置建议：`env/mc/service/app.py` 与 `env/mc/service/env_manager.py`。  
- 核心组件：
  - `EnvManager`：懒加载 `MCGym`，用 `asyncio.Lock`/`threading.Lock` 保证单例。  
  - `@app.post("/reset")`：返回 `obs`、`info`。  
  - `@app.post("/step")`：接收 action（JSON），返回 `obs/reward/terminated/truncated/info`。  
  - `@app.post("/render")`：调用 `env.render()`，以 PNG Base64 或 bytes。  
  - `@app.post("/close")`：释放资源。  
  - `@app.get("/healthz")`：快速检查服务与 jar 可用性。  
- 考虑长步长操作：使用后台任务或同步阻塞方式，保持 `uvicorn --workers 1 --host 0.0.0.0 --port 8000`。如需多 worker，需要在 `EnvManager` 中做进程级隔离。

## 6. Dockerfile 结构（初稿）
```
ARG RAY_BASE_IMAGE=registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base
ARG PROJECT_ROOT=/workspace/AIEvoBox
FROM ${RAY_BASE_IMAGE} AS runtime

# 1. apt 安装系统依赖（含 xvfb + openjdk-8-jre-headless）
# 2. 复制 requirements 并 pip install
# 3. 创建 ${PROJECT_ROOT} 目录并 COPY 全仓库
# 4. pip install -e ${PROJECT_ROOT}/env/mc/MineStudio
# 5. ENV PYTHONPATH=${PROJECT_ROOT}
# 6. ENTRYPOINT: uvicorn env.mc.service.app:app --host 0.0.0.0 --port 8000
```
- 可选多阶段：若需单独构建 jar，可在前一 stage 处理。  
- 若需加速 rebuild，可利用 `pip wheel` 缓存层。

## 7. 待确认事项
- ✅ 基础镜像：`registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/ray:base`。  
- ✅ `offline-mcprec-6.13.jar` 随镜像分发，并保留在 `AIEvoBox/env/mc/build/`。  
- ✅ FastAPI 无需鉴权。  
- ✅ 暂不考虑 GPU/EGL 依赖。  
- ✅ API schema 暂不固定版本。  
- ✅ 动作/观测 JSON 示例已在运行步骤中提供。  
- 尚需讨论：日志/模型输出挂载路径、Ray 集群接入方式等细节。  