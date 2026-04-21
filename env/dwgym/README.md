# DiscoveryWorldEnv

将 [DiscoveryWorld](https://github.com/allenai/discoveryworld) benchmark 集成到 AIEvoBox 框架中，支持文本与视觉双模态、多场景多难度配置。

> 本文档假设你已完成 AIEvoBox 的安装，当前工作目录为 `AIEvoBox/`。

---

## 快速开始

### 第一步：安装 DiscoveryWorld
```bash
cd env/dwgym
git clone https://github.com/allenai/discoveryworld.git
cd discoveryworld
pip install -e .
```

---

### 第二步：配置环境参数

编辑 `env/dwgym/dw_config.yaml`：
```yaml
environments:
  - env_name: discoveryworld
    env_num: 1
    env_params:
      scenario_name: "Proteomics"
      difficulty: "Easy"
      seed: 0
      max_steps: 100
      use_vision: false
      narrate_actions: true
      max_recent_actions: 5
```

完整参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scenario_name` | str | 第一个场景 | 场景名称，见下方场景列表 |
| `difficulty` | str | `"Normal"` | 难度：`"Easy"` / `"Normal"` / `"Challenge"` |
| `seed` | int | `0` | 随机种子 |
| `max_steps` | int | `300` | 每局最大交互步数 |
| `use_vision` | bool | `false` | 是否启用 2D 视觉观察 |
| `capture_frames` | bool | `false` | 是否缓存帧（用于视频导出） |
| `narrate_actions` | bool | `true` | 是否生成动作解说文本 |
| `max_recent_actions` | int | `5` | 观察中保留的最近历史步数 |

---

### 第三步：运行参考
```bash
python launcher.py \
  --mode local \
  --manager-config manager/config.yaml \
  --env-config env/dwgym/dw_config.yaml \
  --llm-base-url <your-base-url> \
  --llm-api-key  <your-api-key> \
  --llm-model    <model-name> \
  --pool-size 1
```

---

### 第四步：查看结果

运行结束后，帧图像会保存到 `AIEvoBox/video/` 目录下。如需启用，请在配置中设置：
```yaml
env_params:
  use_vision: true
  capture_frames: true
```

> `use_vision: false` 时不会生成帧图像。

---

## 场景列表

DiscoveryWorld 包含 8 个科学发现主题，每个主题支持 3 种难度：

| 场景名 |
|--------|
| `"Proteomics"` |
| `"SpaceIllness"` |
| `"RosettaTranslation"` |
| `"Chemistry"` |
| `"RadioisotopeDating"` |
| `"ReactorLab"` |
| `"RocketScience"` |
| `"PlantNutrients"` |

> 场景名以 DiscoveryWorld 源码中的实际定义为准，如有出入请参考 [DiscoveryWorld GitHub](https://github.com/allenai/discoveryworld)。

---

## Docker 部署（环境tag待补充）
```bash
docker build -f env/dwgym/Dockerfile -t dwgym:latest .
```

- 如需预构建镜像，请替换为你自己的镜像仓库地址，例如：
```bash
docker pull <your-registry>/dwgym:latest
```

---

## 文件结构
```
dwgym/
├── __init__.py
├── dw_env.py
├── dw_config.yaml
├── README.md
└── discoveryworld/      # git clone 后生成，不进 repo
```

---

## 相关链接

- [DiscoveryWorld GitHub](https://github.com/allenai/discoveryworld)
- [DiscoveryWorld 论文](https://arxiv.org/abs/2402.03628)

## 许可证

本适配器遵循 AIEvoBox 的许可证。DiscoveryWorld 有其自己的许可证，请参考其官方文档。
