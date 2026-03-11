# DiscoveryWorldEnv

将 [DiscoveryWorld](https://github.com/allenai/discoveryworld) benchmark 集成到 AIEvoBox 框架中，支持文本与视觉双模态、多场景多难度配置。

> 本文档假设你已完成 AIEvoBox 的安装，当前工作目录为 `AIEvoBox/`。

---

## 快速开始

### 第一步：安装 DiscoveryWorld

```bash
# 克隆 DiscoveryWorld 到指定位置
cd env/dwgym
git clone https://github.com/allenai/discoveryworld.git

# 安装 DiscoveryWorld
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
      scenario_name: "Proteomics"   # 场景名称，见下方场景列表
      difficulty: "Easy"            # 建议先用 "Easy" 验证流程
      seed: 0
      max_steps: 100                # Easy 建议 100，Normal/Challenge 建议 1000
      use_vision: false
      narrate_actions: true
      max_recent_actions: 5
```

完整参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scenario_name` | str | 第一个场景 | 场景名称，见下方场景列表 |
| `difficulty` | str | `"Normal"` | 难度：`"Easy"` / `"Normal"` / `"Challenge"` |
| `seed` | int | `0` | 随机种子，控制场景参数化变体 |
| `max_steps` | int | `300` | 每局最大交互步数 |
| `use_vision` | bool | `false` | 是否启用 2D 视觉观察（top-down view） |
| `capture_frames` | bool | `false` | 是否缓存帧（用于视频导出） |
| `narrate_actions` | bool | `true` | 是否生成动作解说文本 |
| `max_recent_actions` | int | `5` | 观察中保留的最近历史步数 |

---

### 第三步：运行

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

每局结束后返回多维度 scorecard：

| 字段 | 说明 |
|------|------|
| `score` | 原始得分 |
| `maxScore` | 满分 |
| `scoreNormalized` | 归一化得分 [0, 1] |
| `completed` | 是否完成任务 |
| `completedSuccessfully` | 是否成功完成 |

> 每步奖励为增量：`reward = scoreNormalized_current - scoreNormalized_previous`

---

## 场景列表

DiscoveryWorld 包含 8 个科学发现主题，每个主题支持 3 种难度，共 120 个任务：

| 主题 | 场景名 |
|------|--------|
| 蛋白质组学 | `"Proteomics"` |
| 平均力场 | `"MeaningOfLife"` |
| 动物适应 | `"AnimalAdaption"` |
| 空气质量 | `"AirQuality"` |
| 仙女座 | `"Andromeda"` |
| 巧克力农场 | `"ChocolateFarm"` |
| 植物育种 | `"PlantBreeding"` |
| 引用机制 | `"IntertwinedCitation"` |

---

## 文件结构

```
dwgym/
├── __init__.py          # 模块初始化
├── dw_env.py            # 核心适配器类
├── dw_config.yaml       # 环境配置文件
├── README.md            # 本文档
└── discoveryworld/      # DiscoveryWorld 源码（git clone 后生成，不进 repo）
```

---

## 相关链接

- [DiscoveryWorld GitHub](https://github.com/allenai/discoveryworld)
- [DiscoveryWorld 论文](https://arxiv.org/abs/2402.03628)

## 许可证

本适配器遵循 AIEvoBox 的许可证。DiscoveryWorld 有其自己的许可证，请参考其官方文档。