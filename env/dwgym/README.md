# DiscoveryWorldEnv

将 [DiscoveryWorld](https://github.com/allenai/discoveryworld) benchmark 集成到 AIEvoBox 框架中，支持文本与视觉双模态、多场景多难度配置。

## 文件结构

```
dwgym/
├── __init__.py          # 模块初始化
├── dw_env.py            # 核心适配器类
├── dw_config.yaml       # 环境配置文件
├── README.md            # 本文档
└── discoveryworld/      # DiscoveryWorld 源码（git clone 后生成，不进 repo）
```

## 快速开始

### 1. 前置依赖

```bash
# 1. 克隆 AIEvoBox 仓库
git clone https://gitee.pjlab.org.cn/L2/safeai/kilab/AIEvoBox.git
cd AIEvoBox

# 2. 安装依赖
pip install -r requirements.txt

# 3. 克隆 DiscoveryWorld 到指定位置（必须是这个路径）
cd env/dwgym
git clone https://github.com/allenai/discoveryworld.git

# 4. 安装 DiscoveryWorld
cd discoveryworld
pip install -e .
```


## 配置说明

### 环境参数

在 `dw_config.yaml` 中可以配置以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scenario_name` | str | 第一个场景 | 场景名称，见下方场景列表 |
| `difficulty` | str | `"Normal"` | 难度，可选 `"Easy"` / `"Normal"` / `"Challenge"` |
| `seed` | int | `0` | 随机种子，控制场景参数化变体 |
| `max_steps` | int | `300` | 每局最大交互步数 |
| `use_vision` | bool | `false` | 是否启用 2D 视觉观察（top-down view） |
| `capture_frames` | bool | `false` | 是否缓存帧（用于视频导出） |
| `narrate_actions` | bool | `true` | 是否生成动作解说文本 |
| `max_recent_actions` | int | `5` | 观察中保留的最近历史步数 |

### 示例配置

```yaml
environments:
  - env_name: discoveryworld
    env_num: 1
    env_params:
      scenario_name: "Proteomics"
      difficulty: "Normal"
      seed: 0
      max_steps: 300
      use_vision: false
      narrate_actions: true
      max_recent_actions: 5
```

## 场景说明

DiscoveryWorld 包含 8 个科学发现主题，每个主题支持 3 种难度，共 120 个任务。

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

论文推荐配置：Easy 模式最多 100 步，Normal / Challenge 最多 1000 步。


## 评分说明

DiscoveryWorld 使用多维度 scorecard 评分：

| 字段 | 说明 |
|------|------|
| `score` | 原始得分 |
| `maxScore` | 满分 |
| `scoreNormalized` | 归一化得分 [0, 1] |
| `completed` | 是否完成任务 |
| `completedSuccessfully` | 是否成功完成 |

每步 `reward = scoreNormalized_current - scoreNormalized_previous`（增量奖励）。

## 相关链接

- [DiscoveryWorld GitHub](https://github.com/allenai/discoveryworld)
- [DiscoveryWorld 论文](https://arxiv.org/abs/2402.03628)

## 许可证

本适配器遵循 AIEvoBox 的许可证。DiscoveryWorld 有其自己的许可证，请参考其官方文档。