# MCGym 环境安装指南

## 📦 安装方式

### 方式 1：使用 requirements.txt（推荐）

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox/env/mc
pip install -r requirements.txt
```

### 方式 2：安装 MineStudio（开发模式）

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox/env/mc/MineStudio
pip install -e .
```

### 方式 3：仅安装关键依赖

```bash
# 最小安装
pip install minecraft-data rich pyyaml pillow numpy

# 如果需要完整功能
pip install -r requirements.txt
```

## 🔧 依赖说明

### 核心依赖

- **minecraft-data** - Minecraft 数据访问库（必需）
- **rich** - 美化输出
- **pyyaml** - YAML 配置解析
- **pillow** - 图像处理
- **numpy** - 数值计算

### MineStudio 依赖

完整的 MineStudio 依赖列表见 `requirements.txt`，包括：
- PyTorch 及相关库
- Gymnasium/Gym
- 各种回调和工具库

## ✅ 验证安装

### 快速测试

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox
python env/mc/test/test_mc_env_simple.py
```

预期输出应该显示大部分测试通过。

### 完整测试

```bash
# 安装所有依赖后
python env/mc/test/test_mc_env.py
```

### Python 检查

```python
# 测试基本导入
import minecraft_data
from env.mc.mc_env import MCGym
print("✓ 安装成功！")
```

## ⚠️ 常见问题

### 0. xvfb-run 命令未找到错误

**问题**：运行环境时报错：
```
xvfb-run: command not found
```
或者完整错误信息：
```
/fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox/env/mc/MineStudio/minestudio/simulator/minerl/env/launchClient.sh: line 38: xvfb-run: command not found
```

**原因**：
- 缺少 Xvfb (X Virtual Framebuffer)，这是在无显示器的服务器环境中运行 Minecraft 图形界面所必需的系统依赖

**解决方案**：

根据你的 Linux 发行版安装 xvfb：

**Ubuntu/Debian 系统：**
```bash
sudo apt-get update
sudo apt-get install -y xvfb
```

**CentOS/RHEL 系统：**
```bash
sudo yum install -y xorg-x11-server-Xvfb
```

**Fedora 系统：**
```bash
sudo dnf install -y xorg-x11-server-Xvfb
```

**验证安装：**
```bash
which xvfb-run
# 应该输出: /usr/bin/xvfb-run
```

安装完成后重新运行你的脚本即可。

### 1. 环境未注册错误

**问题**：运行 `base_eval.py` 时报错：
```
程序运行失败：环境 'mc_gym' 未注册，请先使用 @register_env 装饰器注册
```

**原因**：
- Python 装饰器 `@register_env` 只有在模块被导入时才会执行
- `base_eval.py` 没有导入 `MCGym` 类，导致装饰器未执行，环境未注册

**解决方案**：

在 `AIEvoBox/examples/base_eval.py` 中添加导入（**已修复**）：

```python
from env.mc.mc_env import MCGym  # 导入 Minecraft 环境来注册
```

**验证**：
```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AIEvoBox
bash examples/run_1_mc_envs.sh
```

运行后应该看到 `mc_gym` 在已注册环境列表中：
```
已注册的环境类型：['android_gym', 'core_git_env', 'embodied_alfred', 'mc_gym']
```

---

### 2. ModuleNotFoundError: No module named 'minecraft_data'

**问题**：缺少 minecraft_data 模块

**解决方案**：
```bash
pip install minecraft-data
```

### 3. gym 安装失败 (import gym 错误)

**问题**：`ModuleNotFoundError: No module named 'gym'`

MineStudio 内部代码使用了 `import gym`，但 gym 已停止维护且与新版 setuptools 不兼容。

**解决方案**：

我们使用 gymnasium (gym 的维护版本) + 兼容层：

```bash
# 安装 gymnasium 和兼容层
pip install gymnasium>=0.26.0 shimmy[gym-v21]>=0.2.1
```

`mc_env.py` 中已自动处理兼容性，将 gymnasium 映射为 gym 模块。

### 4. CUDA 运行时错误 (gpu_utils.py)

**问题**：`ImportError: cannot import name 'cuda' from 'cuda'`

这是 MineStudio 运行 Minecraft 模拟器时的 GPU 相关问题。

**原因**：
- `cuda-python` 包安装不正确
- 或者系统没有可用的 CUDA 环境

**解决方案**：

#### 选项 A: 安装 CUDA 支持（推荐，如果有 NVIDIA GPU）

```bash
# 1. 确保系统安装了 CUDA Toolkit (11.x 或 12.x)
nvidia-smi  # 检查 CUDA 版本

# 2. 安装对应版本的 cuda-python
pip install cuda-python

# 3. 验证安装
python -c "from cuda import cuda, cudart; print('CUDA OK')"
```

#### 选项 B: CPU 模式运行（如果没有 GPU）

目前 MineStudio 默认需要 GPU。如果需要 CPU 模式，需要修改 MineStudio 配置或使用模拟测试。

**测试结果说明**：
- ✓ 环境初始化成功 - 所有 Python 依赖已解决
- ✗ 模拟器运行失败 - 需要解决 CUDA 环境问题

### 5. Java JAXB 错误 (Minecraft 运行时)

**问题**：`java.lang.NoClassDefFoundError: javax/xml/bind/JAXBException`

**原因**：
- JAXB (Java XML Binding) 在 Java 9+ 中被移除
- Minecraft Malmo 需要 JAXB 来解析 XML 配置

**解决方案：使用 Java 8 (推荐)**

```bash
# 1. 检查当前 Java 版本
java -version

# 2. 安装 Java 8
sudo apt-get install openjdk-8-jdk

# 3. 设置 Java 8 为默认
sudo update-alternatives --config java
# 选择 java-8 选项

# 4. 验证
java -version  # 应该显示 1.8.x

# 或者，临时使用 Java 8
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

**替代方案：在 Java 11+ 中添加 JAXB**

如果必须使用 Java 11+，需要在启动时添加 JAXB 模块：

```bash
# 修改 MineStudio 的 launchClient.sh
# 添加 --add-modules java.xml.bind 参数
```

### 6. 动作格式转换问题

**问题 A**：`TypeError: string indices must be integers` 在 `action["buttons"]`

**原因**：
- `MCGym.step(action)` 接收的是字符串格式的 action（来自 LLM）
- `MinecraftSim.step()` 期望的是字典格式的 action（包含 `buttons` 和 `camera` 键）

**问题 B**：`AttributeError: 'int' object has no attribute 'ndim'`

**原因**：
- `MinecraftSim` 内部的 `action_mapper.to_factored()` 期望 `action["camera"]` 是 numpy 数组
- `ActionFromLLMConverter` 默认返回 int 类型

**问题 C**：`IndexError: index 220 is out of bounds for axis 0 with size 121`

**原因**：
- `ActionFromLLMConverter` 使用 **21×21=441** camera bins（用于 LLM 输出）
- `MinecraftSim.action_mapper` 期望 **11×11=121** camera bins（VPT 格式）
- Camera 索引不匹配导致越界

**解决方案**：

`mc_env.py` 已经正确配置了动作转换器：

```python
# 初始化转换器（在 __init__ 中）
self.action_converter = ActionFromLLMConverter(
    hfov_deg=self.current_hfov,
    vfov_deg=self.current_vfov,
    return_numpy=True,      # ⭐ 返回 numpy 数组格式
    map_camera_to_11=True   # ⭐ 将 21×21 映射到 11×11 (VPT)
)

# 在 step() 中自动转换字符串 action
if isinstance(action, str):
    image_shape = self.obs_size  # 从配置获取
    action_dict = self.action_converter.convert(action, image_shape)
else:
    action_dict = action
```

### 7. MinecraftSim 返回值类型问题

**问题**：`AttributeError: 'tuple' object has no attribute 'obs'`

**原因**：
- `MinecraftSim.reset()` 和 `MinecraftSim.step()` 可能返回 tuple 或对象
- 不同版本的 MineStudio 可能有不同的返回值格式

**解决方案**：

`mc_env.py` 已经处理了这个兼容性问题：

```python
# reset() 处理 tuple 或对象
if isinstance(result, tuple):
    obs, info = result if len(result) == 2 else (result[0], {})
else:
    obs = result.obs if hasattr(result, 'obs') else result
    info = result.info if hasattr(result, 'info') else {}

# step() 处理不同长度的 tuple
if isinstance(result, tuple):
    # 支持 (obs, reward, done, info) 或 (obs, reward, terminated, truncated, info)
    if len(result) == 4:
        obs, reward, done, info = result
        terminated = done
        truncated = False
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
```

### 8. Minecraft 运行时警告（可忽略）

以下错误和警告是**正常的**，不影响模拟器功能：

#### ✅ 可以忽略的消息：

```
ERROR : Couldn't load Narrator library
```
- **说明**：文本朗读功能，模拟器不需要

```
Error starting SoundSystem
Failed to open OpenAL device
```
- **说明**：无头服务器环境没有音频设备，正常现象

```
Unable to load minecraft:optifine/ctm/default/empty.png
Missing CTM sprite
```
- **说明**：OptiFine 可选材质，不影响功能

```
Could not authorize you against Realms server
```
- **说明**：Minecraft Realms 在线功能，模拟器不需要

```
Ambiguity between arguments [teleport, ...]
```
- **说明**：Minecraft 命令解析器的警告，不影响功能

#### ✅ 成功启动的标志：

当看到以下消息时，说明 Minecraft 已成功启动：

```
INFO - Minecraft process ready
INFO - Logging output of Minecraft to ./logs/mc_...log
```

### 9. cuda-python 安装失败

**问题**：pip install cuda-python 失败

**解决方案**：
```bash
# 跳过 cuda-python，其他依赖仍可正常工作
pip install -r requirements.txt --ignore-installed cuda-python || true
```

### 10. torch 版本冲突

**问题**：PyTorch 版本不兼容

**解决方案**：
```bash
# 先安装 PyTorch（根据你的 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 再安装其他依赖
pip install -r requirements.txt
```

### 11. pyrender 安装失败

**问题**：pyrender==0.1.25 依赖复杂

**解决方案**：
```bash
# 可以尝试更新版本或跳过
pip install pyrender  # 安装最新版本
# 或者
pip install -r requirements.txt --no-deps
```

## 📝 依赖列表

主要依赖包括：

```
minecraft-data>=0.7.0
av
opencv-python
numpy
torch>=2.3.1
gymnasium
rich
pyyaml
pillow
ray
transformers
hydra-core>=1.3.2
...（完整列表见 requirements.txt）
```

## 🎯 最小安装（仅测试配置）

如果只想测试配置和基础功能，不需要完整的 MinecraftSim：

```bash
pip install pyyaml rich pillow numpy pydantic
```

然后运行：
```bash
python env/mc/test/test_mc_env_simple.py
```

## 📚 更多信息

- [MineStudio 官方文档](./MineStudio/README.md)
- [测试指南](./test/README.md)
- [配置说明](./mc_env.yaml)

