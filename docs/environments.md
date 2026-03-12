# Supported Environments

Safactory supports a growing set of simulation environments across multiple domains. All environments run as containerised HTTP services managed by the launcher.

## Overview

| Domain | env_name | Description | Config |
|--------|----------|-------------|--------|
| 📱 Mobile / Android | `android_gym` | Real Android device interaction via ADB (Ghost Bench) | `env/androidgym/android_env.yaml` |
| 🖥️ Desktop / PC | `os_gym` | Desktop task automation (OSWorld / RiOSWorld) | `env/osgym/os_config.yaml` |
| 🤖 Embodied AI | `embodied_alfred` | 3D household tasks (EmbodiedBench / ALFRED) | `env/embodiedgym/` |
| 🔍 Search | `search` | Information retrieval tasks | `env/search/` |
| 💻 Code / Git | `git_gym` | Repository-level coding tasks | `env/gitgym/` |
| 🎮 Game | `mc` | Minecraft-based tasks | `env/mc/config/mc_env.yaml` |

---

## Environment Details

### 📱 Android (`android_gym`)

Drives a real Android emulator over ADB. Tasks are drawn from the Ghost Bench dataset and cover everyday mobile app interactions such as navigation, form filling, and in-app actions.

**Host requirements:**
- Docker with `--privileged` (required for KVM-accelerated emulation)
- Network access to pull the image

**Start the environment:**
```bash
docker pull registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/android-emu-pjlab:and001
docker run -d --name android-env --privileged \
  registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/android-emu-pjlab:and001
```

**Config:** `env/androidgym/android_env.yaml`

---

### 🖥️ Desktop / PC (`os_gym`)

Wraps [OSWorld](https://github.com/xlang-ai/OSWorld) and [RiOSWorld](https://github.com/yjyddq/RiOSWorld) inside Safactory. The agent controls a full Ubuntu desktop via screenshot observations and `pyautogui` actions.

**Host requirements:**
- Docker with `--privileged` (required for QEMU/KVM virtualisation)
- A VM disk image (`Ubuntu.qcow2`) — downloaded automatically from HuggingFace on first run, or [download manually](https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip) and extract to `docker_vm_data/`
- Recommended: ≥ 60 GB RAM, ≥ 20 CPU cores

**Start the environment:**
```bash
docker pull registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/lite-osenv:v1.5
docker run -d --name os-env --privileged \
  registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/lite-osenv:v1.5
```

Full setup guide: [`env/osgym/README.md`](../env/osgym/README.md)

**Config:** `env/osgym/os_config.yaml`

---

### 🎮 Minecraft (`mc`)

Minecraft-based tasks powered by [MineStudio](https://github.com/CraftJarvis/MineStudio). The agent interacts with the game world through pixel observations and discrete action commands.

**Host requirements:**
- Docker for the environment service
- **Java 8** — Java 9+ is incompatible with Minecraft Malmo; install with `sudo apt-get install openjdk-8-jdk`
- **Xvfb** virtual display — install with `sudo apt-get install -y xvfb`
- **CUDA** (optional but recommended for GPU acceleration)

Full installation guide and common troubleshooting: [`env/mc/INSTALL.md`](../env/mc/INSTALL.md)

**Config:** `env/mc/config/mc_env.yaml`

---

### 🤖 Embodied AI (`embodied_alfred`)

3D household task execution based on [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) and [ALFRED](https://github.com/askforalfred/alfred). The agent navigates and manipulates objects in a simulated home environment.

Runs inside Docker. See [`env/embodiedgym/`](../env/embodiedgym/) for setup details.


