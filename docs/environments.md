# Supported Environments

Safactory currently open-sources ten environment integrations in this repository. They cover mobile, desktop, game, embodied agents, QA, data processing, scientific discovery, and multimodal reasoning workloads. Some environments run host-natively, some require Docker, and several depend on upstream datasets or simulators that you install separately.

## Overview

| Domain | env_name | Public name | Description | Config / Docs |
|--------|----------|-------------|-------------|---------------|
| 📱 Mobile | `android_gym` | Android | Real Android emulator interaction over ADB | `env/androidgym/android_env.yaml` |
| 🖥️ Desktop | `os_gym` | OS | Desktop task automation based on OSWorld / RiOSWorld | `env/osgym/os_config.yaml` |
| 🎮 Game | `mc` / `mc_gym` | MC | Minecraft tasks powered by MineStudio | `env/mc/config/mc_env.yaml` |
| 🤖 Embodied | `robotrustbench` | RoboTrustBench | Embodied safety / robustness benchmark | `env/robotrustbench/README.md` |
| 🤖 Embodied | `embodied_alfred` | Embodied ALFRED | 3D household tasks via EmbodiedBench / ALFRED | `env/embodiedgym/embodied_config.yaml` |
| 💬 QA | `qa_gym` | QA | Multi-turn QA attack and evaluation environment | `env/qagym/qa_env.yaml` |
| 🧮 Data Processing | `dabstepgym` | DABStep | Data wrangling and code-execution benchmark | `env/dabstep/dabstep_config.yaml` |
| 🔬 Scientific Discovery | `discoveryworld` | DWGym | DiscoveryWorld science tasks with text / vision observations | `env/dwgym/dw_config.yaml` |
| 👁️ Multimodal Reasoning | `deepeyes_env` | DeepEyes | Multi-turn visual tool-use environment | `env/deepeyes/deepeyes_env.yaml` |
| 📐 Geometric VL | `geo3k_vl_test` | Geo3K-VL | Geometry-focused multimodal reasoning environment | `env/geo3k_vl_test/geo3k_vl_test_env_configs.yaml` |

---

## Environment Details

### 📱 Android (`android_gym`)

Drives a real Android emulator over ADB. Tasks are drawn from the [Ghost Bench](https://arxiv.org/abs/2510.20333) dataset and cover everyday mobile app interactions such as navigation, form filling, and in-app actions.

**Host requirements:**
- `adb` available on the host (or configure `adb_path`)
- Android Emulator available on the host (for standard emulator; must have an AVD named `emulator_name`, default `nexus_safe`)
- Dataset file available at `env/androidgym/cases.jsonl` (provided in the archive mirror; the repo does not ship datasets)

**Config:** `env/androidgym/android_env.yaml`

**Docs:** [`env/androidgym/README_EN.md`](../env/androidgym/README_EN.md)

---

### 🖥️ Desktop / PC (`os_gym`)

Wraps [OSWorld](https://github.com/xlang-ai/OSWorld) and [RiOSWorld](https://github.com/yjyddq/RiOSWorld) inside Safactory. The agent controls a full Ubuntu desktop via screenshot observations and `pyautogui` actions.

**Host requirements:**
- Docker with `--privileged` (required for QEMU/KVM virtualisation)
- A VM disk image (`Ubuntu.qcow2`) — downloaded automatically from HuggingFace on first run, or [download manually](https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip) and extract to `docker_vm_data/`
- Recommended: ≥ 60 GB RAM, ≥ 20 CPU cores

**Docs:** [`env/osgym/README.md`](../env/osgym/README.md)

**Config:** `env/osgym/os_config.yaml`

---

### 🎮 Minecraft (`mc` / `mc_gym`)

Minecraft-based tasks powered by [MineStudio](https://github.com/CraftJarvis/MineStudio). The agent interacts with the game world through pixel observations and discrete action commands.

**Host requirements:**
- Docker for the environment service
- **Java 8** — Java 9+ is incompatible with Minecraft Malmo; install with `sudo apt-get install openjdk-8-jdk`
- **Xvfb** virtual display — install with `sudo apt-get install -y xvfb`
- **CUDA** (optional but recommended for GPU acceleration)

Full installation guide and common troubleshooting: [`env/mc/INSTALL.md`](../env/mc/INSTALL.md)

**Config:** `env/mc/config/mc_env.yaml`
The sample config currently uses the alias `mc_gym`; the factory also accepts `mc`.

---

### 🤖 Embodied (`robotrustbench`)

RoboTrustBench is the embodied safety / robustness benchmark integration in this repository. It currently exposes three public variants: `safety`, `robust`, and `robustd`.

**Typical usage:** run inside a prepared container image with Habitat / simulator dependencies available, then point `launcher.py` at a small YAML config that sets `rt_variant`, `eval_set`, and other runtime parameters.

**Docs:** [`env/robotrustbench/README.md`](../env/robotrustbench/README.md)

---

### 🏠 Embodied (`embodied_alfred`)

3D household task execution based on [EmbodiedBench](https://github.com/embodied-generalist/embodiedbench) and [ALFRED](https://github.com/askforalfred/alfred). The agent navigates and manipulates objects in a simulated home environment.

**Docs:** [`env/embodiedgym/README.md`](../env/embodiedgym/README.md)

**Config:** `env/embodiedgym/embodied_config.yaml`

---

### 💬 QA (`qa_gym`)

QAGym models question-answering interaction as an environment, with configurable attack strategies, environment-side model calls, and judge-based scoring. It is useful for evaluating prompt-injection resistance and QA robustness.

**Docs:** [`env/qagym/README.md`](../env/qagym/README.md)

**Config:** `env/qagym/qa_env.yaml`

---

### 🧮 Data Processing (`dabstepgym`)

DABStep is a code-execution data processing environment. The model receives structured data tasks, writes Python code, executes it inside the environment, and is scored on the produced results and traces.

**Docs:** [`env/dabstep/README.md`](../env/dabstep/README.md)

**Config:** `env/dabstep/dabstep_config.yaml`

---

### 🔬 Scientific Discovery (`discoveryworld`)

DiscoveryWorld exposes interactive scientific discovery tasks with configurable scenario, difficulty, text narration, and optional visual frames. It is the science-discovery environment in the OSS release.

**Docs:** [`env/dwgym/README.md`](../env/dwgym/README.md)

**Config:** `env/dwgym/dw_config.yaml`

---

### 👁️ Multimodal Reasoning (`deepeyes_env`)

DeepEyes is a multi-turn visual tool-use environment. Tasks typically provide prompt messages, image inputs, and a reference answer. The environment can support cropping / rotate-style tool calls and judge-based reward computation.

**Docs:** [`env/deepeyes/README.md`](../env/deepeyes/README.md)

**Config:** `env/deepeyes/deepeyes_env.yaml`

---

### 📐 Geometric VL (`geo3k_vl_test`)

Geo3K-VL is the geometry-focused multimodal reasoning environment in this repository. Each task provides a question, one or more images, and a reference answer; the agent can optionally call the built-in scoring tool before finalising an answer.

**Docs:** [`env/geo3k_vl_test/README.md`](../env/geo3k_vl_test/README.md)

**Config:** `env/geo3k_vl_test/geo3k_vl_test_env_configs.yaml`

---

## Notes

- Some sample YAML files in the repo still require you to replace dataset paths, model endpoints, or runtime parameters with values from your local setup.
- Use `--env-config` to run a single environment config file, or `--env-root env` to load multiple environment YAMLs recursively.
