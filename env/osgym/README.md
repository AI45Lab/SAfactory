# OSGym Guide

OSGym wraps the runtime environments of [OSWorld](https://github.com/xlang-ai/OSWorld) / [RiOSWorld](https://github.com/yjyddq/RiOSWorld) into AIEvoBox, making it convenient to train and evaluate desktop agents or reinforcement learning models.

## 1. Install Dependencies

```bash
# From the AIEvoBox root directory
pip install -r requirements.txt

# Then inside the osgym directory
cd env/osgym && pip install -r requirements.txt
```

## 2. VM Image

The repository does not include the large file `docker_vm_data/Ubuntu.qcow2`. It is downloaded automatically from Hugging Face at runtime:
[download link](https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip)

If automatic download fails, download it manually and extract it into `env/osgym/docker_vm_data/`.

You can also explicitly set the image path in the config, for example `vm_path: "/abs/path/Ubuntu.qcow2"`.
Relative paths are resolved relative to the `env/osgym` directory.

## 3. Configuration Parameters

Configure through `os_config.yaml` or constructor arguments:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset` | Task dataset path; relative paths are resolved from `env/osgym` | `datasets/cases.jsonl` |
| `eval_mode` | Evaluation mode (`standard` / `safety`) | `standard` |
| `provider_name` | Backend provider (`docker` / `containerd`) | `docker` |
| `vm_path` | Explicit VM image path; supports paths relative to `env/osgym` | `None` |
| `prompt_observation_type` | Prompt modality (`screenshot`, `a11y_tree`, `screenshot_a11y_tree`) | `screenshot` |
| `prompt_format` | Prompt protocol format (`kimi`, `qwen`) | `qwen` |
| `action_space` | Action space | `pyautogui` |
| `screen_width/height` | Screen resolution | `1920x1080` |
| `max_steps` | Maximum allowed steps per task | `30` |
| `repeated_click_distance_threshold` | Pixel distance below which consecutive click actions are treated as the same click | `10.0` |
| `repeated_click_limit` | Consecutive repeated click count that truncates the task; set `0` to disable | `2` |

OSGym selects accessibility-tree capture automatically: it is enabled for a11y prompt modes and for safety `popup` / `induced_text` tasks.

## 4. Run Examples

### Local Run

```bash
python launcher.py \
    --mode local \
    --env-config env/osgym/os_config.yaml \
    --llm-base-url your_url \
    --llm-api-key your_key \
    --llm-model model_name \
    --pool-size 2
```

### Container Run

```bash
docker pull safactory/osworld:v0.1.0
sudo docker run --privileged -d --name os_env safactory/osworld:v0.1.0 tail -f /dev/null

# Start the first container terminal
docker exec -it os_env /bin/bash
dockerd

# Start the second container terminal
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
