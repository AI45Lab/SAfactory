# OSGym RL training

Follow the shared [Safactory RL Usage Guide](../../README.md) for the common Buffer Server, Slime, Ray, service, database, and training setup. This document only describes the OSGym-specific differences.

## OSGym components

| Component | OSGym-specific role |
| --- | --- |
| `env.sh` | Configures the OSGym environment, Qwen3.5 model, rollout grouping, GPU placement, and service transport. |
| `run_buffer_server.sh` | Loads the OSGym configuration, prepares the SQLite database directory, and starts the shared Buffer Server. |
| `run_slime_generator.sh` | Replaces the shared Slime launcher because the Qwen3.5 Megatron and SGLang arguments required by OSGym are not exposed by the shared launcher. |
| `slime_generator.py` | Preserves complete variable-length OSGym trajectories, validates Qwen3.5 actions, and pads flattened optimizer batches. |
| `trajectory_rewards.py` | Computes trajectory-level GRPO advantages and masks invalid actions. |

## Configuration differences

OSGym uses `AIEVOBOX_MODE=remote` and `AIEVOBOX_ENV_TRANSPORT=http` because its environments are served independently. Configure the task dataset, VM image, and OSGym runtime in `env/osgym/os_config.yaml` before starting training.

## Start training

Start the OSGym Buffer Server launcher in the first terminal:

```bash
bash rl/examples/osgym/run_buffer_server.sh
```

Start the OSGym-specific Slime launcher in the second terminal:

```bash
bash rl/examples/osgym/run_slime_generator.sh
```
