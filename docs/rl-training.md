# Reinforcement Learning Training

Safactory integrates with [Slime](https://github.com/THUDM/slime) (GRPO-based VL training) via a **Buffer Server** that bridges the rollout engine with the Slime training loop.

## Architecture

```
┌─────────────────┐    rollout trajectories    ┌───────────────────┐
│  Safactory      │ ─────────────────────────► │  Buffer Server    │
│  (launcher.py)  │                            │  (rl/buffer_      │
└─────────────────┘                            │   server.py)      │
                                               └────────┬──────────┘
                                                        │ training batches
                                               ┌────────▼──────────┐
                                               │  Slime / GRPO     │
                                               │  Training Process  │
                                               └───────────────────┘
```

The Safactory runner collects environment rollouts and pushes them to the Buffer Server. The Slime training process pulls batches from the Buffer Server and performs GRPO gradient updates. The two processes can run on separate machines as long as they can communicate over the network.

## Prerequisites

- [Slime](https://github.com/THUDM/slime) installed and configured (Terminal 1 only):
  - **Option A** — install from scratch using the [build script](https://github.com/THUDM/slime/blob/main/build_conda.sh)
  - **Option B** — use the official [Slime Docker image](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md#pull-and-start-docker-container)
- Safactory dependencies installed (`pip install -r requirements.txt`) on the Buffer Server machine (Slime is not required there)
- At least one environment running and reachable

## Setup

### Step 1 — Configure environment variables

```bash
cp rl/.env.example rl/.env
# Edit rl/.env and fill in your actual configuration
```

See [Key Parameters](#key-parameters) below for a full reference.

### Step 2 — Start the Slime training process (Terminal 1)

```bash
cd rl

# For text-only models:
./run_slime_generator.sh

# For vision-language models:
./run_slime_generator_vl.sh
```

### Step 3 — Start the Buffer Server (Terminal 2)

```bash
cd rl && ./run_buffer_server.sh
```

Once started, the Buffer Server automatically launches the Safactory runner and begins collecting rollout data. Completed trajectories are held in the buffer until the Slime training process requests a batch.

> Terminals 1 and 2 can run on different machines as long as they can reach each other over the network.

## Key Parameters

Set these in `rl/.env`:

### Environment selection (choose one)

| Variable | Description |
|----------|-------------|
| `SAFACTORY_ENV_CONFIG` | Path to a single env YAML — useful for debugging or single-environment runs |
| `SAFACTORY_ENV_ROOT` | Path to a directory of env YAMLs — all configs are loaded at once for multi-env training |

### Training parameters

| Variable | Description |
|----------|-------------|
| `RL_GROUP_SIZE` | Samples per prompt (maps to `n_samples_per_prompt` in Slime) |
| `RL_EPOCH` | Number of training epochs |
| `RL_OFF_BY_N` | Max weight-version lag allowed (`0` = strictly on-policy) |
| `SLIME_GLOBAL_BATCH_SIZE` | Global training batch size |

### Service ports

| Service | Default Port | Override Variable |
|---------|-------------|-------------------|
| Buffer Server | `18889` | `BUFFER_SERVER_PORT` |
| LLM Proxy | `18890` | `LLM_PROXY_PORT` |

Complete configuration reference: `rl/.env.example`. Additional Slime training parameters are documented in the [Slime repository](https://github.com/THUDM/slime).
