# Configuration

Safactory's configuration is layered: **CLI arguments take precedence over `manager/config.yaml`**.

---

## CLI Flags

### Essential flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `local` | `local` (connects to a locally running env service) or `remote` (Ray cluster) |
| `--env-transport` | `inproc` | Environmental Data Transmission Mode |
| `--env-config` | — | Path to a single environment YAML |
| `--env-root` | `env` | Load all env YAMLs found under this directory tree |
| `--pool-size` | `0` (from YAML) | Number of parallel environment instances |
| `--llm-base-url` | — | OpenAI-compatible LLM endpoint URL |
| `--llm-model` | — | Model identifier string |
| `--llm-api-key` | `EMPTY` | API key (set to `EMPTY` when not required) |
| `--max-steps` | `1000` | Maximum agent steps per episode |
| `--db-path` | `sqlite://test_envs.db` | SQLite database path |
| `--log-dir` | `logs` | Directory that stores run-specific log directories |

### Full CLI reference

| Category | Flag | Default | Description |
|----------|------|---------|-------------|
| Job | `--job-id` | auto-generated | Unique ID recorded in the DB for this run |
| Config | `--manager-config` | `./manager/config.yaml` | Manager config YAML path |
| DB | `--storage-type` | `sqlite` | Storage backend |
| | `--warmup-count` | `100` | The number of environment configs to pre-store in the manager |
| | `--save-batch-size` | `100` | Size of environment configs to store in the manager |
| | `--disable-buffer` | — | Disable buffered record storage (buffer is enabled by default) |
| | `--buffer-size` | `100` | Size of the buffer for storing records |
| | `--flush-interval` | `5.0` | Interval (in seconds) for flushing buffered records |
| | `--rebuild-table` | `false` | Drop and recreate DB tables before run |
| Pool | `--multiplier` | `1.2` | Pre-warm `ceil(multiplier × pool_size)` actors |
| Local | `--local-upstream-port` | `36663` | Port for the local env HTTP service |
| | `--wait-timeout` | `60.0` | Seconds to wait for local service to become ready |
| Interactor | `--message-cut` | `3` | Recent dialogue turns kept in LLM context |
| | `--env-http-timeout-s` | `300.0` | Per-request timeout for env HTTP calls |
| | `--http-retries` | `2` | Retry count for failed env HTTP calls |
| LLM | `--llm-temperature` | `0.3` | Sampling temperature |
| RL | `--rl-group-size` | `0` | Override `env_num` for all environments |
| | `--rl-epoch` | `1` | Repeat configs N times with distinct group IDs |
| | `--rl-use-session-suffix-url` | `false` | Use session-routed LLM proxy (for RL) |
| Logging | `--run-name` | `""` | Prefix for the run log directory name |
| | `--console-log-level` | `INFO` | Console verbosity |
| | `--file-log-level` | `DEBUG` | File log verbosity |

Notes:

- Buffered record storage is **enabled by default**.
- Pass `--disable-buffer` to turn it off. It is a simple presence-based switch (no value needed).

---

## `manager/config.yaml` Reference

```yaml
mode: remote          # overridden by --mode; "local" or "remote"

pool_size: 2          # overridden by --pool-size

database:
  driver: sqlite
  sqlite_path: test_envs.db

cluster:
  http:
    port: 36663        # all env services share this port
    timeout_s: 100     # HTTP connection timeout (seconds)
    concurrency: 200   # max concurrent HTTP calls during actor pre-warming
  env_types:
    android_gym:
      quotagroup: "evobox_cpu_task"       # Ray cluster quota group
      entrypoint: "bash /path/to/run.sh"  # command used to start the env service
      limit: 10                           # max concurrent instances of this env type
    os_gym:
      quotagroup: "evobox_proxy"
      entrypoint: "bash /path/to/run.sh"
      resources:
        head:
          cpu: 20
          gpu: 0
          memory: 60Gi
          privileged: true    # required for QEMU/KVM
      limit: 5

rayjob:               # remote mode only
  domain: "https://your-rayjob-platform.com"
  tenant: "your-tenant"
  access_key: "xxx"
  secret_key: "xxx"
  project: "your-project"
```

---

## Environment YAML Reference

Each environment has its own YAML config file. Multiple environments can be listed in a single file, or point `--env-root` at a directory to load all YAMLs at once.

```yaml
environments:
  - env_name: android_gym       # must match a registered env key (see env/env_factory.py)
    env_image: your_optional_docker_image
    env_num: 1                  # number of parallel instances
    dataset: cases.jsonl        # task dataset (JSON/YAML array or plain text, one task per line)
    env_params:
      # all fields here are passed as kwargs to the environment's __init__
      max_step: 10
      seed: 1234
```

Built-in OSS environments in this repository:

| env_name | Environment | Example config / docs |
|----------|-------------|-----------------------|
| `android_gym` | Android | `env/androidgym/android_env.yaml` |
| `os_gym` | OS | `env/osgym/os_config.yaml` |
| `mc` / `mc_gym` | Minecraft | `env/mc/config/mc_env.yaml` |
| `robotrustbench` | RoboTrustBench | `env/robotrustbench/README.md` |
| `embodied_alfred` | Embodied ALFRED | `env/embodiedgym/embodied_config.yaml` |
| `qa_gym` | QA | `env/qagym/qa_env.yaml` |
| `dabstepgym` | DABStep | `env/dabstep/dabstep_config.yaml` |
| `discoveryworld` | DWGym | `env/dwgym/dw_config.yaml` |
| `deepeyes_env` | DeepEyes | `env/deepeyes/deepeyes_env.yaml` |
| `geo3k_vl_test` | Geo3K-VL | `env/geo3k_vl_test/geo3k_vl_test_env_configs.yaml` |
