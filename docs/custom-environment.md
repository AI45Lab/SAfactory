# Custom Environment

Adding a new environment to Safactory requires four steps and fewer than 50 lines of Python.

## Step 1 — Implement `BaseEnv`

Create a new file, e.g. `env/mygym/my_env.py`, and subclass `BaseEnv`:

```python
from core.env.base_env import BaseEnv
from core.env.env_register import register_env

@register_env("my_env")
class MyEnv(BaseEnv):

    def __init__(self, my_param: str = "default", **kwargs):
        super().__init__(**kwargs)
        # initialise your environment state here

    # --- Prompt interface (tell the LLM what it sees and can do) ---

    def observation_space(self) -> str:
        """Describe the observable state format in natural language."""
        return "A text description of the current environment state."

    def action_space(self) -> str:
        """Describe the available actions in natural language."""
        return "Available actions: action_a, action_b, ..."

    def get_task_prompt(self) -> list:
        """Return the task instruction as a list of chat messages."""
        return [{"role": "user", "content": "Your task is to ..."}]

    # --- Gym-style interface ---

    def reset(self, seed=None):
        """Reset to initial state and return the first observation."""
        ...

    def step(self, action: str):
        """Execute action and return (obs, reward, done, truncated, info)."""
        ...

    def render(self):
        """(Optional) Return a screenshot or frame for GIF logging."""
        ...

    def close(self):
        """(Optional) Release any held resources."""
        ...
```

### `BaseEnv` interface contract

| Method | Required | Description |
|--------|----------|-------------|
| `observation_space()` | Yes | Natural-language description of the observation format |
| `action_space()` | Yes | Natural-language description of available actions |
| `get_task_prompt()` | Yes | List of chat messages forming the task instruction |
| `reset(seed=None)` | Yes | Reset env, return initial observation |
| `step(action)` | Yes | Apply action, return `(obs, reward, done, truncated, info)` |
| `render()` | No | Return a frame/screenshot (used for GIF logging) |
| `close()` | No | Clean up resources |

---

## Step 2 — Register the environment

Registration is handled in two places:

- **`env/registry.py`** — defines a lazy-import function for each environment to avoid loading heavy dependencies at startup.
- **`env/env_factory.py`** — maps environment names to those import functions in `_ENV_IMPORTERS`.

**`env/registry.py`** — add a lazy-import function:

```python
def _import_my_env() -> Type:
    from env.mygym.my_env import MyEnv
    return MyEnv
```

**`env/env_factory.py`** — import the function and add an entry to `_ENV_IMPORTERS`:

```python
from env.registry import (..., _import_my_env)

_ENV_IMPORTERS: Dict[str, Callable[[], Type]] = {
    ...
    "my_env": _import_my_env,   # key must match env_name in your YAML
}
```

The `@register_env("my_env")` decorator lets the class self-register once it is imported, and the lazy importer ensures the module is only imported on first use.

---

## Step 3 — Write a config YAML

Create `env/mygym/my_config.yaml`:

```yaml
environments:
  - env_name: my_env              # must match the key in env/env_factory.py
    env_image: your_docker_image_url
    env_num: 2                    # number of parallel instances
    dataset: cases.jsonl          # task dataset file
    env_params:
      my_param: value             # passed as kwargs to MyEnv.__init__
```

**`dataset` notes:**
- Supports JSON arrays, YAML arrays, and plain text (one task per line).
- Each row is merged into `env_params` as a dictionary when the environment is reset.

---

## Step 4 — Run

```bash
python launcher.py \
  --env-config env/mygym/my_config.yaml \
  --llm-base-url http://YOUR_LLM_HOST/v1 \
  --llm-api-key YOUR_API_KEY \
  --llm-model YOUR_MODEL \
  --pool-size 1
```

---

## Tips

- Keep `env_num` small (e.g. `1`) during development to reduce startup time.
- Implement `render()` to get a visual GIF log of agent trajectories — useful for debugging.
- Use `env_params` to pass dataset paths, credentials, or any other per-environment configuration.
- Check existing environments (e.g. `env/androidgym/`, `env/dabstep/`, `env/deepeyes/`) for reference implementations.
