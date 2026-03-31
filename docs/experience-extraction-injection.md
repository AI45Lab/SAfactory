# Experience Extraction and Injection

Safactory can reuse past trajectories in two stages:

- **Experience extraction** reads historical trajectories from the database and summarizes reusable lessons into a local experience library.
- **Experience injection** loads that library at episode start and appends relevant experience to the agent prompt.


## Files You Need

- `exp_service_config.yaml`: config for the extraction service
- `core/exp/config.yaml`: config for prompt-time experience injection
- `experiences/`: the generated experience library directory

## 1. Extract Experiences

Make sure your historical runs have already been written to the Safactory database, then run:

```bash
python3 -m exp_service run-once --config exp_service_config.yaml
```

This command reads trajectories from the database and writes experience files into the directory configured by `exp_service_config.yaml`.

## 2. Enable Experience Injection

Edit `core/exp/config.yaml` and point it to the same experience directory:

```yaml
enabled: true
dir: ./experiences
top_k: 3
mode: template
embedding_model: null
```

Common modes:

- `template`: selects the highest-ranked experience templates
- `ucb`: uses a simple bandit strategy based on usage feedback
- `contextual_ucb`: uses task text plus feedback history to choose one experience

If you want to turn the feature off again, set `enabled: false`.

## 3. Run Safactory With Injection

By default, `launcher.py` reads `core/exp/config.yaml`. A normal run will automatically use experience injection when `enabled: true`.

If you want to override the config path for one run, use:

```bash
python3 launcher.py --exp-config ./core/exp/config.yaml
```

## 4. Practical Tips

- Keep the extraction output directory and the injection `dir` setting consistent.
- Start with `mode: template` if you want the most predictable behavior.
- Use a separate experience directory for experiments if you do not want to mix statistics across runs.
