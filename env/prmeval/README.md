# PRMEval environment

`prmeval` is a metrics-only SAfactory environment for one progress trajectory
per JSONL row. The adapter preserves the native `mse` and `pearson` values; it
does not define an RL reward or support cross-trajectory aggregation.

## Runtime prerequisite

Build the runtime image outside this repository and install the published
`prmeval` wheel into it. The image needs Python 3.10+, the `prmeval` command,
and every dependency used by the selected PRMEval infer backend. Set its actual
tag or digest in `prmeval_config.yaml`.

The current PRMEval loader imports `cv2`; include `opencv-python-headless` in
the wheel's runtime dependencies (preferred) or in the runtime image.

## Smoke run

The bundled `datasets/prmeval_smoke.jsonl` is a self-contained, three-frame
synthetic trajectory. With a ready Gateway route, run:

```bash
python launcher.py \
  --mode docker \
  --agent-config env/prmeval/prmeval_config.yaml \
  --agent-start-config env/prmeval/prmeval_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model YOUR_ROUTE_KEY \
  --db-path sqlite://env_trajs.db \
  --job-id prmeval-smoke \
  --pool-size 1 \
  --max-workers 1
```

Do not add `--enable-evaluation` until a scalar reward policy is chosen. The
runner writes the generated PRMEval config, one-row input, metrics summary, and
metric detail JSONL beneath `results/<job-id>/<session-id>/prmeval/`.

## Production datasets

For a path-based `frames` value, mount the media root read-only in
`prmeval_start.yaml` and set the matching container path in
`env_params.media_root` (or `trajectory_base_dir`). The runner makes that path
absolute before creating its per-session JSONL file.
