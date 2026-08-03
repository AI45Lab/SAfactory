# Docker Evaluation Workflow

Use this reference when the user asks to run evaluation for an SAfactory environment in Docker mode.

Canonical docs:

- `README.md`
- `README_CN.md`
- `docs/guides/evaluation.md`
- `docs/guides/evaluation_CN.md`
- `docs/reference/gateway.md`
- `docs/reference/gateway_CN.md`
- `docs/reference/environments.md`
- `docs/reference/environments_CN.md`
- `docs/reference/configuration.md`
- `docs/reference/configuration_CN.md`

Standard baseline:

- Geo3K is the standard smoke-test environment.
- For other environments, replace `geo3k` paths and model route with the requested environment name.

## Preflight

Check these before running:

- Docker is available and the current user can run `docker build`, `docker run`, and `docker exec`.
- Environment config exists: `env/<env>/<env>_config.yaml`.
- Start config exists: `env/<env>/<env>_start.yaml`.
- Required image, dataset, and runner files are available.
- Gateway is running at a session root such as `http://127.0.0.1:8000/v1/sessions`.
- The requested model route key exists in Gateway `llm_routes`.
- Gateway storage and Launcher `--db-path` point to the same backend when telemetry/results are expected in the same DB.
- `rule_evaluator.py` exists if the user expects `--enable-evaluation` reward output.

Do not add private `base_url` or `api_key` values to committed files. Use placeholders when preparing config.

## Command Shape

Use the root README as the command source. For a generic environment:

```bash
python launcher.py \
  --mode docker \
  --agent-config env/<env>/<env>_config.yaml \
  --agent-start-config env/<env>/<env>_start.yaml \
  --gateway-base-url http://127.0.0.1:8000/v1/sessions \
  --llm-model <route_key> \
  --enable-evaluation \
  --db-path sqlite:///<env>_eval.db \
  --job-id <env>-docker-smoke
```

Use a small dataset, small sample count, or smoke-test config when available. Avoid launching a full benchmark unless the user explicitly asks for it.

## Gateway Handling

When Gateway config is needed:

1. Start from `gateway/config.example.yaml`.
2. Copy to `gateway/config.local.yaml` only if the user does not already have a local config or explicitly wants one.
3. Keep route values replaceable:

```yaml
llm_routes:
  YOUR_ROUTE_KEY:
    base_url: http://YOUR_LLM_HOST/v1
    api_key: YOUR_API_KEY
```

4. Ensure `--llm-model` equals the selected route key.

Start Gateway with:

```bash
python -m gateway --config gateway/config.local.yaml
```

If Gateway is already running, verify the session root and route key instead of starting a second server on the same port.

## Result Checks

After a run, inspect:

- Launcher stdout/stderr for runtime failures.
- Gateway logs for route, session, admission, or upstream errors.
- The configured SQLite DB or storage backend for completed rows.
- Evaluator output and final reward when `--enable-evaluation` is used.

Common failures:

- `--llm-model` does not match a Gateway route key.
- Gateway base URL points to an LLM proxy instead of `/v1/sessions`.
- Docker image or mounted runner path is missing.
- Gateway and Launcher write to different storage backends.
- Runner exits non-zero for benchmark-level failure instead of returning a failed result.
