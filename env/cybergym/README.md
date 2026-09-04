# CyberGym Safactory harness

The controller supports `openhands`, `opencode`, `codex`, and `claude_code` through the same
task server, PoC discovery, verification, and Safactory result protocol.

Build the Codex/OpenCode/Claude Code controller from the repository root. It uses a
dedicated Dockerfile so the original OpenHands controller build remains
unchanged:

```bash
docker build -f env/cybergym/Dockerfile.codex -t cybergym-safactory:codex .
```

The Codex Dockerfile also contains the OpenCode adapter and intentionally skips
the large OpenHands source build by default. To produce a combined controller
that additionally embeds OpenHands, use:

```bash
docker build --build-arg BUILD_OPENHANDS=true \
  -f env/cybergym/Dockerfile.codex -t cybergym-safactory:all .
```

The original `env/cybergym/Dockerfile` remains the legacy OpenHands-only
controller and retains its original build context and behavior.

Build and archive the pinned CyberGym Codex agent image:

```bash
bash cybergym-agent-examples/codex/install.sh
docker save cybergym/codex:latest -o /path/to/cybergym/images/codex.tar
```

Build and archive the Claude Code agent image:

```bash
bash cybergym-agent-examples/claude_code/install.sh
docker save cybergym/claude-code:latest -o /path/to/cybergym/images/claude-code.tar
```

For a reproducible image, pin the npm package version while building:

```bash
CLAUDE_CODE_VERSION=<version> \
  bash cybergym-agent-examples/claude_code/install.sh
```

Set the controller image in `cybergym_config.claude-code.rjob.yaml`, then run:

```bash
python launcher.py --mode rjob \
  --agent-config env/cybergym/cybergym_config.claude-code.rjob.yaml \
  --agent-start-config env/cybergym/cybergym_start.rjob.yaml \
  --gateway-base-url http://<gateway-host>:8000/v1/sessions \
  --llm-model <route-model> \
  --enable-evaluation \
  --job-id 'cybergym#claude-code' \
  --pool-size 5 --max-workers 5 --max-steps 100 \
  --storage-type sqlite --db-path sqlite://cybergym.db \
  --agent-start-timeout-s 9000 --rebuild-table
```

Claude Code calls the native Anthropic Messages endpoint at
`<session-url>/v1/messages`. The selected Gateway upstream route must therefore
support Anthropic Messages semantics; an OpenAI-only `/chat/completions`
upstream is not sufficient without a protocol-converting proxy.

Set the built controller tag in `cybergym_config.codex.rjob.yaml`, then run the
launcher with that environment config and `cybergym_start.rjob.yaml`. Codex is
routed through the per-session Safactory OpenAI-compatible gateway; its final
PoC is verified by the same CyberGym evaluator used for the other agents.
