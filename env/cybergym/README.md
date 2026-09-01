# CyberGym Safactory harness

The controller supports `openhands`, `opencode`, and `codex` through the same
task server, PoC discovery, verification, and Safactory result protocol.

Build the Codex/OpenCode controller from the repository root. It uses a
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

Set the built controller tag in `cybergym_config.codex.rjob.yaml`, then run the
launcher with that environment config and `cybergym_start.rjob.yaml`. Codex is
routed through the per-session Safactory OpenAI-compatible gateway; its final
PoC is verified by the same CyberGym evaluator used for the other agents.
