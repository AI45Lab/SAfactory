#!/usr/bin/env python3
"""Copy the guide-derived templates without overwriting an existing environment."""
import argparse
from pathlib import Path
import re

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "environment"


def scaffold(name, mode, env_root, enable_evaluation=False):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("name must start with a lowercase letter and contain only a-z, 0-9, _")
    if mode not in {"docker", "rjob"}:
        raise ValueError("mode must be docker or rjob")
    env_root = Path(env_root)
    env_root.mkdir(parents=True, exist_ok=True)
    destination = env_root / name
    # Docker bind-mount sources are resolved by SAfactory from the launcher's
    # current working directory (the repository root for the live-smoke
    # workflow), while runner/embedded sources are resolved from the config
    # file.  Preserve that distinction in generated Docker YAML.
    env_root_ref = env_root.expanduser().as_posix()
    if not Path(env_root_ref).is_absolute() and not env_root_ref.startswith("."):
        env_root_ref = f"./{env_root_ref}"
    sources = {
        "runner.py": "runner.py", "adapter.py": "adapter.py",
        "request.smoke.json": "request.smoke.json.tmpl",
        # Docker is the base contract.  RJob adds two files; generating both
        # pairs up front keeps an environment portable between deployment modes.
        f"{name}_config.yaml": "config.yaml.tmpl",
        f"{name}_start.yaml": "start.docker.yaml.tmpl",
        f"{name}_config.rjob.yaml": "config.rjob.yaml.tmpl",
        f"{name}_start.rjob.yaml": "start.rjob.yaml.tmpl",
    }
    if enable_evaluation:
        sources["rule_evaluator.py"] = "rule_evaluator.py"
    contents = {target: (ASSETS / source).read_text(encoding="utf-8")
                .replace("__ENV_NAME__", name)
                .replace("__ENV_ROOT__", env_root_ref)
                for target, source in sources.items()}
    destination.mkdir(parents=True, exist_ok=False)
    for target, content in contents.items():
        (destination / target).write_text(content, encoding="utf-8")
    (destination / "datasets").mkdir()
    (destination / "datasets" / "smoke.jsonl").write_text(
        '{"task_id": "hello-001", "prompt": "Write one short greeting."}\n', encoding="utf-8"
    )
    (destination / "results").mkdir()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--mode", required=True, choices=["docker", "rjob"])
    parser.add_argument("--env-root", type=Path, default=Path("env"))
    parser.add_argument("--enable-evaluation", action="store_true")
    args = parser.parse_args()
    path = scaffold(args.name, args.mode, args.env_root, args.enable_evaluation)
    print(
        f"Created {path} for {args.mode} (Docker and RJob config pairs included). "
        "Fill adapter.py and deployment values; replace the example dataset."
    )
    if args.enable_evaluation:
        print("Fill score_metrics in rule_evaluator.py before running evaluation.")


if __name__ == "__main__":
    main()
