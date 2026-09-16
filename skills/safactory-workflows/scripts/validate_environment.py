#!/usr/bin/env python3
"""Static consistency checks for a SAfactory environment directory.

Usage (from the repository root, matching launcher path resolution):

    python skills/safactory-workflows/scripts/validate_environment.py env/<name> [--expect-evaluation] [--json]

Every finding carries a *side* label that maps the problem to an owner:

- ``config``  — integration wiring in this directory (config/start YAMLs, file
  set, naming).  Fix by editing the environment's files.
- ``env``     — environment/benchmark side (adapter code, dataset rows,
  request fixture).  Fix in adapter.py, the dataset, or the benchmark image.
- ``safactory`` — framework side; if a check fails here the integration is
  fine and the issue belongs to SAfactory itself.

Exit code 0 when there are no error-severity findings (warnings allowed).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import sys
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - degrade like live_smoke does
    yaml = None

SIDE_ENV = "env"
SIDE_CONFIG = "config"
SIDE_SAFATORY = "safactory"
SEV_ERROR = "error"
SEV_WARN = "warn"
SEV_INFO = "info"

# Loose protocol markers of the standard runner shell.  A missing marker means
# someone hand-edited the standard part; restore it from the skill assets.
RUNNER_MARKERS = (
    "def read_request",
    "SAFACTORY_START_REQUEST_JSON",
    "SAFACTORY_RESULT_PATH",
    "from adapter import run_case",
    "redirect_stdout",
    "SAFACTORY_RUNNER_DIAGNOSTIC",
)


@dataclass
class Finding:
    code: str
    side: str
    severity: str
    file: str
    message: str
    hint: str = ""

    def line(self) -> str:
        text = f"[{self.severity}] [{self.side}] {self.file}: {self.message}"
        return f"{text}\n        hint: {self.hint}" if self.hint else text


class Report:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, code: str, side: str, severity: str, file: str, message: str, hint: str = "") -> None:
        self.findings.append(Finding(code, side, severity, file, message, hint))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEV_ERROR]

    def ok(self) -> bool:
        return not self.errors


def validate_environment(env_dir: Path, expect_evaluation: bool = False) -> Report:
    report = Report()
    env_dir = env_dir.resolve()
    name = env_dir.name

    if not env_dir.is_dir():
        report.add("env-dir", SIDE_CONFIG, SEV_ERROR, str(env_dir), "environment directory does not exist")
        return report
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        report.add(
            "env-name", SIDE_CONFIG, SEV_ERROR, name,
            "environment name must start with a lowercase letter and contain only a-z, 0-9, _",
        )

    _check_file_set(env_dir, name, expect_evaluation, report)
    _check_python_files(env_dir, report)
    if yaml is None:
        report.add(
            "yaml-missing", SIDE_SAFATORY, SEV_INFO, "-",
            "PyYAML is not installed; YAML consistency checks skipped",
            "pip install pyyaml to enable config/start/dataset checks",
        )
        return report

    docker_cfg = _load_yaml(env_dir / f"{name}_config.yaml", report)
    rjob_cfg = _load_yaml(env_dir / f"{name}_config.rjob.yaml", report)
    docker_start = _load_yaml(env_dir / f"{name}_start.yaml", report)
    rjob_start = _load_yaml(env_dir / f"{name}_start.rjob.yaml", report)

    docker_env = _check_config(docker_cfg, env_dir, name, report, docker=True)
    rjob_env = _check_config(rjob_cfg, env_dir, name, report, docker=False)
    docker_mounts = _check_docker_start(docker_start, env_dir, name, docker_env, report)
    rjob_targets, rjob_adapter_target = _check_rjob_start(rjob_start, env_dir, name, rjob_env, report)
    _check_cross(docker_env, rjob_env, docker_cfg, rjob_cfg, report)

    docker_adapter_target = next(
        (m["target"] for m in docker_mounts if m["target"].endswith("adapter.py")), ""
    )
    if docker_adapter_target and rjob_adapter_target and docker_adapter_target != rjob_adapter_target:
        report.add(
            "adapter-path-drift", SIDE_CONFIG, SEV_ERROR, "adapter.py",
            f"docker mounts adapter at {docker_adapter_target!r} but rjob embeds it at {rjob_adapter_target!r}",
            "keep the container path identical so `from adapter import run_case` works in both modes",
        )

    rows = _check_dataset(docker_env, env_dir, docker_cfg, report)
    _check_dataset_paths(rows, docker_mounts, rjob_targets, report)
    _check_smoke_request(env_dir, report)
    return report


def _check_file_set(env_dir: Path, name: str, expect_evaluation: bool, report: Report) -> None:
    required = [
        "runner.py",
        "adapter.py",
        "request.smoke.json",
        f"{name}_config.yaml",
        f"{name}_start.yaml",
        f"{name}_config.rjob.yaml",
        f"{name}_start.rjob.yaml",
    ]
    for rel in required:
        if not (env_dir / rel).is_file():
            report.add(
                "file-set", SIDE_CONFIG, SEV_ERROR, rel, f"required file {rel!r} is missing",
                "generate it from skills/safactory-workflows/assets/environment/ templates",
            )
    if expect_evaluation and not (env_dir / "rule_evaluator.py").is_file():
        report.add(
            "evaluator-missing", SIDE_CONFIG, SEV_ERROR, "rule_evaluator.py",
            "--expect-evaluation is set but rule_evaluator.py is missing",
            "with --enable-evaluation a missing evaluator marks every episode as failed",
        )
    if not (env_dir / "datasets").is_dir():
        report.add("datasets-dir", SIDE_CONFIG, SEV_WARN, "datasets/", "datasets/ directory is missing")
    if not (env_dir / "results").is_dir():
        report.add("results-dir", SIDE_CONFIG, SEV_WARN, "results/",
                   "results/ directory is missing", "add results/.gitkeep so the host mount target exists")


def _check_python_files(env_dir: Path, report: Report) -> None:
    evaluator = env_dir / "rule_evaluator.py"
    for path in [env_dir / "runner.py", env_dir / "adapter.py", evaluator]:
        if not path.is_file():
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            report.add("py-compile", SIDE_ENV, SEV_ERROR, path.name, f"Python syntax error: {exc}")
    runner = env_dir / "runner.py"
    if runner.is_file():
        source = runner.read_text(encoding="utf-8")
        missing = [m for m in RUNNER_MARKERS if m not in source]
        if missing:
            report.add(
                "runner-drift", SIDE_CONFIG, SEV_WARN, "runner.py",
                f"standard protocol shell markers missing: {', '.join(missing)}",
                "runner.py is a fixed standard part; restore it from the skill assets or env/prmeval",
            )
    if evaluator.is_file():
        source = evaluator.read_text(encoding="utf-8")
        if not any(marker in source for marker in ("def evaluate_rule", "class RuleEvaluator", "def evaluate")):
            report.add(
                "evaluator-contract", SIDE_ENV, SEV_ERROR, "rule_evaluator.py",
                "exports none of evaluate_rule / RuleEvaluator / evaluate",
                "the framework discovers the evaluator by these entry points",
            )


def _load_yaml(path: Path, report: Report) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        report.add("yaml-parse", SIDE_CONFIG, SEV_ERROR, path.name, f"invalid YAML: {exc}")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _config_entry(config: dict[str, Any]) -> dict[str, Any]:
    entries = config.get("environments")
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        return entries[0]
    return {}


def _check_config(
    config: dict[str, Any], env_dir: Path, name: str, report: Report, *, docker: bool
) -> dict[str, Any]:
    suffix = "_config.yaml" if docker else "_config.rjob.yaml"
    if not config:
        report.add("config-empty", SIDE_CONFIG, SEV_ERROR, name + suffix,
                   "missing `environments:` list or file unparseable")
        return {}
    entry = _config_entry(config)
    rel = name + suffix

    if entry.get("env_name") != name:
        report.add(
            "env-name", SIDE_CONFIG, SEV_ERROR, rel,
            f"env_name {entry.get('env_name')!r} does not match directory name {name!r}",
            "the scheduler looks up start configs by env_name",
        )
    image = str(entry.get("env_image") or "")
    if not image:
        report.add("env-image", SIDE_CONFIG, SEV_ERROR, rel, "env_image is empty")
    elif docker and (":" not in image or "/" in image and image.rsplit("/", 1)[0].startswith("gpfs")):
        report.add("env-image", SIDE_CONFIG, SEV_WARN, rel,
                   f"docker env_image {image!r} should carry an explicit tag")
    env_num = entry.get("env_num", 1)
    if not isinstance(env_num, int) or env_num < 1:
        report.add("env-num", SIDE_CONFIG, SEV_WARN, rel, "env_num must be an integer >= 1")

    mode = str(entry.get("dataset_load_mode") or "")
    if mode and mode not in {"eager", "parquet_row_ref"}:
        report.add("dataset-mode", SIDE_CONFIG, SEV_ERROR, rel,
                   f"dataset_load_mode {mode!r} is not one of eager / parquet_row_ref")

    env_params = entry.get("env_params")
    if not isinstance(env_params, dict):
        report.add("env-params", SIDE_CONFIG, SEV_ERROR, rel,
                   "env_params must be a mapping; it is serialized into every request")
        env_params = {}
    else:
        results_root = str(env_params.get("results_root") or "")
        if not results_root:
            report.add("results-root", SIDE_CONFIG, SEV_WARN, rel,
                       "env_params.results_root is not set",
                       "SAFACTORY_RESULT_PATH then defaults to /app/results and may not match any mount")

    dataset = str(entry.get("dataset") or "")
    if dataset and not dataset.startswith(("http", "hf://")) and mode != "parquet_row_ref":
        dataset_path = Path(dataset)
        if not dataset_path.is_absolute():
            dataset_path = env_dir / dataset_path
        if not dataset_path.is_file():
            report.add("dataset-file", SIDE_CONFIG, SEV_ERROR, rel,
                       f"dataset file {dataset!r} not found",
                       "dataset paths resolve relative to the config file's directory")
    return env_params


def _mounts_of(start: dict[str, Any]) -> list[dict[str, str]]:
    container = start.get("container") or {}
    mounts = container.get("mounts") or []
    result = []
    for mount in mounts:
        if isinstance(mount, str):
            source, _, target = mount.partition(":")
            result.append({"source": source, "target": target, "mode": "ro"})
        elif isinstance(mount, dict):
            result.append({
                "source": str(mount.get("source") or mount.get("hostPath") or mount.get("host_path") or ""),
                "target": str(mount.get("target") or mount.get("containerPath") or mount.get("container_path") or ""),
                "mode": str(mount.get("mode") or "ro"),
            })
    return result


def _check_docker_start(
    start: dict[str, Any], env_dir: Path, name: str, env_params: dict[str, Any], report: Report
) -> list[dict[str, str]]:
    rel = f"{name}_start.yaml"
    if not start:
        report.add("start-empty", SIDE_CONFIG, SEV_ERROR, rel, "file missing or unparseable")
        return []
    agent_names = {str(start.get("agent_name") or "")} if start.get("agent_name") else set(start.get("agents") or {})
    if name not in agent_names:
        report.add(
            "agent-name", SIDE_CONFIG, SEV_ERROR, rel,
            f"agent_name {sorted(a for a in agent_names if a)} does not match env_name {name!r}",
            "agent_name must equal env_name so the scheduler finds this start config",
        )

    container = start.get("container") or {}
    entrypoint = container.get("runner_entrypoint") or {}
    if isinstance(entrypoint, dict):
        source = str(entrypoint.get("source") or "")
        if source:
            source_path = Path(source)
            if not source_path.is_absolute():
                # runner_entrypoint sources resolve from the config file.
                source_path = (env_dir / source_path).resolve()
            if not source_path.is_file():
                report.add("entrypoint-source", SIDE_CONFIG, SEV_ERROR, rel,
                           f"runner_entrypoint source {source!r} not found")
        if not str(entrypoint.get("command") or ""):
            report.add("entrypoint-command", SIDE_CONFIG, SEV_ERROR, rel,
                       "runner_entrypoint.command is empty")

    mounts = _mounts_of(start)
    if not any(m["target"].endswith("adapter.py") for m in mounts):
        report.add(
            "adapter-mount", SIDE_CONFIG, SEV_WARN, rel,
            "adapter.py is not mounted into the container",
            "the runner does `from adapter import run_case`; mount it beside the runner target",
        )
    for mount in mounts:
        source = mount["source"]
        if not source or source.startswith(("gpfs://", "nfs://")) or source.startswith("/"):
            if source.startswith(("gpfs://", "nfs://")) or (source.startswith("/") and not Path(source).exists()):
                report.add("mount-source", SIDE_CONFIG, SEV_WARN, rel,
                           f"mount source {source!r} cannot be checked from this workstation")
            continue
        # Plain bind sources resolve from the launcher cwd (repository root).
        resolved = Path(source)
        if not resolved.exists():
            report.add(
                "mount-source", SIDE_CONFIG, SEV_ERROR, rel,
                f"mount source {source!r} not found from the current directory",
                "docker bind sources resolve from the launcher's cwd; run from the repository root",
            )
        if mount["target"]:
            _check_mount_mode_rw(rel, mount, report)

    results_root = str(env_params.get("results_root") or "")
    if results_root and not any(m["target"].rstrip("/") == results_root.rstrip("/") for m in mounts):
        report.add(
            "results-mount", SIDE_CONFIG, SEV_ERROR, rel,
            f"env_params.results_root {results_root!r} is not a mount target",
            "SAFACTORY_RESULT_PATH artifacts would stay inside the container; mount the host results dir there",
        )
    return mounts


def _check_mount_mode_rw(rel: str, mount: dict[str, str], report: Report) -> None:
    if "results" in mount["target"] and mount["mode"] not in {"rw", "w"}:
        report.add("results-mount-ro", SIDE_CONFIG, SEV_WARN, rel,
                   f"results mount {mount['target']!r} is read-only", "artifacts cannot be written")


def _check_rjob_start(
    start: dict[str, Any], env_dir: Path, name: str, env_params: dict[str, Any], report: Report
) -> tuple[list[str], str]:
    rel = f"{name}_start.rjob.yaml"
    if not start:
        report.add("start-empty", SIDE_CONFIG, SEV_ERROR, rel, "file missing or unparseable")
        return [], ""
    agent_name = str(start.get("agent_name") or "")
    if agent_name and agent_name != name:
        report.add("agent-name", SIDE_CONFIG, SEV_ERROR, rel,
                   f"agent_name {agent_name!r} does not match env_name {name!r}")
    rjob = start.get("rjob")
    if not isinstance(rjob, dict):
        report.add("rjob-section", SIDE_CONFIG, SEV_ERROR, rel, "rjob: section is missing")
        return [], ""

    embedded = [e for e in rjob.get("embedded_files") or [] if isinstance(e, dict)]
    if not embedded and not str((start.get("container") or {}).get("runner_entrypoint", {}).get("source") or ""):
        report.add("embedded-files", SIDE_CONFIG, SEV_WARN, rel, "no embedded_files declared")
    adapter_targets = []
    for item in embedded:
        source = str(item.get("source") or "")
        target = str(item.get("target") or "")
        if target.endswith("adapter.py"):
            adapter_targets.append(target)
        if source:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = (env_dir / source_path).resolve()
            if not source_path.is_file():
                report.add("embedded-source", SIDE_CONFIG, SEV_ERROR, rel,
                           f"embedded file source {source!r} not found")
    if not adapter_targets:
        report.add("adapter-embed", SIDE_CONFIG, SEV_WARN, rel,
                   "adapter.py is not in embedded_files",
                   "the runner does `from adapter import run_case`; embed it at the same path as the docker mount")

    mount_config = rjob.get("mount_config") or rjob.get("mount") or []
    if isinstance(mount_config, str):
        mount_config = [mount_config]
    targets = []
    results_root = str(env_params.get("results_root") or "")
    has_results_mount = False
    cluster_storage_reported = False
    for spec in mount_config if isinstance(mount_config, list) else []:
        text = str(spec)
        # mount specs look like "gpfs://SRC:/container/target"; keep the part after the last ':'.
        target = text.rsplit(":", 1)[-1] if ":" in text else ""
        targets.append(target)
        if "CLUSTER_STORAGE" in text and not cluster_storage_reported:
            cluster_storage_reported = True
            report.add("cluster-storage", SIDE_CONFIG, SEV_WARN, rel,
                       "CLUSTER_STORAGE placeholder is still present",
                       "replace it with a path visible to the RJob cluster before live runs")
        if results_root and target.rstrip("/") == results_root.rstrip("/"):
            has_results_mount = True
    if results_root and mount_config and not has_results_mount:
        report.add("results-mount", SIDE_CONFIG, SEV_WARN, rel,
                   f"env_params.results_root {results_root!r} is not an rjob mount_config target",
                   "results stay on the node-local filesystem otherwise")
    return targets, (adapter_targets[0] if adapter_targets else "")


def _check_cross(
    docker_env: dict[str, Any], rjob_env: dict[str, Any],
    docker_cfg: dict[str, Any], rjob_cfg: dict[str, Any], report: Report
) -> None:
    docker_entry, rjob_entry = _config_entry(docker_cfg), _config_entry(rjob_cfg)
    if docker_entry and rjob_entry:
        for key in ("dataset", "dataset_load_mode"):
            if docker_entry.get(key) != rjob_entry.get(key):
                report.add(
                    "config-drift", SIDE_CONFIG, SEV_WARN, key,
                    f"docker {key} {docker_entry.get(key)!r} != rjob {key} {rjob_entry.get(key)!r}",
                    "keep task parameters identical across the two config files",
                )
    docker_params = {k: v for k, v in docker_env.items() if k != "results_root"}
    rjob_params = {k: v for k, v in rjob_env.items() if k != "results_root"}
    if docker_params != rjob_params:
        report.add("config-drift", SIDE_CONFIG, SEV_WARN, "env_params",
                   "env_params (excluding results_root) differ between docker and rjob configs",
                   "native benchmark options must behave identically in both modes")


def _check_dataset(
    env_params: dict[str, Any], env_dir: Path, config: dict[str, Any], report: Report
) -> list[dict[str, Any]]:
    entry = _config_entry(config)
    dataset = str(entry.get("dataset") or "")
    if not dataset or dataset.startswith(("http", "hf://")):
        return []
    path = Path(dataset)
    if not path.is_absolute():
        path = env_dir / path
    if not path.is_file() or path.suffix not in {".jsonl", ".ndjson", ".json", ".yaml", ".yml"}:
        return []
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix in {".jsonl", ".ndjson"}:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        elif path.suffix == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            rows = loaded if isinstance(loaded, list) else [loaded]
        else:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            rows = loaded if isinstance(loaded, list) else [loaded]
    except (ValueError, yaml.YAMLError) as exc:
        report.add("dataset-parse", SIDE_ENV, SEV_ERROR, path.name, f"dataset row is not valid JSON/YAML: {exc}")
        return []
    bad = [i for i, row in enumerate(rows) if not isinstance(row, dict)]
    if bad:
        report.add("dataset-row", SIDE_ENV, SEV_ERROR, path.name,
                   f"rows {bad[:5]} are not JSON objects", "one dataset row = one episode and must be an object")
        return []
    if rows:
        report.add("dataset-info", SIDE_ENV, SEV_INFO, path.name,
                   f"{len(rows)} row(s); first-row keys: {', '.join(sorted(map(str, rows[0].keys())))[:160]}")
    return rows


def _iter_path_values(row: dict[str, Any], prefix: str = ""):
    for key, value in row.items():
        field = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _iter_path_values(value, f"{field}.")
        elif isinstance(value, str) and value.startswith("/"):
            yield field, value


def _check_dataset_paths(
    rows: list[dict[str, Any]], docker_mounts: list[dict[str, str]], rjob_targets: list[str], report: Report
) -> None:
    if not rows:
        return
    targets = [m["target"].rstrip("/") for m in docker_mounts if m["target"]]
    targets += [t.rstrip("/") for t in rjob_targets if t]
    for row in rows[:50]:
        for field, value in _iter_path_values(row):
            if not any(value == t or value.startswith(t + "/") for t in targets):
                report.add(
                    "dataset-path", SIDE_ENV, SEV_ERROR, field,
                    f"row {row.get('id', rows.index(row))} field {field!r} points to {value!r}, "
                    "which is under no mount target",
                    "absolute paths in dataset rows must match a container mount target in both start files",
                )
                break  # one finding per row is enough to act on


def _check_smoke_request(env_dir: Path, report: Report) -> None:
    path = env_dir / "request.smoke.json"
    if not path.is_file():
        return
    try:
        request = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        report.add("smoke-request", SIDE_ENV, SEV_ERROR, path.name, f"invalid JSON: {exc}")
        return
    if not isinstance(request, dict):
        report.add("smoke-request", SIDE_ENV, SEV_ERROR, path.name, "must be a JSON object")
        return
    for key in ("session_id", "model"):
        if not str(request.get(key) or ""):
            report.add("smoke-request", SIDE_ENV, SEV_ERROR, path.name, f"field {key!r} is empty")
    dataset = (request.get("env_params") or {}).get("dataset")
    if not isinstance(dataset, dict) or not dataset:
        report.add("smoke-request", SIDE_ENV, SEV_ERROR, path.name,
                   "env_params.dataset must be a non-empty object (one case row)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_dir", type=Path, help="environment directory, e.g. env/mybench")
    parser.add_argument("--expect-evaluation", action="store_true",
                        help="require rule_evaluator.py to exist")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    args = parser.parse_args(argv)

    report = validate_environment(args.env_dir, expect_evaluation=args.expect_evaluation)
    if args.json:
        print(json.dumps({"ok": report.ok(), "findings": [asdict(f) for f in report.findings]},
                         ensure_ascii=False, indent=2))
    else:
        for finding in report.findings:
            if finding.severity != SEV_INFO:
                print(finding.line())
        errors, warns = len(report.errors), sum(1 for f in report.findings if f.severity == SEV_WARN)
        print(f"validate: {errors} error(s), {warns} warning(s) -> {'FAIL' if not report.ok() else 'PASS'}")
    return 0 if report.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
