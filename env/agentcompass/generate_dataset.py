"""Build a SAfactory task manifest using the pinned native SciCode loader.

Deployment preparation and native loading use the pinned AgentCompass Python.
Daily --catalog selection only needs standard Python. --prepare may download a
missing official asset; no mode starts a model, environment session or database.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

PINNED_COMMIT = "d2c3e148902e948db3270fa34b2198fb1b10beb7"
SCICODE_URL = "http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/agentcompass/scicode.zip"
# Existing GPFS bytes verified against the official archive size/CRC32 on
# 2026-10-22; retain SHA256 so subsequent preparation is entirely offline.
H5_SIZE = 1049345865
H5_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
H5_CRC32 = "9fe0ae53"
ASSET_FILES = ("problems_dev.jsonl", "problems_test.jsonl", "background_comment_template.txt",
               "multistep_template.txt", "13.6.txt", "62.1.txt", "76.3.txt")
SOURCE_HASHES = {
    "SciCodeBenchmark": "fdc4aa30db3ead957b05a7d0135591cb6c5ec7984d86f2af8071371c859a11e1",
    "SciCodeToolUseHarness": "0447a0b452e949f6b9f76e68d410000f61c66fab14db6d7bcfe82344cfe82e5a",
}
SPLIT_FILES = {
    "validation": ("problems_dev.jsonl",),
    "test": ("problems_test.jsonl",),
    "all": ("problems_dev.jsonl", "problems_test.jsonl"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_h5(path: Path, digest: str | None = None) -> None:
    if path.stat().st_size != H5_SIZE or (digest or sha256(path)) != H5_SHA256:
        raise ValueError(f"{path} is not the verified official SciCode HDF5 asset")


def verify_h5_structure(native_dir: Path) -> dict:
    import h5py

    expected = {}
    for filename in SPLIT_FILES["all"]:
        for line in (native_dir / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            problem = json.loads(line)
            for step in problem["sub_steps"]:
                key = str(step["step_number"])
                if key not in {"13.6", "62.1", "76.3"}:  # Official prefilled steps.
                    expected[key] = len(step["test_cases"])
    with h5py.File(native_dir / "test_data.h5", "r") as data:
        if set(data) != set(expected):
            raise ValueError("SciCode HDF5 subproblem groups differ from the pinned definitions")
        for key, count in expected.items():
            for index in range(1, count + 1):
                if not isinstance(data.get(f"{key}/test{index}"), h5py.Group):
                    raise ValueError(f"Missing SciCode HDF5 group: {key}/test{index}")
    return {"subproblems": len(expected), "test_groups": sum(expected.values())}


def prepare_shared_data(data_dir: Path, reuse_h5: Path | None = None) -> dict:
    """Deployment only: publish immutable inputs, reusing verified assets."""
    from agentcompass.benchmarks.scicode import SciCodeBenchmark

    package_dir = Path(inspect.getfile(SciCodeBenchmark)).parent / "data"
    if sha256(Path(inspect.getfile(SciCodeBenchmark))) != SOURCE_HASHES["SciCodeBenchmark"]:
        raise ValueError(f"Preparation requires AgentCompass {PINNED_COMMIT}")
    data_dir.mkdir(parents=True, exist_ok=True)
    native_dir = data_dir / "scicode"
    with (data_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reused = native_dir.exists()
        if not reused:
            with tempfile.TemporaryDirectory(prefix=".scicode-", dir=data_dir) as tmp:
                stage = Path(tmp) / "scicode"
                stage.mkdir(mode=0o755)
                for name in ASSET_FILES:
                    shutil.copyfile(package_dir / name, stage / name)
                if reuse_h5 is not None:
                    verify_h5(reuse_h5)
                    shutil.copyfile(reuse_h5, stage / "test_data.h5")
                else:
                    archive = Path(tmp) / "scicode.zip"
                    with urllib.request.urlopen(SCICODE_URL, timeout=60) as source, archive.open("wb") as out:
                        shutil.copyfileobj(source, out)
                    with zipfile.ZipFile(archive) as bundle:
                        # Extract only the known asset. Templates/definitions
                        # come from the pinned source, not a mutable archive.
                        with bundle.open("scicode/test_data.h5") as source, (stage / "test_data.h5").open("wb") as out:
                            shutil.copyfileobj(source, out)
                verify_h5(stage / "test_data.h5")
                verify_h5_structure(stage)
                for path in stage.iterdir():
                    path.chmod(0o644)
                stage.rename(native_dir)
        # Never overwrite or 'repair' an existing deployment silently.
        assets = {}
        for name in ASSET_FILES:
            actual = sha256(native_dir / name)
            if actual != sha256(package_dir / name):
                raise ValueError(f"Existing shared asset differs from pinned source: {name}")
            assets[name] = actual
        verify_h5(native_dir / "test_data.h5")
        assets["test_data.h5"] = H5_SHA256
        structure = verify_h5_structure(native_dir)
        tasks, _ = load_native_tasks(data_dir, "validation")
        catalog_dir = data_dir / "manifests"
        catalog_dir.mkdir(mode=0o755, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".catalog-", dir=data_dir) as tmp:
            output = Path(tmp) / "scicode-validation.jsonl"
            report = write_manifest(output, tasks, split="validation",
                                    container_data_dir="/app/data/agentcompass", assets=assets)
            report.update({"task_manifest_sha256": sha256(output), "hdf5_structure": structure,
                           "official_archive_url": SCICODE_URL, "official_hdf5_crc32": H5_CRC32})
            metadata = output.with_suffix(".metadata.json")
            metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for path in (output, metadata):
                target = catalog_dir / path.name
                if target.exists():
                    if target.read_bytes() != path.read_bytes():
                        raise ValueError(f"Shared catalog differs; use a new version directory: {target}")
                else:
                    path.chmod(0o644)
                    path.rename(target)
        return {"data_dir": str(data_dir), "reused_complete_data": reused, **report}


def load_catalog(path: Path):
    """Read a deployment-produced catalog without native/model dependencies."""
    report = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if report.get("agentcompass_commit") != PINNED_COMMIT or report.get("task_manifest_sha256") != sha256(path):
        raise ValueError("Shared catalog version or checksum mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = []
    for row in rows:
        if (row.get("benchmark"), row.get("harness"), row.get("environment")) != (
            "scicode", "scicode_tool_use", "host_process"
        ) or row.get("harness_params", {}).get("mode") != "naive":
            raise ValueError("Shared catalog contains an unsupported component combination")
        count = row.get("generation_calls_upper_bound")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Shared catalog is missing the native substep budget")
        tasks.append(SimpleNamespace(task_id=row["sample_id"], metadata={"sub_steps": [None] * count}))
    return tasks, report


def load_native_tasks(data_dir: Path, split: str):
    from agentcompass.benchmarks.scicode import SciCodeBenchmark
    from agentcompass.harnesses.scicode_tool_use import SciCodeToolUseHarness
    from agentcompass.runtime import RunRequest
    from agentcompass.runtime.config.settings import bootstrap_runtime
    import h5py

    for cls in (SciCodeBenchmark, SciCodeToolUseHarness):
        if sha256(Path(inspect.getfile(cls))) != SOURCE_HASHES[cls.__name__]:
            raise ValueError(f"{cls.__name__} does not match AgentCompass {PINNED_COMMIT}")

    # Require staged definitions, rather than silently using packaged fallback
    # data or letting the native loader download a missing HDF5 archive.
    native_dir = data_dir / "scicode"
    package_dir = Path(inspect.getfile(SciCodeBenchmark)).parent / "data"
    assets = {}
    for name in (*SPLIT_FILES[split], "test_data.h5"):
        path = native_dir / name
        if not path.is_file():
            raise ValueError(f"Missing native SciCode asset: {path}")
        assets[name] = sha256(path)
        if name.endswith(".jsonl") and assets[name] != sha256(package_dir / name):
            raise ValueError(f"{name} differs from the pinned official problem definitions")
    verify_h5(native_dir / "test_data.h5", assets["test_data.h5"])
    with h5py.File(native_dir / "test_data.h5", "r") as data:
        if not len(data):
            raise ValueError("SciCode test_data.h5 is empty")

    bootstrap_runtime(data_dir=str(data_dir), file_log_level=None, force=True)
    request = RunRequest.from_params({
        "benchmark": {"id": "scicode", "params": {"split": split}},
        "harness": {"id": "scicode_tool_use", "params": {"mode": "naive"}},
        "environment": {"id": "host_process"},
        "model": {"id": "offline-manifest", "api_protocol": "openai-chat"},
    })
    harness = SciCodeToolUseHarness()
    if not harness.supports(request.environment, request.model):
        raise ValueError("Pinned SciCode harness does not support this combination")
    tasks = SciCodeBenchmark().load_tasks(request)
    if not tasks:
        raise ValueError("Native SciCode loader returned no tasks")
    return tasks, assets


def select_tasks(tasks, *, sample_ids=None, limit=2):
    catalog = {}
    for task in tasks:
        key = str(task.task_id)
        if not key or key in catalog:
            raise ValueError(f"Empty or duplicate native sample ID: {key!r}")
        catalog[key] = task
    if sample_ids:
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Duplicate --sample-id")
        unknown = set(sample_ids) - catalog.keys()
        if unknown:
            raise ValueError(f"Unknown native sample IDs: {sorted(unknown)}")
        return [catalog[key] for key in sample_ids]
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive; use --all explicitly for the full split")
    return list(catalog.values()) if limit is None else list(catalog.values())[:limit]


def write_manifest(output: Path, tasks, *, split: str, container_data_dir: str,
                   assets: dict, timeout_seconds: int = 3600):
    data_root = PurePosixPath(container_data_dir)
    if not data_root.is_absolute() or ".." in data_root.parts:
        raise ValueError("--container-data-dir must be an absolute container path")
    if data_root == PurePosixPath("/app/results") or PurePosixPath("/app/results") in data_root.parents:
        raise ValueError("Native data must be separate from /app/results")
    if timeout_seconds < 1 or not tasks:
        raise ValueError("Positive timeout and at least one task are required")
    rows = []
    for task in tasks:
        rows.append({
            "task_id": f"scicode-{split}-{task.task_id}",
            "benchmark": "scicode", "harness": "scicode_tool_use",
            "environment": "host_process", "sample_id": str(task.task_id),
            "data_dir": str(data_root),
            "benchmark_params": {
                "split": split, "h5py_file": str(data_root / "scicode/test_data.h5"),
                "timeout": 1800,
            },
            "harness_params": {"mode": "naive", "with_background": True},
            "environment_params": {},
            "generation_calls_upper_bound": len(task.metadata.get("sub_steps") or []),
            "model_params": {"max_retries": 0},
            "timeout_seconds": timeout_seconds, "contract_only": False,
        })
    # Native naive mode makes one generation per non-prefilled substep. Counting
    # all substeps is a conservative bound; retries are disabled in the manifest.
    max_calls = max(len(task.metadata.get("sub_steps") or []) for task in tasks)
    provenance = {
        "agentcompass_commit": PINNED_COMMIT, "source_sha256": SOURCE_HASHES,
        "native_asset_sha256": assets, "split": split,
        "sample_ids": [str(task.task_id) for task in tasks],
        "harness_mode": "naive", "generation_calls_upper_bound_per_episode": max_calls,
        "recommended_max_steps": max(2, max_calls + 1),
        "note": "Asset hashes are provenance, not proof of benchmark or HDF5 evaluation success.",
    }
    # Data preparation only: scheduling remains in the standard Launcher.
    # Exclusive creation protects an earlier task manifest.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("scicode",), required=True,
                        help="Only combinations checked against the pinned runtime are supported")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path,
                        help="Native data root containing scicode/, visible to this process")
    source.add_argument("--catalog", type=Path, help="Deployment catalog; no native runtime or download needed")
    parser.add_argument("--prepare", action="store_true", help="Deployment only: prepare shared assets and validation catalog")
    parser.add_argument("--reuse-h5", type=Path, help="Deployment only: reuse an existing official HDF5 instead of downloading")
    parser.add_argument("--container-data-dir", default="/app/data/agentcompass")
    parser.add_argument("--split", choices=tuple(SPLIT_FILES), default="validation")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=2)
    selection.add_argument("--sample-id", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path, help="Task JSONL; must not exist (not required for --prepare)")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            if args.catalog or args.output:
                raise ValueError("--prepare requires --data-dir and cannot use --catalog or --output")
            print(json.dumps(prepare_shared_data(args.data_dir.resolve(), args.reuse_h5), indent=2))
            return 0
        if args.reuse_h5 or not args.output:
            raise ValueError("Task selection requires --output; --reuse-h5 is only valid with --prepare")
        if args.catalog:
            tasks, catalog = load_catalog(args.catalog)
            args.split = catalog["split"]
            assets = catalog["native_asset_sha256"]
        else:
            tasks, assets = load_native_tasks(args.data_dir.resolve(), args.split)
        selected = select_tasks(tasks, sample_ids=args.sample_id, limit=None if args.all else args.limit)
        if not args.catalog and any(str(t.task_id) == "80" for t in selected) and importlib.util.find_spec("matplotlib") is None:
            raise ValueError("SciCode sample 80 requires matplotlib in the runtime image")
        report = write_manifest(args.output, selected, split=args.split,
                                container_data_dir=args.container_data_dir,
                                assets=assets, timeout_seconds=args.timeout_seconds)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
