"""Deployment-side shared data for AgentCompass, and task selection for users.

One tool for every benchmark at the pinned AgentCompass revision:

* ``prepare`` (deployer, once, inside the pinned image with the shared data root
  mounted read-write): fetch each benchmark's complete native data tree from the
  source pinned by the benchmark module (or record the snapshot actually fetched
  when the module does not pin one), verify integrity, publish it atomically
  under its native ``<data_dir>`` location, then load it with the pinned native
  loader and publish a catalog of the loader's own task IDs. Re-running reuses a
  complete published tree and never overwrites it.
* ``status``: report what each profile still lacks.
* ``select`` (users, any Python): pick tasks from a catalog into a manifest for
  the standard ``launcher.py``.

Container images used by sandboxed tasks are recorded separately; they are
never copied into the shared data root.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import urllib.request
from pathlib import Path, PurePosixPath

PINNED_COMMIT = "d2c3e148902e948db3270fa34b2198fb1b10beb7"
STATE_DIR = ".published"
# Catalogs written by this tool; earlier catalogs in manifests/ stay untouched
# because existing manifests and acceptance records reference them.
CATALOG_DIR = "manifests/v2"
# Lock files the datasets/HF libraries leave beside caches; never assets.
RUNTIME_LOCK_SUFFIXES = (".lock",)
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"


def _pinned(module: str, name: str):
    import importlib

    return getattr(importlib.import_module(f"agentcompass.benchmarks.{module}"), name)


# Each profile states where the native loader reads data (``root``, relative to
# data_dir), how to obtain that tree, and which catalogs to publish. Source
# values are read from the pinned benchmark module at prepare time.
#   kind=zip      official archive; ``url`` names the pinned module constant
#   kind=hf       Hugging Face dataset snapshot
#   kind=git      git checkout at a pinned ref (or a recorded HEAD)
#   kind=tar_md5  checksum-pinned tar archive handled by the native loader
#   kind=native   the pinned loader itself fetches and verifies the data
#   kind=external no public download source at the pinned revision
PROFILES: dict[str, dict] = {
    "browsecomp": {"kind": "zip", "url": ("browsecomp", "DATASET_ZIP_URL"), "root": "browsecomp"},
    "browsecomp_zh": {"kind": "zip", "url": ("browsecomp_zh", "DATASET_ZIP_URL"), "root": "browsecomp_zh"},
    "deepsearchqa": {"kind": "zip", "url": ("deepsearchqa", "DATASET_ZIP_URL"), "root": "deepsearchqa"},
    "frontierscience": {"kind": "zip", "url": ("frontierscience", "DATASET_ZIP_URL"), "root": "frontierscience"},
    "sgi_deep_research": {"kind": "zip", "url": ("sgi_deep_research", "DATASET_ZIP_URL"),
                          "root": "sgi_deep_research"},
    # Attachments (gaia) and images (hle, screenspot) live in the same archive tree.
    "gaia": {"kind": "zip", "url": ("gaia", "DATASET_ZIP_URL"), "root": "gaia",
             "catalogs": {"llm": {"modality": "llm"}, "multimodal": {"modality": "vlm"}}},
    "hle": {"kind": "zip", "url": ("hle", "DATASET_ZIP_URL"), "root": "hle",
            "catalogs": {"llm": {"modality": "llm"}, "multimodal": {"modality": "vlm"}}},
    "screenspot": {"kind": "zip", "url": ("screenspot", "DATASET_ZIP_URL"), "root": "screenspot"},
    "swebench_verified": {"kind": "zip", "url": ("swebench_verified", "_DATASET_ZIP_URL"),
                          "root": "swe_bench_verified"},
    "scicode": {"kind": "scicode", "root": "scicode",
                "catalogs": {"validation": {"split": "validation"}, "test": {"split": "test"}}},
    "sealqa": {"kind": "hf", "repo": "vtllms/sealqa", "revision": ("sealqa", "DATASET_REVISION"),
               "root": "sealqa/267b8197ae75680ee0db180c4c2e96bd4e1001b4",
               # Files published before records existed; adopted only when they
               # match the LFS SHA256 of the pinned HF revision.
               "adopt_sha256": {
                   "seal-0.parquet": "53331f95a4579d2f707ea098774b41e5b7145abfbbc4b704694fab2b84d60e73",
                   "seal-hard.parquet": "efad00e6420ffebc35d45c47100c96105c5e83fad1e1c7ff31b451c8a78b216d",
                   "longseal.parquet": "f5d3ae72604f0131b1307308fbb16eb5e39236d81f770d6995156d326093d3a1",
               },
               "catalogs": {c: {"category": c} for c in ("seal_0", "seal_hard", "longseal")}},
    "hle_verified": {"kind": "hf", "repo": "skylenage-ai/HLE-Verified", "revision": None,
                     "root": "hle_verified_snapshot", "load_dataset_cache": "hle_verified",
                     "catalogs": {"llm": {"modality": "llm"}, "multimodal": {"modality": "vlm"}}},
    "swebench_multilingual": {"kind": "hf", "repo": "SWE-bench/SWE-bench_Multilingual", "revision": None,
                              "root": "swe_bench_multilingual"},
    "swebench_pro": {"kind": "hf", "repo": "ScaleAI/SWE-bench_Pro", "revision": None,
                     "root": "swe_bench_pro", "extra": "swebench_pro_run_scripts"},
    "gdpval_ac": {"kind": "hf", "repo": ("gdpval_ac.dataset", "DEFAULT_HF_REPO_ID"), "revision": None,
                  "root": "gdpval_ac", "extra": "gdpval_baseline"},
    "xbench_deepsearch": {"kind": "url_files", "root": "xbench_deepsearch",
                          "catalogs": {"2505": {"version": "2505"}, "2510": {"version": "2510"}}},
    "deepresearch_bench": {"kind": "zip_ref", "repo": "https://github.com/Ayanami0730/deep_research_bench",
                           "root": "deep_research_bench-main",
                           # Loading only: the default metrics need a Jina key at config
                           # time. Scoring metrics are never written into the catalog.
                           "load_only_params": {"metrics": ["race"]}},
    "researchclawbench": {"kind": "zip_ref", "repo": "https://github.com/InternScience/ResearchClawBench",
                          "root": "ResearchClawBench-main"},
    "deepswe": {"kind": "native", "root": "deepswe"},
    "frontier_engineering": {"kind": "native", "root": "frontier_engineering",
                             "catalogs": {t: {"task_set": t} for t in ("v1", "v1_non_gpu", "v1_filtered", "v1_lite")}},
    "pinchbench": {"kind": "native", "root": "pinchbench"},
    "taubench": {"kind": "native", "root": "taubench"},
    "wildclawbench": {"kind": "tar_md5", "root": "wildclawbench"},
    "terminal_bench_2": {"kind": "git", "repo": ("terminalbench2.terminalbench2", "_REPO_URL"),
                         "root": "terminalbench2/terminal-bench-2"},
    "terminal_bench_2_1": {"kind": "native", "root": "terminalbench2_1"},
    "terminal_bench_2_verified": {"kind": "git_lfs", "repo": ("terminalbench2.terminalbench2_verified", "_REPO_URL"),
                                  "root": "terminalbench2_verified"},
    # No download source at the pinned revision: the deployer supplies data in
    # the native layout via --source, and the native loader validates it.
    "skillsbench": {"kind": "external", "root": "skillsbench",
                    "reason": "pinned loader has no default download source (dataset_zip_url is empty)",
                    "layout": "<source>/tasks/<task_id>/{instruction.md,task.toml,...} as read by "
                              "SkillsBenchmark._resolve_tasks_root"},
    "special_pattern_check": {"kind": "external", "root": "special_pattern_check",
                              "reason": "loader reads a deployer-supplied dataset_dir/<version>.jsonl",
                              "layout": "<source>/<version>.jsonl with chat-format records",
                              "catalog_from_files": "*.jsonl"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value):
    return _pinned(*value) if isinstance(value, tuple) else value


def _tree_manifest(root: Path) -> dict[str, str]:
    """SHA256 of every published file; the integrity record for reuse checks."""
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and not path.name.endswith(RUNTIME_LOCK_SUFFIXES)
    }


def _verify_tree(root: Path, recorded: dict[str, str]) -> dict:
    actual = _tree_manifest(root)
    return {
        "missing": sorted(set(recorded) - set(actual)),
        "unrecorded": sorted(set(actual) - set(recorded)),
        "changed": sorted(name for name in set(actual) & set(recorded) if actual[name] != recorded[name]),
    }


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=300) as source, target.open("wb") as out:
        shutil.copyfileobj(source, out, length=8 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Staging: build the complete native tree for one profile under ``stage``.
# Each function returns provenance describing exactly what was fetched.
# ---------------------------------------------------------------------------

def _stage_zip(profile: dict, stage: Path) -> dict:
    from agentcompass.benchmarks.utils.data import _extract_zip, _zip_is_valid

    url = _resolve(profile["url"])
    archive = stage / ".archive.zip"
    _download(url, archive)
    if not _zip_is_valid(archive):
        raise ValueError(f"Official archive is invalid: {url}")
    provenance = {"source": url, "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size}
    # Same path-checked extractor the native helper uses; the archive carries
    # its own top-level directory, so the native root appears under stage.
    _extract_zip(archive, stage)
    archive.unlink()
    return provenance


def _stage_zip_ref(profile: dict, stage: Path) -> dict:
    """GitHub branch archive used by the module: pin the commit actually fetched."""
    repo = profile["repo"]
    head = subprocess.run(["git", "ls-remote", f"{repo}.git", "refs/heads/main"],
                          capture_output=True, text=True, check=True, timeout=120).stdout.split()[0]
    url = f"{repo}/archive/{head}.zip"
    provenance = _stage_zip({"url": url}, stage)
    extracted = stage / f"{repo.rsplit('/', 1)[1]}-{head}"
    extracted.rename(stage / profile["root"])
    return {**provenance, "source_commit": head, "module_default": f"{repo}/archive/refs/heads/main.zip"}


def _stage_hf(profile: dict, stage: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    repo = _resolve(profile["repo"])
    revision = _resolve(profile["revision"]) or HfApi().dataset_info(repo).sha
    target = stage / profile["root"]
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=str(target),
                      cache_dir=str(stage / ".hf-cache"))
    shutil.rmtree(stage / ".hf-cache", ignore_errors=True)
    shutil.rmtree(target / ".cache", ignore_errors=True)
    provenance = {"source": f"hf://datasets/{repo}", "revision": revision,
                  "revision_pinned_by_module": bool(_resolve(profile["revision"]))}
    if profile.get("extra") == "swebench_pro_run_scripts":
        provenance["run_scripts"] = _stage_swebench_pro_scripts(stage)
    if profile.get("extra") == "gdpval_baseline":
        url = _pinned("gdpval_ac.gdpval_ac", "BASELINE_ZIP_URL")
        provenance["baseline"] = _stage_zip({"url": url}, stage)
    return provenance


def _stage_swebench_pro_scripts(stage: Path) -> dict:
    """Run scripts the loader fetches from the unpinned main branch; pin the commit."""
    repo = "https://github.com/scaleapi/SWE-bench_Pro-os"
    head = subprocess.run(["git", "ls-remote", f"{repo}.git", "refs/heads/main"],
                          capture_output=True, text=True, check=True, timeout=120).stdout.split()[0]
    checkout = stage / ".swebench-pro-os"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "fetch", "-q", "--depth", "1", f"{repo}.git", head], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    (stage / "swe_bench_pro").mkdir(exist_ok=True)
    shutil.copytree(checkout / "run_scripts", stage / "swe_bench_pro" / "run_scripts")
    shutil.rmtree(checkout)
    return {"source": repo, "source_commit": head}


def _stage_git(profile: dict, stage: Path, *, lfs: bool = False) -> dict:
    repo = _resolve(profile["repo"])
    target = stage / profile["root"]
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", repo, str(target)], check=True, timeout=3600)
    if lfs:
        subprocess.run(["git", "-C", str(target), "lfs", "pull"], check=True, timeout=3600)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    shutil.rmtree(target / ".git")
    return {"source": repo, "source_commit": head, "revision_pinned_by_module": False}


def _stage_url_files(profile: dict, stage: Path) -> dict:
    revision = _pinned("xbench_deepsearch", "UPSTREAM_REVISION")
    urls = _pinned("xbench_deepsearch", "DATASET_URLS")
    target = stage / profile["root"]
    target.mkdir(parents=True)
    files = {}
    for version, url in sorted(urls.items()):
        path = target / f"DeepSearch-{version}.csv"
        _download(url, path)
        files[path.name] = url
    return {"source_revision": revision, "files": files}


def _stage_frontier_engineering(stage: Path) -> dict:
    """Fetch the pinned source tree and task assets the plan step would fetch per task."""
    from agentcompass.runtime.registry import BENCHMARKS, load_builtin_components

    load_builtin_components()
    native = BENCHMARKS.create("frontier_engineering")
    request = _native_request("frontier_engineering", {"task_set": "v1"}, stage)
    config = native.build_config(request)
    source_root = native._resolve_source_root(config)
    records = native._load_batch_records("v1")
    native._ensure_external_assets(source_root, records)
    return {"source": _pinned("frontier_engineering.frontier_engineering", "_DEFAULT_SOURCE_REPO_URL"),
            "source_commit": _pinned("frontier_engineering.frontier_engineering", "_DEFAULT_SOURCE_REPO_REF"),
            "external_assets": "fetched by the pinned scripts/bootstrap/fetch_task_assets.py for task_set v1"}


def _stage_native(benchmark: str, profile: dict, stage: Path) -> dict:
    """Let the pinned loader fetch into a private data_dir; it verifies its own pins."""
    if benchmark == "frontier_engineering":
        return _stage_frontier_engineering(stage)
    # Native loaders shell out to CLIs installed in the runtime venv (e.g. harbor).
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    load_native_tasks(benchmark, stage, profile.get("load_params") or {})
    if not (stage / profile["root"]).exists():
        raise ValueError(f"Pinned {benchmark} loader did not create {profile['root']}")
    return {"source": f"pinned {benchmark} loader", "note": "source and checksum enforced by native code"}


def _stage_external(profile: dict, stage: Path, source: Path | None) -> dict:
    if source is None:
        raise ValueError(f"{profile['reason']}; provide native data with --source ({profile['layout']})")
    if not source.is_dir():
        raise ValueError(f"--source must be a directory: {source}")
    shutil.copytree(source, stage / profile["root"], symlinks=False)
    return {"source": "deployer-supplied", "source_path": str(source), "layout": profile["layout"]}


def _stage(benchmark: str, profile: dict, stage: Path, source: Path | None = None) -> dict:
    kind = profile["kind"]
    if kind == "external":
        return _stage_external(profile, stage, source)
    if kind == "zip":
        return _stage_zip(profile, stage)
    if kind == "zip_ref":
        return _stage_zip_ref(profile, stage)
    if kind == "hf":
        return _stage_hf(profile, stage)
    if kind == "git":
        return _stage_git(profile, stage)
    if kind == "git_lfs":
        return _stage_git(profile, stage, lfs=True)
    if kind == "url_files":
        return _stage_url_files(profile, stage)
    if kind in {"native", "tar_md5"}:
        return _stage_native(benchmark, profile, stage)
    raise ValueError(f"{benchmark}: unsupported staging kind {kind!r}")


# ---------------------------------------------------------------------------
# Native loading and catalogs
# ---------------------------------------------------------------------------

def _native_request(benchmark: str, benchmark_params: dict, data_dir: Path):
    from agentcompass.runtime import RunRequest
    from agentcompass.runtime.config.settings import bootstrap_runtime

    from dataclasses import fields

    from agentcompass.runtime.registry import BENCHMARKS

    bootstrap_runtime(data_dir=str(data_dir), file_log_level=None, force=True)
    params = dict(benchmark_params)
    # load_tasks never calls a model. The placeholder judge ID only satisfies
    # config validation for benchmarks that declare judge_model; it has no
    # endpoint and is never written into a catalog row.
    config_class = getattr(BENCHMARKS.create(benchmark), "config_class", None)
    if config_class is not None and "judge_model" in {f.name for f in fields(config_class)}:
        params.setdefault("judge_model", {"id": "catalog-only"})
    return RunRequest.from_params({
        "benchmark": {"id": benchmark, "params": params},
        "harness": {"id": "openai_chat"},
        "environment": {"id": "host_process"},
        "model": {"id": "catalog-only", "api_protocol": "openai-chat"},
    })


def load_native_tasks(benchmark: str, data_dir: Path, benchmark_params: dict | None = None):
    from agentcompass.runtime.registry import BENCHMARKS, load_builtin_components

    load_builtin_components()
    tasks = BENCHMARKS.create(benchmark).load_tasks(
        _native_request(benchmark, dict(benchmark_params or {}), data_dir))
    if not tasks:
        raise ValueError(f"Native {benchmark} loader returned no tasks from {data_dir}")
    return tasks


def _catalog_params(benchmark: str, profile: dict, data_dir: Path, params: dict) -> dict:
    """Params that point the loader at the published tree instead of a download."""
    rendered = dict(params)
    root = data_dir / profile["root"]
    if benchmark == "sealqa":
        rendered["dataset_path"] = str(root / f"{params['category'].replace('_', '-')}.parquet")
    if benchmark == "xbench_deepsearch":
        rendered["dataset_path"] = str(root / f"DeepSearch-{params['version']}.csv")
    if benchmark in {"deepresearch_bench"}:
        rendered["data_dir"] = str(root)
    if benchmark in {"researchclawbench"}:
        rendered["tasks_dir"] = str(root)
    if benchmark == "scicode":
        rendered["h5py_file"] = str(root / "test_data.h5")
    if benchmark == "skillsbench":
        rendered["dataset_source_dir"] = str(data_dir)
    if benchmark == "special_pattern_check":
        rendered["dataset_dir"] = str(root)
    return rendered


def _variants(profile: dict, root: Path) -> dict:
    if profile.get("catalog_from_files"):
        variants = {path.stem: {"version": path.stem} for path in sorted(root.glob(profile["catalog_from_files"]))}
        if not variants:
            raise ValueError(f"No {profile['catalog_from_files']} files under {root}")
        return variants
    return profile.get("catalogs") or {"default": {}}


def _load_for_catalog(benchmark: str, profile: dict, data_dir: Path, params: dict):
    params = {**params, **(profile.get("load_only_params") or {})}
    # Loaders that build a datasets cache under data_dir run against a scratch
    # copy-free view so nothing is written into the published tree.
    if benchmark in {"sealqa", "xbench_deepsearch"}:
        with tempfile.TemporaryDirectory(prefix=f".{benchmark}-catalog-", dir=data_dir) as scratch:
            return load_native_tasks(benchmark, Path(scratch), params)
    return load_native_tasks(benchmark, data_dir, params)


def _row(benchmark: str, variant: str, sample_id: str, params: dict, data_dir: str) -> dict:
    suffix = "" if variant == "default" else f"-{variant}"
    return {
        "task_id": f"{benchmark}{suffix}-{sample_id}",
        "benchmark": benchmark, "environment": "host_process", "sample_id": sample_id,
        "data_dir": data_dir, "benchmark_params": dict(params),
        "harness_params": {}, "environment_params": {}, "model_params": {"max_retries": 0},
        "timeout_seconds": 3600, "contract_only": False,
    }


def _container_params(params: dict, data_dir: Path, container_data_dir: str) -> dict:
    rendered = dict(params)
    for key in ("dataset_path", "data_dir", "tasks_dir", "h5py_file", "dataset_source_dir", "dataset_dir"):
        if key in rendered and Path(rendered[key]).is_relative_to(data_dir):
            relative = Path(rendered[key]).relative_to(data_dir).as_posix()
            rendered[key] = str(PurePosixPath(container_data_dir) / relative)
    return rendered


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Shared file differs; publish to a new version directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, path)


def _publish_catalog(data_dir: Path, benchmark: str, variant: str, rows: list[dict], record: dict) -> str:
    # Catalogs never carry model configuration; users choose the judge at select time.
    if any("judge_model" in row["benchmark_params"] for row in rows):
        raise ValueError("catalog rows must not contain judge_model")
    name = benchmark if variant == "default" else f"{benchmark}-{variant}"
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    metadata = {
        "agentcompass_commit": PINNED_COMMIT, "benchmark": benchmark, "variant": variant,
        "data_record_sha256": record["record_sha256"], "task_count": len(rows),
        "task_manifest_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    catalog = data_dir / CATALOG_DIR / f"{name}.jsonl"
    _write_once(catalog, body)
    _write_once(catalog.with_suffix(".metadata.json"), json.dumps(metadata, indent=2) + "\n")
    return str(catalog)


def _adopt_existing(benchmark: str, profile: dict, root: Path) -> dict:
    """Record files published before records existed, only against independent pins."""
    if profile["kind"] == "scicode":
        import importlib.util

        spec = importlib.util.spec_from_file_location("scicode_prepare", Path(__file__).with_name("generate_dataset.py"))
        scicode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scicode)
        scicode.verify_h5(root / "test_data.h5")
        package_dir = Path(_pinned("scicode", "__file__")).parent / "data"
        for name in scicode.ASSET_FILES:
            if sha256(root / name) != sha256(package_dir / name):
                raise ValueError(f"Existing SciCode {name} differs from the pinned package file")
        return {"source": "adopted: official SciCode HDF5 (size+SHA256 pinned) and pinned package definitions",
                "hdf5_sha256": scicode.H5_SHA256, "official_archive": scicode.SCICODE_URL}
    expected = profile.get("adopt_sha256")
    if not expected:
        raise ValueError(f"Unrecorded {benchmark} data at {root}; publish to a new version directory")
    actual = {path.name: sha256(path) for path in root.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(f"Existing {benchmark} files do not match the pinned source checksums")
    return {"source": f"adopted: hf://datasets/{_resolve(profile['repo'])}", "revision": _resolve(profile["revision"]),
            "verified_against": "pinned LFS SHA256 per file"}


def _amend_record(record_path: Path, record: dict, root: Path, names: list[str],
                  removed: list[str] = ()) -> dict:
    files = {name: digest for name, digest in record["files"].items() if name not in set(removed)}
    files.update({name: sha256(root / name) for name in names})
    body = json.dumps(files, sort_keys=True)
    amended = {**record, "files": files, "file_count": len(files),
               "total_bytes": sum((root / name).stat().st_size for name in files),
               "record_sha256": hashlib.sha256(body.encode()).hexdigest(),
               "amendments": [*record.get("amendments", []),
                              {"added": sorted(names), "removed_runtime_locks": sorted(removed)}]}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=record_path.parent, delete=False) as tmp:
        tmp.write(json.dumps(amended, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, record_path)
    return amended


def prepare(benchmark: str, data_dir: Path, container_data_dir: str, source: Path | None = None,
            reconcile: bool = False) -> dict:
    """Deployment only: publish one benchmark's complete native tree, then its catalogs."""
    profile = PROFILES[benchmark]
    data_dir.mkdir(parents=True, exist_ok=True)
    record_path = data_dir / STATE_DIR / f"{benchmark}.json"
    with (data_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = data_dir / profile["root"]
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            drift = _verify_tree(root, record["files"])
            stale_locks = [name for name in drift["missing"] if name.endswith(RUNTIME_LOCK_SUFFIXES)]
            if reconcile and not drift["changed"] and drift["missing"] == stale_locks and (
                    stale_locks or drift["unrecorded"]):
                # Recorded assets are intact. Records written before lock files
                # were excluded listed them as assets; drop only those entries,
                # and append files that appeared after recording (e.g. a loader
                # README). Both changes are kept in the record's amendments.
                record = _amend_record(record_path, record, root, drift["unrecorded"], stale_locks)
                drift = _verify_tree(root, record["files"])
            if any(drift.values()):
                raise ValueError(f"Published {benchmark} tree differs from its record: "
                                 f"{ {k: v[:5] for k, v in drift.items() if v} }")
            reused = True
        else:
            if root.exists():
                provenance = _adopt_existing(benchmark, profile, root)
                for params in _variants(profile, root).values():
                    _load_for_catalog(benchmark, profile, data_dir, _catalog_params(benchmark, profile, data_dir, params))
            else:
                with tempfile.TemporaryDirectory(prefix=f".{benchmark}-", dir=data_dir) as tmp:
                    stage = Path(tmp)
                    provenance = _stage(benchmark, profile, stage, source)
                    staged_root = stage / profile["root"]
                    if not staged_root.exists():
                        raise ValueError(f"{benchmark} staging produced no {profile['root']}")
                    # Validate with the native loader before anything is published.
                    for params in _variants(profile, staged_root).values():
                        _load_for_catalog(benchmark, profile, stage, _catalog_params(benchmark, profile, stage, params))
                    for extra in ("gdpval_baseline", "swe_bench_pro/run_scripts"):
                        if extra.split("/")[0] != profile["root"] and (stage / extra).exists():
                            (data_dir / extra).parent.mkdir(parents=True, exist_ok=True)
                            os.replace(stage / extra, data_dir / extra)
                    root.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged_root, root)
            files = _tree_manifest(root)
            body = json.dumps(files, sort_keys=True)
            record = {"agentcompass_commit": PINNED_COMMIT, "benchmark": benchmark, "root": profile["root"],
                      "provenance": provenance, "file_count": len(files),
                      "total_bytes": sum((root / name).stat().st_size for name in files),
                      "record_sha256": hashlib.sha256(body.encode()).hexdigest(), "files": files}
            _write_once(record_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
            reused = False
        catalogs = {}
        for variant, params in _variants(profile, root).items():
            native_params = _catalog_params(benchmark, profile, data_dir, params)
            tasks = _load_for_catalog(benchmark, profile, data_dir, native_params)
            ids = [str(task.task_id) for task in tasks]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Native {benchmark}/{variant} loader returned duplicate task IDs")
            container = _container_params(native_params, data_dir, container_data_dir)
            rows = [_row(benchmark, variant, sid, container, container_data_dir) for sid in ids]
            catalogs[variant] = {"path": _publish_catalog(data_dir, benchmark, variant, rows, record),
                                 "native_task_count": len(rows)}
        return {"benchmark": benchmark, "reused_complete_data": reused,
                "provenance": record["provenance"], "file_count": record["file_count"],
                "total_bytes": record["total_bytes"], "catalogs": catalogs}


def status(data_dir: Path) -> list[dict]:
    rows = []
    for benchmark, profile in sorted(PROFILES.items()):
        record_path = data_dir / STATE_DIR / f"{benchmark}.json"
        entry = {"benchmark": benchmark, "kind": profile["kind"]}
        if not record_path.exists() and profile["kind"] == "external":
            entry.update(state="needs_deployer_data", reason=profile["reason"], layout=profile["layout"])
        elif not record_path.exists():
            entry.update(state="not_published")
        else:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            catalogs = sorted(p.name for p in (data_dir / CATALOG_DIR).glob(f"{benchmark}*.jsonl"))
            entry.update(state="published", file_count=record["file_count"],
                         total_bytes=record["total_bytes"], catalogs=catalogs)
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# User-side selection (standard Python only)
# ---------------------------------------------------------------------------

def load_catalog(path: Path) -> list[dict]:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata.get("agentcompass_commit") != PINNED_COMMIT or metadata.get("task_manifest_sha256") != sha256(path):
        raise ValueError("Shared catalog version or checksum mismatch")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select(rows: list[dict], *, harness: str, sample_ids=None, limit=None, judge_model: str = "",
           harness_params: dict | None = None) -> list[dict]:
    by_id = {}
    for row in rows:
        if row["sample_id"] in by_id:
            raise ValueError(f"Duplicate native sample ID: {row['sample_id']!r}")
        by_id[row["sample_id"]] = row
    if sample_ids:
        unknown = sorted(set(sample_ids) - by_id.keys())
        if unknown or len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"Unknown or duplicate sample IDs: {unknown or sample_ids}")
        selected = [by_id[key] for key in sample_ids]
    else:
        if limit is not None and limit < 1:
            raise ValueError("--limit must be positive; use --all explicitly")
        selected = list(by_id.values()) if limit is None else list(by_id.values())[:limit]
    out = []
    for row in selected:
        params = dict(row["benchmark_params"])
        if judge_model:
            params["judge_model"] = {"id": judge_model}
        out.append({**row, "harness": harness, "benchmark_params": params,
                    "harness_params": dict(harness_params or {})})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Deployment only, inside the pinned image")
    prep.add_argument("--benchmark", choices=sorted(PROFILES), action="append", required=True)
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--container-data-dir", default="/app/data/agentcompass")
    prep.add_argument("--source", type=Path, help="Deployer-supplied native data (skillsbench, special_pattern_check)")
    prep.add_argument("--reconcile", action="store_true",
                      help="Record files added after publication when all recorded files are unchanged")
    stat = sub.add_parser("status", help="Report published data per profile")
    stat.add_argument("--data-dir", type=Path, required=True)
    pick = sub.add_parser("select", help="Write a task manifest from a shared catalog")
    pick.add_argument("--catalog", type=Path, required=True)
    pick.add_argument("--harness", required=True, help="Native harness ID for the selected tasks")
    group = pick.add_mutually_exclusive_group()
    group.add_argument("--sample-id", action="append")
    group.add_argument("--limit", type=int, default=2)
    group.add_argument("--all", action="store_true")
    pick.add_argument("--judge-model", default="",
                      help=f"Judge model ID routed through the session Gateway (e.g. {DEFAULT_JUDGE_MODEL})")
    pick.add_argument("--harness-params", type=json.loads, default=None, help="JSON object")
    pick.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            reports = []
            for name in args.benchmark:
                try:
                    reports.append(prepare(name, args.data_dir.resolve(), args.container_data_dir, args.source,
                                           args.reconcile))
                except Exception as exc:  # report every benchmark; one failure must not hide others
                    reports.append({"benchmark": name, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(limit=8)})
            print(json.dumps(reports, ensure_ascii=False, indent=2))
            failed = [r["benchmark"] for r in reports if "error" in r]
            if failed:
                print(f"prepare failed for: {', '.join(failed)}", file=sys.stderr)
            return 1 if failed else 0
        if args.command == "status":
            print(json.dumps(status(args.data_dir.resolve()), ensure_ascii=False, indent=2))
            return 0
        rows = select(load_catalog(args.catalog), harness=args.harness, sample_ids=args.sample_id,
                      limit=None if args.all else args.limit, judge_model=args.judge_model,
                      harness_params=args.harness_params)
        with args.output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"tasks": len(rows), "sample_ids": [row["sample_id"] for row in rows]}))
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
