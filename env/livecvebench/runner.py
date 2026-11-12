#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
SUMMARY_FILENAME = "safactory_result.json"
DEFAULT_DATASET_PATH = "/benchmark/tasks/LiveCVEBench-verified"
DEFAULT_DATASET_PATHS = {
    "livecvebench": DEFAULT_DATASET_PATH,
    "patcheval": "/benchmark/tasks/PatchEval-verified",
}
DEFAULT_OUTPUT_ROOT = "/benchmark/runs"
# Docker Hub is reachable through the RJob HTTP proxy. Third-party registry
# mirrors can stall indefinitely during token exchange (and have emitted
# cfgHost/host mismatches), so do not put them in front of the canonical
# registry.
DOCKER_REGISTRY_MIRRORS: tuple[str, ...] = ()
COMMON_IMAGE_CACHE_DIR = Path(
    "/mnt/shared-storage-gpfs2/trustcyberdata/private/docker-infra/tmp/"
    "puyuan/cve-factory/common-images"
)


def main() -> int:
    started_at = time.perf_counter()
    dockerd_proc: subprocess.Popen[str] | None = None
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}
    task_id = ""
    suite = ""
    output_dir: Path | None = None

    try:
        task_id = _required_text(
            dataset.get("task_id") or dataset.get("id") or env_params.get("task_id"),
            "env_params.dataset.task_id",
        )
        suite = _first_text(dataset.get("suite"), env_params.get("suite"), "livecvebench").lower()
        dataset_path = _resolve_dataset_path(dataset, env_params, suite)
        output_root = Path(
            _first_text(dataset.get("output_root"), env_params.get("output_root"), DEFAULT_OUTPUT_ROOT)
        )
        job_id = _safe_name(str(request.get("job_id") or "job"))
        output_dir = output_root / "safactory" / job_id / _safe_name(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not (dataset_path / task_id).is_dir():
            raise RuntimeError(f"{suite} task does not exist: {dataset_path / task_id}")

        agent = _first_text(dataset.get("agent"), env_params.get("agent"), "oracle")
        agent_import_path = _first_text(
            dataset.get("agent_import_path"),
            env_params.get("agent_import_path"),
        )
        model = _first_text(dataset.get("model"), env_params.get("model"))
        n_concurrent = _positive_int(
            dataset.get("n_concurrent", env_params.get("n_concurrent", 1)), default=1
        )
        test_timeout = _positive_int(
            dataset.get(
                "global_test_timeout_sec",
                env_params.get("global_test_timeout_sec", 900),
            ),
            default=900,
        )
        agent_timeout = _positive_int(
            dataset.get(
                "global_agent_timeout_sec",
                env_params.get("global_agent_timeout_sec", 1800),
            ),
            default=1800,
        )
        timeout_s = _runner_timeout(request, env_params, dataset, test_timeout)
        cmd = [
            "tb",
            "run",
            "--dataset-path",
            str(dataset_path),
            "--task-id",
            task_id,
            "--n-concurrent",
            str(n_concurrent),
            "--global-test-timeout-sec",
            str(test_timeout),
            "--global-agent-timeout-sec",
            str(agent_timeout),
            "--output-path",
            str(output_dir),
        ]
        if agent_import_path:
            cmd.extend(["--agent-import-path", agent_import_path])
        else:
            cmd.extend(["--agent", agent])
        if model:
            cmd.extend(["--model", model])

        log_path = output_dir / "tb-run.log"
        run_env = _docker_client_env(output_dir)
        dockerd_proc = _ensure_docker_daemon(output_dir, run_env)
        image_cache_dir = _resolve_image_cache_dir(dataset, env_params, suite)
        if image_cache_dir:
            loaded_image = _load_cached_task_image(
                Path(image_cache_dir),
                task_id,
                output_dir / "docker-cache-load.log",
                run_env,
            )
            _load_cached_compose_dependency_images(
                dataset_path / task_id / "docker-compose.yaml",
                COMMON_IMAGE_CACHE_DIR,
                output_dir / "docker-dependency-cache-load.log",
                run_env,
            )
            # Terminal-Bench requires tmux/asciinema regardless of which suite
            # supplied the image. Probe image capabilities instead of assuming
            # only PatchEval images need the compatibility layer.
            loaded_image = _ensure_terminal_tools_image(
                loaded_image, output_dir, run_env
            )
            dataset_path = _prepare_load_only_task_dataset(
                dataset_path,
                task_id,
                loaded_image,
                output_dir,
            )
            cmd[cmd.index("--dataset-path") + 1] = str(dataset_path)
        elif suite == "cve-factory-3k":
            registry_image = (
                "registry.h.pjlab.org.cn/ailab-llmagent-llmagent_cpu_task/"
                f"cve_factory:{task_id.lower()}"
            )
            local_archive = dataset_path / task_id / f"{task_id.lower()}-image.tar"
            if local_archive.is_file():
                _load_local_task_image(
                    local_archive,
                    registry_image,
                    output_dir / "docker-cache-load.log",
                    run_env,
                )
            else:
                _pull_task_images_with_retry(
                    dataset_path / task_id / "docker-compose.yaml",
                    output_dir / "docker-pull.log",
                    run_env,
                )
        else:
            _pull_task_images_with_retry(
                dataset_path / task_id / "docker-compose.yaml",
                output_dir / "docker-pull.log",
                run_env,
            )
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("command: " + shlex.join(cmd) + "\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd="/benchmark",
                env=run_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )

        metadata_path = _find_metadata(output_dir)
        metadata = _read_json(metadata_path) if metadata_path else {}
        result_paths = _result_paths(metadata_path, task_id)
        trial_results = []
        for path in result_paths:
            trial_result = _read_json(path)
            usage = _claude_usage_from_trial(path)
            if usage["message_count"]:
                trial_result.update(
                    {
                        "total_input_tokens": usage["input_tokens"],
                        "total_output_tokens": usage["output_tokens"],
                        "total_cache_creation_input_tokens": usage[
                            "cache_creation_input_tokens"
                        ],
                        "total_cache_read_input_tokens": usage[
                            "cache_read_input_tokens"
                        ],
                        "token_usage_source": "claude_stream_json",
                    }
                )
                _write_json(path, trial_result)
            trial_results.append(trial_result)
        total_usage = _sum_trial_usage(trial_results)
        resolved = [bool(item.get("is_resolved")) for item in trial_results]
        failure_modes = [item.get("failure_mode") for item in trial_results]
        agent_timed_out = any(
            _is_agent_timeout_failure_mode(failure_mode)
            for failure_mode in failure_modes
        )
        score = (
            sum(1.0 for value in resolved if value) / len(resolved)
            if resolved
            else _float_or_default(metadata.get("accuracy"), 0.0)
        )

        error_text: str | None = None
        status = "succeeded"
        if proc.returncode != 0:
            status = "failed"
            error_text = f"tb run exited with code {proc.returncode}"
        elif not metadata_path:
            status = "failed"
            error_text = _with_log_tail("tb run produced no new run_metadata.json", log_path)
        elif not result_paths:
            status = "failed"
            error_text = _with_log_tail(
                f"tb run produced no results.json for {task_id}",
                log_path,
            )

        result = {
            "session_id": session_id,
            "status": status,
            "total_reward": score if status == "succeeded" else 0.0,
            "step_count": max(1, len(trial_results)),
            "terminated": True,
            "truncated": agent_timed_out,
            "error_text": error_text,
            "metrics": {
                "bench": "livecvebench",
                "suite": suite,
                "task_id": task_id,
                "agent": agent,
                "agent_import_path": agent_import_path or None,
                "model": model or None,
                "score": score,
                "is_resolved": resolved[0] if len(resolved) == 1 else None,
                "resolved_trials": sum(1 for value in resolved if value),
                "trial_count": len(trial_results),
                "failure_modes": failure_modes,
                "total_input_tokens": total_usage["input_tokens"],
                "total_output_tokens": total_usage["output_tokens"],
                "total_cache_creation_input_tokens": total_usage[
                    "cache_creation_input_tokens"
                ],
                "total_cache_read_input_tokens": total_usage[
                    "cache_read_input_tokens"
                ],
                "token_usage_source": (
                    "claude_stream_json"
                    if any(
                        item.get("token_usage_source") == "claude_stream_json"
                        for item in trial_results
                    )
                    else "terminal_bench"
                ),
                "dataset_path": str(dataset_path),
                "output_root": str(output_root),
                "output_dir": str(output_dir),
                "safactory_result_path": str(output_dir / SUMMARY_FILENAME),
                "run_metadata_path": str(metadata_path) if metadata_path else None,
                "result_paths": [str(path) for path in result_paths],
                "log_path": str(log_path),
                "returncode": proc.returncode,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            },
        }
        _write_summary(output_dir, result)
        _write_result(result)
        return 0
    except subprocess.TimeoutExpired as exc:
        result = _failure_result(
            session_id,
            f"tb run timed out after {float(exc.timeout or 0):.1f}s",
            started_at,
            suite=suite,
            task_id=task_id,
            truncated=True,
        )
        _write_summary(output_dir, result)
        _write_result(result)
        return 0
    except Exception as exc:
        result = _failure_result(
            session_id,
            str(exc),
            started_at,
            suite=suite,
            task_id=task_id,
        )
        _write_summary(output_dir, result)
        _write_result(result)
        return 0
    finally:
        _stop_docker_daemon(dockerd_proc)
        _cleanup_temporary_claude_binary(output_dir, task_id)


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "").strip()
    if not raw:
        raise RuntimeError("SimulationStartRequest JSON was not provided on stdin")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return value


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _resolve_dataset_path(
    dataset: dict[str, Any],
    env_params: dict[str, Any],
    suite: str,
) -> Path:
    explicit_path = _first_text(dataset.get("dataset_path"), env_params.get("dataset_path"))
    if explicit_path:
        return Path(explicit_path)

    configured_paths = env_params.get("dataset_paths")
    if isinstance(configured_paths, dict):
        configured_path = _first_text(configured_paths.get(suite))
        if configured_path:
            return Path(configured_path)

    default_path = DEFAULT_DATASET_PATHS.get(suite)
    if default_path:
        return Path(default_path)
    supported = ", ".join(sorted(DEFAULT_DATASET_PATHS))
    raise RuntimeError(f"unsupported suite {suite!r}; expected one of: {supported}")


def _resolve_image_cache_dir(
    dataset: dict[str, Any],
    env_params: dict[str, Any],
    suite: str,
) -> str:
    explicit_path = _first_text(dataset.get("image_cache_dir"))
    if explicit_path:
        return explicit_path

    configured_paths = env_params.get("image_cache_dirs")
    if isinstance(configured_paths, dict):
        configured_path = _first_text(configured_paths.get(suite))
        if configured_path:
            return configured_path

    return _first_text(env_params.get("image_cache_dir"))


def _positive_int(value: Any, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def _runner_timeout(
    request: dict[str, Any],
    env_params: dict[str, Any],
    dataset: dict[str, Any],
    test_timeout: int,
) -> float:
    outer = _float_or_default(request.get("agent_start_timeout_s"), float(test_timeout + 60))
    configured = _float_or_default(
        dataset.get("timeout_s", env_params.get("timeout_s")),
        float(test_timeout + 60),
    )
    return max(1.0, min(outer, configured))


def _docker_client_env(output_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    http_proxy = _first_text(env.get("HTTP_PROXY"), env.get("http_proxy"))
    https_proxy = _first_text(env.get("HTTPS_PROXY"), env.get("https_proxy"), http_proxy)
    if not http_proxy and not https_proxy:
        return env

    no_proxy = _first_text(env.get("NO_PROXY"), env.get("no_proxy"))
    docker_config_dir = output_dir / ".docker"
    docker_config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "proxies": {
            "default": {
                "httpProxy": http_proxy,
                "httpsProxy": https_proxy,
                "noProxy": no_proxy,
            }
        }
    }
    auth_config_path = _first_text(env.get("SAFACTORY_DOCKER_AUTH_CONFIG"))
    if auth_config_path:
        try:
            auth_config = json.loads(Path(auth_config_path).read_text(encoding="utf-8"))
            if isinstance(auth_config.get("auths"), dict):
                config["auths"] = auth_config["auths"]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to load Docker auth config from {auth_config_path}"
            ) from exc
    (docker_config_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False),
        encoding="utf-8",
    )
    env["DOCKER_CONFIG"] = str(docker_config_dir)
    # Compose v2 does not render useful progress through Terminal-Bench's
    # redirected output in interactive mode. Plain progress keeps pull/build
    # diagnostics visible in the run artifacts. The timeout variables also
    # prevent a single registry request from occupying the whole task timeout.
    env["BUILDKIT_PROGRESS"] = "plain"
    env["COMPOSE_PROGRESS"] = "plain"
    env["DOCKER_CLIENT_TIMEOUT"] = "300"
    env["COMPOSE_HTTP_TIMEOUT"] = "300"
    return env


def _ensure_docker_daemon(
    output_dir: Path,
    env: dict[str, str],
) -> subprocess.Popen[str] | None:
    if subprocess.run(
        ["docker", "info"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0:
        return None

    for legacy_binary, generic_binary in (
        ("/usr/sbin/iptables-legacy", "iptables"),
        ("/usr/sbin/ip6tables-legacy", "ip6tables"),
    ):
        if Path(legacy_binary).is_file():
            subprocess.run(
                ["update-alternatives", "--set", generic_binary, legacy_binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    log_path = output_dir / "dockerd.log"
    log_file = log_path.open("w", encoding="utf-8")
    try:
        dockerd_cmd = ["dockerd", "--host=unix:///var/run/docker.sock"]
        for mirror in DOCKER_REGISTRY_MIRRORS:
            dockerd_cmd.extend(["--registry-mirror", mirror])
        proc = subprocess.Popen(
            dockerd_cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_file.close()

    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(
                _with_log_tail("the inner Docker daemon failed to start", log_path)
            )
        if subprocess.run(
            ["docker", "info"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            return proc
        time.sleep(1)

    proc.terminate()
    raise RuntimeError(
        _with_log_tail("timed out waiting for the inner Docker daemon", log_path)
    )


def _stop_docker_daemon(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)




def _load_local_task_image(
    archive: Path,
    expected_image: str,
    log_path: Path,
    env: dict[str, str],
) -> None:
    result = subprocess.run(
        ["docker", "load", "--input", str(archive)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = result.stdout or ""
    log_path.write_text(
        f"loading local task image archive {archive}\n{output}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            _with_log_tail(
                f"docker load failed for local archive {archive} "
                f"with code {result.returncode}",
                log_path,
            )
        )
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", expected_image],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspect_result.returncode != 0:
        loaded_images = [
            line.strip()[len("Loaded image:") :].strip()
            for line in output.splitlines()
            if line.strip().startswith("Loaded image:")
        ]
        if len(loaded_images) != 1:
            raise RuntimeError(
                _with_log_tail(
                    f"loaded local archive did not expose one tagged image; "
                    f"loaded={loaded_images!r}",
                    log_path,
                )
            )
        tag_result = subprocess.run(
            ["docker", "tag", loaded_images[0], expected_image],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if tag_result.returncode != 0:
            raise RuntimeError(
                _with_log_tail(
                    f"failed to tag loaded image {loaded_images[0]} as "
                    f"{expected_image}",
                    log_path,
                )
            )
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"tagged loaded image {loaded_images[0]} as {expected_image}\n"
            )
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"local task image loaded successfully: {expected_image}\n")

def _pull_task_images_with_retry(
    compose_path: Path,
    log_path: Path,
    env: dict[str, str],
    *,
    attempts: int = 3,
    timeout_s: int = 300,
) -> None:
    """Pre-pull non-buildable task images with bounded retries."""
    if not compose_path.is_file():
        return
    images: list[str] = []
    for line in compose_path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("image:"):
            continue
        image = stripped.split(":", 1)[1].split("#", 1)[0].strip().strip("'\"")
        # Terminal-Bench fills its client image variables only after it creates
        # the trial. Pull only literal service images such as mysql:5.7 here.
        if image and "$" not in image and image not in images:
            images.append(image)
    if not images:
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        for image in images:
            if subprocess.run(
                ["docker", "image", "inspect", image],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0:
                log_file.write(f"use cached local image {image}\n")
                log_file.flush()
                continue
            command = ["docker", "pull", image]
            for attempt in range(1, attempts + 1):
                log_file.write(f"pull {image} attempt {attempt}/{attempts}\n")
                log_file.flush()
                try:
                    result = subprocess.run(
                        command,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=timeout_s,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    log_file.write(
                        f"pull {image} attempt {attempt} timed out after {timeout_s}s\n"
                    )
                    log_file.flush()
                else:
                    if result.returncode == 0:
                        log_file.write(f"pull {image} attempt {attempt} succeeded\n")
                        log_file.flush()
                        break
                    log_file.write(
                        f"pull {image} attempt {attempt} exited with code "
                        f"{result.returncode}\n"
                    )
                    log_file.flush()
                if attempt < attempts:
                    time.sleep(min(30, 5 * (2 ** (attempt - 1))))
            else:
                raise RuntimeError(
                    _with_log_tail(
                        f"failed to pull {image} after {attempts} attempts",
                        log_path,
                    )
                )


def _load_cached_task_image(
    cache_dir: Path,
    task_id: str,
    log_path: Path,
    env: dict[str, str],
) -> str:
    """Load the verified per-task Docker archive before Terminal-Bench runs."""
    candidates = (
        cache_dir / f"{task_id}.tar",
        cache_dir / f"{task_id}.tar.gz",
        cache_dir / "image-tars" / f"{task_id}.tar",
        cache_dir / "images" / f"{task_id}.tar.gz",
    )
    archive = next((path for path in candidates if path.is_file()), None)
    if archive is None:
        checked = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"cached Docker archive does not exist; checked: {checked}")
    result = subprocess.run(
        ["docker", "load", "--input", str(archive)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = result.stdout or ""
    log_path.write_text(
        f"loading cached Docker archive {archive}\n{output}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            _with_log_tail(
                f"docker load failed for cached archive {archive} "
                f"with code {result.returncode}",
                log_path,
            )
        )

    loaded_images = []
    for line in output.splitlines():
        prefix = "Loaded image:"
        if line.strip().startswith(prefix):
            image = line.strip()[len(prefix):].strip()
            if image:
                loaded_images.append(image)
    loaded_images = list(dict.fromkeys(loaded_images))
    expected_task_image = f"cve-factory-{task_id.lower()}-client:latest"
    if len(loaded_images) == 1:
        loaded_image = loaded_images[0]
    elif expected_task_image in loaded_images:
        loaded_image = expected_task_image
    else:
        raise RuntimeError(
            _with_log_tail(
                "cached Docker archive must expose either one tagged image or "
                f"the task client image {expected_task_image!r}; found "
                f"{loaded_images!r}",
                log_path,
            )
        )
    inspect_result = subprocess.run(
        ["docker", "image", "inspect", loaded_image],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspect_result.returncode != 0:
        raise RuntimeError(f"loaded Docker image is not inspectable: {loaded_image}")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"cached Docker archive loaded successfully: {loaded_image}\n")
    return loaded_image


def _ensure_terminal_tools_image(
    source_image: str,
    output_dir: Path,
    env: dict[str, str],
) -> str:
    """Add Terminal-Bench tools to any cached task image that lacks them."""
    log_path = output_dir / "tmux-image-patch.log"
    probe = subprocess.run(
        [
            "docker", "run", "--rm", "--user", "0", "--entrypoint", "/bin/sh",
            source_image, "-lc", "command -v tmux && command -v asciinema",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        log_path.write_text(
            f"Terminal-Bench tools already available in {source_image}\n{probe.stdout or ''}",
            encoding="utf-8",
        )
        return source_image

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*", source_image):
        raise RuntimeError(f"unsafe Docker image reference: {source_image!r}")
    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Config.User}}", source_image],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        raise RuntimeError(f"could not inspect Docker image user for {source_image}")
    original_user = (inspect.stdout or "").strip()
    suffix = hashlib.sha256(source_image.encode("utf-8")).hexdigest()[:16]
    patched_image = f"safactory/terminal-tools:{suffix}"
    context_dir = output_dir / "tmux-image-context"
    context_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = [
        f"FROM {source_image}",
        "USER root",
        "ARG PATCH_HTTP_PROXY",
        "ARG PATCH_HTTPS_PROXY",
        "ARG PATCH_NO_PROXY",
        "RUN set -eu; \\",
        "    export http_proxy=\"$PATCH_HTTP_PROXY\" https_proxy=\"$PATCH_HTTPS_PROXY\" no_proxy=\"mirrors.i.h.pjlab.org.cn,pypi.i.h.pjlab.org.cn,$PATCH_NO_PROXY\"; \\",
        "    mkdir -p /etc/pip; printf '%s\\n' '[global]' 'timeout = 60' 'index-url = http://mirrors.i.h.pjlab.org.cn/repository/pypi-tsinghua/simple' 'extra-index-url = http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/ http://pypi.i.h.pjlab.org.cn/brain/dev/+simple' 'trusted-host = mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn' > /etc/pip.conf; \\",
        "    if command -v tmux >/dev/null 2>&1 && command -v asciinema >/dev/null 2>&1; then exit 0; \\",
        "    elif command -v apt-get >/dev/null 2>&1; then . /etc/os-release; codename=\"${VERSION_CODENAME:-jammy}\"; case \"$codename\" in bionic) apt_base=http://mirrors.i.h.pjlab.org.cn/repository/apt-bionic-proxy/ubuntu;; focal) apt_base=http://mirrors.i.h.pjlab.org.cn/ubuntu;; jammy) apt_base=http://mirrors.i.h.pjlab.org.cn/repository/apt-jammy-proxy/ubuntu;; *) apt_base=;; esac; if [ -n \"$apt_base\" ]; then printf 'deb %s %s main restricted universe multiverse\\ndeb %s %s-security main restricted universe multiverse\\ndeb %s %s-updates main restricted universe multiverse\\ndeb %s %s-backports main restricted universe multiverse\\n' \"$apt_base\" \"$codename\" \"$apt_base\" \"$codename\" \"$apt_base\" \"$codename\" \"$apt_base\" \"$codename\" > /etc/apt/sources.list; find /etc/apt/sources.list.d -type f -name '*.list' -delete 2>/dev/null || true; fi; http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= apt-get update && DEBIAN_FRONTEND=noninteractive http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*; \\",
        "    elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux asciinema; \\",
        "    elif command -v dnf >/dev/null 2>&1; then dnf install -y tmux asciinema && dnf clean all; \\",
        "    elif command -v yum >/dev/null 2>&1; then yum install -y tmux asciinema && yum clean all; \\",
        "    else echo 'no supported package manager found' >&2; exit 42; fi; \\",
        "    command -v tmux && command -v asciinema",
    ]
    if original_user:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", original_user):
            raise RuntimeError(f"unsafe Docker image user: {original_user!r}")
        dockerfile.append(f"USER {original_user}")
    (context_dir / "Dockerfile").write_text(
        "\n".join(dockerfile) + "\n", encoding="utf-8"
    )
    command = [
        "docker", "build", "--pull=false", "--tag", patched_image,
        "--file", str(context_dir / "Dockerfile"),
    ]
    for target, upper, lower in (
        ("PATCH_HTTP_PROXY", "HTTP_PROXY", "http_proxy"),
        ("PATCH_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy"),
        ("PATCH_NO_PROXY", "NO_PROXY", "no_proxy"),
    ):
        value = env.get(upper) or env.get(lower)
        if value:
            command.extend(["--build-arg", f"{target}={value}"])
    command.append(str(context_dir))
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            f"source image lacks Terminal-Bench tools; building {patched_image}\n"
        )
        log_file.flush()
        result = subprocess.run(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            _with_log_tail(
                f"failed to add Terminal-Bench tools to task image {source_image}",
                log_path,
            )
        )
    return patched_image


def _prepare_load_only_task_dataset(
    dataset_path: Path,
    task_id: str,
    loaded_image: str,
    output_dir: Path,
) -> Path:
    """Create a private task copy whose Compose file uses only the loaded image."""
    source_task = dataset_path / task_id
    private_dataset = output_dir / "load-only-dataset"
    private_task = private_dataset / task_id
    if private_dataset.exists():
        shutil.rmtree(private_dataset)
    shutil.copytree(source_task, private_task)
    patch_report = _patch_task_test_package_sources(private_task)
    (output_dir / "test-source-patch.log").write_text(
        "\n".join(f"{key}={value}" for key, value in patch_report.items()) + "\n",
        encoding="utf-8",
    )

    compose_path = private_task / "docker-compose.yaml"
    if not compose_path.is_file():
        raise RuntimeError(f"task Compose file does not exist: {compose_path}")
    claude_binary_source = Path(
        "/mnt/shared-storage-gpfs2/trustcyberdata/private/qiupanjia/"
        "SAfactory/.cache/claude-code/claude-2.1.241"
    )
    claude_binary = private_task / "claude"
    if claude_binary_source.is_file():
        _link_or_copy_claude_binary(claude_binary_source, claude_binary)
    compose_text = _compose_load_only_text(
        compose_path.read_text(encoding="utf-8"),
        loaded_image,
    )
    if claude_binary.is_file():
        compose_text = _inject_claude_binary_mount(compose_text, claude_binary)
    compose_path.write_text(
        compose_text,
        encoding="utf-8",
    )
    (output_dir / "load-only-image.txt").write_text(
        loaded_image + "\n",
        encoding="utf-8",
    )
    return private_dataset


def _load_cached_compose_dependency_images(
    compose_path: Path,
    cache_dir: Path,
    log_path: Path,
    run_env: dict[str, str],
) -> None:
    """Load fixed Compose service images from the shared offline cache."""
    if not compose_path.is_file() or not cache_dir.is_dir():
        return
    images: list[str] = []
    for match in re.finditer(r"(?m)^\s*image:\s*['\"]?([^'\"\s#]+)", compose_path.read_text(encoding="utf-8")):
        image = match.group(1).strip()
        if image and "${" not in image and image not in images:
            images.append(image)
    if not images:
        return
    with log_path.open("a", encoding="utf-8") as log_file:
        for image in images:
            digest = hashlib.sha256(image.encode("utf-8")).hexdigest()
            archive = cache_dir / f"{digest}.tar"
            if not archive.is_file():
                log_file.write(f"missing cached Compose dependency: {image}\n")
                continue
            log_file.write(f"loading cached Compose dependency: {image} <- {archive}\n")
            log_file.flush()
            result = subprocess.run(
                ["docker", "load", "-i", str(archive)],
                env=run_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"failed to load cached Compose dependency {image} from {archive}"
                )


def _link_or_copy_claude_binary(source: Path, destination: Path) -> None:
    """Expose the cached Claude binary without duplicating it on the same filesystem."""
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        destination.chmod(0o755)


def _cleanup_temporary_claude_binary(
    output_dir: Path | None,
    task_id: str,
) -> None:
    """Remove the per-task Claude mount source after nested Docker has stopped."""
    if output_dir is None or not task_id:
        return
    claude_binary = output_dir / "load-only-dataset" / task_id / "claude"
    try:
        claude_binary.unlink(missing_ok=True)
    except OSError as exc:
        print(
            f"warning: failed to remove temporary Claude binary {claude_binary}: {exc}",
            file=sys.stderr,
        )


def _inject_claude_binary_mount(compose_text: str, claude_binary: Path) -> str:
    """Insert the Claude bind mount using the Compose file's list indentation."""
    volumes_marker = "    volumes:\n"
    if volumes_marker not in compose_text:
        raise RuntimeError("client Compose service has no volumes section")
    volumes_tail = compose_text.split(volumes_marker, 1)[1]
    volume_item = re.search(r"(?m)^([ \t]*)-\s", volumes_tail)
    if volume_item is None:
        raise RuntimeError("client Compose volumes section has no list item")
    volume_indent = volume_item.group(1)
    return compose_text.replace(
        volumes_marker,
        f"{volumes_marker}{volume_indent}- "
        f"{claude_binary}:/benchmark/claude:ro\n",
        1,
    )


def _patch_task_test_package_sources(private_task: Path) -> dict[str, int]:
    """Route copied benchmark test installers through H-cluster mirrors."""
    preamble = (
        'export NO_PROXY="${NO_PROXY:+$NO_PROXY,}'
        'deploy.i.h.pjlab.org.cn,mirrors.i.h.pjlab.org.cn,'
        'pypi.i.h.pjlab.org.cn"\n'
        'export no_proxy="${no_proxy:+$no_proxy,}'
        'deploy.i.h.pjlab.org.cn,mirrors.i.h.pjlab.org.cn,'
        'pypi.i.h.pjlab.org.cn"\n'
        'export UV_INSTALLER_GITHUB_BASE_URL="http://deploy.i.h.pjlab.org.cn"\n'
        'export UV_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/'
        'repository/pypi-proxy/simple/"\n'
        'export PIP_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/'
        'repository/pypi-tsinghua/simple"\n'
        'export PIP_EXTRA_INDEX_URL="http://mirrors.i.h.pjlab.org.cn/'
        'repository/pypi-proxy/simple/ '
        'http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"\n'
        'export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn '
        'pypi.i.h.pjlab.org.cn"\n'
        'safactory_ensure_uv() {\n'
        '  if ! command -v uv >/dev/null 2>&1; then\n'
        '    curl -LsSf "http://deploy.i.h.pjlab.org.cn/astral-sh/uv/'
        'uv-installer.sh" | sh\n'
        '    export PATH="$HOME/.local/bin:$PATH"\n'
        '  fi\n'
        '}\n'
        'safactory_ensure_uv_venv() {\n'
        '  safactory_ensure_uv\n'
        '  if [ ! -x .venv/bin/python ]; then '
        'uv venv .venv --python python3 --system-site-packages; fi\n'
        '}\n'
        'safactory_python_pytest() {\n'
        '  if python3 -c "import pytest" >/dev/null 2>&1; then\n'
        '    python3 -m pytest "$@"\n'
        '  else\n'
        '    safactory_ensure_uv_venv\n'
        '    uv pip install pytest\n'
        '    uv run python -m pytest "$@"\n'
        '  fi\n'
        '}\n'
        'safactory_bare_pytest() {\n'
        '  if command -v pytest >/dev/null 2>&1; then\n'
        '    pytest "$@"\n'
        '  else\n'
        '    safactory_python_pytest "$@"\n'
        '  fi\n'
        '}\n'
    )
    internal_installer = (
        "http://deploy.i.h.pjlab.org.cn/astral-sh/uv/uv-installer.sh"
    )
    report = {
        "scripts_seen": 0,
        "scripts_patched": 0,
        "installers_rewritten": 0,
        "remaining_astral_urls": 0,
    }
    installer_pattern = r"https?://astral\.sh/uv/(?:[^\s'\"]+/)?install\.sh"
    script_paths = [private_task / "run-tests.sh"]
    script_paths.extend((private_task / "tests").glob("**/*.sh"))
    for script_path in script_paths:
        if not script_path.is_file():
            continue
        report["scripts_seen"] += 1
        text = script_path.read_text(encoding="utf-8", errors="replace")
        installer_count = len(re.findall(installer_pattern, text))
        if not installer_count and not re.search(
            r"\b(?:uv\s+(?:add|run|pip)|pytest|python3?\s+-m\s+pytest)\b",
            text,
        ):
            continue
        text = re.sub(
            installer_pattern,
            internal_installer,
            text,
        )
        has_uv_project_install = bool(re.search(r"(?m)^\s*uv\s+add\b", text))
        needs_uv_venv = bool(
            re.search(r"(?m)^\s*uv\s+pip\s+install[^\n]*\s--system(?:\s|$)", text)
        )
        if needs_uv_venv:
            # Keep every original option and dependency, changing only the
            # PEP-668-sensitive target from system Python to the project venv.
            text = re.sub(
                r"(?m)^(?P<i>[ \t]*)(?P<c>uv\s+pip\s+install)"
                r"(?P<a>[^\n]*)$",
                lambda match: (
                    f"{match.group('i')}safactory_ensure_uv_venv\n"
                    f"{match.group('i')}{match.group('c')}"
                    + re.sub(r"(?<!\S)--system(?=\s|$)", "", match.group("a"))
                ),
                text,
            )
        if has_uv_project_install or needs_uv_venv:
            # Packages installed by uv must also be executed in uv's project
            # environment. Existing `uv run` commands are deliberately intact.
            text = re.sub(
                r"(?m)^(?P<i>[ \t]*)(?:python3?|python)\s+-m\s+pytest\b",
                r"\g<i>uv run python -m pytest",
                text,
            )
            text = re.sub(
                r"(?m)^(?P<i>[ \t]*)pytest\b",
                r"\g<i>uv run pytest",
                text,
            )
        else:
            # Preserve known-good system environments. The fallback is entered
            # only when pytest is genuinely absent in the task container.
            text = re.sub(
                r"(?m)^(?P<i>[ \t]*)(?:python3?|python)\s+-m\s+pytest\b",
                r"\g<i>safactory_python_pytest",
                text,
            )
            text = re.sub(
                r"(?m)^(?P<i>[ \t]*)pytest\b",
                r"\g<i>safactory_bare_pytest",
                text,
            )
        lines = text.splitlines(keepends=True)
        insert_at = 1 if lines and lines[0].startswith("#!") else 0
        lines.insert(insert_at, preamble)
        patched_text = "".join(lines)
        script_path.write_text(patched_text, encoding="utf-8")
        report["scripts_patched"] += 1
        report["installers_rewritten"] += installer_count
        report["remaining_astral_urls"] += len(
            re.findall(r"https?://astral\.sh/uv/", patched_text)
        )
    return report


def _compose_load_only_text(compose_text: str, loaded_image: str) -> str:
    """Replace one Compose build service without requiring a YAML package."""
    lines = compose_text.splitlines(keepends=True)
    build_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*build\s*:", line) and not line.lstrip().startswith("#")
    ]
    if not build_indexes:
        raise RuntimeError(
            "cached task image requires a Compose build entry; found none"
        )

    candidates = []
    for build_index in build_indexes:
        build_indent = len(lines[build_index]) - len(lines[build_index].lstrip())
        service_start = None
        service_indent = None
        for index in range(build_index - 1, -1, -1):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(lines[index]) - len(lines[index].lstrip())
            if indent < build_indent and stripped.endswith(":"):
                service_start = index
                service_indent = indent
                break
        if service_start is None or service_indent is None:
            continue

        service_end = len(lines)
        for index in range(service_start + 1, len(lines)):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(lines[index]) - len(lines[index].lstrip())
            if indent <= service_indent:
                service_end = index
                break
        image_indexes = [
            index
            for index in range(service_start + 1, service_end)
            if re.match(r"^\s*image\s*:", lines[index])
            and not lines[index].lstrip().startswith("#")
        ]
        if len(image_indexes) == 1:
            candidates.append(
                (build_index, build_indent, service_end, image_indexes[0])
            )

    if not candidates:
        raise RuntimeError(
            "cached task image requires at least one Compose service containing "
            "both build and image"
        )
    for build_index, build_indent, service_end, image_index in reversed(candidates):
        original_value = lines[image_index].split(":", 1)[1].strip().strip("'\"")
        service_image = _cached_compose_image(original_value, loaded_image)
        image_indent = lines[image_index][
            : len(lines[image_index]) - len(lines[image_index].lstrip())
        ]
        newline = "\r\n" if lines[image_index].endswith("\r\n") else "\n"
        lines[image_index] = (
            f"{image_indent}image: {json.dumps(service_image)}{newline}"
        )

        build_end = build_index + 1
        while build_end < service_end:
            stripped = lines[build_end].strip()
            if stripped:
                indent = len(lines[build_end]) - len(lines[build_end].lstrip())
                if indent <= build_indent:
                    break
            build_end += 1
        del lines[build_index:build_end]
    return "".join(lines)


def _cached_compose_image(compose_image: str, loaded_image: str) -> str:
    variable = "${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}"
    if compose_image == variable:
        return loaded_image
    if compose_image.startswith(variable + "-"):
        suffix = compose_image[len(variable):]
        repository, separator, tag = loaded_image.rpartition(":")
        if not separator:
            repository, tag = loaded_image, "latest"
        return f"{repository}{suffix}:{tag}"
    raise RuntimeError(
        "cached build service has an unsupported image expression: "
        f"{compose_image!r}"
    )


def _find_metadata(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("*/run_metadata.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _result_paths(metadata_path: Path | None, task_id: str) -> list[Path]:
    if metadata_path is None:
        return []
    paths = list((metadata_path.parent / task_id).glob("*/results.json"))
    top_level_results = metadata_path.parent / "results.json"
    if top_level_results.is_file():
        paths.append(top_level_results)
    return sorted(paths)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _claude_usage_from_trial(result_path: Path) -> dict[str, int]:
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "message_count": 0,
    }
    agent_log = result_path.parent / "sessions" / "agent.log"
    try:
        lines = agent_log.open(encoding="utf-8", errors="replace")
    except OSError:
        return usage

    # Claude stream-json can emit several events for the same assistant message
    # (for example thinking followed by tool_use). Keep the largest counters per
    # message id so those events are not charged more than once.
    messages: dict[str, dict[str, int]] = {}
    final_usage: dict[str, int] | None = None
    with lines:
        for line in lines:
            start = line.find("{")
            if start < 0:
                continue
            try:
                event = json.loads(line[start:])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "result":
                model_usage = event.get("modelUsage")
                if isinstance(model_usage, dict):
                    exact = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }
                    for counters in model_usage.values():
                        if not isinstance(counters, dict):
                            continue
                        exact["input_tokens"] += _nonnegative_int(
                            counters.get("inputTokens")
                        )
                        exact["output_tokens"] += _nonnegative_int(
                            counters.get("outputTokens")
                        )
                        exact["cache_creation_input_tokens"] += _nonnegative_int(
                            counters.get("cacheCreationInputTokens")
                        )
                        exact["cache_read_input_tokens"] += _nonnegative_int(
                            counters.get("cacheReadInputTokens")
                        )
                    final_usage = exact
                elif isinstance(event.get("usage"), dict):
                    counters = event["usage"]
                    final_usage = {
                        key: _nonnegative_int(counters.get(key))
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        )
                    }
                continue
            if event.get("type") != "assistant":
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            counters = message.get("usage")
            if not isinstance(counters, dict):
                continue
            message_id = str(message.get("id") or event.get("uuid") or "").strip()
            if not message_id:
                continue
            current = messages.setdefault(message_id, {})
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                current[key] = max(
                    current.get(key, 0),
                    _nonnegative_int(counters.get(key)),
                )

    usage["message_count"] = len(messages)
    if final_usage is not None:
        usage.update(final_usage)
        # A valid result event proves that at least one model response completed.
        usage["message_count"] = max(1, usage["message_count"])
        return usage
    for counters in messages.values():
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            usage[key] += counters.get(key, 0)
    return usage


def _sum_trial_usage(trial_results: list[dict[str, Any]]) -> dict[str, int]:
    result = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    for trial in trial_results:
        for key, result_key in (
            ("total_input_tokens", "input_tokens"),
            ("total_output_tokens", "output_tokens"),
            ("total_cache_creation_input_tokens", "cache_creation_input_tokens"),
            ("total_cache_read_input_tokens", "cache_read_input_tokens"),
        ):
            result[result_key] += _nonnegative_int(trial.get(key))
    return result


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _is_agent_timeout_failure_mode(value: Any) -> bool:
    return str(value or "").strip().lower() == "agent_timeout"


def _with_log_tail(message: str, log_path: Path, *, max_chars: int = 6000) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return message
    if not text:
        return message
    return f"{message}\n\nLast tb output:\n{text[-max_chars:]}"


def _safe_name(value: str) -> str:
    # Keep this normalization identical to manager.episode_common.safe_path_part.
    # The launcher uses the normalized job/session components to locate the
    # result artifact written on the shared RJob filesystem.
    text = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(value or "").strip()
    )
    return text.strip("._") or "item"


def _failure_result(
    session_id: str,
    error_text: str,
    started_at: float,
    *,
    suite: str = "",
    task_id: str = "",
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "failed",
        "total_reward": 0.0,
        "step_count": 0,
        "terminated": True,
        "truncated": truncated,
        "error_text": error_text,
        "metrics": {
            "bench": "livecvebench",
            "suite": suite or None,
            "task_id": task_id or None,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        },
    }


def _write_summary(output_dir: Path | None, result: dict[str, Any]) -> None:
    if output_dir is None:
        return
    path = output_dir / SUMMARY_FILENAME
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metrics.setdefault("safactory_result_path", str(path))
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_result(result: dict[str, Any]) -> None:
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    artifact = str(os.environ.get(RESULT_PATH_ENV) or "").strip()
    if artifact:
        path = Path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(RESULT_JSON_PREFIX + encoded, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
