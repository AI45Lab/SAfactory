from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LLM_DATASET = (
    Path(__file__).resolve().parent
    / "PatchEval"
    / "patcheval"
    / "exp_agent"
    / "sweagent"
    / "dataset.jsonl"
)
DEFAULT_CLAUDECODE_DATASET = (
    Path(__file__).resolve().parent
    / "PatchEval"
    / "patcheval"
    / "exp_agent"
    / "claudecode"
    / "dataset.jsonl"
)
DEFAULT_OPENHANDS_DATASET = DEFAULT_CLAUDECODE_DATASET
OFFICIAL_DATASET = (
    Path(__file__).resolve().parent
    / "PatchEval"
    / "patcheval"
    / "datasets"
    / "input.json"
)
SETTINGS = ("s1.1", "s1.2", "s1.3", "s1.4")
BASELINES = ("llm", "claudecode", "openhands")
AGENT_EXPERIMENTS = ("exp1",)
TEMPLATE_DIR = OFFICIAL_DATASET.parent.parent / "exp_llm" / "prompt_templates"
SETTING_TEMPLATES = {
    "s1.1": TEMPLATE_DIR / "Default.txt",
    "s1.2": TEMPLATE_DIR / "Ablation_without_CoT.txt",
    "s1.3": TEMPLATE_DIR / "Ablation_without_Knowledge.txt",
    "s1.4": TEMPLATE_DIR / "Default.txt",
}
DEFAULT_OFFICIAL_RUNTIME = Path(
    "/mnt/shared-storage-user/evobox-share/leishanzhe/dataset/patcheval-runtime"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAfactory configs for the Docker-backed PatchEval set")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 means all tasks")
    parser.add_argument("--official-dataset", type=Path, default=OFFICIAL_DATASET)
    parser.add_argument("--official-runtime-dir", type=Path, default=DEFAULT_OFFICIAL_RUNTIME)
    parser.add_argument("--baseline", choices=BASELINES, default="llm")
    parser.add_argument("--setting", choices=SETTINGS, default="s1.1")
    parser.add_argument("--agent-experiment", choices=AGENT_EXPERIMENTS, default="exp1")
    parser.add_argument("--agent-tool-limit", type=int, default=100)
    parser.add_argument("--claude-install-timeout-s", type=float, default=900.0)
    parser.add_argument("--claude-gateway-base-url", default="")
    parser.add_argument("--claude-model", default="")
    parser.add_argument("--claude-max-thinking-tokens", type=int, default=0)
    parser.add_argument("--openhands-install-timeout-s", type=float, default=900.0)
    parser.add_argument("--http-proxy", default="")
    parser.add_argument("--evaluation-timeout-s", type=float, default=3600.0)
    parser.add_argument("--shared-tmp", default="")
    parser.add_argument(
        "--no-proxy",
        default="host.docker.internal,localhost,127.0.0.1,::1",
    )
    return parser.parse_args()


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_cves: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [
                field
                for field in ("cve_id", "image_name", "work_dir", "problem_statement")
                if not str(record.get(field) or "").strip()
            ]
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {', '.join(missing)}")
            cve_id = str(record["cve_id"]).strip().upper()
            if cve_id in seen_cves:
                raise ValueError(f"{path}:{line_number} duplicate CVE: {cve_id}")
            seen_cves.add(cve_id)
            record["cve_id"] = cve_id
            records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No PatchEval records found in {path}")
    return records


def load_official_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("is_poc"):
            continue
        cve_id = str(record.get("cve_id") or "").strip().upper()
        if cve_id:
            result[cve_id] = record
    return result


def archive_name(image: str) -> str:
    image_name = image.rsplit("/", 1)[-1]
    if ":" in image_name:
        repository, tag = image_name.rsplit(":", 1)
    else:
        repository, tag = image_name, "latest"
    return f"{repository}-{tag}.tar"


def env_name(cve_id: str) -> str:
    return "patcheval_" + re.sub(r"[^a-z0-9]+", "_", cve_id.lower()).strip("_")


def write_configs(
    records: list[dict[str, Any]],
    official_records: dict[str, dict[str, Any]],
    baseline: str,
    setting: str,
    prompt_template: str,
    agent_experiment: str,
    output_dir: Path,
    archive_dir: Path | None,
    http_proxy: str,
    no_proxy: str,
    evaluation_timeout_s: float,
    shared_tmp: str,
    official_runtime_dir: Path,
    agent_tool_limit: int,
    claude_install_timeout_s: float,
    claude_gateway_base_url: str,
    claude_model: str,
    claude_max_thinking_tokens: int,
    openhands_install_timeout_s: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    environments: list[dict[str, Any]] = []
    agents: dict[str, Any] = {}
    rule_evaluator_path = Path(__file__).resolve().with_name("rule_evaluator.py")
    docker_adapter_path = Path(__file__).resolve().parent / "docker_archive_adapter" / "sitecustomize.py"
    official_root = OFFICIAL_DATASET.parent.parent
    runner_name = {
        "claudecode": "claudecode_runner.py",
        "openhands": "openhands_runner.py",
    }.get(baseline, "strict_runner.py")
    runner_path = Path(__file__).resolve().with_name(runner_name)
    container_env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATCHEVAL_OFFICIAL_ROOT": "/opt/patcheval",
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    }
    if baseline == "claudecode":
        if not claude_gateway_base_url:
            raise ValueError("--claude-gateway-base-url is required for the Claude Code baseline")
        if not claude_model:
            raise ValueError("--claude-model is required for the Claude Code baseline")
        container_env.update(
            {
                "PATCHEVAL_CLAUDE_TOOL_LIMIT": str(agent_tool_limit),
                "PATCHEVAL_CLAUDE_INSTALL_TIMEOUT_S": str(claude_install_timeout_s),
                "PATCHEVAL_CLAUDE_GATEWAY_BASE_URL": claude_gateway_base_url,
                "PATCHEVAL_CLAUDE_MODEL": claude_model,
            }
        )
        if claude_max_thinking_tokens > 0:
            container_env["PATCHEVAL_CLAUDE_MAX_THINKING_TOKENS"] = str(
                claude_max_thinking_tokens
            )
    elif baseline == "openhands":
        if not claude_gateway_base_url:
            raise ValueError("--claude-gateway-base-url is required for the OpenHands baseline")
        if not claude_model:
            raise ValueError("--claude-model is required for the OpenHands baseline")
        container_env.update(
            {
                "PATCHEVAL_OPENHANDS_TOOL_LIMIT": str(agent_tool_limit),
                "PATCHEVAL_OPENHANDS_INSTALL_TIMEOUT_S": str(openhands_install_timeout_s),
                "PATCHEVAL_OPENHANDS_MODEL": claude_model,
                "PATCHEVAL_OPENHANDS_GATEWAY_BASE_URL": claude_gateway_base_url,
            }
        )
    if http_proxy:
        container_env.update(
            {
                "HTTP_PROXY": http_proxy,
                "HTTPS_PROXY": http_proxy,
                "http_proxy": http_proxy,
                "https_proxy": http_proxy,
            }
        )

    common_agent = {
        "container": {
            "workdir": "/workspace",
            "runner_entrypoint": {
                "source": str(runner_path),
                "target": "/tmp/safactory-patcheval-runner.py",
                "command": "python /tmp/safactory-patcheval-runner.py",
            },
            "install_runner_script": True,
            "env": container_env,
            "extra_args": ["--add-host=host.docker.internal:host-gateway"],
            "volumes": [
                {
                    "source": str(official_runtime_dir),
                    "target": "/opt/patcheval",
                    "read_only": True,
                }
            ],
            "idle_command": "tail -f /dev/null",
        }
    }

    missing_archives: list[str] = []
    for record in records:
        cve_id = str(record["cve_id"])
        image = str(record["image_name"]).strip()
        name = env_name(cve_id)
        dataset_filename = f"{cve_id.lower()}.jsonl"
        dataset_path = dataset_dir / dataset_filename
        official_record = official_records.get(cve_id)
        if official_record is None:
            raise ValueError(f"Official PatchEval input.json has no PoC record for {cve_id}")
        task: dict[str, Any] = {
            "cve_id": cve_id,
            "work_dir": str(record["work_dir"]),
            "official_record": official_record,
        }
        if baseline in {"claudecode", "openhands"}:
            task.update(
                {
                    "agent_framework": "claude-code" if baseline == "claudecode" else "openhands",
                    "agent_experiment": agent_experiment,
                    "problem_statement": str(record["problem_statement"]),
                }
            )
        else:
            task.update({"setting": setting, "prompt_template": prompt_template})
        dataset_path.write_text(json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8")

        environments.append(
            {
                "env_name": name,
                "env_image": image,
                "env_num": 1,
                "dataset": f"./datasets/{dataset_filename}",
                "dataset_load_mode": "eager",
                "env_params": {
                    "task_family": "patcheval",
                    "rule_evaluator_timeout_s": evaluation_timeout_s,
                    "patcheval_official_root": str(official_root),
                    "patcheval_docker_adapter": str(docker_adapter_path),
                    "patcheval_image_archive_dir": str(archive_dir or ""),
                    "patcheval_http_proxy": http_proxy,
                    "patcheval_no_proxy": no_proxy,
                    "patcheval_shared_tmp": shared_tmp,
                },
            }
        )
        agents[name] = common_agent
        evaluator_dir = output_dir / name
        evaluator_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rule_evaluator_path, evaluator_dir / "rule_evaluator.py")

        if archive_dir is not None:
            base = archive_dir / archive_name(image)
            candidates = (base, Path(f"{base}.gz"), base.with_suffix(".tgz"))
            if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
                missing_archives.append(" or ".join(str(path) for path in candidates))

    if missing_archives:
        sample = "\n".join(f"  - {path}" for path in missing_archives[:10])
        suffix = "" if len(missing_archives) <= 10 else f"\n  ... and {len(missing_archives) - 10} more"
        raise FileNotFoundError(f"Missing {len(missing_archives)} image archive(s):\n{sample}{suffix}")

    with (output_dir / "patcheval_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"environments": environments}, handle, sort_keys=False, allow_unicode=True)
    with (output_dir / "patcheval_start.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"agents": agents}, handle, sort_keys=False, allow_unicode=True)
def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.evaluation_timeout_s <= 0:
        raise SystemExit("--evaluation-timeout-s must be positive")
    if args.agent_tool_limit <= 0:
        raise SystemExit("--agent-tool-limit must be positive")
    if args.claude_install_timeout_s <= 0:
        raise SystemExit("--claude-install-timeout-s must be positive")
    if args.claude_max_thinking_tokens < 0:
        raise SystemExit("--claude-max-thinking-tokens must be non-negative")
    if args.openhands_install_timeout_s <= 0:
        raise SystemExit("--openhands-install-timeout-s must be positive")
    if args.baseline == "llm":
        default_dataset = DEFAULT_LLM_DATASET
    elif args.baseline == "claudecode":
        default_dataset = DEFAULT_CLAUDECODE_DATASET
    else:
        default_dataset = DEFAULT_OPENHANDS_DATASET
    dataset = (args.dataset or default_dataset).expanduser().resolve()
    official_dataset = args.official_dataset.expanduser().resolve()
    official_runtime_dir = args.official_runtime_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    archive_dir = args.archive_dir.expanduser().resolve() if args.archive_dir else None
    if archive_dir is not None and not archive_dir.is_dir():
        raise SystemExit(f"Archive directory does not exist: {archive_dir}")
    required_runtime_files: list[Path] = []
    if args.baseline == "claudecode":
        required_runtime_files.append(
            official_runtime_dir / "exp_agent" / "claudecode" / "templates" / "default.md"
        )
    elif args.baseline == "llm":
        required_runtime_files.extend(
            [
                official_runtime_dir / "exp_llm" / "helper" / "llm_suite.py",
                official_runtime_dir / "exp_llm" / "helper" / "func_replacer.py",
                official_runtime_dir / "exp_llm" / "helper" / "__init__.py",
            ]
        )
    missing_runtime_files = [path for path in required_runtime_files if not path.is_file()]
    if missing_runtime_files:
        missing = "\n".join(f"  - {path}" for path in missing_runtime_files)
        raise SystemExit(f"Official PatchEval runtime is incomplete:\n{missing}")

    records = load_records(dataset, args.limit)
    official_records = load_official_records(official_dataset)
    prompt_template = (
        SETTING_TEMPLATES[str(args.setting)].read_text(encoding="utf-8")
        if args.baseline == "llm"
        else ""
    )
    write_configs(
        records,
        official_records,
        str(args.baseline),
        str(args.setting),
        prompt_template,
        str(args.agent_experiment),
        output_dir,
        archive_dir,
        str(args.http_proxy).strip(),
        str(args.no_proxy).strip(),
        float(args.evaluation_timeout_s),
        str(args.shared_tmp).strip(),
        official_runtime_dir,
        int(args.agent_tool_limit),
        float(args.claude_install_timeout_s),
        str(args.claude_gateway_base_url).strip(),
        str(args.claude_model).strip(),
        int(args.claude_max_thinking_tokens),
        float(args.openhands_install_timeout_s),
    )
    print(
        f"Generated PatchEval {args.baseline} configuration "
        f"for {len(records)} task(s) in {output_dir}"
    )


if __name__ == "__main__":
    main()
