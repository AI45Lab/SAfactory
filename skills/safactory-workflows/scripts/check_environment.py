#!/usr/bin/env python3
"""One-command environment verification: static checks -> contract smoke -> optional live run.

Usage (from the repository root):

    python skills/safactory-workflows/scripts/check_environment.py --env env/<name> [options]

Stages:

1. Static validation (validate_environment.py): file set, naming, and
   config <-> start <-> dataset invariants, each finding labeled with the
   owning side (config / env).
2. Contract smoke (contract_smoke.py): runs the real runner against an owned
   mock model endpoint.  Native dependencies must be installed locally; pass
   ``--fixture-adapter`` to verify the protocol shell only.  Skipped when
   static validation reports errors.
3. Live deployment (live_smoke.py), only with ``--live``: starts a Gateway,
   runs launcher.py, and cleans up.  ``--live`` consumes all remaining
   arguments and forwards them to live_smoke.py verbatim, e.g.:

       check_environment.py --env env/prmeval --live \\
           --gateway-config gateway.yaml -- \\
           --agent-config env/prmeval/prmeval_config.yaml \\
           --agent-start-config env/prmeval/prmeval_start.yaml --llm-model <route>

Exit code 0 only when every executed stage passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from contract_smoke import run_smoke  # noqa: E402
from validate_environment import (  # noqa: E402
    SEV_INFO,
    validate_environment,
)

LIVE_SMOKE = SCRIPTS / "live_smoke.py"

DEPS_MARKERS = ("ModuleNotFoundError", "ImportError", "No module named")


def check(env_dir: Path, *, expect_evaluation: bool = False, request_path: Path | None = None,
          fixture_adapter: Path | None = None, timeout: float = 30,
          require_model_call: bool = True, live_args: list[str] | None = None) -> dict:
    stages: dict[str, dict] = {}

    # Stage 1: static validation.
    report = validate_environment(env_dir, expect_evaluation=expect_evaluation)
    stages["static"] = {
        "ok": report.ok(),
        "findings": [asdict(f) for f in report.findings if f.severity != SEV_INFO],
    }

    # Stage 2: contract smoke (skipped when static errors exist).
    if not report.ok():
        stages["contract"] = {"ok": False, "skipped": True,
                              "reason": "static validation failed; fix [config] findings first"}
    else:
        stages["contract"] = _contract_stage(
            env_dir, request_path=request_path, fixture_adapter=fixture_adapter,
            timeout=timeout, require_model_call=require_model_call,
        )

    # Stage 3: live deployment (opt-in).
    if live_args is None:
        stages["live"] = {"ok": True, "skipped": True, "reason": "not requested (pass --live)"}
    elif not stages["static"]["ok"] or not stages["contract"]["ok"]:
        stages["live"] = {"ok": False, "skipped": True,
                          "reason": "earlier stages failed; live run would only add noise"}
    else:
        stages["live"] = _live_stage(live_args)

    stages["summary"] = {
        "ok": stages["static"]["ok"] and stages["contract"]["ok"] and stages["live"]["ok"]
    }
    return stages


def _contract_stage(env_dir: Path, *, request_path, fixture_adapter, timeout,
                    require_model_call) -> dict:
    runner = env_dir / "runner.py"
    request_file = request_path or (env_dir / "request.smoke.json")
    try:
        request = json.loads(request_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"cannot read smoke request {request_file}: {exc}",
                "side": "env"}
    try:
        outcome = run_smoke(
            runner, request, adapter=fixture_adapter, timeout=timeout,
            require_model_call=require_model_call,
        )
    except ValueError as exc:
        message = str(exc)
        side = "env"
        hint = ""
        if any(marker in message for marker in DEPS_MARKERS):
            hint = ("native dependencies are missing locally; install them, or pass "
                    "--fixture-adapter <path> to verify the protocol shell only")
        elif "runner process failed" in message:
            hint = "the runner must always exit 0 and report failures inside the result JSON"
        return {"ok": False, "error": message, "side": side, "hint": hint}
    return {"ok": True, "outcome": {"model_calls": outcome.get("model_calls"),
                                    "result": outcome.get("result")}}


def _live_stage(live_args: list[str]) -> dict:
    completed = subprocess.run([sys.executable, str(LIVE_SMOKE), *live_args], check=False)
    return {"ok": completed.returncode == 0, "exit_code": completed.returncode}


def _print_stages(stages: dict) -> None:
    order = [("static", "1/3 static validation"), ("contract", "2/3 contract smoke"),
             ("live", "3/3 live deployment")]
    for key, title in order:
        stage = stages[key]
        status = "SKIPPED" if stage.get("skipped") else ("PASS" if stage["ok"] else "FAIL")
        print(f"== {title}: {status}")
        if key == "static":
            for finding in stage["findings"]:
                severity, side = finding["severity"], finding["side"]
                print(f"  [{severity}] [{side}] {finding['file']}: {finding['message']}")
                if finding["hint"]:
                    print(f"      hint: {finding['hint']}")
        elif not stage.get("skipped"):
            detail = stage.get("error") or stage.get("outcome") or {"exit_code": stage.get("exit_code")}
            print(f"  {json.dumps(detail, ensure_ascii=False)[:400]}")
        elif stage.get("reason"):
            print(f"  {stage['reason']}")
    summary = stages["summary"]
    print(f"SUMMARY: {'OK' if summary['ok'] else 'FAILED'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=Path, required=True, help="environment directory, e.g. env/mybench")
    parser.add_argument("--expect-evaluation", action="store_true",
                        help="require rule_evaluator.py to exist (static stage)")
    parser.add_argument("--request", type=Path, default=None,
                        help="smoke request JSON (default: <env>/request.smoke.json)")
    parser.add_argument("--fixture-adapter", type=Path, default=None,
                        help="swap in a fixture adapter to verify the protocol shell without native deps")
    parser.add_argument("--timeout", type=float, default=30, help="contract runner timeout in seconds")
    parser.add_argument("--no-require-model-call", action="store_true",
                        help="do not require the adapter to route through the mock session")
    parser.add_argument("--live", nargs=argparse.REMAINDER, default=None, metavar="ARGS",
                        help="run the live stage; all remaining arguments are forwarded to live_smoke.py")
    parser.add_argument("--json", action="store_true", help="print the stage report as JSON")
    args = parser.parse_args(argv)

    live_args = list(args.live) if args.live is not None else None
    if live_args and live_args[0] == "--":
        live_args = live_args[1:]
    stages = check(
        args.env, expect_evaluation=args.expect_evaluation, request_path=args.request,
        fixture_adapter=args.fixture_adapter, timeout=args.timeout,
        require_model_call=not args.no_require_model_call, live_args=live_args,
    )
    if args.json:
        print(json.dumps(stages, ensure_ascii=False, indent=2, default=str))
    else:
        _print_stages(stages)
    return 0 if stages["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
