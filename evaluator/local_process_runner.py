from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from evaluator.container_runner import CommandResult, ContainerCommandError


class LocalProcessRunner:
    """Run evaluator agents as local subprocesses.

    An empty command means "run the configured Codex CLI default". A non-empty
    command is executed through the local shell for advanced templates.
    """

    def __init__(self) -> None:
        self._last_results: dict[str, CommandResult] = {}

    async def exec(
        self,
        *,
        lease: Any,
        command: str,
        timeout_s: float,
    ) -> str:
        if str(command or "").strip():
            result = await self._run_shell(lease=lease, command=command, timeout_s=timeout_s)
        else:
            result = await self._run_codex_cli(lease=lease, timeout_s=timeout_s)
        self._last_results[str(getattr(lease, "lease_id", ""))] = result
        if result.exit_code != 0:
            raise ContainerCommandError(
                f"local evaluator command failed with exit code {result.exit_code}",
                exit_code=result.exit_code,
                stderr=result.stderr,
            )
        return result.stdout

    async def read_file(
        self,
        *,
        lease: Any,
        path: str,
        timeout_s: float,
    ) -> str:
        del lease, timeout_s
        return await asyncio.to_thread(Path(path).read_text, encoding="utf-8")

    async def write_text(
        self,
        *,
        lease: Any,
        path: str,
        text: str,
        timeout_s: float,
    ) -> None:
        del lease, timeout_s
        target = Path(path)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_text, text, encoding="utf-8")

    def get_last_result(self, lease: Any) -> CommandResult | None:
        return self._last_results.get(str(getattr(lease, "lease_id", "")))

    async def _run_codex_cli(self, *, lease: Any, timeout_s: float) -> CommandResult:
        spec = lease.spec
        prompt_path = Path(spec.prompt_path)
        prompt = await asyncio.to_thread(prompt_path.read_text, encoding="utf-8")
        result_path = Path(spec.result_path)
        await asyncio.to_thread(result_path.parent.mkdir, parents=True, exist_ok=True)

        args = [spec.cli_bin, "exec", "--color", "never"]
        cli_args = list(spec.cli_args or [])
        if "--json" not in cli_args:
            args.append("--json")
        if spec.ephemeral and "--ephemeral" not in cli_args:
            args.append("--ephemeral")
        if spec.sandbox and "--sandbox" not in cli_args and "-s" not in cli_args:
            args.extend(["--sandbox", spec.sandbox])
        if spec.approval_policy and "--ask-for-approval" not in cli_args and "-a" not in cli_args:
            args.extend(["--ask-for-approval", spec.approval_policy])
        if spec.cli_profile and "--profile" not in cli_args and "-p" not in cli_args:
            args.extend(["--profile", spec.cli_profile])
        if spec.workdir and "--cd" not in cli_args and "-C" not in cli_args:
            args.extend(["-C", spec.workdir])
        if spec.model and "--model" not in cli_args and "-m" not in cli_args:
            args.extend(["-m", spec.model])
        if "--output-last-message" not in cli_args and "-o" not in cli_args:
            args.extend(["--output-last-message", str(result_path)])
        args.extend(cli_args)
        args.append("-")
        return await self._run_exec(
            args=args,
            input_text=prompt,
            cwd=spec.workdir or None,
            env={**os.environ, **dict(spec.env or {})},
            timeout_s=timeout_s,
        )

    async def _run_shell(self, *, lease: Any, command: str, timeout_s: float) -> CommandResult:
        spec = lease.spec
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=spec.workdir or None,
            env={**os.environ, **dict(spec.env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"local evaluator command timed out after {timeout_s}s")
        return CommandResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=int(proc.returncode or 0),
        )

    async def _run_exec(
        self,
        *,
        args: list[str],
        input_text: str | None,
        cwd: str | None,
        env: dict[str, str],
        timeout_s: float,
    ) -> CommandResult:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input_text.encode("utf-8") if input_text is not None else None),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"local evaluator command timed out after {timeout_s}s: {args!r}")
        return CommandResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=int(proc.returncode or 0),
        )
