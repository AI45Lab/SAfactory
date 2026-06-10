from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass
from typing import Any


class ContainerCommandError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int | None = None, stderr: str = ""):
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


class DockerContainerRunner:
    """Small async wrapper around docker exec/cp-style operations."""

    def __init__(self, docker_bin: str = "docker") -> None:
        self.docker_bin = docker_bin

    async def exec(
        self,
        *,
        lease: Any,
        command: str,
        timeout_s: float,
    ) -> str:
        container = _container_name_or_id(lease)
        docker_bin = str(getattr(lease, "docker_bin", None) or self.docker_bin)
        result = await self._run(
            [docker_bin, "exec", container, "sh", "-lc", command],
            timeout_s=timeout_s,
        )
        if result.exit_code != 0:
            raise ContainerCommandError(
                f"container command failed with exit code {result.exit_code}",
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
        return await self.exec(lease=lease, command=f"cat {shlex.quote(path)}", timeout_s=timeout_s)

    async def write_text(
        self,
        *,
        lease: Any,
        path: str,
        text: str,
        timeout_s: float,
    ) -> None:
        container = _container_name_or_id(lease)
        docker_bin = str(getattr(lease, "docker_bin", None) or self.docker_bin)
        quoted_path = shlex.quote(path)
        command = f"mkdir -p $(dirname {quoted_path}) && cat > {quoted_path}"
        result = await self._run(
            [docker_bin, "exec", "-i", container, "sh", "-lc", command],
            input_text=text,
            timeout_s=timeout_s,
        )
        if result.exit_code != 0:
            raise ContainerCommandError(
                f"writing {path} failed with exit code {result.exit_code}",
                exit_code=result.exit_code,
                stderr=result.stderr,
            )

    async def _run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout_s: float,
    ) -> CommandResult:
        proc = await asyncio.create_subprocess_exec(
            *args,
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
            raise TimeoutError(f"command timed out after {timeout_s}s: {args!r}")
        return CommandResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=int(proc.returncode or 0),
        )


def _container_name_or_id(lease: Any) -> str:
    container = (
        getattr(lease, "container_id", None)
        or getattr(lease, "container_name", None)
        or getattr(lease, "name", None)
    )
    if not container:
        raise ValueError("lease does not include container_id/container_name")
    return str(container)
