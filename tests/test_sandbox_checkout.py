"""Package from the Git index so untracked files cannot conceal missing runtime code."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = {"core", "manager", "clusters", "evaluator", "gateway", "rl"}
ENTRYPOINTS = {"launcher.py", "args.py", "log_setup.py"}
REQUIRED_RUNTIME_FILES = ENTRYPOINTS | {
    "core/result_artifact.py", "evaluator/result_file.py", "evaluator/factory.py",
    "manager/sandbox_episode_runner.py", "manager/simulation_config.py",
    "manager/simulation_flow.py", "manager/simulation_worker.py",
    "manager/resume_cleanup.py", "manager/runtime_allocator.py", "manager/types.py",
    "clusters/sandbox_cluster.py",
}


@unittest.skipUnless(shutil.which("git") and (REPO_ROOT / ".git").exists(), "requires a Git checkout to inspect the staged package")
class SandboxCheckoutTest(unittest.TestCase):
    def git(self, *arguments, **kwargs):
        return subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs,
        ).stdout

    def tracked_runtime(self):
        paths = self.git("ls-files", "-z").decode().split("\0")
        return {
            path for path in paths if path.endswith(".py") and (
                path in ENTRYPOINTS or path.split("/", 1)[0] in RUNTIME_ROOTS
            )
        }

    def test_required_runtime_modules_are_in_git_index(self):
        missing = REQUIRED_RUNTIME_FILES - self.tracked_runtime()
        self.assertFalse(missing, f"Runtime modules missing from Git index: {sorted(missing)}")

    def test_index_checkout_imports_launcher_without_reference_directory(self):
        tracked = self.tracked_runtime()
        self.assertFalse(REQUIRED_RUNTIME_FILES - tracked, "Stage every required runtime module before packaging")
        with tempfile.TemporaryDirectory(prefix="safactory-checkout-") as directory:
            checkout = Path(directory).resolve()
            self.git(
                "checkout-index", "--stdin", "-z", f"--prefix={checkout}{os.sep}",
                input=("\0".join(sorted(tracked)) + "\0").encode(),
            )
            self.assertFalse((checkout / "sandbox").exists())
            # Also catch lazy script dependencies which importing launcher alone cannot execute.
            for relative in tracked:
                tree = ast.parse((checkout / relative).read_text(encoding="utf-8"), filename=relative)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").split(".")[0] == "sandbox", relative)
                    if isinstance(node, ast.Import):
                        self.assertFalse(any(item.name.split(".")[0] == "sandbox" for item in node.names), relative)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertFalse(node.value.replace("\\", "/").startswith("sandbox/"), relative)
            script = """
import importlib
import socket
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path
root = Path(sys.argv[1]).resolve()
assert not (root / 'sandbox').exists()
sys.path.insert(0, str(root))
def offline(*args, **kwargs):
    raise AssertionError('clean-checkout imports must not access the network')
socket.create_connection = offline
socket.socket.connect = offline
class BlockOptionalSandboxSDK(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'opensandbox' or fullname.startswith('opensandbox.'):
            raise ModuleNotFoundError('OpenSandbox SDK disabled for import regression', name=fullname)
        return None
sys.meta_path.insert(0, BlockOptionalSandboxSDK())
for name in ('launcher', 'args', 'core.result_artifact', 'clusters.sandbox_cluster',
             'manager.sandbox_episode_runner', 'evaluator.result_file'):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root), (name, module.__file__)
from args import parse_simulation_args
for mode in ('docker', 'rjob', 'sandbox'):
    assert parse_simulation_args(['--mode', mode]).mode == mode
print('clean-checkout imports passed')
"""
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(checkout)], cwd=checkout,
                text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("clean-checkout imports passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
