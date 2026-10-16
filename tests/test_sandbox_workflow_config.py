from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml

from args import parse_simulation_args
from manager.resume_cleanup import prepare_sandbox_resume
from manager.simulation_config import load_agent_start_config, load_simulation_run_config
from manager.simulation_flow import SimulationFlow
from rl.gateway_autostart import build_gateway_config


class SandboxConfigTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "runner.py").write_text("print('runner')")
        self.path = self.root / "demo.start.sandbox.yaml"
        self.start = {
            "agent_name": "demo",
            "sandbox": {
                "runner_entrypoint": {"source": "runner.py", "target": "/tmp/runner.py", "command": "python /tmp/runner.py"},
                "storage": {
                    "inputs": [{"host_path": "data", "afs_subpath": "inputs/demo/v1", "mount_path": "/data"}],
                    "outputs": [{"afs_subpath": "outputs/demo", "mount_path": "/app/results", "read_only": False}],
                },
            },
        }

    def load(self):
        self.path.write_text(yaml.safe_dump(self.start))
        return load_agent_start_config(str(self.path), mode="sandbox")["demo"]

    def test_standalone_sandbox_paths_and_upload_deduplication(self):
        self.start["sandbox"]["embedded_files"] = [{"source": "runner.py", "target": "/tmp/runner.py", "permissions": "0755"}]
        result = self.load()
        self.assertEqual(set(result), {"sandbox"})
        cfg = result["sandbox"]
        self.assertEqual(len(cfg["embedded_files"]), 1)
        self.assertEqual(cfg["embedded_files"][0]["permissions"], "0755")
        self.assertEqual(cfg["storage"]["inputs"][0]["host_path"], str((self.root / "data").resolve()))
        self.assertTrue(cfg["storage"]["inputs"][0]["read_only"])
        self.assertFalse(cfg["storage"]["outputs"][0]["read_only"])

    def test_direct_runner_preserves_executable_permissions(self):
        self.start["sandbox"]["runner_entrypoint"].update(command="/tmp/runner.py", permissions="0755")
        cfg = self.load()["sandbox"]
        self.assertEqual(cfg["embedded_files"][0]["permissions"], "0755")
        self.assertEqual(cfg["run_command"], "/tmp/runner.py")

    def test_invalid_storage_and_uploads_fail_before_allocation(self):
        original = copy.deepcopy(self.start)
        mutations = [
            lambda cfg: cfg["sandbox"]["storage"]["outputs"][0].update(read_only=True),
            lambda cfg: cfg["sandbox"]["storage"]["outputs"][0].update(mount_path="/other"),
            lambda cfg: cfg["sandbox"]["storage"]["inputs"][0].update(afs_subpath="../escape"),
            lambda cfg: cfg["sandbox"]["runner_entrypoint"].update(source="missing.py"),
            lambda cfg: cfg["sandbox"].update(gateway_base_url="http://another-gateway"),
            lambda cfg: cfg["sandbox"]["runner_entrypoint"].update(permissions=0o644),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.start = copy.deepcopy(original)
                mutate(self.start)
                with self.assertRaises(ValueError):
                    self.load()

    def test_global_gateway_is_shared_and_sandbox_requires_explicit_profiles(self):
        global_path = self.root / "sandbox-config.yaml"
        global_path.write_text(yaml.safe_dump({"sandbox": {"gateway_base_url": "http://gateway:8000/v1/sessions"}}))
        argv = ["--mode", "sandbox", "--sandbox-config", str(global_path)]
        with self.assertRaisesRegex(ValueError, "explicit"):
            load_simulation_run_config(parse_simulation_args(argv))
        argv += ["--agent-config", "demo.config.sandbox.yaml", "--agent-start-config", str(self.path)]
        cfg = load_simulation_run_config(parse_simulation_args(argv))
        self.assertEqual(cfg.gateway_base_url, cfg.sandbox_config["gateway_base_url"])
        self.assertTrue(cfg.evaluation_enabled)
        self.assertFalse(cfg.enable_buffer)
        cfg = load_simulation_run_config(parse_simulation_args(argv + ["--gateway-base-url", "http://override:8000/v1/sessions"]))
        self.assertEqual(cfg.sandbox_config["gateway_base_url"], "http://override:8000/v1/sessions")
        self.assertEqual(load_simulation_run_config(parse_simulation_args([])).gateway_base_url, "http://127.0.0.1:8080/v1/sessions")

    def test_rl_autostart_uses_durable_sandbox_settings_only_in_sandbox(self):
        with patch.dict(os.environ, {"AIEVOBOX_MODE": "sandbox"}):
            cfg = build_gateway_config(aievobox_root=str(self.root))
        self.assertFalse(cfg["storage_config"]["enable_buffer"])
        self.assertFalse(cfg["storage_config"]["enable_upsert_batching"])
        self.assertEqual(cfg["telemetry"], {"mode": "strict", "async_cloud_writes": False})
        with patch.dict(os.environ, {"AIEVOBOX_MODE": "docker"}):
            self.assertNotIn("telemetry", build_gateway_config(aievobox_root=str(self.root)))


class SandboxPreflightTest(unittest.TestCase):
    def setUp(self):
        self.flow = SimulationFlow.__new__(SimulationFlow)
        self.flow.cfg = SimpleNamespace(mode="sandbox", storage_type="sqlite", db_url="sqlite:///tmp/shared.db")
        self.payload = {
            "instance_id": "gateway-1",
            "storage_type": "sqlite", "storage_identity": {"db_url": "sqlite:///tmp/shared.db"},
            "durability": {"telemetry_mode": "strict", "async_cloud_writes": False, "enable_buffer": False, "enable_upsert_batching": False},
        }

    def validate(self):
        self.flow._validate_gateway_storage(json.dumps(self.payload), "http://gateway/readyz")

    def test_rejects_buffers_async_writes_and_mismatched_database(self):
        self.validate()
        for key in ("async_cloud_writes", "enable_buffer", "enable_upsert_batching"):
            with self.subTest(key=key):
                self.payload["durability"][key] = True
                with self.assertRaises(RuntimeError):
                    self.validate()
                self.payload["durability"][key] = False
        self.payload["storage_identity"]["db_url"] = "sqlite:///tmp/another.db"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.validate()

    def test_cloud_compares_resolved_tables(self):
        self.flow.cfg.storage_type = "cloud"
        self.flow.data_manager = SimpleNamespace(_strategy=SimpleNamespace(landing_table="landing_prod", env_config_table="env_prod"))
        self.payload.update(storage_type="cloud", storage_identity={"landing_table": "landing_prod", "env_config_table": "env_prod", "namespace": "gateway"})
        self.validate()
        self.payload["storage_identity"]["landing_table"] = "landing_test"
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.validate()


class SandboxResumeGuardTest(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_recoveries_and_requires_drain_before_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(job_id="job", gateway_base_url="http://gateway/v1/sessions", gateway_close_timeout_s=1, gateway_close_retry_backoff_s=0, sandbox_config={"results": {"local_root": directory}})
            row = {"env_id": "session"}
            backend = SimpleNamespace(prepare_resume=AsyncMock(return_value=([row], {"saved": {"kind": "local"}})), close=AsyncMock())
            client = SimpleNamespace(get_session_status=AsyncMock(return_value={"status": "active"}), close_session=AsyncMock(return_value={"session_id": "session", "status": "closed", "drained": True, "telemetry_status": "flushed"}), aclose=AsyncMock())
            manager = SimpleNamespace(list_session_steps=AsyncMock(return_value=[]))
            with patch("manager.resume_cleanup.SandboxClusterBackend", return_value=backend), patch("manager.resume_cleanup.GatewayClient", return_value=client):
                rerun, recovery = await prepare_sandbox_resume(cfg=cfg, manager_cfg={}, environment_rows=[row], data_manager=manager)
                self.assertEqual(rerun, [row])
                self.assertIn("saved", recovery)
                client.close_session.assert_awaited_once()
                client.close_session.return_value["telemetry_status"] = "failed"
                with self.assertRaisesRegex(RuntimeError, "undrained"):
                    await prepare_sandbox_resume(cfg=cfg, manager_cfg={}, environment_rows=[row], data_manager=manager)
                manager.list_session_steps.return_value = [{"step_id": 1}]
                with self.assertRaisesRegex(RuntimeError, "without a runtime reference"):
                    await prepare_sandbox_resume(cfg=cfg, manager_cfg={}, environment_rows=[row], data_manager=manager)
            self.assertEqual(client.aclose.await_count, 3)
            self.assertEqual(backend.close.await_count, 3)
