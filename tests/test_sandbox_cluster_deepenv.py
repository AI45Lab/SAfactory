from __future__ import annotations

import unittest
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from clusters.sandbox_cluster import (
    ProjectAfsVolume, SandboxClusterBackend, SandboxCommandOutput,
    _normalize_command_endpoint, _storage_volumes, run_sandbox_command,
)


class FakeEndpoint:
    def __init__(self, endpoint: str, headers: dict[str, str] | None = None) -> None:
        self.endpoint = endpoint
        self.headers = dict(headers or {})

    def model_copy(self, *, update: dict[str, object]) -> "FakeEndpoint":
        return FakeEndpoint(
            endpoint=str(update.get("endpoint", self.endpoint)),
            headers=dict(update.get("headers", self.headers)),
        )


class FakeConnection:
    instances: list["FakeConnection"] = []

    def __init__(
        self,
        *,
        domain: str = "deepenv.sandbox.pjlab.org.cn",
        api_key: str = "key",
        protocol: str = "http",
        use_server_proxy: bool = True,
        request_timeout: object = None,
        close_state: dict[str, int] | None = None,
        headers: dict | None = None,
    ) -> None:
        self.domain = domain
        self.api_key = api_key
        self.protocol = protocol
        self.use_server_proxy = use_server_proxy
        self.request_timeout = request_timeout
        self.close_state = close_state or {"calls": 0}
        self.headers = headers or {}
        self.__class__.instances.append(self)

    def with_transport_if_missing(self) -> "FakeConnection":
        return self

    def model_copy(self, *, update: dict[str, object]) -> "FakeConnection":
        return FakeConnection(
            domain=self.domain,
            api_key=self.api_key,
            protocol=str(update.get("protocol", self.protocol)),
            use_server_proxy=self.use_server_proxy,
            request_timeout=self.request_timeout,
            close_state=self.close_state,
        )

    async def close_transport_if_owned(self) -> None:
        self.close_state["calls"] += 1


class FakeSandboxService:
    def __init__(self, *, state: str = "Running", endpoint: FakeEndpoint | None = None) -> None:
        self.state = state
        self.endpoint = endpoint or FakeEndpoint("sandbox.local:44772")
        self.create_kwargs: dict[str, object] | None = None
        self.endpoint_calls: list[tuple[str, int, bool]] = []
        self.kill_calls: list[str] = []

    async def create_sandbox(self, **kwargs: object) -> SimpleNamespace:
        self.create_kwargs = kwargs
        return SimpleNamespace(id="sandbox-1")

    async def get_sandbox_info(self, sandbox_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=sandbox_id,
            status=SimpleNamespace(state=self.state, reason="test-reason", message="test-message"),
        )

    async def get_sandbox_endpoint(
        self,
        sandbox_id: str,
        port: int,
        use_server_proxy: bool,
    ) -> FakeEndpoint:
        self.endpoint_calls.append((sandbox_id, port, use_server_proxy))
        return self.endpoint

    async def kill_sandbox(self, sandbox_id: str) -> None:
        self.kill_calls.append(sandbox_id)


class FakeHealth:
    def __init__(self) -> None:
        self.ping_calls: list[str] = []

    async def ping(self, sandbox_id: str) -> bool:
        self.ping_calls.append(sandbox_id)
        return True


class AdapterHarness:
    def __init__(self, service: FakeSandboxService) -> None:
        self.service = service
        self.commands = object()
        self.files = object()
        self.health = FakeHealth()
        self.command_inputs: list[tuple[FakeConnection, FakeEndpoint]] = []
        self.health_inputs: list[tuple[FakeConnection, FakeEndpoint]] = []

    def factory_type(self):
        harness = self

        class FakeAdapterFactory:
            def __init__(self, connection: FakeConnection) -> None:
                self.connection = connection

            def create_sandbox_service(self) -> FakeSandboxService:
                return harness.service

            def create_command_service(self, endpoint: FakeEndpoint) -> object:
                harness.command_inputs.append((self.connection, endpoint))
                return harness.commands

            def create_filesystem_service(self, endpoint: FakeEndpoint) -> object:
                return harness.files

            def create_health_service(self, endpoint: FakeEndpoint) -> FakeHealth:
                harness.health_inputs.append((self.connection, endpoint))
                return harness.health

        return FakeAdapterFactory


def sandbox_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "domain": "deepenv.sandbox.pjlab.org.cn",
        "api_key": "ak:sk",
        "protocol": "http",
        "use_server_proxy": True,
        "request_timeout_s": 1.0,
        "create_timeout_s": 2.0,
        "lifecycle_minutes": 120,
        "command_port": 44772,
        "skip_health_check": False,
        "extensions": {},
    }
    config.update(overrides)
    return config


class SandboxEndpointNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeConnection.instances.clear()

    def test_full_proxy_url_is_normalized_only_for_adapter(self) -> None:
        connection = FakeConnection(protocol="http")
        public_endpoint = FakeEndpoint(
            "https://deepenv.sandbox.pjlab.org.cn/v1/sandboxes/sandbox-1/proxy/44772/",
            {"X-Sandbox-Route": "token"},
        )

        command_connection, adapter_endpoint = _normalize_command_endpoint(
            connection,
            public_endpoint,
        )

        self.assertEqual(command_connection.protocol, "https")
        self.assertEqual(
            adapter_endpoint.endpoint,
            "deepenv.sandbox.pjlab.org.cn/v1/sandboxes/sandbox-1/proxy/44772",
        )
        self.assertEqual(adapter_endpoint.headers, {"X-Sandbox-Route": "token"})
        self.assertEqual(
            public_endpoint.endpoint,
            "https://deepenv.sandbox.pjlab.org.cn/v1/sandboxes/sandbox-1/proxy/44772/",
        )

    def test_host_port_endpoint_keeps_connection_protocol(self) -> None:
        connection = FakeConnection(protocol="http")

        command_connection, adapter_endpoint = _normalize_command_endpoint(
            connection,
            FakeEndpoint("10.0.0.7:44772"),
        )

        self.assertIs(command_connection, connection)
        self.assertEqual(adapter_endpoint.endpoint, "10.0.0.7:44772")


class SandboxDeepEnvCreationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeConnection.instances.clear()

    async def test_environment_create_only_queries_44772_and_keeps_public_endpoint(self) -> None:
        public_endpoint = FakeEndpoint(
            "https://deepenv.sandbox.pjlab.org.cn/v1/sandboxes/sandbox-1/proxy/44772",
            {"X-Sandbox-Route": "token"},
        )
        service = FakeSandboxService(endpoint=public_endpoint)
        harness = AdapterHarness(service)
        backend = SandboxClusterBackend()
        cfg = sandbox_config(
            extensions={
                "project": "must-be-filtered",
                "environmentId": "stale-environment",
                "environmentName": "must-be-filtered",
                "fuse": "true",
            }
        )

        with patch(
            "clusters.sandbox_cluster._load_opensandbox",
            return_value=(harness.factory_type(), FakeConnection),
        ):
            handle = await backend._create_sandbox(
                image="ignored-by-environment",
                environment_id="environment-1",
                cfg=cfg,
            )

        self.assertIsNotNone(service.create_kwargs)
        request = service.create_kwargs or {}
        self.assertIsNone(request["spec"])
        self.assertIsNone(request["entrypoint"])
        self.assertIsNone(request["volumes"])
        self.assertEqual(request["env"], {})
        self.assertEqual(request["resource"], {})
        self.assertEqual(
            request["extensions"],
            {"fuse": "true", "environmentId": "environment-1"},
        )
        self.assertEqual(service.endpoint_calls, [("sandbox-1", 44772, True)])
        self.assertEqual(harness.health.ping_calls, ["sandbox-1"])
        self.assertEqual(len(harness.command_inputs), 1)
        command_connection, command_endpoint = harness.command_inputs[0]
        self.assertEqual(command_connection.protocol, "https")
        self.assertFalse(command_endpoint.endpoint.startswith(("http://", "https://")))
        self.assertIs(await handle.get_endpoint(44772), public_endpoint)
        self.assertEqual(service.endpoint_calls, [("sandbox-1", 44772, True)])
        self.assertIs(handle.commands, harness.commands)

        await handle.close()
        await handle.close()
        self.assertEqual(FakeConnection.instances[0].close_state["calls"], 1)

    async def test_explicit_resource_override_and_skip_health_check(self) -> None:
        service = FakeSandboxService()
        harness = AdapterHarness(service)
        backend = SandboxClusterBackend()

        with patch(
            "clusters.sandbox_cluster._load_opensandbox",
            return_value=(harness.factory_type(), FakeConnection),
        ):
            await backend._create_sandbox(
                image="ignored-by-environment",
                environment_id="environment-1",
                cfg=sandbox_config(
                    resource={"cpu": "4", "memory": "8Gi"},
                    skip_health_check=True,
                ),
            )

        self.assertEqual(
            (service.create_kwargs or {})["resource"],
            {"cpu": "4", "memory": "8Gi"},
        )
        self.assertEqual(harness.health_inputs, [])
        self.assertEqual(harness.health.ping_calls, [])
        self.assertEqual(service.endpoint_calls, [("sandbox-1", 44772, True)])

    async def test_terminal_state_kills_sandbox_and_closes_transport(self) -> None:
        service = FakeSandboxService(state="Failed")
        harness = AdapterHarness(service)
        backend = SandboxClusterBackend()

        with patch(
            "clusters.sandbox_cluster._load_opensandbox",
            return_value=(harness.factory_type(), FakeConnection),
        ):
            with self.assertRaisesRegex(RuntimeError, "terminal state 'failed'"):
                await backend._create_sandbox(
                    image="ignored-by-environment",
                    environment_id="environment-1",
                    cfg=sandbox_config(),
                )

        self.assertEqual(service.kill_calls, ["sandbox-1"])
        self.assertEqual(service.endpoint_calls, [])
        self.assertEqual(FakeConnection.instances[0].close_state["calls"], 1)


if __name__ == "__main__":
    unittest.main()


class FakeFiles:
    def __init__(self, *, corrupt: bool = False) -> None:
        self.data: dict[str, bytes] = {}
        self.corrupt = corrupt
        self.batches = []
        self.modes = {}

    async def create_directories(self, entries):
        return None

    async def write_files(self, entries):
        self.batches.append(entries)
        for entry in entries:
            self.data[entry.path] = bytes(entry.data)

    async def set_permissions(self, entries):
        self.modes.update({entry.path: entry.mode for entry in entries})

    async def read_bytes(self, path):
        return b'corrupted' if self.corrupt else self.data[path]


class SandboxFileAndIdentityTest(unittest.IsolatedAsyncioTestCase):
    async def test_batched_sdk_upload_preserves_bytes_and_explicit_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'runner.py', Path(directory) / 'data.bin'
            first.write_bytes(b'print("ok")\n')
            second.write_bytes(b'\x00\xffdata')
            files = FakeFiles()
            await SandboxClusterBackend()._prepare_instance(SimpleNamespace(files=files), {
                'embedded_files': [
                    {'source': str(first), 'target': '/tmp/runner.py', 'permissions': '0755'},
                    {'source': str(second), 'target': '/tmp/data.bin', 'permissions': '0644'},
                ],
            })
            self.assertEqual(len(files.batches), 1)
            self.assertEqual(files.data['/tmp/data.bin'], second.read_bytes())
            self.assertEqual(files.modes, {'/tmp/runner.py': 755, '/tmp/data.bin': 644})

    async def test_corrupted_upload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'runner.py'
            source.write_text('runner')
            with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                await SandboxClusterBackend()._prepare_instance(SimpleNamespace(files=FakeFiles(corrupt=True)), {
                    'embedded_files': [{'source': str(source), 'target': '/tmp/runner.py'}],
                })

    def test_full_volume_override_is_sdk_compatible_and_output_is_writable(self):
        from opensandbox.adapters.converter.sandbox_model_converter import SandboxModelConverter
        volumes = _storage_volumes({
            'inputs': [{'afs_subpath': 'inputs/v1', 'mount_path': '/app/data', 'read_only': True}],
            'outputs': [{'afs_subpath': 'outputs', 'mount_path': '/app/results', 'read_only': False}],
        })
        payloads = [SandboxModelConverter.to_api_volume(volume).to_dict() for volume in volumes]
        self.assertEqual(payloads[0], {'name': 'project-default-afs', 'mountPath': '/app/data', 'readOnly': True, 'subPath': 'inputs/v1'})
        self.assertFalse(payloads[1]['readOnly'])
        with self.assertRaisesRegex(ValueError, 'read_only'):
            _storage_volumes({'outputs': [{'afs_subpath': 'outputs', 'mount_path': '/app/results', 'read_only': True}]})

    async def test_allocation_writes_identity_before_create_and_passes_complete_mounts(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = sandbox_config(job_id='job-1', results={'local_root': directory})
            backend = SandboxClusterBackend(cluster_cfg={'sandbox': cfg, 'env_types': {
                'openrt': {'sandbox': {'environment_id': 'env-1', 'run_command': 'python runner.py',
                    'storage': {'outputs': [{'afs_subpath': 'results', 'mount_path': '/app/results'}]}}},
            }})
            handle = SimpleNamespace(id='sandbox-1', retain=False, _command_endpoint=FakeEndpoint('host:44772'))
            async def create(**kwargs):
                reference = json.loads((Path(directory) / 'job-1/session-1/runtime.json').read_text())
                self.assertIsNone(reference['sandbox_id'])
                self.assertEqual(reference['attempt_id'], kwargs['metadata']['attempt_id'])
                self.assertFalse(kwargs['volumes'][0].read_only)
                return handle
            with patch.object(backend, '_create_sandbox', side_effect=create), patch.object(backend, '_prepare_instance', new=AsyncMock()):
                entry = await backend.allocate(row_id=1, env_name='openrt', env_id='session-1', image='image', env_params={}, group_id='g')
            reference = json.loads(Path(entry.runtime_config['runtime_ref_path']).read_text())
            self.assertEqual(reference['sandbox_id'], 'sandbox-1')
            self.assertEqual(entry.runtime_config['attempt_id'], reference['attempt_id'])

    async def test_retain_protects_both_release_and_backend_close(self):
        from manager.types import PoolEntry
        backend = SandboxClusterBackend()
        retained = SimpleNamespace(retain=True, close=AsyncMock(), kill=AsyncMock())
        backend._instances['one'] = retained
        await backend.release(PoolEntry(env_name='env', env_id='s', row_id=1, image='image', job_name='one', resource_id='one', runtime_handle=retained), succeeded=False)
        retained.kill.assert_not_awaited()
        backend._instances['two'] = retained
        await backend.close()
        retained.kill.assert_not_awaited()
        self.assertEqual(retained.close.await_count, 2)


class SandboxCommandLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def test_missing_exit_code_never_defaults_to_success(self):
        with self.assertRaisesRegex(RuntimeError, 'trustworthy'):
            SandboxCommandOutput.from_sdk(SimpleNamespace(exit_code=None, error=None, logs=None))
        with self.assertRaisesRegex(RuntimeError, 'trustworthy'):
            SandboxCommandOutput.from_sdk(SimpleNamespace(exit_code=0, error='execution failure', logs=None))

    async def test_sse_init_is_preserved_and_cancellation_stops_remote_command(self):
        initialized = asyncio.Event()
        seen = []
        async def run(command, *, opts, handlers):
            self.assertEqual(opts.timeout.total_seconds(), 120)
            await handlers.on_init(SimpleNamespace(id='command-1'))
            initialized.set()
            await asyncio.Event().wait()
        commands = SimpleNamespace(run=run, interrupt=AsyncMock(), get_command_status=AsyncMock(return_value=SimpleNamespace(running=False)))
        handle = SimpleNamespace(commands=commands, retain=True)
        async def remember(command_id):
            seen.append(command_id)
        task = asyncio.create_task(run_sandbox_command(handle, 'runner', timeout_s=120, on_init=remember))
        await initialized.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        commands.interrupt.assert_awaited_once_with('command-1')
        self.assertEqual(seen, ['command-1'])
        self.assertTrue(handle.execution_stopped)

    async def test_lost_init_does_not_claim_remote_writer_stopped(self):
        async def run(command, *, opts, handlers):
            raise TimeoutError('transport lost')
        handle = SimpleNamespace(commands=SimpleNamespace(run=run), retain=False)
        with self.assertRaisesRegex(RuntimeError, 'state is unknown'):
            await run_sandbox_command(handle, 'runner', timeout_s=1)
        self.assertFalse(handle.execution_stopped)
        self.assertTrue(handle.retain)

    async def test_cancel_before_init_is_not_converted_to_runtime_failure(self):
        entered = asyncio.Event()
        async def run(command, *, opts, handlers):
            entered.set()
            await asyncio.Event().wait()
        handle = SimpleNamespace(commands=SimpleNamespace(run=run), retain=False)
        task = asyncio.create_task(run_sandbox_command(handle, 'runner', timeout_s=120))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(handle.execution_stopped)
        self.assertTrue(handle.retain)

    async def test_failed_interrupt_does_not_replace_cancellation(self):
        initialized = asyncio.Event()
        async def run(command, *, opts, handlers):
            await handlers.on_init(SimpleNamespace(id='command-1'))
            initialized.set()
            await asyncio.Event().wait()
        commands = SimpleNamespace(run=run, interrupt=AsyncMock(side_effect=OSError('connection lost')))
        handle = SimpleNamespace(commands=commands, retain=False)
        task = asyncio.create_task(run_sandbox_command(handle, 'runner', timeout_s=120))
        await initialized.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(handle.execution_stopped)
        self.assertTrue(handle.retain)

    async def test_cancellation_during_error_cleanup_is_not_swallowed(self):
        cleanup_started = asyncio.Event()
        async def run(command, *, opts, handlers):
            await handlers.on_init(SimpleNamespace(id='command-1'))
            raise OSError('connection lost')
        async def interrupt(command_id):
            cleanup_started.set()
            await asyncio.Event().wait()
        commands = SimpleNamespace(run=run, interrupt=interrupt)
        handle = SimpleNamespace(commands=commands, retain=False)
        task = asyncio.create_task(run_sandbox_command(handle, 'runner', timeout_s=120))
        await cleanup_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(handle.retain)

    async def test_repeated_cancellation_during_remote_stop_still_propagates(self):
        initialized = asyncio.Event()
        cleanup_started = asyncio.Event()
        async def run(command, *, opts, handlers):
            await handlers.on_init(SimpleNamespace(id='command-1'))
            initialized.set()
            await asyncio.Event().wait()
        async def interrupt(command_id):
            cleanup_started.set()
            await asyncio.Event().wait()
        handle = SimpleNamespace(commands=SimpleNamespace(run=run, interrupt=interrupt), retain=False)
        task = asyncio.create_task(run_sandbox_command(handle, 'runner', timeout_s=120))
        await initialized.wait()
        task.cancel()
        await cleanup_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(handle.execution_stopped)
        self.assertTrue(handle.retain)

    async def test_nonzero_exit_remains_nonzero_and_unknown_terminal_is_queried(self):
        from opensandbox.models.execd import Execution, ExecutionError
        execution = Execution(id='cmd', exit_code=None, error=ExecutionError(name='exit', value='7', timestamp=1))
        commands = SimpleNamespace(run=AsyncMock(return_value=execution), get_command_status=AsyncMock(return_value=SimpleNamespace(running=False, exit_code=7)))
        handle = SimpleNamespace(commands=commands, retain=False)
        result = await run_sandbox_command(handle, 'runner', timeout_s=1)
        self.assertEqual(result.exit_code, 7)
        self.assertTrue(handle.execution_stopped)


class SandboxResumeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cfg = sandbox_config(job_id='job-1', results={'local_root': self.temporary.name})
        self.cfg.update(environment_id='env-1', run_command='python runner.py')
        self.backend = SandboxClusterBackend(cluster_cfg={'sandbox': self.cfg})
        self.row = {'env_name': 'openrt', 'env_id': 'session-1', 'env_image': 'image'}
        self.reference = {
            'job_id': 'job-1', 'session_id': 'session-1', 'attempt_id': 'original-attempt',
            'sandbox_id': 'original-sandbox', 'image': 'image', 'environment_id': 'env-1',
            'config_digest': self.backend._config_digest(self.backend._config_for_env('openrt'), 'image'),
            'remote_result_path': '/app/results/job-1/session-1/safactory_result.json',
            'command_finished': True, 'exit_code': 0,
        }
        self.path = self.backend._runtime_path(self.cfg, 'session-1')
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(self.reference))
        self.body = {
            'schema_version': 'safactory.result.v2', 'job_id': 'job-1', 'session_id': 'session-1',
            'attempt_id': 'original-attempt', 'status': 'succeeded', 'step_count': 1,
            'terminated': True, 'truncated': False, 'total_reward': 0, 'error_text': None, 'metrics': {},
            'evaluation': {'session_id': 'session-1', 'status': 'succeeded', 'eval_id': 'test',
                'method': 'rule_evaluator', 'raw_score': 0, 'normalized_score_10': 0,
                'normalizer_id': 'test-v1', 'reason': '', 'artifacts': {}, 'error_text': None},
        }

    async def test_valid_local_result_resumes_without_allocating_or_rewriting_identity(self):
        (self.path.parent / 'safactory_result.json').write_text(json.dumps(self.body))
        before = self.path.read_bytes()
        with patch.object(self.backend, '_recover_handle', new=AsyncMock()) as connect:
            reruns, recoveries = await self.backend.prepare_resume([self.row])
        connect.assert_not_awaited()
        self.assertEqual(reruns, [])
        self.assertEqual(recoveries['session-1']['kind'], 'local')
        self.assertEqual(self.path.read_bytes(), before)
        self.backend._sandbox_cfg['recovery_by_session'] = recoveries
        with patch.object(self.backend, '_create_sandbox', new=AsyncMock()) as create:
            entry = await self.backend.allocate(row_id=1, env_name='openrt', env_id='session-1', image='image', env_params={}, group_id='g')
        create.assert_not_awaited()
        self.assertIsNone(entry.runtime_handle)
        self.assertEqual(entry.runtime_config['attempt_id'], 'original-attempt')
        self.assertEqual(self.path.read_bytes(), before)

    async def test_remote_final_file_is_recoverable_without_rerun(self):
        handle = SimpleNamespace(id='original-sandbox', execution_stopped=True, retain=True,
            files=SimpleNamespace(read_bytes=AsyncMock(return_value=json.dumps(self.body).encode())),
            close=AsyncMock(), kill=AsyncMock())
        with patch.object(self.backend, '_recover_handle', new=AsyncMock(return_value=handle)):
            reruns, recoveries = await self.backend.prepare_resume([self.row])
        self.assertEqual(reruns, [])
        self.assertEqual(recoveries['session-1']['kind'], 'remote')
        self.assertEqual(recoveries['session-1']['runtime_ref']['attempt_id'], 'original-attempt')
        handle.kill.assert_not_awaited()
        self.assertFalse((self.path.parent / 'safactory_result.json').exists())

    async def test_missing_final_file_reruns_only_after_confirmed_stop(self):
        self.reference.update(command_finished=False, command_id='command-1')
        self.path.write_text(json.dumps(self.reference))
        missing = RuntimeError('file missing')
        missing.status_code = 404
        handle = SimpleNamespace(id='original-sandbox', execution_stopped=False, retain=True,
            files=SimpleNamespace(read_bytes=AsyncMock(side_effect=missing)),
            commands=SimpleNamespace(get_command_status=AsyncMock(return_value=SimpleNamespace(running=False, exit_code=1, error=None)), interrupt=AsyncMock()),
            close=AsyncMock(), kill=AsyncMock())
        with patch.object(self.backend, '_recover_handle', new=AsyncMock(return_value=handle)):
            reruns, recoveries = await self.backend.prepare_resume([self.row])
        self.assertEqual(reruns, [self.row])
        self.assertEqual(recoveries, {})
        handle.commands.interrupt.assert_not_awaited()
        handle.kill.assert_awaited_once()

    async def test_unknown_command_identity_does_not_rerun_or_delete(self):
        self.reference['command_finished'] = False
        self.path.write_text(json.dumps(self.reference))
        handle = SimpleNamespace(id='original-sandbox', execution_stopped=False, retain=True, close=AsyncMock(), kill=AsyncMock())
        with patch.object(self.backend, '_recover_handle', new=AsyncMock(return_value=handle)):
            with self.assertRaisesRegex(RuntimeError, 'Missing command ID'):
                await self.backend.prepare_resume([self.row])
        handle.kill.assert_not_awaited()

    async def test_corrupt_or_stale_remote_file_is_not_automatically_rerun(self):
        body = dict(self.body, attempt_id='old-attempt')
        handle = SimpleNamespace(id='original-sandbox', execution_stopped=True, retain=True,
            files=SimpleNamespace(read_bytes=AsyncMock(return_value=json.dumps(body).encode())),
            close=AsyncMock(), kill=AsyncMock())
        with patch.object(self.backend, '_recover_handle', new=AsyncMock(return_value=handle)):
            with self.assertRaisesRegex(ValueError, 'attempt_id'):
                await self.backend.prepare_resume([self.row])
        handle.kill.assert_not_awaited()

    async def test_unallocated_row_requires_empty_metadata_lookup(self):
        self.path.unlink()
        service = SimpleNamespace(list_sandboxes=AsyncMock(return_value=SimpleNamespace(sandbox_infos=[])))
        connection = SimpleNamespace(close_transport_if_owned=AsyncMock())
        with patch.object(self.backend, '_connection', return_value=(connection, service)):
            reruns, recoveries = await self.backend.prepare_resume([self.row])
        self.assertEqual(reruns, [self.row])
        self.assertEqual(recoveries, {})
        service.list_sandboxes.return_value = SimpleNamespace(sandbox_infos=[SimpleNamespace(id='old', metadata={'job_id': 'job-1', 'session_id': 'session-1'})])
        with patch.object(self.backend, '_connection', return_value=(connection, service)):
            with self.assertRaisesRegex(RuntimeError, 'Missing runtime.json'):
                await self.backend.prepare_resume([self.row])

    async def test_changed_configuration_is_rejected_before_remote_access(self):
        self.backend._sandbox_cfg['run_command'] = 'different runner'
        with patch.object(self.backend, '_recover_handle', new=AsyncMock()) as connect:
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                await self.backend.prepare_resume([self.row])
        connect.assert_not_awaited()


class SandboxCreateRecoveryAndRenewalTest(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_create_response_is_found_by_identity_without_second_create(self):
        service = FakeSandboxService()
        metadata = {'job_id': 'job-1', 'session_id': 'session-1', 'attempt_id': 'attempt-1'}
        service.create_sandbox = AsyncMock(side_effect=TimeoutError('response lost'))
        service.list_sandboxes = AsyncMock(return_value=SimpleNamespace(sandbox_infos=[SimpleNamespace(id='sandbox-1', metadata=metadata)]))
        harness = AdapterHarness(service)
        backend = SandboxClusterBackend()
        with patch('clusters.sandbox_cluster._load_opensandbox', return_value=(harness.factory_type(), FakeConnection)):
            handle = await backend._create_sandbox(image='image', environment_id='env-1', cfg=sandbox_config(), metadata=metadata)
        service.create_sandbox.assert_awaited_once()
        self.assertEqual(handle.id, 'sandbox-1')
        self.assertIs(handle.files, harness.files)
        await handle.close()

    async def test_ambiguous_create_response_never_retries(self):
        service = FakeSandboxService()
        metadata = {'job_id': 'job-1', 'session_id': 'session-1', 'attempt_id': 'attempt-1'}
        service.create_sandbox = AsyncMock(side_effect=TimeoutError('response lost'))
        service.list_sandboxes = AsyncMock(return_value=SimpleNamespace(sandbox_infos=[]))
        harness = AdapterHarness(service)
        with patch('clusters.sandbox_cluster._load_opensandbox', return_value=(harness.factory_type(), FakeConnection)):
            with self.assertRaisesRegex(RuntimeError, 'outcome is unknown'):
                await SandboxClusterBackend()._create_sandbox(image='image', environment_id='env-1', cfg=sandbox_config(), metadata=metadata)
        service.create_sandbox.assert_awaited_once()
        self.assertEqual(service.kill_calls, [])

    async def test_renewal_uses_aware_expiration_and_close_cancels_heartbeat(self):
        from datetime import datetime, timezone
        service = FakeSandboxService()
        service.renew_sandbox_expiration = AsyncMock()
        harness = AdapterHarness(service)
        with patch('clusters.sandbox_cluster._load_opensandbox', return_value=(harness.factory_type(), FakeConnection)):
            handle = await SandboxClusterBackend()._create_sandbox(image='image', environment_id='env-1', cfg=sandbox_config())
        before = datetime.now(timezone.utc)
        await handle.renew(600)
        sandbox_id, expiry = service.renew_sandbox_expiration.await_args.args
        self.assertEqual(sandbox_id, 'sandbox-1')
        self.assertIsNotNone(expiry.tzinfo)
        self.assertGreater((expiry - before).total_seconds(), 599)
        task = handle._renewal_task
        await handle.close()
        self.assertTrue(task.done())


class SandboxStoragePreparationTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_stages_shared_inputs_only_once_and_keeps_each_full_volume_set(self):
        cfg = sandbox_config(job_id='job', project='project', afs={'webdav_url': 'http://webdav.local'})
        storage = {'inputs': [{'host_path': '/h/data', 'afs_subpath': 'inputs/v1', 'mount_path': '/data'}],
                   'outputs': [{'afs_subpath': 'outputs', 'mount_path': '/app/results'}]}
        backend = SandboxClusterBackend(cluster_cfg={'sandbox': cfg, 'env_types': {
            'one': {'sandbox': {'environment_id': 'env-one', 'storage': storage}},
            'two': {'sandbox': {'environment_id': 'env-two', 'storage': storage}},
        }})
        with patch.object(backend, '_validate_environment', new=AsyncMock()), patch('clusters.sandbox_cluster.prepare_storage', new=AsyncMock()) as prepare:
            await backend.start(SimpleNamespace(env_to_image={'one': 'image-one', 'two': 'image-two'}))
        self.assertEqual(prepare.await_count, 2)
        self.assertEqual(sum(len(call.kwargs['inputs']) for call in prepare.await_args_list), 1)
        self.assertEqual(prepare.await_args_list[0].kwargs['webdav_url'], 'http://webdav.local')
        self.assertEqual([v.mount_path for v in backend._volumes['two']], ['/data', '/app/results'])

    async def test_prepare_storage_passes_credentials_only_in_environment(self):
        from clusters.sandbox_cluster import prepare_storage
        calls = []
        async def process(arguments, *, env, stdin=None):
            calls.append((arguments, dict(env), stdin))
            return b'obscured\n' if arguments[:2] == ['rclone', 'obscure'] else b''
        with patch('clusters.sandbox_cluster._transfer_process', side_effect=process), patch(
            'clusters.sandbox_cluster._publish_afs_input', new=AsyncMock(),
        ) as publish:
            volumes = await prepare_storage(inputs=[{'host_path': '/h/data', 'afs_subpath': 'inputs/v1', 'mount_path': '/data'}], outputs=[{'afs_subpath': 'outputs', 'mount_path': '/app/results'}], project='project', webdav_url='http://webdav.local', api_key='access:very-secret')
        self.assertTrue(volumes[0].read_only)
        self.assertFalse(volumes[1].read_only)
        self.assertEqual(calls[0][0], ['rclone', 'obscure', '-'])
        self.assertEqual(calls[0][1]['SK'], 'very-secret')
        self.assertEqual(calls[0][2], b'very-secret')
        self.assertEqual(calls[1][0], ['rclone', 'mkdir', 'afs:outputs'])
        publish.assert_awaited_once()
        self.assertEqual(publish.await_args.kwargs['env']['RCLONE_CONFIG_AFS_PASS'], 'obscured')
        self.assertNotIn('very-secret', repr([call[0] for call in calls]))
