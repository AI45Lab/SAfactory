from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import unquote

import httpx

import clusters.sandbox_cluster as backend


FAKE_RCLONE = r'''import os, pathlib, shutil, sys, time, json
root = pathlib.Path(os.environ['FAKE_AFS'])
args = sys.argv[1:]
with open(os.environ['FAKE_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\n')
def path(value):
    return root / value[4:] if value.startswith('afs:') else pathlib.Path(value)
def copy(source, destination):
    if destination.exists() and '--immutable' in args:
        if source.stat().st_size != destination.stat().st_size:
            raise SystemExit('immutable destination differs')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
command = args[0]
if command == 'obscure':
    sys.stdin.read()
    print('obscured-placeholder')
elif command == 'mkdir':
    path(args[1]).mkdir(parents=True, exist_ok=True)
elif command in ('copy', 'check'):
    if command == 'copy' and os.environ.get('FAKE_COPY_FAILURE'):
        raise SystemExit('simulated copy failure')
    if command == 'copy' and os.environ.get('FAKE_COPY_PAUSE'):
        pause = pathlib.Path(os.environ['FAKE_COPY_PAUSE'])
        pause.with_suffix('.pid').write_text(str(os.getpid()))
        pause.with_suffix('.entered').touch()
        while not pause.with_suffix('.release').exists():
            time.sleep(.01)
    source, destination = path(args[1]), path(args[2])
    for parent, directories, names in os.walk(source):
        relative = pathlib.Path(parent).relative_to(source)
        if command == 'copy':
            (destination / relative).mkdir(parents=True, exist_ok=True)
        for name in names:
            item = pathlib.Path(parent) / name
            if item.is_symlink() and '--copy-links' not in args:
                continue
            target = destination / relative / name
            if command == 'copy':
                copy(item, target)
            elif not target.exists() or target.read_bytes() != item.read_bytes():
                raise SystemExit('destination bytes differ')
elif command == 'cat':
    sys.stdout.buffer.write(path(args[1]).read_bytes())
elif command == 'rcat':
    path(args[1]).write_bytes(sys.stdin.buffer.read())
elif command == 'lsjson':
    directory = path(args[1])
    print(json.dumps([{'Path': str(item.relative_to(directory)), 'Size': item.stat().st_size}
                     for item in directory.rglob('*') if item.is_file()]))
else:
    raise SystemExit('unsupported command')
'''


class AfsRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.remote = self.root / 'remote'
        self.remote.mkdir()
        self.log = self.root / 'rclone.jsonl'
        executable = self.root / 'rclone'
        executable.write_text(f'#!{sys.executable}\n' + FAKE_RCLONE)
        executable.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{self.root}:{os.environ["PATH"]}',
                        FAKE_AFS=str(self.remote), FAKE_LOG=str(self.log))
        self.http_requests = []
        self.fail_lock = False
        self.fail_unlock = False
        self.lose_lock_response = False
        real_client = httpx.AsyncClient
        self.environment_patch = patch.dict(os.environ, self.env)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.client_patch = patch.object(backend.httpx, 'AsyncClient', side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(self.handle_request), **kwargs))
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def handle_request(self, request):
        self.http_requests.append(request.method)
        target = self.remote / unquote(request.url.path.lstrip('/'))
        if request.method == 'MKCOL':
            if self.fail_lock:
                return httpx.Response(500)
            try:
                target.mkdir()
            except FileExistsError:
                return httpx.Response(405)
            if self.lose_lock_response:
                raise httpx.ReadTimeout('lost lock response', request=request)
            return httpx.Response(201)
        if request.method == 'DELETE':
            if self.fail_unlock:
                return httpx.Response(500)
            target.rmdir()
            return httpx.Response(204)
        raise AssertionError(request.method)

    async def prepare(self, *, source=None, subpath='inputs/version-1'):
        return await backend.prepare_storage(
            inputs=[{'host_path': str(source or self.source), 'afs_subpath': subpath, 'mount_path': '/app/data'}],
            outputs=[{'afs_subpath': 'results', 'mount_path': '/app/results'}],
            project='test-project', webdav_url='http://test-webdav.local', api_key='access:test-secret',
        )

    def commands(self):
        return [json.loads(line)[0] for line in self.log.read_text().splitlines()]

    async def test_runtime_materializes_links_and_publishes_verified_manifest(self):
        (self.source / 'payload').write_bytes(b'actual-data')
        (self.source / 'alias').symlink_to('payload')
        (self.source / 'empty').mkdir()
        volumes = await self.prepare()
        destination = self.remote / 'inputs/version-1'
        self.assertEqual((destination / 'alias').read_bytes(), b'actual-data')
        self.assertFalse((destination / 'alias').is_symlink())
        self.assertTrue((destination / 'empty').is_dir())
        manifest = json.loads((destination / backend._AFS_READY).read_bytes())
        self.assertEqual({entry['path'] for entry in manifest['files']}, {'payload', 'alias'})
        self.assertTrue(all(entry['sha256'] == hashlib.sha256(b'actual-data').hexdigest() for entry in manifest['files']))
        self.assertTrue(volumes[0].read_only)
        self.assertFalse(volumes[1].read_only)
        self.assertTrue((self.remote / 'results').is_dir())
        self.assertNotIn('test-secret', self.log.read_text())
        self.assertEqual(self.http_requests, ['MKCOL', 'DELETE'])

    async def test_directory_links_and_legacy_gpfs_alias_are_materialized(self):
        actual = self.root / 'available'
        actual.mkdir()
        (actual / 'data').mkdir()
        (actual / 'data/payload').write_text('A')
        (self.source / 'legacy').symlink_to('/mnt/shared-storage-gpfs2/data', target_is_directory=True)
        with patch.dict(backend._AFS_PATH_ALIASES, {Path('/mnt/shared-storage-gpfs2'): actual}):
            await self.prepare()
        self.assertEqual((self.remote / 'inputs/version-1/legacy/payload').read_text(), 'A')

    async def test_published_version_cannot_change_bytes_or_lose_source_files(self):
        (self.source / 'kept').write_text('A')
        removed = self.source / 'removed'
        removed.write_text('B')
        await self.prepare()
        marker = self.remote / 'inputs/version-1' / backend._AFS_READY
        before = marker.read_bytes()
        removed.unlink()
        with self.assertRaisesRegex(RuntimeError, 'complete source manifest'):
            await self.prepare()
        self.assertEqual(marker.read_bytes(), before)
        removed.write_text('C')
        with self.assertRaisesRegex(RuntimeError, 'complete source manifest'):
            await self.prepare()
        self.assertEqual((marker.parent / 'removed').read_text(), 'B')

    async def test_remote_byte_corruption_and_extra_files_are_rejected(self):
        (self.source / 'payload').write_text('A')
        await self.prepare()
        destination = self.remote / 'inputs/version-1'
        (destination / 'payload').write_text('B')
        with self.assertRaisesRegex(RuntimeError, 'destination bytes differ'):
            await self.prepare()
        (destination / 'payload').write_text('A')
        (destination / 'extra').write_text('C')
        with self.assertRaisesRegex(RuntimeError, 'inventory differs'):
            await self.prepare()

    async def test_broken_link_cycle_and_special_file_are_rejected(self):
        (self.source / 'broken').symlink_to('missing')
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            await self.prepare()
        (self.source / 'broken').unlink()
        (self.source / 'cycle').symlink_to('.')
        with self.assertRaisesRegex(ValueError, 'cycle'):
            await self.prepare()
        (self.source / 'cycle').unlink()
        os.mkfifo(self.source / 'fifo')
        with self.assertRaisesRegex(ValueError, 'special input file'):
            await self.prepare()
        self.assertNotIn('copy', self.commands())
        self.assertEqual(self.http_requests, [])

    async def test_traversal_destination_is_rejected_before_remote_access(self):
        with self.assertRaisesRegex(ValueError, 'subpath'):
            await self.prepare(subpath='../escape')
        self.assertFalse(self.log.exists())
        self.assertEqual(self.http_requests, [])

    async def test_foreign_or_unknown_lock_never_allows_copy_or_delete(self):
        (self.source / 'payload').write_text('A')
        lock = self.remote / 'inputs/version-1' / backend._AFS_LOCK
        lock.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'HTTP 405'):
            await self.prepare()
        self.assertTrue(lock.is_dir())
        self.assertEqual(self.http_requests, ['MKCOL'])
        self.fail_lock = True
        with self.assertRaisesRegex(RuntimeError, 'HTTP 500'):
            await self.prepare()
        self.assertNotIn('copy', self.commands())
        self.assertNotIn('DELETE', self.http_requests)

    async def test_failed_copy_releases_only_its_lock_and_does_not_publish_ready(self):
        (self.source / 'payload').write_text('A')
        with patch.dict(os.environ, {'FAKE_COPY_FAILURE': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'simulated copy failure'):
                await self.prepare()
        destination = self.remote / 'inputs/version-1'
        self.assertFalse((destination / backend._AFS_LOCK).exists())
        self.assertFalse((destination / backend._AFS_READY).exists())
        self.assertEqual(self.http_requests, ['MKCOL', 'DELETE'])

    async def test_lost_lock_response_never_claims_ownership_or_starts_copy(self):
        (self.source / 'payload').write_text('A')
        self.lose_lock_response = True
        with self.assertRaises(httpx.ReadTimeout):
            await self.prepare()
        self.assertTrue((self.remote / 'inputs/version-1' / backend._AFS_LOCK).exists())
        self.assertNotIn('copy', self.commands())
        self.assertEqual(self.http_requests, ['MKCOL'])

    async def test_source_change_while_copying_cannot_publish_a_ready_marker(self):
        payload = self.source / 'payload'
        payload.write_text('A')
        pause = self.root / 'pause'
        with patch.dict(os.environ, {'FAKE_COPY_PAUSE': str(pause)}):
            task = asyncio.create_task(self.prepare())
            await self.wait_for_file(pause.with_suffix('.entered'))
            payload.write_text('B')
            pause.with_suffix('.release').touch()
            with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                await task
        self.assertFalse((self.remote / 'inputs/version-1' / backend._AFS_READY).exists())
        self.assertEqual(self.http_requests, ['MKCOL', 'DELETE'])

    async def test_competing_publishers_cannot_mix_inputs(self):
        (self.source / 'first').write_text('A')
        pause = self.root / 'pause'
        first = asyncio.create_task(self.prepare())
        with patch.dict(os.environ, {'FAKE_COPY_PAUSE': str(pause)}):
            # The task sees the process environment when it begins execution.
            await self.wait_for_file(pause.with_suffix('.entered'))
            other = self.root / 'other'
            other.mkdir()
            (other / 'second').write_text('B')
            try:
                with self.assertRaisesRegex(RuntimeError, 'HTTP 405'):
                    await self.prepare(source=other)
            finally:
                pause.with_suffix('.release').touch()
                await first
        self.assertFalse((self.remote / 'inputs/version-1/second').exists())

    async def test_cancel_stops_transfer_process_and_releases_owned_lock(self):
        (self.source / 'payload').write_text('A')
        pause = self.root / 'pause'
        self.fail_unlock = True  # A cleanup failure must not turn cancellation into a runtime error.
        with patch.dict(os.environ, {'FAKE_COPY_PAUSE': str(pause)}):
            task = asyncio.create_task(self.prepare())
            await self.wait_for_file(pause.with_suffix('.entered'))
            pid = int(pause.with_suffix('.pid').read_text())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertEqual(self.http_requests, ['MKCOL', 'DELETE'])
        self.assertFalse((self.remote / 'inputs/version-1' / backend._AFS_READY).exists())

    async def wait_for_file(self, path):
        deadline = asyncio.get_running_loop().time() + 5
        while not path.exists():
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(f'File was not created: {path}')
            await asyncio.sleep(.01)

    async def test_runtime_works_in_a_checkout_with_no_sandbox_directory(self):
        (self.source / 'payload').write_text('independent')
        checkout = self.root / 'clean-checkout'
        checkout.mkdir()
        repository = Path(backend.__file__).resolve().parents[1]
        for relative in ('clusters/sandbox_cluster.py', 'clusters/base.py', 'core/result_artifact.py', 'manager/types.py'):
            target = checkout / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(repository / relative, target)
            (target.parent / '__init__.py').touch()
        self.assertFalse((checkout / 'sandbox').exists())
        program = '''
import asyncio, os
from pathlib import Path
from urllib.parse import unquote
import httpx
import clusters.sandbox_cluster as backend
real_client = httpx.AsyncClient
root = Path(os.environ['FAKE_AFS'])
def handler(request):
    path = root / unquote(request.url.path.lstrip('/'))
    if request.method == 'MKCOL':
        path.mkdir()
        return httpx.Response(201)
    path.rmdir()
    return httpx.Response(204)
backend.httpx.AsyncClient = lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
asyncio.run(backend.prepare_storage(
    inputs=[{'host_path': os.environ['INPUT_SOURCE'], 'afs_subpath': 'inputs/clean', 'mount_path': '/data'}],
    outputs=[{'afs_subpath': 'results', 'mount_path': '/app/results'}], project='test-project',
    webdav_url='http://test-webdav.local', api_key='access:test-secret'))
assert not Path('sandbox').exists()
assert (root / 'inputs/clean/payload').read_text() == 'independent'
print('no-sandbox-runtime-ok')
'''
        env = dict(self.env, PYTHONPATH=str(checkout), INPUT_SOURCE=str(self.source))
        process = await asyncio.create_subprocess_exec(sys.executable, '-c', program, cwd=checkout, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertIn(b'no-sandbox-runtime-ok', stdout)
        # The external tool log contains only rclone arguments; no reference script was invoked.
        self.assertNotIn('bash', self.commands())
        self.assertNotIn('python', self.commands())


class AfsProcessCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_during_error_cleanup_takes_precedence_and_waits_for_stop(self):
        stopping, release = asyncio.Event(), asyncio.Event()
        class Process:
            pid = 12345
            async def communicate(self, stdin):
                raise OSError('broken transfer pipe')
            async def wait(self):
                stopping.set()
                await release.wait()
                return 1
        with patch.object(backend.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=Process())), patch.object(backend.os, 'killpg') as kill:
            task = asyncio.create_task(backend._transfer_process(['rclone', 'copy'], env={}))
            await stopping.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])

    async def test_unconfirmed_process_cleanup_marks_the_original_error(self):
        class Process:
            pid = 12345
            async def communicate(self, stdin):
                raise OSError('broken transfer pipe')
        with patch.object(backend.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=Process())), patch.object(
            backend.os, 'killpg', side_effect=PermissionError('cannot stop process'),
        ):
            with self.assertRaises(OSError) as raised:
                await backend._transfer_process(['rclone', 'copy'], env={})
        self.assertIs(raised.exception.afs_writer_stopped, False)
