"""Controller-only planning of immutable, task-scoped S3 input downloads.

Only a short-lived signed plan crosses into the Sandbox. Credentials, clients,
and storage profiles stay in this process and must never be serialized with a
start request, lease, or result.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
import time
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

from env.harbor.input_bundle import parse_image_lock, resolve_archive_path


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA256 = re.compile(r"[a-fA-F0-9]{64}\Z")
_PROFILE_KEYS = {
    "endpoint_url", "region_name", "addressing_style", "bucket", "prefix",
    "access_key_id_env", "secret_access_key_env", "session_token_env",
    "auth_mode", "url_ttl_s", "loader_files",
}
_MAX_LOCK_BYTES = 4 * 1024 * 1024
_MAX_FILES = 10000
_PRIVATE_S3_PLANNING: ContextVar[bool] = ContextVar("private_s3_planning", default=False)
_SDK_LOG_ROOTS = ("boto3", "botocore", "s3transfer", "urllib3")


class _PrivateS3LogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Botocore DEBUG output includes authorization headers and session tokens
        # before our exception boundary. Suppress only this planning context.
        return not _PRIVATE_S3_PLANNING.get()


_SDK_LOG_FILTER = _PrivateS3LogFilter()


def _install_sdk_log_filters() -> None:
    # Ancestor logger filters do not apply to records emitted by descendants.
    # Include known sensitive loggers and every SDK logger imported so far.
    names = {*_SDK_LOG_ROOTS, "botocore.auth", "botocore.endpoint", "botocore.parsers",
             "botocore.credentials", "botocore.httpsession", "urllib3.connectionpool"}
    names.update(name for name in list(logging.Logger.manager.loggerDict)
                 if any(name == root or name.startswith(root + ".") for root in _SDK_LOG_ROOTS))
    for name in names:
        logging.getLogger(name).addFilter(_SDK_LOG_FILTER)


class S3InputError(RuntimeError):
    """A stable failure code and a message safe to persist in run results."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _text(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f"{label} must be a nonempty string without outer whitespace or controls")
    return value


def _segment(value: Any, label: str) -> str:
    value = _text(value, label)
    if value.startswith("-") or value in {".", ".."} or any(c in value for c in "/\\:"):
        raise ValueError(f"{label} must be a single path segment")
    return value


def _relative(value: Any, label: str) -> str:
    value = _text(value, label)
    if (value.startswith("/") or "\\" in value or ":" in value
            or any(p in {"", ".", ".."} for p in value.split("/"))):
        raise ValueError(f"{label} must be a safe relative POSIX path")
    return value


def _endpoint(value: Any) -> str:
    value = _text(value, "endpoint_url")
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"} or "\\" in value or parsed.port == 0
                or any(c.isspace() for c in parsed.hostname)):
            raise ValueError
        # Accessing .port also rejects malformed ports without echoing the URL.
        parsed.port
    except ValueError:
        raise ValueError("endpoint_url must be an HTTPS origin without credentials, path, query, or fragment") from None
    return value.rstrip("/")


def normalize_input_storage(value: Any) -> dict[str, Any]:
    """Validate profiles without resolving or reading credential environment variables."""
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {"s3_profiles"}:
        raise ValueError("input_storage only accepts s3_profiles")
    profiles = value.get("s3_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("input_storage.s3_profiles must be a mapping")
    normalized = {}
    for name, raw in profiles.items():
        _segment(name, "S3 profile name")
        if not isinstance(raw, dict) or set(raw) - _PROFILE_KEYS:
            raise ValueError("S3 profile contains unsupported fields")
        profile = {
            "endpoint_url": _endpoint(raw.get("endpoint_url")),
            "region_name": _text(raw.get("region_name", "us-east-1"), "region_name"),
            "addressing_style": raw.get("addressing_style", "path"),
            "bucket": _segment(raw.get("bucket"), "bucket"),
            "auth_mode": raw.get("auth_mode", "presigned"),
        }
        if profile["addressing_style"] != "path":
            raise ValueError("S3 input addressing_style must be path")
        if profile["auth_mode"] != "presigned":
            raise ValueError("S3 input auth_mode must be presigned")
        prefix = raw.get("prefix", "")
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")
        profile["prefix"] = _relative(prefix.rstrip("/"), "prefix") if prefix else ""
        for key in ("access_key_id_env", "secret_access_key_env", "session_token_env"):
            env_name = raw.get(key)
            if key == "session_token_env" and (env_name is None or env_name == ""):
                continue
            if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name):
                raise ValueError(f"{key} must name an environment variable")
            profile[key] = env_name
        credential_envs = [profile[key] for key in (
            "access_key_id_env", "secret_access_key_env", "session_token_env",
        ) if key in profile]
        if len(set(credential_envs)) != len(credential_envs):
            raise ValueError("S3 credential environment variable names must be distinct")
        ttl = raw.get("url_ttl_s", 3600)
        if type(ttl) is not int or not 60 <= ttl <= 604800:
            raise ValueError("url_ttl_s must be an integer between 60 and 604800")
        profile["url_ttl_s"] = ttl
        loaders = raw.get("loader_files", ["bin/load_image.py"])
        if not isinstance(loaders, list) or not loaders:
            raise ValueError("loader_files must be a nonempty list")
        loaders = [_relative(path, "loader_files entry") for path in loaders]
        if any(not path.startswith("bin/") for path in loaders) or "bin/load_image.py" not in loaders:
            raise ValueError("loader_files must be under bin/ and include bin/load_image.py")
        profile["loader_files"] = list(dict.fromkeys(loaders))
        normalized[name] = profile
    return {"s3_profiles": normalized} if normalized else {}


class S3InputPlanner:
    """Lazy credential resolution; bounded SDK calls; no credential-bearing repr."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = normalize_input_storage(config)
        # Each planning call owns a client; cancelling the caller cannot close
        # a client while its worker thread is still using it.

    def _client(self, name: str, profile: dict[str, Any]) -> Any:
        del name
        credentials = {}
        for config_key, client_key in (
            ("access_key_id_env", "aws_access_key_id"),
            ("secret_access_key_env", "aws_secret_access_key"),
            ("session_token_env", "aws_session_token"),
        ):
            env_name = profile.get(config_key)
            secret = os.environ.get(env_name, "") if env_name else None
            if env_name and (not secret or not secret.strip()):
                raise S3InputError("credentials_missing", "S3 input credential environment variable is missing or empty")
            credentials[client_key] = secret
        try:
            import boto3
            from botocore.config import Config
            _install_sdk_log_filters()
            return boto3.client(
                "s3", endpoint_url=profile["endpoint_url"],
                region_name=profile["region_name"], **credentials,
                config=Config(
                    signature_version="s3v4", s3={"addressing_style": "path"},
                    connect_timeout=10, read_timeout=60,
                    retries={"mode": "standard", "total_max_attempts": 3},
                ),
            )
        except Exception:
            raise S3InputError("client_initialization", "S3 input client initialization failed") from None

    def plan(
        self, profile_name: str, task_id: str, profile_subdir: str, *,
        session_id: str, attempt_id: str,
        cancel_event: threading.Event | None = None, deadline: float | None = None,
    ) -> dict[str, Any]:
        """Plan only this task's profile, lock, all locked image roles, and explicit loaders."""
        token = _PRIVATE_S3_PLANNING.set(True)
        try:
            _install_sdk_log_filters()
            return self._plan(
                profile_name, task_id, profile_subdir, session_id=session_id,
                attempt_id=attempt_id, cancel_event=cancel_event, deadline=deadline,
            )
        finally:
            _PRIVATE_S3_PLANNING.reset(token)

    def _plan(
        self, profile_name: str, task_id: str, profile_subdir: str, *,
        session_id: str, attempt_id: str,
        cancel_event: threading.Event | None, deadline: float | None,
    ) -> dict[str, Any]:
        self._check_budget(cancel_event, deadline)
        _segment(profile_name, "S3 profile name")
        _segment(task_id, "task_id")
        _segment(profile_subdir, "profile_subdir")
        _text(session_id, "session_id")
        _text(attempt_id, "attempt_id")
        profile = self._config.get("s3_profiles", {}).get(profile_name)
        if profile is None:
            raise S3InputError("profile_unknown", "unknown S3 input profile")
        client = self._client(profile_name, profile)
        bucket = profile["bucket"]
        prefix = profile["prefix"] + "/" if profile["prefix"] else ""
        phase = "reading image lock"
        try:
            self._check_budget(cancel_event, deadline)
            lock_relative = f"{task_id}/image-lock.json"
            response = client.get_object(Bucket=bucket, Key=prefix + lock_relative)
            stream = response["Body"]
            try:
                self._check_budget(cancel_event, deadline)
                body = stream.read(_MAX_LOCK_BYTES + 1)
            finally:
                stream.close()
            self._check_budget(cancel_event, deadline)
            if len(body) > _MAX_LOCK_BYTES or response.get("ContentLength") != len(body):
                raise ValueError("invalid lock length")
            lock_sha = hashlib.sha256(body).hexdigest()
            images = parse_image_lock(body, task_id)
            metadata = {lock_relative: self._fixed_metadata(response)}
            hashes = {lock_relative: lock_sha}

            phase = "listing task profile"
            profile_prefix = f"{task_id}/{profile_subdir}/"
            listed = set()
            paginator = client.get_paginator("list_objects_v2")
            pages = iter(paginator.paginate(Bucket=bucket, Prefix=prefix + profile_prefix))
            while True:
                # Check before advancing: pagination can issue another HTTP call.
                self._check_budget(cancel_event, deadline)
                try:
                    page = next(pages)
                except StopIteration:
                    break
                self._check_budget(cancel_event, deadline)
                for item in page.get("Contents", []):
                    key = item.get("Key")
                    if not isinstance(key, str) or not key.startswith(prefix + profile_prefix):
                        raise ValueError("object escaped task profile")
                    if key.endswith("/"):
                        continue
                    relative = _relative(key[len(prefix):], "profile object")
                    listed.add(relative)
                    if len(listed) > _MAX_FILES:
                        raise ValueError("too many input objects")
            if not listed:
                raise ValueError("task profile is empty")
            if (any(profile_prefix + name not in listed
                    for name in ("task.toml", "instruction.md", "tests/test.sh"))
                    or not any(path.startswith(profile_prefix + "environment/") for path in listed)):
                raise S3InputError("profile_incomplete", "Harbor task profile is incomplete")

            relatives = listed | set(profile["loader_files"])
            for row in images:  # Never filter by role: attacker/verifier/base can all be required.
                relative = resolve_archive_path(task_id, row["archive"])
                relatives.add(relative)
                digest = row.get("sha256")
                if digest is not None:
                    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                        raise ValueError("invalid archive digest")
                    digest = digest.lower()
                    if relative in hashes and hashes[relative] != digest:
                        raise ValueError("conflicting archive digest")
                    hashes[relative] = digest
            if len(relatives | {lock_relative}) > _MAX_FILES:
                raise ValueError("too many input objects")

            phase = "resolving immutable input objects"
            for relative in sorted(relatives - {lock_relative}):
                self._check_budget(cancel_event, deadline)
                metadata[relative] = self._fixed_metadata(
                    client.head_object(Bucket=bucket, Key=prefix + relative)
                )

            phase = "signing download plan"
            # Capture before the first signature. This slightly understates the
            # lifetime of later URLs and avoids an overstated plan expiry.
            expires_at = time.time() + profile["url_ttl_s"]
            files = []
            for relative, meta in sorted(metadata.items()):
                self._check_budget(cancel_event, deadline)
                key = prefix + relative
                params: dict[str, Any] = {"Bucket": bucket, "Key": key}
                headers = {}
                if meta["version_id"] is not None:
                    params["VersionId"] = meta["version_id"]
                else:
                    params["IfMatch"] = meta["etag"]
                    headers["If-Match"] = meta["etag"]
                url = client.generate_presigned_url(
                    "get_object", Params=params, ExpiresIn=profile["url_ttl_s"], HttpMethod="GET",
                )
                expected, signed = urlsplit(profile["endpoint_url"]), urlsplit(url)
                if ((signed.scheme, signed.hostname, signed.port or 443)
                        != (expected.scheme, expected.hostname, expected.port or 443)
                        or signed.username is not None or signed.password is not None or signed.fragment):
                    raise ValueError("signed object URL changed endpoint origin")
                item = {"key": key, "relative_path": relative, **meta, "url": url, "headers": headers}
                if relative in hashes:
                    item["sha256"] = hashes[relative]
                files.append(item)
            self._check_budget(cancel_event, deadline)
            return {
                "schema": 1, "session_id": session_id, "attempt_id": attempt_id,
                "task_id": task_id, "profile_subdir": profile_subdir, "expires_at": expires_at,
                "endpoint_url": profile["endpoint_url"], "files": files,
                "lock_sha256": lock_sha,
            }
        except Exception as error:
            # SDK exception strings may contain endpoints, signed URLs, headers,
            # and credential material. Do not retain a chained exception either.
            code = self._error_code(error, phase)
            raise S3InputError(code, f"S3 input planning failed while {phase}") from None
        finally:
            try:
                client.close()
            except Exception:
                pass

    @staticmethod
    def _check_budget(cancel_event: threading.Event | None, deadline: float | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise S3InputError("planning_cancelled", "S3 input planning cancelled")
        if deadline is not None:
            if isinstance(deadline, bool) or not isinstance(deadline, (float, int)) or not math.isfinite(deadline):
                raise S3InputError("invalid_deadline", "S3 input planning deadline is invalid")
            if time.monotonic() >= deadline:
                raise S3InputError("planning_timeout", "S3 input planning deadline expired")

    @staticmethod
    def _error_code(error: Exception, phase: str) -> str:
        if isinstance(error, S3InputError):
            return error.code
        response = getattr(error, "response", None)
        detail = response.get("Error", {}) if isinstance(response, dict) else {}
        code = str(detail.get("Code", "")) if isinstance(detail, dict) else ""
        if code in {"NoSuchKey", "NoSuchVersion", "NotFound", "404"}:
            return "object_missing"
        if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "403"}:
            return "s3_access_denied"
        if code in {"PreconditionFailed", "412"}:
            return "snapshot_changed"
        if code in {"ExpiredToken", "InvalidToken", "TokenRefreshRequired"}:
            return "credentials_expired"
        if phase == "reading image lock" and isinstance(error, (ValueError, RuntimeError)):
            return "invalid_lock"
        return "input_plan_failed"

    @staticmethod
    def _fixed_metadata(response: dict[str, Any]) -> dict[str, Any]:
        size = response.get("ContentLength")
        etag = response.get("ETag")
        version = response.get("VersionId")
        if version in {None, "", "null"}:
            version = None
        if type(size) is not int or size < 0:
            raise ValueError("invalid S3 object length")
        if etag is not None and (not isinstance(etag, str) or not etag
                                 or any(ord(c) < 32 or ord(c) == 127 for c in etag)):
            raise S3InputError("snapshot_unbound", "invalid S3 object ETag")
        if version is not None and (not isinstance(version, str)
                                    or any(ord(c) < 32 or ord(c) == 127 for c in version)):
            raise ValueError("invalid S3 object version")
        if version is None and (not etag or not re.fullmatch(r'"[^"\\\x00-\x20\x7f]+"', etag)):
            raise S3InputError("snapshot_unbound", "S3 object has no immutable version or strong ETag")
        return {"size": size, "etag": etag, "version_id": version}

    def close(self) -> None:
        # No shared clients: each plan() closes its own client in finally.
        pass
