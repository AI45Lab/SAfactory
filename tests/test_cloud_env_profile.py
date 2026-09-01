import asyncio
from types import SimpleNamespace

import pytest

from core.data_manager.strategy import cloud_strategy_impl as cloud


class _FakeGatewayConfig:
    def __init__(self, **_kwargs):
        profile = cloud.os.environ.get("WT_SDK_PROFILE", "test").strip().lower()
        profile = "production" if profile in {"prod", "production"} else "test"
        self.tables = SimpleNamespace(
            profile=profile,
            db_uri="s3://trajectory-test",
            landing_table=(
                "wind_tunnel_landing" if profile == "production" else "landing_test"
            ),
        )
        self.s3 = SimpleNamespace(to_storage_options=lambda: {})


class _FakeGatewayClient:
    def __init__(self, config):
        self.config = config

    def close(self):
        return None


class _FakeEnvConfigManager:
    calls = []

    def __init__(self, *, table_name=None, profile=None, **_kwargs):
        self.calls.append({"table_name": table_name, "profile": profile})
        self.table_name = table_name or (
            "evaluation_env_config" if profile == "production" else "env_config_test"
        )

    def close(self):
        return None


class _FakeS3Uploader:
    pass


def _install_cloud_fakes(monkeypatch):
    _FakeEnvConfigManager.calls.clear()
    monkeypatch.setattr(cloud, "_load_wt_sdk", lambda: None)
    monkeypatch.setattr(cloud, "GatewayConfig", _FakeGatewayConfig)
    monkeypatch.setattr(cloud, "WTGatewayClient", _FakeGatewayClient)
    monkeypatch.setattr(cloud, "EnvConfigManager", _FakeEnvConfigManager)
    monkeypatch.setattr(cloud, "S3Uploader", _FakeS3Uploader)


@pytest.mark.parametrize(
    ("profile", "expected_landing", "expected_env"),
    [
        ("test", "landing_test", "env_config_test"),
        ("production", "wind_tunnel_landing", "evaluation_env_config"),
        ("prod", "wind_tunnel_landing", "evaluation_env_config"),
    ],
)
def test_cloud_strategy_uses_profile_for_landing_and_env_tables(
    monkeypatch,
    profile,
    expected_landing,
    expected_env,
):
    _install_cloud_fakes(monkeypatch)
    monkeypatch.setenv("WT_SDK_PROFILE", profile)
    strategy = cloud.CloudStrategy(job_id="profile-test")

    asyncio.run(strategy.init())

    assert strategy.landing_table == expected_landing
    assert strategy.env_config_table == expected_env
    assert _FakeEnvConfigManager.calls == [
        {
            "table_name": None,
            "profile": "production" if profile == "prod" else profile,
        }
    ]
    asyncio.run(strategy.close())


def test_explicit_env_table_still_overrides_profile(monkeypatch):
    _install_cloud_fakes(monkeypatch)
    monkeypatch.setenv("WT_SDK_PROFILE", "test")
    strategy = cloud.CloudStrategy(
        job_id="profile-test",
        env_config_table=" custom_env_config ",
    )

    asyncio.run(strategy.init())

    assert strategy.env_config_table == "custom_env_config"
    assert _FakeEnvConfigManager.calls == [
        {"table_name": "custom_env_config", "profile": "test"}
    ]
    asyncio.run(strategy.close())


@pytest.mark.parametrize(
    ("profile", "expected"),
    [("test", "test"), ("prod", "production")],
)
def test_mock_sdk_fallback_exposes_resolved_profile(monkeypatch, profile, expected):
    monkeypatch.setenv("WT_SDK_PROFILE", profile)
    for name in (
        "GatewayConfig",
        "LandingRecord",
        "generate_deterministic_id",
        "S3Uploader",
        "S3Downloader",
    ):
        monkeypatch.setattr(cloud, name, None)

    cloud._install_mock_wt_sdk_fallbacks()

    assert cloud.GatewayConfig().tables.profile == expected
