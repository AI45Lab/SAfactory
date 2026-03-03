from __future__ import annotations

from typing import Optional
from core.data_manager.manager import SessionContext


class BaseURLProvider:
    """
    抽象的 base_url 生成器接口。
    """

    def get_base_url(self, session: Optional[SessionContext] = None) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class StaticBaseURLProvider(BaseURLProvider):
    """
    返回固定 base_url，不随 session 变化。
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get_base_url(self, session: Optional[SessionContext] = None) -> str:
        return self.base_url


class SessionSuffixBaseURLProvider(BaseURLProvider):
    """
    为每个 session 生成带 session_id 后缀的 base_url：
      base_url_root/<session_id>
    """

    def __init__(self, base_url_root: str) -> None:
        self.base_url_root = base_url_root.rstrip("/")

    def get_base_url(self, session: Optional[SessionContext] = None) -> str:
        if session is None:
            raise ValueError("SessionSuffixBaseURLProvider requires a session")
        return f"{self.base_url_root}/{session.session_id}"

