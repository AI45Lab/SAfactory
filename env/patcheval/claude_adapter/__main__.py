from __future__ import annotations

import os

import uvicorn

from .app import create_app


def main() -> None:
    uvicorn.run(
        create_app(),
        host=os.environ.get("CLAUDE_ADAPTER_HOST", "0.0.0.0"),
        port=int(os.environ.get("CLAUDE_ADAPTER_PORT", "18001")),
        log_level=os.environ.get("CLAUDE_ADAPTER_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
