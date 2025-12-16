from __future__ import annotations

import base64
from typing import Any, Dict, Optional, Union

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from env.mc.service.env_manager import EnvManager


app = FastAPI(
    title="MCGym FastAPI Service",
    version="0.1.0",
    description="Expose reset/step/render/close endpoints for the Minecraft simulator.",
)

manager = EnvManager()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    env_config: Optional[str] = None
    force_recreate: bool = False


class StepRequest(BaseModel):
    action: Union[str, Dict[str, Any]]


@app.get("/healthz")
async def health() -> Dict[str, Any]:
    return manager.health()


@app.post("/reset")
async def reset(request: ResetRequest | None = Body(default=None)) -> Dict[str, Any]:
    try:
        payload = request or ResetRequest()
        output = manager.reset(
            seed=payload.seed,
            env_config=payload.env_config,
            force_recreate=payload.force_recreate,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return output.model_dump()


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    try:
        output = manager.step(action=request.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return output.model_dump()


@app.post("/render")
async def render() -> Dict[str, Any]:
    try:
        render_output = manager.render()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = render_output.model_dump(exclude_none=True)
    if "image_data" in payload and isinstance(payload["image_data"], (bytes, bytearray, memoryview)):
        payload["image_data"] = base64.b64encode(payload["image_data"]).decode("utf-8")
    return payload


@app.post("/close")
async def close() -> Dict[str, str]:
    manager.close()
    return {"detail": "Environment closed"}


@app.on_event("shutdown")
def _shutdown():
    manager.close()

