"""
Multiprocess HTTP server using the `mymp` backend.

Endpoints match the Ray-based server so that existing clients (including
`scripts/test_multi_api.py`) work unchanged.
"""

import uuid
import base64
import io
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import asyncio
import time
from copy import deepcopy

from .mymp.global_pool import get_global_env_pool


# ============================================================================
# 数据模型
# ============================================================================

class BatchCreateRequest(BaseModel):
    count: int = Field(..., ge=1, description="要创建的环境数量 >=1")
    env_name: str = Field(default="minecraft", description="环境名称")
    env_kwargs: List[Dict[str, Any]] = Field(default_factory=list, description="环境参数")


class BatchCreateResponse(BaseModel):
    env_ids: list


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    action: str = Field(..., description="JSON 格式的 action 字符串")


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    ray_initialized: bool
    num_environments: int


class EnvStatusResponse(BaseModel):
    env_id: str
    status: str  # "idle", "busy", "failed", "not_found"
    ready: bool  # 是否已经初始化完成，可以接受 step 请求


class StepResultResponse(BaseModel):
    status: str           # "success" / "pending" / "error"
    observation: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    truncated: Optional[bool] = None
    info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============================================================================
# 工具函数
# ============================================================================

def serialize_value(value: Any) -> Any:
    """序列化 numpy 与其他特殊对象."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    else:
        return value


def serialize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """将 observation 中的 RGB 数据压缩为 JPEG base64."""
    result = {}
    for key, value in obs.items():
        if key == 'rgb' and isinstance(value, np.ndarray):
            pil_image = Image.fromarray(value)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            jpeg_bytes = buffer.getvalue()
            result[key] = {
                'type': 'jpeg',
                'data': base64.b64encode(jpeg_bytes).decode('utf-8')
            }
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = serialize_value(value)
    return result


# ============================================================================
# FastAPI 生命周期
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 纯 multiprocess 后端，无需 Ray 初始化
    get_global_env_pool()
    yield


app = FastAPI(
    title="RayCraft GPU HTTP API (MP)",
    version="1.0.0",
    description="Multiprocess Minecraft environment API",
    lifespan=lifespan
)


# ============================================================================
# API 路由
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    try:
        env_pool = get_global_env_pool()
        num_envs = env_pool.get_num_envs()
        return HealthResponse(
            status="healthy",
            ray_initialized=False,
            num_environments=num_envs
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/batch/envs", response_model=BatchCreateResponse)
async def batch_create_envs(request: BatchCreateRequest):
    """批量创建环境，并在后台触发 reset。"""
    try:
        env_pool = get_global_env_pool()

        env_kwargs = request.env_kwargs or []
        env_ids = [str(uuid.uuid4()) for _ in range(request.count * len(env_kwargs))]

        configs = []
        for i in range(len(env_kwargs)):
            for _ in range(request.count):
                configs.append(env_kwargs[i])

        success = env_pool.create_envs(env_ids, configs)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create some environments"
            )

        # 触发异步 reset，不等待完成
        for idx, env_id in enumerate(env_ids):
            start_moment = time.perf_counter()
            print(f'[INFO] start to init, env_id={env_id}, start_moment={start_moment}')
            env_pool.trigger_reset(env_id)

        return BatchCreateResponse(env_ids=env_ids)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create environments: {str(e)}"
        )


@app.get("/envs/{env_id}/status", response_model=EnvStatusResponse)
async def get_env_status(env_id: str):
    """查询环境状态和是否已完成首次 reset。"""
    try:
        env_pool = get_global_env_pool()
        env_ref = env_pool.get_env(env_id)

        if env_ref is None:
            return EnvStatusResponse(
                env_id=env_id,
                status="not_found",
                ready=False
            )

        status_dict = env_pool.get_env_status(env_id)
        env_status = status_dict.get("status", "unknown")
        is_ready = env_pool.is_env_ready(env_id)

        return EnvStatusResponse(
            env_id=env_id,
            status=env_status,
            ready=is_ready
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get environment status: {str(e)}"
        )


@app.post("/envs/{env_id}/reset", response_model=ResetResponse)
async def reset_env(env_id: str):
    """同步 reset，返回 obs/info。"""
    try:
        env_pool = get_global_env_pool()
        env_ref = env_pool.get_env(env_id)

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        result = env_pool.reset_env(env_id, timeout=600)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}

        return ResetResponse(
            observation=serialize_observation(obs),
            info=serialize_value(info)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset: {str(e)}"
        )


@app.post("/envs/{env_id}/step", response_model=StepResponse)
async def step_env(env_id: str, request: StepRequest):
    """同步 step（主要用于调试）。"""
    from datetime import datetime
    print(f"{env_id=}, {datetime.now()}")
    try:
        env_pool = get_global_env_pool()
        env_ref = env_pool.get_env(env_id)

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        result = env_pool.step_sync(env_id, request.action)

        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        return StepResponse(
            observation=serialize_observation(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=serialize_value(info)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to step: {str(e)}"
        )


# ===== 1. 提交 step =====
@app.post("/envs/{env_id}/step_async", response_model=StepResultResponse)
async def submit_step(env_id: str, request: StepRequest):
    """异步提交 step，服务端只负责投递。"""
    from datetime import datetime
    t0 = time.time()
    print(f"[submit_step] recv env_id={env_id}, now={datetime.now()}")

    env_pool = get_global_env_pool()

    env_ref = env_pool.get_env(env_id)
    if env_ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment {env_id} not found"
        )

    t2 = time.time()
    env_pool.step(env_id, request.action)
    t3 = time.time()

    print(
        f"[submit_step] {env_id} timings - "
        f"get_env_ref: {t2 - t0:.3f}s, "
        f"dispatch_step: {t3 - t2:.3f}s, "
        f"total: {t3 - t0:.3f}s"
    )

    return StepResultResponse(status="submitted")


# ===== 2. 轮询结果 =====
@app.get("/envs/{env_id}/get_step_async", response_model=StepResultResponse)
async def get_step_result(env_id: str):
    """轮询 step 结果。"""
    t0 = time.time()
    env_pool = get_global_env_pool()
    t1 = time.time()

    step_res = env_pool.get_step_res()
    t2 = time.time()

    if env_id in step_res.keys():
        if len(step_res[env_id]) == 4:
            obs, reward, done, info = deepcopy(step_res[env_id])
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = deepcopy(step_res[env_id])
        env_pool.rm_step_res(env_id)
        t3 = time.time()
        print(
            f"[get_step_result] {env_id} timings - "
            f"pool_lookup: {t1 - t0:.3f}s, "
            f"get_res: {t2 - t1:.3f}s, "
            f"serialize+cleanup: {t3 - t2:.3f}s, "
            f"total: {t3 - t0:.3f}s"
        )
        return StepResultResponse(
            status="success",
            observation=serialize_observation(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=serialize_value(info)
        )
    else:
        t3 = time.time()
        print(
            f"[get_step_result] {env_id} pending - "
            f"pool_lookup: {t1 - t0:.3f}s, "
            f"get_res: {t2 - t1:.3f}s, "
            f"total: {t3 - t0:.3f}s"
        )
        return StepResultResponse(status="pending")


@app.post("/envs/{env_id}/close")
async def close_env(env_id: str):
    """关闭并销毁环境。"""
    try:
        env_pool = get_global_env_pool()
        success = env_pool.close_env(env_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        return success

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close: {str(e)}"
        )


@app.get("/envs/{env_id}/reset_result", response_model=ResetResponse)
async def get_reset_result(env_id: str, wait: int = 10):
    """获取后台 reset 结果（用于 batch_create_envs 后的异步 reset）。"""
    try:
        import time
        env_pool = get_global_env_pool()
        env_ref = env_pool.get_env(env_id)

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        start_time = time.time()
        while True:
            obs, info = env_pool.get_reset_result(env_id)
            print(env_id)

            if obs is not None:
                return ResetResponse(
                    observation=serialize_observation(obs),
                    info=serialize_value(info)
                )

            elapsed = time.time() - start_time
            if elapsed >= wait:
                print("getting observation results")

            await asyncio.sleep(1)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reset result: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=8)

