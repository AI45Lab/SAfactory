"""
RayCraft GPU HTTP Server

?? GPU MineStudio ? HTTP API ??

???
- fastapi
- uvicorn
- ray
- pillow
"""

import uuid
import base64
import io
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import ray
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .myray.global_pool import get_global_env_pool
import asyncio
import time
from copy import deepcopy



# ============================================================================
# 全局配置
# ============================================================================

# 默认配置文件路径（与 deepeyes 版本保持一致）
DEFAULT_CONFIG_PATH = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/configs/kill/base.yaml"

# 默认数据文件路径
DEFAULT_DATA_PATH = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/datasets/data.json"


# ============================================================================
# 数据模型
# ============================================================================

class BatchCreateRequest(BaseModel):
    count: int = Field(..., ge=1, description="???????>=1")
    env_name: str = Field(default="minecraft", description="????")
    env_kwargs: List[Dict[str, Any]] = Field(default_factory=list, description="????")


class BatchCreateResponse(BaseModel):
    env_ids: list


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    action: str = Field(..., description="JSON???action???")


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
    status: str           # "done" / "pending" / "error"
    observation: Optional[Dict[str, Any]] = None 
    reward: Optional[float] = None
    terminated: Optional[bool] = None
    truncated: Optional[bool] = None
    info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============================================================================
# ????
# ============================================================================

def serialize_value(value: Any) -> Any:
    """?????????numpy???Ray??"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif 'ray' in str(type(value).__module__):
        # Ray????ActorID, ObjectRef???????
        return str(value)
    else:
        return value


def serialize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """???observation???RGB??"""
    result = {}
    for key, value in obs.items():
        if key == 'rgb' and isinstance(value, np.ndarray):
            # ???JPEG
            pil_image = Image.fromarray(value)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            jpeg_bytes = buffer.getvalue()
            result[key] = {
                'type': 'jpeg',
                'data': base64.b64encode(jpeg_bytes).decode('utf-8')
            }
        elif isinstance(value, np.ndarray):
            # ??numpy???list
            result[key] = value.tolist()
        else:
            result[key] = serialize_value(value)
    return result


# ============================================================================
# FastAPI ??
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """??????Ray??GPU???"""
    if not ray.is_initialized():
        print("[INFO] ray init 8")
        ray.init(num_gpus=8, ignore_reinit_error=True)
    else:
        print("[INFO] no need init")
    get_global_env_pool()  # ????actor??
    yield


app = FastAPI(
    title="RayCraft GPU HTTP API",
    version="1.0.0",
    description="GPU-accelerated Minecraft environment API",
    lifespan=lifespan
)


# ============================================================================
# API ??
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """????"""
    try:
        env_pool = get_global_env_pool()
        num_envs = ray.get(env_pool.get_num_envs.remote())
        return HealthResponse(
            status="healthy",
            ray_initialized=ray.is_initialized(),
            num_environments=num_envs
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )



@app.post("/batch/envs", response_model=BatchCreateResponse)
async def batch_create_envs(request: BatchCreateRequest):
    """批量创建环境（并行利用GPU硬件加速渲染）"""
    try:
        env_pool = get_global_env_pool()

        # 生成UUIDs
        env_kwargs = request.env_kwargs or []
        env_ids = [str(uuid.uuid4()) for _ in range(request.count*len(env_kwargs))]
        
        
        configs = []
        for i in range(len(env_kwargs)):
            for _ in range(request.count):
                configs.append(env_kwargs[i])

        # 创建环境（注册 Actor，不初始化）
        # 真正的初始化会延迟到第一次 reset 调用

        success = ray.get(env_pool.create_envs.remote(env_ids, configs))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create some environments"
            )
        
        for env_id in env_ids:
            env_ref = ray.get(env_pool.get_env.remote(env_id))
            import time
            start_moment = time.perf_counter()
            print(f'[INFO] start to init, {env_id=}, {start_moment=}')
            env_ref.reset.remote()

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
    """Get environment status (for checking if env is ready)"""
    try:
        env_pool = get_global_env_pool()
        
        # Check if environment exists in registry
        env_ref = ray.get(env_pool.get_env.remote(env_id))
        
        if env_ref is None:
            return EnvStatusResponse(
                env_id=env_id,
                status="not_found",
                ready=False
            )
        
        # Get status from pool
        status_dict = ray.get(env_pool.get_env_status.remote(env_id))
        env_status = status_dict.get("status", "unknown")
        
        # Check if environment is ready (has been reset at least once)
        is_ready = ray.get(env_pool.is_env_ready.remote(env_id))
        
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
    """Reset environment (first reset will initialize, may take ~8 minutes)"""
    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        result = ray.get(env_ref.reset.remote())

        # MCEnvActor.reset() ??? obs??? (obs, info)
        # ????
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

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
    """Step??"""
    from datetime import datetime
    print(f"{env_id=}, {datetime.now()}")
    # try:
    env_pool = get_global_env_pool()
    env_ref = await env_pool.get_env.remote(env_id)

    if env_ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment {env_id} not found"
        )
    

    start_t = time.time()
    result = await env_pool.step.remote(env_id, request.action)
    elpased_t = time.time() - start_t
    print(f"[INFO], server 1 , {elpased_t=}")

    # MCEnvActor.step() ????Gym?? (obs, reward, done, info)
    # ????????? (obs, reward, terminated, truncated, info)
    if len(result) == 4:
        obs, reward, done, info = result
        terminated = done
        truncated = False  # ????truncated???False
    else:
        obs, reward, terminated, truncated, info = result
    elpased_t = time.time() - start_t
    print(f"[INFO], server 2 , {elpased_t=}")

    return StepResponse(
        observation=serialize_observation(obs),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=serialize_value(info)
    )

    # except HTTPException:
    #     raise
    # except Exception as e:
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail=f"Failed to step: {str(e)}"
    #     )


# ===== 1. 提交 step =====
@app.post("/envs/{env_id}/step_async", response_model=StepResultResponse)
async def submit_step(env_id: str, request: StepRequest):
    """Step??"""
    from datetime import datetime
    print(f"{env_id=}, {datetime.now()}")
    # try:
    env_pool = get_global_env_pool()
    env_ref = env_pool.get_env.remote(env_id)

    if env_ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Environment {env_id} not found"
        )
    

    start_t = time.time()
    await env_pool.step.remote(env_id, request.action)
    elpased_t = time.time() - start_t
    print(f"[INFO], server 1 , {elpased_t=}")

    return StepResultResponse(status="submitted")

# ===== 2. 轮询结果 =====
@app.get("/envs/{env_id}/get_step_async", response_model=StepResultResponse)
async def get_step_result(env_id: str):
    env_pool = get_global_env_pool()
    step_res = await env_pool.get_step_res.remote()
    if env_id in step_res.keys():
        if len(step_res[env_id])==4:
            obs, reward, done, info = deepcopy(step_res[env_id])
            terminated = done
            truncated = False  # ????truncated???False
        else:
            obs, reward, terminated, truncated, info = deepcopy(step_res[env_id])
        env_pool.rm_step_res.remote(env_id)
        return StepResultResponse(
                status="success", 
                observation=serialize_observation(obs),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=serialize_value(info)
            )
    else:
        return StepResultResponse(status="pending")



@app.post("/envs/{env_id}/close")
async def close_env(env_id: str):
    """????"""
    try:
        env_pool = get_global_env_pool()
        success = ray.get(env_pool.close_env.remote(env_id))

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
    """获取后台 reset 的结果（用于 batch_create_envs 后的异步 reset）

    Args:
        env_id: 环境ID
        wait: 等待时间（秒），如果reset未完成，最多等待这么多秒
    """
    try:
        import time
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        # 轮询等待 reset 完成
        start_time = time.time()
        while True:
            obs, info = ray.get(env_ref.get_reset_result.remote())

            if obs is not None:
                # reset 完成
                return ResetResponse(
                    observation=serialize_observation(obs),
                    info=serialize_value(info)
                )

            # 检查是否超时
            elapsed = time.time() - start_time

            if elapsed >= wait:
                print("getting observation results")
            #     # 返回202表示reset正在进行中
            #     raise HTTPException(
            #         status_code=status.HTTP_202_ACCEPTED,
            #         detail=f"Environment {env_id} reset is still in progress"
            #     )

            # 等待一小段时间后重试
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
