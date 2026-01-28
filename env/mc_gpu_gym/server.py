#!/usr/bin/env python3
"""
gRPC 服务端，语义与原来的 FastAPI 完全一致。
"""
import os
import sys
import json
import uuid
import base64
import io
import time
import asyncio
from datetime import datetime
from typing import Any, Dict

import numpy as np
from PIL import Image
import grpc
from grpc import aio

# 把生成的代码放 generated/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
import minecraft_pb2 as pb
import minecraft_pb2_grpc as pb_grpc

# 复用原来的工具函数与后端
from mymp.global_pool import get_global_env_pool

# ---------- 工具 ----------
def serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    return value

def serialize_observation(obs: Dict[str, Any]) -> str:
    """obs -> json（RGB 压成 JPEG base64）"""
    result = {}
    for k, v in obs.items():
        if k == "rgb" and isinstance(v, np.ndarray):
            pil = Image.fromarray(v)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            result[k] = {
                "type": "jpeg",
                "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
            }
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = serialize_value(v)
    return json.dumps(result)

# ---------- 服务端实现 ----------
class MinecraftServicer(pb_grpc.MinecraftEnvServicer):
    def __init__(self, env_list_id):
        self.pool = get_global_env_pool()
        self.env_list_id = env_list_id

    # 1. 健康检查
    async def Health(self, request: pb.Empty, context) -> pb.HealthResp:
        num = self.pool.get_num_envs()
        return pb.HealthResp(
            status="healthy", ray_initialized=False, num_environments=num
        )

    # 2. 批量创建
    async def BatchCreate(
        self, request: pb.BatchCreateReq, context
    ) -> pb.BatchCreateResp:
        count = request.count
        env_name = request.env_name or "minecraft"
        # 反序列化 env_kwargs
        kwargs_list = [json.loads(k) for k in request.env_kwargs] or [{}]
        env_ids = [str(uuid.uuid4()) for _ in range(count * len(kwargs_list))]
        configs = []
        for kw in kwargs_list:
            for _ in range(count):
                configs.append(kw)
        ok = self.pool.create_envs(self.env_list_id, env_ids, configs)
        if not ok:
            await context.abort(grpc.StatusCode.INTERNAL, "create envs failed")
        # 触发异步 reset
        for eid in env_ids:
            print(f"[INFO] trigger reset {eid}")
            self.pool.trigger_reset(eid)
        return pb.BatchCreateResp(env_ids=env_ids)

    # 3. 环境状态
    async def EnvStatus(self, request: pb.EnvIdReq, context) -> pb.EnvStatusResp:
        eid = request.env_id
        ref = self.pool.get_env(eid)
        if ref is None:
            return pb.EnvStatusResp(env_id=eid, status="not_found", ready=False)
        st = self.pool.get_env_status(eid).get("status", "unknown")
        ready = self.pool.is_env_ready(eid)
        return pb.EnvStatusResp(env_id=eid, status=st, ready=ready)

    # 4. 同步 reset
    async def Reset(self, request: pb.EnvIdReq, context) -> pb.ResetResp:
        eid = request.env_id
        config = json.loads(request.config)
        if self.pool.get_env(eid) is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "env not found")
        self.pool.reset_env(eid, config, timeout=600)
        return pb.StepResultResp(status="submitted")

    # 5. 同步 step
    async def Step(self, request: pb.StepReq, context) -> pb.StepResp:
        eid = request.env_id
        if self.pool.get_env(eid) is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "env not found")
        result = self.pool.step_sync(eid, request.action_json)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        return pb.StepResp(
            observation_json=serialize_observation(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info_json=json.dumps(serialize_value(info)),
        )

    # 6. 异步 step：提交
    async def SubmitStep(self, request: pb.StepReq, context) -> pb.StepResultResp:
        eid = request.env_id
        if self.pool.get_env(eid) is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "env not found")
        t0 = time.time()
        self.pool.step(eid, request.action_json)
        print(f"[submit_step] {eid} dispatch {time.time()-t0:.3f}s")
        return pb.StepResultResp(status="submitted")

    # 7. 异步 step：轮询结果
    async def GetStepResult(
        self, request: pb.EnvIdReq, context
    ) -> pb.StepResultResp:
        eid = request.env_id
        t0 = time.time()
        step_res = self.pool.get_step_res()
        t1 = time.time()
        if eid in step_res:
            if len(step_res[eid]) == 4:
                obs, reward, done, info = step_res[eid]
                terminated, truncated = done, False
            else:
                obs, reward, terminated, truncated, info = step_res[eid]
            self.pool.rm_step_res(eid)
            print(f"[get_step_result] {eid} total {time.time()-t0:.3f}s")
            return pb.StepResultResp(
                status="success",
                observation_json=serialize_observation(obs),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info_json=json.dumps(serialize_value(info)),
            )
        return pb.StepResultResp(status="pending")

    # 8. 关闭环境
    async def CloseEnv(self, request: pb.EnvIdReq, context) -> pb.CloseResp:
        eid = request.env_id
        ok = self.pool.close_env(eid)
        if not ok:
            await context.abort(grpc.StatusCode.NOT_FOUND, "env not found")
        return pb.CloseResp(success=True)

    async def FastCloseEnv(self, request: pb.EnvIdReq, context) -> pb.CloseResp:
        eid = request.env_id
        ok = self.pool.fast_close_env(eid)
        if not ok:
            await context.abort(grpc.StatusCode.NOT_FOUND, "env not found")
        return pb.CloseResp(success=True)

    # 9. 获取异步 reset 结果
    async def GetResetResult(
        self, request: pb.GetResetResultReq, context
    ) -> pb.ResetResp:
        eid = request.env_id
        if self.pool.get_env(eid) is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "env not found")
        start = time.time()
        while True:
            obs, info = self.pool.get_reset_result(eid)
            if obs is not None:
                print(f"[get_reset_result] {eid} total {time.time()-start:.3f}s")
                return pb.ResetResp(
                    observation_json=serialize_observation(obs),
                    info_json=json.dumps(serialize_value(info)),
                )
            if time.time() - start >= request.wait:
                await asyncio.sleep(1)

# ---------- 启动 ----------
async def serve(env_list_id):
    server = aio.server()
    pb_grpc.add_MinecraftEnvServicer_to_server(MinecraftServicer(env_list_id), server)
    listen_addr = "0.0.0.0:8000"
    server.add_insecure_port(listen_addr)
    print("gRPC Minecraft server start on", listen_addr)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    env_list_id = 2
    asyncio.run(serve(env_list_id))