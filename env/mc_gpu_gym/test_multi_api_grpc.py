#!/usr/bin/env python3
"""
测试 RayCraft GPU **gRPC** API

测试 BatchCreate 端点，创建多个环境，并测试并发 step 性能
"""
import os
import sys
import json
import time
import asyncio
import multiprocessing
from datetime import datetime


# 把生成的 proto 路径加进来
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "generated"))

import grpc
import minecraft_pb2 as pb
import minecraft_pb2_grpc as pb_grpc
import numpy as np
from PIL import Image

def load_screen(path: str) -> np.ndarray:
    """
    返回 RGB 顺序、uint8、形状 (H, W, 3) 的 numpy 数组
    """
    return np.asarray(Image.open(path))      # 零拷贝，只读




# gRPC 配置
TARGET = "10.102.236.18:8000"

# 测试配置
ENV_COUNT = 1            # 公共的环境数量配置
CONCURRENT_STEPS = 6    # 每个环境并发执行的 step 数量
STEP_INTERVAL = 1.0      # 每个 step 之间的间隔时间（秒），0 表示无间隔


# ---------- 工具 ----------
def now():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def stub_channel():
    """返回复用的 stub（短连接，演示用，不想写连接池）"""
    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),   # 50 MiB
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(TARGET, options=options)
    return pb_grpc.MinecraftEnvStub(channel)


# ==========================
#  1. 健康检查
# ==========================
def test_health():
    print("\n" + "=" * 60)
    print("测试 1: Health Check")
    print("=" * 60)
    stub = stub_channel()
    try:
        resp = stub.Health(pb.Empty())
        print(f"状态: {resp.status} | ray_initialized: {resp.ray_initialized} | num_environments: {resp.num_environments}")
        return True
    except grpc.RpcError as e:
        print(f"❌ gRPC 错误: {e.code()} - {e.details()}")
        return False


# ==========================
#  2. 批量创建环境
# ==========================
def test_batch_create_envs(count=None):
    if count is None:
        count = ENV_COUNT
    print("\n" + "=" * 60)
    print(f"测试 2: Batch Create Envs (创建 {count} 个环境)")
    print("=" * 60)

    stub = stub_channel()
    # 与原 HTTP 请求体保持一致：两个 env_kwargs

    dict_1 = {
        "id": 1,
        "scene": "001",
        "target_type": "Door",
        "target_id_str": "tmeo_ultra:shafazhuanjiao",
        "task_str": "find and open the door",
        "start_pos": [2306,101,984],
        "start_rotation": [-59.8 , 5.8],
        "goal_obj_pos": [2308, 101, 992],
        "goal_obj_id": "tmeo_ultra:woshimenjijian_2baisezuokai", 
        "start_obj_status": "tmeo_ultra:woshimenjijian_2baisezuo",
        "goal_obj_status": "tmeo_ultra:woshimenjijian_2baisezuokai",
        "interact_list": ["Door", "Button"],
        "interact_pos": [[2308, 101, 993], [2310, 102, 991.5]]
    }
    dict_2 = {
        "id": 1,
        "scene": "001",
        "target_type": "Door",
        "target_id_str": "tmeo_ultra:shafazhuanjiao",
        "task_str": "find and open the door",
        "start_pos": [2306,101,984],
        "start_rotation": [-59.8 , 5.8],
        "goal_obj_pos": [2308, 101, 992],
        "goal_obj_id": "tmeo_ultra:woshimenjijian_2baisezuokai", 
        "start_obj_status": "tmeo_ultra:woshimenjijian_2baisezuo",
        "goal_obj_status": "tmeo_ultra:woshimenjijian_2baisezuokai",
        "interact_list": ["Door", "Button"],
        "interact_pos": [[2308, 101, 993], [2310, 102, 991.5]]
    }
    kwargs_json = [json.dumps(dict_1), json.dumps(dict_2)]
    req = pb.BatchCreateReq(count=count, env_name="minecraft", env_kwargs=kwargs_json)

    start = time.time()
    try:
        resp = stub.BatchCreate(req)
        elapsed = time.time() - start
        env_ids = list(resp.env_ids)
        print(f"✅ 成功创建 {len(env_ids)} 个环境，耗时 {elapsed:.2f}s")
        for i, eid in enumerate(env_ids, 1):
            print(f"  环境 {i}: {eid}")
        return env_ids
    except grpc.RpcError as e:
        print(f"❌ 创建失败: {e.code()} - {e.details()}")
        return None


# ==========================
#  3. 查询环境状态
# ==========================
def test_env_status(env_id):
    stub = stub_channel()
    try:
        resp = stub.EnvStatus(pb.EnvIdReq(env_id=env_id))
        print(f"环境 {env_id[:8]}...  status={resp.status}  ready={resp.ready}")
        return resp.status, resp.ready
    except grpc.RpcError as e:
        print(f"❌ 查询状态失败: {e.code()} - {e.details()}")
        return None, None


# ==========================
#  4. 获取异步 reset 结果
# ==========================
def test_getreset_obs(env_id, timeout=600):
    print("\n" + "=" * 60)
    print(f"Reset Environment (初始化可能需要 ~8 分钟)")
    print("=" * 60)
    stub = stub_channel()
    req = pb.GetResetResultReq(env_id=env_id, wait=timeout)
    start = time.time()
    try:
        resp = stub.GetResetResult(req)
        elapsed = time.time() - start
        obs_path = json.loads(resp.observation_json)["pov"]
        obs = load_screen(obs_path)
        info = json.loads(resp.info_json)
        print(f"✅ 环境 {env_id[:8]}... 初始化成功，耗时 {elapsed:.1f}s")
        print(f"   观察键: {obs_path}")
        print(f"   信息键: {list(info.keys())}")
        return True
    except grpc.RpcError as e:
        elapsed = time.time() - start
        print(f"❌ 初始化失败 / 超时，耗时 {elapsed:.1f}s: {e.code()} - {e.details()}")
        return False


# ==========================
#  5. 关闭环境
# ==========================
def test_close_env(env_id):
    print("\n" + "=" * 60)
    print(f"测试 5: Close Environment")
    print("=" * 60)
    stub = stub_channel()
    try:
        resp = stub.CloseEnv(pb.EnvIdReq(env_id=env_id))
        print(f"✅ 环境 {env_id[:8]}... 已关闭")
        return resp.success
    except grpc.RpcError as e:
        print(f"❌ 关闭失败: {e.code()} - {e.details()}")
        return False


# ==========================
#  6. 单环境顺序 step（多进程版）
# ==========================
def mp_step_env(env_id, step_num, action="turn_right"):
    """单进程执行一个 step，供进程池调用"""
    stub = stub_channel()
    print(f"[发送] 环境: {env_id[:8]}... | Step {step_num} | 动作: {action}")

    start = time.time()
    try:
        # 1. 提交
        stub.SubmitStep(pb.StepReq(env_id=env_id, action_json=json.dumps(action)))
        # 2. 轮询
        while True:
            resp = stub.GetStepResult(pb.EnvIdReq(env_id=env_id))
            if resp.status == "success":
                break
            time.sleep(0.01)
        elapsed = time.time() - start
        print(f"[接收] 环境: {env_id[:8]}... | Step {step_num} | 耗时: {elapsed:.3f}s | 奖励: {resp.reward:.2f}")
        return {
            "env_id": env_id,
            "step_num": step_num,
            "status": "success",
            "elapsed": elapsed,
            "reward": resp.reward,
            "terminated": resp.terminated,
        }
    except grpc.RpcError as e:
        elapsed = time.time() - start
        print(f"[错误] 环境: {env_id[:8]}... | Step {step_num} | 耗时: {elapsed:.3f}s | gRPC 错误: {e.details()}")
        return {
            "env_id": env_id,
            "step_num": step_num,
            "status": "error",
            "elapsed": elapsed,
            "error": str(e),
        }


def mp_step_env_sequence(env_id, steps_count, step_interval):
    """单个环境在子进程中顺序执行 step"""
    results = []
    # actions = ["look_right", "look_right", "look_right", "walk_forward", "walk_forward", "walk_forward", "look_left", "look_left", "look_left", "look_left", "walk_forward", "walk_forward", "walk_forward"]
    actions = ["look_right", "look_right", "walk_forward", "walk_forward", "walk_forward", "operate door"]
    for step_num in range(steps_count):
        act = actions[step_num]
        results.append(mp_step_env(env_id, step_num, act))
        if step_num < steps_count - 1 and step_interval > 0:
            time.sleep(step_interval)
    return results


# ==========================
#  7. 并发测试（多进程）
# ==========================
def test_concurrent_steps(env_ids, steps_per_env=None, step_interval=None, processes=None):
    if steps_per_env is None:
        steps_per_env = CONCURRENT_STEPS
    if step_interval is None:
        step_interval = STEP_INTERVAL
    if processes is None:
        processes = len(env_ids)

    print("\n" + "=" * 60)
    print(f"并发性能测试: {len(env_ids)} 个环境 × {steps_per_env} 步")
    print("=" * 60)
    print(f"总请求数: {len(env_ids) * steps_per_env}")
    print(f"Step 间隔: {step_interval} s")
    print(f"并发模式: 多进程（环境间并发，环境内顺序）")
    print(f"进程数: {processes}")

    start = time.time()
    with multiprocessing.Pool(processes=processes) as pool:
        env_results_list = pool.starmap(
            mp_step_env_sequence,
            [(eid, steps_per_env, step_interval) for eid in env_ids],
        )
    total_elapsed = time.time() - start

    # 合并
    all_results = []
    for env_res in env_results_list:
        all_results.extend(env_res)

    success_count = sum(1 for r in all_results if r["status"] == "success")
    error_count = len(all_results) - success_count
    throughput = len(all_results) / total_elapsed

    # 按环境统计
    env_stats = {}
    for r in all_results:
        eid = r["env_id"]
        st = env_stats.setdefault(eid, {"success": 0, "error": 0, "total_time": 0, "min_time": 1e9, "max_time": 0})
        if r["status"] == "success":
            st["success"] += 1
        else:
            st["error"] += 1
        t = r["elapsed"]
        st["total_time"] += t
        st["min_time"] = min(st["min_time"], t)
        st["max_time"] = max(st["max_time"], t)

    # 打印
    print("\n" + "=" * 60)
    print("并发测试结果")
    print("=" * 60)
    print(f"总耗时: {total_elapsed:.2f} s")
    print(f"总请求: {len(all_results)}")
    print(f"成功/失败: {success_count}/{error_count}")
    print(f"平均吞吐量: {throughput:.2f} 请求/s")

    print("\n每个环境统计:")
    for i, (eid, st) in enumerate(env_stats.items(), 1):
        avg = st["total_time"] / (st["success"] + st["error"])
        print(f"环境 {i}: {eid[:8]}... | 成功: {st['success']} | 平均耗时: {avg:.3f}s")

    errors = [r for r in all_results if r["status"] != "success"]
    if errors:
        print("\n错误样例 (前5):")
        for er in errors[:5]:
            print(f"  {er['env_id'][:8]}... step{er['step_num']} | {er.get('error', 'unknown')[:80]}")
    return {
        "total_elapsed": total_elapsed,
        "total_requests": len(all_results),
        "success_count": success_count,
        "error_count": error_count,
        "throughput": throughput,
        "env_stats": env_stats,
        "results": all_results,
    }


def test_fastreset_env(env_id):
    print("\n" + "=" * 60)
    print(f"FastReset Environment")
    print("=" * 60)

    dict_1 = {
        "id": 1,
        "scene": "001",
        "target_type": "Door",
        "target_id_str": "tmeo_ultra:shafazhuanjiao",
        "task_str": "find and open the door",
        "start_pos": [2306,101,984],
        "start_rotation": [-59.8 , 5.8],
        "goal_obj_pos": [2308, 101, 992],
        "goal_reward_pos": [2308, 101, 993],
        "goal_obj_id": "tmeo_ultra:woshimenjijian_2baisezuokai", 
        "start_obj_status": "tmeo_ultra:woshimenjijian_2baisezuo",
        "goal_obj_status": "tmeo_ultra:woshimenjijian_2baisezuokai",
        "interact_list": ["Door", "Button"],
        "interact_id": ["tmeo_ultra:woshimenjijian_2baisezuo", "minecraft:oak_button"]
    }

    stub = stub_channel()
    print(env_id)
    print(json.dumps(dict_1))
    req = pb.FastResetReq(env_id=env_id, config=json.dumps(dict_1))
    stub.Reset(req)



# ==========================
#  主流程
# ==========================
def main():
    print("\n" + "=" * 60)
    print("RayCraft GPU gRPC API 并发性能测试")
    print("=" * 60)
    print(f"gRPC 目标: {TARGET}")
    print(f"测试配置:")
    print(f"  - 环境数量: {ENV_COUNT}")
    print(f"  - 每个环境 step 数: {CONCURRENT_STEPS}")
    print(f"  - 总 step 请求: {ENV_COUNT * CONCURRENT_STEPS}")

    try:
        if not test_health():
            print("\n❌ 健康检查失败，请确保 gRPC 服务端已运行")
            return

        env_ids = test_batch_create_envs()
        if not env_ids:
            print("\n❌ 创建环境失败")
            return

        # 初始化所有环境
        print("\n" + "=" * 60)
        print("初始化所有环境")
        print("=" * 60)
        initialized = []
        for idx, eid in enumerate(env_ids, 1):
            print(f"[{idx}/{len(env_ids)}] 初始化 {eid[:8]}...")
            if test_getreset_obs(eid):
                initialized.append(eid)
        print(f"\n✅ 成功初始化 {len(initialized)}/{len(env_ids)} 个环境")

        # 并发 step 测试
        print("\n" + "=" * 60)
        print("开始并发 Step 性能测试")
        print("=" * 60)
        results = test_concurrent_steps(initialized)

        flag = 0
        for idx, eid in enumerate(env_ids, 1):
            test_fastreset_env(eid)
            if test_getreset_obs(eid):
                flag+=1
        if flag == ENV_COUNT*2:
            results_2 = test_concurrent_steps(initialized)

        # 清理
        print("\n" + "=" * 60)
        print("清理：关闭所有环境")
        print("=" * 60)
        for idx, eid in enumerate(env_ids, 1):
            print(f"[{idx}/{len(env_ids)}] 关闭 {eid[:8]}...")
            test_close_env(eid)

        # 总结
        print("\n" + "=" * 60)
        print("✅ 并发性能测试完成")
        print("=" * 60)
        print(f"总耗时: {results['total_elapsed']:.2f} s")
        print(f"成功率: {results['success_count']}/{results['total_requests']} ({100 * results['success_count'] / results['total_requests']:.1f}%)")
        print(f"平均吞吐量: {results['throughput']:.2f} 请求/s")

    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()