import sys
import os
import base64
import json
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent  # AIEvoBox 根目录
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 确保包形式导入（env.osgym.os_env 使用了相对导入，需要包上下文）
try:
    from env.osgym.os_env import OSGym
except ImportError as e:
    print(f"ImportError: {e}")
    print("请在 AIEvoBox 根目录运行：python -m env.osgym.test_osgym")
    sys.exit(1)

def test_osgym():
    print("="*50)
    print("开始测试 OSGym 环境...")
    print("="*50)

    # 1. 初始化环境
    print("\n[1/4] 初始化环境 (OSGym)...")
    try:
        # 检查 VM 镜像是否存在（仓库未包含大文件 Ubuntu.qcow2，需要自行下载放置）
        # 优先尝试自动下载（与 os_env 中逻辑一致）
        try:
            from desktop_env.providers.docker.manager import DockerVMManager
            vm_manager = DockerVMManager()
            vm_path = vm_manager.get_vm_path(os_type="Ubuntu", region=None)
        except Exception as exc:
            vm_path = os.path.join(CURRENT_DIR, "docker_vm_data", "Ubuntu.qcow2")
            if not os.path.exists(vm_path):
                print(f"缺少 VM 镜像文件: {vm_path}")
                print(f"自动下载失败，原因: {exc}")
                print("可从 HuggingFace 手动下载 Ubuntu.qcow2.zip 并解压到 docker_vm_data/ 后重试：")
                print("https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip")
                return

        # 加载任务配置文件 (从 datasets/osworld_cases.jsonl 读取第一个任务)
        task_config_path = os.path.join(
            CURRENT_DIR, "datasets", "osworld_cases.jsonl"
        )
        with open(task_config_path, "r", encoding="utf-8") as f:
            # JSONL 格式：每行一个 JSON 对象，读取第一行作为测试任务
            first_line = f.readline().strip()
            task_config = json.loads(first_line)

        print(f"加载任务配置: {task_config.get('id')} (from osworld_cases.jsonl)")

        env = OSGym(
            provider_name="docker",
            headless=True,
            action_space="pyautogui",
            dataset=task_config  # 通过 dataset 参数传递任务配置
        )
        print("环境初始化成功！")
    except Exception as e:
        print(f"环境初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 重置环境
    print("\n[2/4] 重置环境 (reset)...")
    try:
        ret = env.reset()
        # print(f"env.reset() 返回类型: {type(ret)}")
        # print(f"env.reset() 返回值: {ret}")
        
        if isinstance(ret, tuple):
            obs, info = ret
        elif hasattr(ret, 'observation') and hasattr(ret, 'info'):
             # 处理 ResetOutput 对象
             obs = ret.observation
             info = ret.info
        else:
             print("无法识别的返回类型")
             return

        print("环境重置成功！")
        print(f"info 类型: {type(info)}")
        print(f"info 值: {info}")
        
        if isinstance(info, dict):
            print(f"当前任务 ID: {info.get('task_id')}")
            print(f"当前任务 Domain: {info.get('domain')}")
        else:
            print("Info 不是字典，跳过任务ID打印")

        print(f"任务指令: {env.current_instruction}")
        print(f"观测空间包含的键: {list(obs.keys())}")
        
        if 'accessibility_tree' in obs:
            print(f"A11y Tree 长度: {len(obs['accessibility_tree'])}")
        
        # 保存截图
        if 'screenshot' in obs:
            print(f"截图数据类型: {type(obs['screenshot'])}")
            try:
                screenshot_data = obs['screenshot']
                if isinstance(screenshot_data, str):
                    img_data = base64.b64decode(screenshot_data)
                elif isinstance(screenshot_data, bytes):
                    img_data = screenshot_data
                else:
                    img_data = None
                    print("未知截图格式，跳过保存")

                if img_data:
                    save_path = os.path.join(CURRENT_DIR, "test_osgym_reset.png")
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                    print(f"Reset 截图已保存为: {save_path}")
            except Exception as e:
                print(f"保存截图失败: {e}")
            
    except Exception as e:
        print(f"环境重置失败: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return

    # 3. 执行动作
    print("\n[3/4] 执行动作 (step)...")
    # 构造一个简单的 Python 动作
    action_code = """```python
    import pyautogui
    import time
    print('Hello from OSGym Test Agent!')
    pyautogui.moveTo(100, 100)
    ```"""
    print(f"发送动作:\n{action_code}")
    
    try:
        step_ret = env.step(action_code)
        
        # 正确处理 StepOutput 对象
        if hasattr(step_ret, 'observation'):
            obs = step_ret.observation
            reward = step_ret.reward
            done = step_ret.terminated
            truncated = step_ret.truncated
            info = step_ret.info
        else:
            # 兼容旧式 tuple 返回
            obs, reward, done, truncated, info = step_ret

        print("动作执行成功！")
        print(f"奖励 (Reward): {reward}")
        print(f"结束状态 (Done): {done}")
        print(f"截断状态 (Truncated): {truncated}")
        print(f"信息 (Info): {info}")
        
        # 再次检查观测
        if 'terminal' in obs and obs['terminal']:
            print(f"终端输出片段: {obs['terminal'][:200]}...")
            
    except Exception as e:
        print(f"动作执行失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. 关闭环境
    print("\n[4/4] 关闭环境 (close)...")
    try:
        env.close()
        print("环境关闭成功！")
    except Exception as e:
        print(f"环境关闭失败: {e}")

    print("\n" + "="*50)
    print("测试结束")
    print("="*50)

if __name__ == "__main__":
    test_osgym()
