import sys
import os
import time
import base64
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from os_env import OSGym
except ImportError as e:
    print(f"ImportError: {e}")
    print("请确保环境依赖已安装。")
    sys.exit(1)

def test_osgym():
    print("="*50)
    print("开始测试 OSGym 环境...")
    print("="*50)

    # 1. 初始化环境
    print("\n[1/4] 初始化环境 (OSGym)...")
    try:
        # 计算配置文件的绝对路径
        # 使用 test_simple.json 以避免 Google Drive 认证问题
        config_path = os.path.join(CURRENT_DIR, "evaluation_risk_examples", "test_simple.json")
        
        env = OSGym(
            provider_name="docker",
            headless=True,
            action_space="pyautogui",
            task_config_path=config_path
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
