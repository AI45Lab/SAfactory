"""
测试 MCGym (Minecraft Environment) 环境是否正常工作
"""

import os
import sys

# 添加项目根目录到路径
current_file_path = os.path.abspath(__file__)
test_dir = os.path.dirname(current_file_path)
mc_dir = os.path.dirname(test_dir)
env_dir = os.path.dirname(mc_dir)
aievobox_dir = os.path.dirname(env_dir)
sys.path.insert(0, aievobox_dir)

from env.mc.mc_env import MCGym
from core.env.env_register import list_registered_envs

def test_environment_registration():
    """测试环境注册"""
    print("="*80)
    print("测试 1: 环境注册")
    print("="*80)
    
    try:
        registered_envs = list_registered_envs()
        print(f"已注册的环境: {list(registered_envs.keys())}")
        
        if "mc_gym" in registered_envs:
            print("✓ mc_gym 环境已成功注册")
            return True
        else:
            print("✗ mc_gym 环境未注册")
            return False
    except Exception as e:
        print(f"✗ 环境注册检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_initialization():
    """测试环境初始化"""
    print("\n" + "="*80)
    print("测试 2: 环境初始化")
    print("="*80)
    
    try:
        # 使用 kill_zombie 配置进行测试
        config_path = "env/mc/config/kill/kill_zombie.yaml"
        env = MCGym(
            env_config=config_path,
            env_id="test_mc_1",
            env_name="mc_gym"
        )
        print("✓ 环境初始化成功")
        print(f"  - 环境 ID: {env.env_id}")
        print(f"  - 环境名称: {env.env_name}")
        print(f"  - 任务指令: {env.instructions[:100] if env.instructions else 'None'}...")
        print(f"  - 当前步数: {env.current_step}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ 环境初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_reset():
    """测试环境 reset"""
    print("\n" + "="*80)
    print("测试 3: 环境 Reset")
    print("="*80)
    
    try:
        config_path = "env/mc/config/kill/kill_zombie.yaml"
        env = MCGym(
            env_config=config_path,
            env_id="test_mc_2",
            env_name="mc_gym"
        )
        
        reset_output = env.reset()
        print("✓ Reset 成功")
        print(f"  - 观测键: {list(reset_output.observation.keys())}")
        print(f"  - Info 键: {list(reset_output.info.keys())}")
        
        # 检查观测中的图像
        if 'image' in reset_output.observation:
            image = reset_output.observation['image']
            if isinstance(image, list):
                import numpy as np
                image = np.array(image)
            print(f"  - 图像类型: {type(image)}")
            if hasattr(image, 'shape'):
                print(f"  - 图像形状: {image.shape}")
                print(f"  - 图像数据类型: {image.dtype}")
        
        # 检查 info 中的关键信息
        if 'pov' in reset_output.info:
            pov = reset_output.info['pov']
            if hasattr(pov, 'shape'):
                print(f"  - POV 形状: {pov.shape}")
        
        print(f"  - 当前步数重置为: {env.current_step}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Reset 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_step():
    """测试环境 step"""
    print("\n" + "="*80)
    print("测试 4: 环境 Step")
    print("="*80)
    
    try:
        config_path = "env/mc/config/kill/kill_zombie.yaml"
        env = MCGym(
            env_config=config_path,
            env_id="test_mc_3",
            env_name="mc_gym"
        )
        
        env.reset()
        
        # 执行一个简单的前进动作
        action = '[{"action": "forward", "yaw": 0, "pitch": 0}]'
        step_output = env.step(action)
        
        print("✓ Step 成功")
        print(f"  - 观测键: {list(step_output.observation.keys())}")
        print(f"  - 奖励: {step_output.reward}")
        print(f"  - 终止: {step_output.terminated}")
        print(f"  - 截断: {step_output.truncated}")
        print(f"  - 当前步数: {env.current_step}")
        
        # 再执行一个动作
        action2 = '[{"action": "attack", "yaw": 0, "pitch": 0}]'
        step_output2 = env.step(action2)
        print(f"  - 第二步奖励: {step_output2.reward}")
        print(f"  - 当前步数: {env.current_step}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Step 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_render():
    """测试环境 render"""
    print("\n" + "="*80)
    print("测试 5: 环境 Render")
    print("="*80)
    
    try:
        config_path = "env/mc/config/kill/kill_zombie.yaml"
        env = MCGym(
            env_config=config_path,
            env_id="test_mc_4",
            env_name="mc_gym"
        )
        
        env.reset()
        render_output = env.render()
        
        print("✓ Render 成功")
        print(f"  - 步骤: {render_output.step}")
        print(f"  - 图像数据大小: {len(render_output.image_data)} bytes")
        if render_output.image_base64:
            print(f"  - Base64 数据大小: {len(render_output.image_base64)} chars")
        
        # 执行一步后再渲染
        action = '[{"action": "forward", "yaw": 0, "pitch": 0}]'
        env.step(action)
        render_output2 = env.render()
        print(f"  - 第二次渲染步骤: {render_output2.step}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Render 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_build_prompt():
    """测试 build_prompt"""
    print("\n" + "="*80)
    print("测试 6: Build Prompt")
    print("="*80)
    
    try:
        config_path = "env/mc/config/kill/kill_zombie.yaml"
        env = MCGym(
            env_config=config_path,
            env_id="test_mc_5",
            env_name="mc_gym"
        )
        
        env.reset()
        prompt_output = env.build_prompt()
        
        print("✓ Build Prompt 成功")
        print(f"  - System message role: {prompt_output.system_message.role}")
        print(f"  - System message 内容数: {len(prompt_output.system_message.content)}")
        print(f"  - User message role: {prompt_output.user_message.role}")
        print(f"  - User message 内容数: {len(prompt_output.user_message.content)}")
        
        # 打印部分 system prompt
        if prompt_output.system_message.content:
            sys_text = prompt_output.system_message.content[0].root.text
            print(f"  - System prompt 预览: {sys_text[:200]}...")
        
        # 检查是否包含任务指令
        if env.instructions:
            print(f"  - 任务指令已加载: {env.instructions[:100]}...")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Build Prompt 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_action_conversion():
    """测试动作转换"""
    print("\n" + "="*80)
    print("测试 7: 动作格式转换")
    print("="*80)
    
    try:
        config_path = "env/mc/config/kill/kill_zombie.yaml"
        env = MCGym(
            env_config=config_path,
            env_id="test_mc_6",
            env_name="mc_gym"
        )
        
        env.reset()
        
        # 测试 LLM 格式的动作
        llm_action = '[{"action": "forward", "yaw": 10.0, "pitch": -5.0}]'
        converted_action = env.action_string_to_dict(llm_action)
        print("✓ LLM 格式动作转换成功")
        print(f"  - 输入: {llm_action}")
        print(f"  - 输出键: {list(converted_action.keys())}")
        
        # 测试多个动作
        multi_action = '[{"action": "attack", "yaw": 0, "pitch": 0}]'
        converted_multi = env.action_string_to_dict(multi_action)
        print("✓ 攻击动作转换成功")
        print(f"  - 动作类型: attack")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ 动作转换失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_different_configs():
    """测试不同的配置文件"""
    print("\n" + "="*80)
    print("测试 8: 不同任务配置")
    print("="*80)
    
    configs = [
        ("Kill Zombie", "env/mc/config/kill/kill_zombie.yaml"),
        ("Collect Wood", "env/mc/config/collect/collect_wood.yaml"),
        ("Craft Table", "env/mc/config/craft/craft_crafting_table.yaml"),
    ]
    
    all_passed = True
    for name, config_path in configs:
        try:
            env = MCGym(
                env_config=config_path,
                env_id=f"test_{name.lower().replace(' ', '_')}",
                env_name="mc_gym"
            )
            env.reset()
            print(f"✓ {name} 配置加载成功")
            print(f"  - 指令: {env.instructions[:80] if env.instructions else 'None'}...")
            env.close()
        except Exception as e:
            print(f"✗ {name} 配置加载失败: {str(e)}")
            all_passed = False
    
    return all_passed

def test_fov_initialization():
    """测试 FOV 初始化"""
    print("\n" + "="*80)
    print("测试 9: FOV 参数初始化")
    print("="*80)
    
    try:
        config_path = "env/mc/config/kill/kill_zombie.yaml"
        env = MCGym(
            env_config=config_path,
            env_id="test_mc_fov",
            env_name="mc_gym"
        )
        
        print("✓ FOV 参数初始化成功")
        print(f"  - HFOV: {env.current_hfov}°")
        print(f"  - VFOV: {env.current_vfov}°")
        print(f"  - HFOV half: {env.hfov_half}°")
        print(f"  - VFOV half: {env.vfov_half}°")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ FOV 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🚀 开始测试 MCGym (Minecraft) 环境\n")
    
    results = []
    results.append(("环境注册", test_environment_registration()))
    results.append(("环境初始化", test_initialization()))
    results.append(("环境 Reset", test_reset()))
    results.append(("环境 Step", test_step()))
    results.append(("环境 Render", test_render()))
    results.append(("构建 Prompt", test_build_prompt()))
    results.append(("动作转换", test_action_conversion()))
    results.append(("多配置测试", test_different_configs()))
    results.append(("FOV 初始化", test_fov_initialization()))
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status} - {name}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\n总计: {total_passed}/{len(results)} 测试通过")
    
    if total_passed == len(results):
        print("\n🎉 所有测试通过！Minecraft 环境已准备就绪。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")

