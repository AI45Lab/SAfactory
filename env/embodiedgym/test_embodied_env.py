"""
测试 EmbodiedAlfredGym 环境是否正常工作
"""

import os
import sys

# 添加项目根目录到路径
current_file_path = os.path.abspath(__file__)
embodiedgym_dir = os.path.dirname(current_file_path)
env_dir = os.path.dirname(embodiedgym_dir)
aievobox_dir = os.path.dirname(env_dir)
sys.path.insert(0, aievobox_dir)

from embodied_env import EmbodiedAlfredGym

def test_initialization():
    """测试环境初始化"""
    print("="*80)
    print("测试 1: 环境初始化")
    print("="*80)
    
    try:
        env = EmbodiedAlfredGym(
            eval_set='base',
            down_sample_ratio=0.01,  # 只使用 1% 的数据进行快速测试
            resolution=300,  # 降低分辨率加快速度
            max_episode_steps=5,
            exp_name='test_init'
        )
        print("✓ 环境初始化成功")
        print(f"  - Episode 总数: {env.alfred_env.number_of_episodes}")
        print(f"  - 动作空间大小: {len(env.language_skill_set)}")
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
    print("测试 2: 环境 Reset")
    print("="*80)
    
    try:
        env = EmbodiedAlfredGym(
            eval_set='base',
            down_sample_ratio=0.01,
            resolution=300,
            max_episode_steps=5,
            exp_name='test_reset'
        )
        
        reset_output = env.reset()
        print("✓ Reset 成功")
        print(f"  - 观测键: {list(reset_output.observation.keys())}")
        print(f"  - 任务指令: {reset_output.info['instruction'][:100]}...")
        head_rgb = reset_output.observation['head_rgb']
        if hasattr(head_rgb, 'shape'):
            print(f"  - 图像形状: {head_rgb.shape}")
        else:
            print(f"  - 图像类型: {type(head_rgb)}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Reset 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_render():
    """测试环境 render"""
    print("\n" + "="*80)
    print("测试 3: 环境 Render")
    print("="*80)
    
    try:
        env = EmbodiedAlfredGym(
            eval_set='base',
            down_sample_ratio=0.01,
            resolution=300,
            max_episode_steps=5,
            exp_name='test_render'
        )
        
        env.reset()
        render_output = env.render()
        
        print("✓ Render 成功")
        print(f"  - 步骤: {render_output.step}")
        print(f"  - 图像数据大小: {len(render_output.image_data)} bytes")
        print(f"  - Base64 数据大小: {len(render_output.image_base64)} chars")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Render 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_prompt():
    """测试 get_task_prompt"""
    print("\n" + "="*80)
    print("测试 4: Get Task Prompt")
    print("="*80)
    
    try:
        env = EmbodiedAlfredGym(
            eval_set='base',
            down_sample_ratio=0.01,
            resolution=300,
            max_episode_steps=5,
            exp_name='test_prompt'
        )
        
        env.reset()
        prompt_output = env.get_task_prompt()
        
        print("✓ Get Task Prompt 成功")
        print(f"  - System message 长度: {len(prompt_output.system_message.content[0].root.text)}")
        print(f"  - User message 内容数: {len(prompt_output.user_message.content)}")
        print(f"  - 包含图像: {any(hasattr(c.root, 'image_url') for c in prompt_output.user_message.content)}")
        
        # 打印部分文本
        user_text = next((c.root.text for c in prompt_output.user_message.content if hasattr(c.root, 'text')), None)
        if user_text:
            print(f"  - User 文本预览: {user_text[:200]}...")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Get Task Prompt 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_parse_response():
    """测试 LLM 响应解析"""
    print("\n" + "="*80)
    print("测试 5: Parse LLM Response")
    print("="*80)
    
    try:
        env = EmbodiedAlfredGym(
            eval_set='base',
            down_sample_ratio=0.01,
            resolution=300,
            max_episode_steps=5,
            exp_name='test_parse'
        )
        
        # 测试有效的 JSON 响应
        valid_json = '''
        {
            "reasoning": "I need to find an apple first",
            "executable_plan": [
                {"action_id": 0, "description": "find a apple"},
                {"action_id": 50, "description": "pick up the apple"}
            ]
        }
        '''
        
        action_id, reasoning, is_valid = env.parse_llm_response(valid_json)
        print(f"✓ 有效 JSON 解析: action_id={action_id}, is_valid={is_valid}")
        
        # 测试无效的响应
        invalid_response = "This is not a valid JSON"
        action_id, reasoning, is_valid = env.parse_llm_response(invalid_response)
        print(f"✓ 无效响应处理: action_id={action_id}, is_valid={is_valid}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Parse 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🚀 开始测试 EmbodiedAlfredGym 环境\n")
    
    results = []
    results.append(("环境初始化", test_initialization()))
    results.append(("环境 Reset", test_reset()))
    results.append(("环境 Render", test_render()))
    results.append(("Task Prompt", test_prompt()))
    results.append(("响应解析", test_parse_response()))
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status} - {name}")
    
    total_passed = sum(1 for _, p in results if p)
    print(f"\n总计: {total_passed}/{len(results)} 测试通过")
    
    if total_passed == len(results):
        print("\n🎉 所有测试通过！环境已准备就绪。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")

