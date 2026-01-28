import json
import os
from pathlib import Path
import cv2
import numpy as np
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image

# 假设 MCSimulator 已经定义好了，直接导入
from mc_simulator import MCSimulator

def read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"File content (first 500 characters): {open(file_path, 'r').read(500)}")
        raise

# 初始化 LLM 模型
def init_llm(model_path):
    llm = LLM(model_path)
    return llm

# 调用 memory agent
def call_memory_agent(previous_obs, current_obs, response_text, current_memory, processor, tokenizer):
    """
    调用 memory agent 生成新的记忆和思考内容
    
    Args:
        previous_obs: 之前观测的图片的列表
        current_obs: 当前观测（图片）
        current_memory: 当前记忆内容（字符串）
        processor: 图像处理器
        tokenizer: 分词器
    
    Returns:
        new_memory: 更新后的记忆内容
        new_think: 新的思考内容
    """
    if previous_obs == current_obs and 'operate' in response_text.lower():
        new_memory = (
            "The current observation appears nearly identical to the previous one (no discernible change in layout, objects, or perspective), you are likely stuck. "
            "Consider the following possibilities and take appropriate actions:\n"
            "1. You might be too far away from the object, which is beyond the interaction range. Move closer to the object.\n"
            "2. Your current perspective might be too skewed, and the target object is not visible in the current view, preventing selection. Adjust your rotation to get a better view.\n"
            "3. The object might be non-interactive, meaning it does not have the necessary properties for interaction. Check if there are other objects that can be used to achieve your goal.\n"
            "Based on the context, determine the most likely scenario and execute the corresponding action."
        )
        new_think = ""
    else:
        new_memory = ""
        new_think = ""
    
    return new_memory, new_think

# 生成 prompt 并调用 LLM 获取动作
def get_action_from_llm(tokenizer, llm, task, observation, current_memory, previous_obs, processor):
    prompt = f"""You are an embodied agent capable of performing the following discrete actions. Your objective is to select the most appropriate action to complete the specific **Task** provided below, based strictly on the visual information from your current **Observation**.
                Note that each rotation or directional change corresponds to a **30-degree** movement:
                - wait: Do nothing and stay still.
                - walk_forward: Move forward one step.
                - walk_backward: Move backward one step.
                - look_left: Rotate your body 30 degrees to the left.
                - look_right: Rotate your body 30 degrees to the right.
                - look_up: Tilt your view 30 degrees upward.
                - look_down: Tilt your view 30 degrees downward.
                - operate <object>: interact with corresponding object, you should describe the object in <object>.

                # Critical Navigation Strategies
                1. **Approach Target**: Finding the target is not enough. You must navigate to the target's location and get close to it.
                2. **Obstacle Avoidance**: Prioritize wide, open pathways. If the path ahead is blocked, you should prioritize `look_right` to find a spacious road. Move onto that spacious road to ensure you are on a clear path, and **then** proceed toward the target.
                3. **Visual Recovery**: If your current observation lacks meaningful information (e.g., you are staring closely at a wall, the floor, or the ceiling, and cannot see any distinct objects or pathways), you must assume your view is obstructed. In this scenario, priority is to regain situational awareness by rotating to find a valid visual cue. You should strictly choose to `look_right` repeatedly until a meaningful object, open space, or your target comes into view.
                4. **Operate Object**: You can only operate an object if it is directly in your field of view and you are close enough to interact with it. If the target object is not in front of you or you are not close enough, you must first navigate to its location.

                # Output Requirements
                1. **Decision Basis**: You must analyze the provided `Task` and inspect the `Observation`. Your decision must bridge the gap between what you see and what you need to achieve.
                2. **Consistency**: Your chosen action must logically result from your reasoning process.
                3. **Format**: You should think first and then choose the right action at this moment. Your output should be strictly in the following format: <think>Your step-by-step reasoning process...</think><action>selected_action</action>
                4. **Constraint**: You should only choose the exact action name from the list above without any other words or explanations outside the tags.

                Your Task is: {task}\n{current_memory}
                """

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,  
        stop_token_ids=[], 
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": observation},
            ],
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    llm_inputs = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "image": observation # Qwen2-VL 接收 PIL 对象
            },
        }
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    output = outputs[0].outputs[0].text
    return output

# 测试模型
def test_model(json_file_path, model_path, mc_root, working_dir):
    # 读取数据集
    data = read_json(json_file_path)
    
    # 初始化 LLM 模型
    llm = init_llm(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,  
        trust_remote_code=True,
    )
    
    # 初始化 Minecraft Simulator
    sim = MCSimulator(config=data[0], mc_root=mc_root, working_dir=working_dir, display_port=9001, xvfb=True)
    
    # 创建输出目录
    output_dir = Path("./test_results")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化 memory agent 相关变量
    current_memory = ""
    previous_obs = None
    processor = None  # 如果需要使用 processor，可以在这里初始化

    # 测试每个场景
    for i in range(len(data)):
        if i == 0:
            obs, info = sim.reset()
        else:
            obs, info = sim.fast_reset(data[i])
        
        # 保存初始观察图像
        # img_bgr = cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2BGR)
        # img_path = output_dir / f"scene_{scene_id}_step_0.png"
        # cv2.imwrite(str(img_path), img_bgr)
        
        # 测试模型
        step_count = 0
        while True:
            task = f'find and navigate to the {data[i]["target_type"]}'
            pov_img = Image.open(obs['pov']).convert('RGB') 
            pov_img = pov_img.resize((540, 360), resample=Image.BILINEAR)
            
            # 调用 memory agent 更新记忆
            current_memory, _ = call_memory_agent(
                previous_obs=previous_obs,
                current_obs=pov_img,
                response_text="",
                current_memory=current_memory,
                processor=processor,
                tokenizer=tokenizer
            )
            
            action_str = get_action_from_llm(tokenizer, llm, task, pov_img, current_memory, previous_obs, processor)
            print(obs['pov'])
            print(action_str)
            
            # 解析动作
            action_match = re.search(r'<action>(.*?)</action>', action_str)
            if action_match:
                action = action_match.group(1)
            else:
                print("Invalid action format")
                break
            
            # 执行动作
            obs, reward, terminated, truncated, info = sim.step(action)
            step_count += 1
            
            # 更新 previous_obs
            previous_obs = pov_img
            
            # 保存观察图像
            # img_bgr = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
            # img_path = output_dir / f"scene_{scene_id}_step_{step_count}.png"
            # cv2.imwrite(str(img_path), img_bgr)
            
            if terminated or truncated or step_count == 120:
                sim.close()
                print(f"Episode finished at step {step_count}")
                break
        
        print(f"Test completed! Total steps: {step_count}")
    
    print(f"\n[Test] All scenes completed! Results saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    jsonl_file_path = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/data_test.json"  # 替换为你的 JSONL 文件路径
    model_path = "/mnt/shared-storage-user/steai_share/wuxiongbin/checkpoints/steai_origin_door"  # 替换为你的 VLLM 模型路径
    mc_root = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/grpc_raycraft_gpu/7/.minecraft"  # 替换为你的 Minecraft 根目录
    working_dir = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/grpc_raycraft_gpu/test"  # 替换为你的工作目录
    
    test_model(jsonl_file_path, model_path, mc_root, working_dir)