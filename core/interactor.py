import asyncio
import base64
import json
import os
from typing import List, Dict, Tuple, Type
from .agent.base_agent import APIAgent
from .data_manager.manager import DataManager
from .data_manager.models import EnvironmentConfig, InteractionSession
from .env.env_register import get_env_class

class Interactor:
    def __init__(
        self,
        agent: APIAgent,
        data_manager: DataManager,
        max_workers: int = 5,
        max_steps: int = 1000,
        visual_save_path: str = None
    ):
        self.agent = agent
        self.data_manager = data_manager
        self.max_workers = max_workers  # 最大并行环境数
        self.max_steps = max_steps      # 每个环境最大交互步数
        self.visual_save_path = visual_save_path

    async def _init_environment(
        self, 
        env_config: EnvironmentConfig
    ) -> object:
        """根据配置初始化环境实例"""
        # 1. 从注册表获取环境类
        env_class: Type[object] = get_env_class(env_config.env_name)
        
        # 2. 解析环境参数
        env_params = env_config.env_params.copy()
        env_id = env_config.env_id
        env_name = env_config.env_name
        
        # 3. 动态传入所有环境参数
        try:
            return env_class(env_id=env_id, env_name=env_name, **env_params)
        except TypeError as e:
            raise ValueError(
                f"初始化环境 {env_config.env_name} 失败：参数不匹配。"
                f"环境所需参数与env_params中的参数不兼容。错误详情：{str(e)}"
            ) from e

    async def _run_single_environment(
        self, 
        env_config: EnvironmentConfig
    ) -> Tuple[InteractionSession, float]:
        """在单个环境中运行Agent交互循环（核心逻辑不变，略作调整）"""
        # 1. 初始化环境（通过注册机制获取的环境类）
        env = await self._init_environment(env_config)
        session = await self.data_manager.create_session(
            env_config=env_config,
            agent_model=self.agent.model
        )

        total_reward = 0.0
        step_id = 1
        
        try:
            # 2. 重置环境获取初始状态（假设所有环境都实现了reset方法）
            obs, info = env.reset()
            done = False

            # 3. 交互循环（假设所有环境都实现了step方法）
            while not done and step_id <= self.max_steps:
                prompt = env.get_task_prompt()
                
                # Agent生成响应
                response = await asyncio.to_thread(
                    self.agent.generate, 
                    prompt_output=prompt
                )

                # 环境执行动作（统一接口假设：step返回(state, reward, done, info)）
                step_output = env.step(response)
                reward = step_output.reward
                terminated = step_output.terminated
                truncated = step_output.truncated
                done = terminated or truncated

                img_filename = f"env_{env_config.env_id}/step_{step_id:04d}.png"
                render_output = env.render()
                base64_str = render_output.image_base64
                text_content = render_output.text_content
                text_dict = render_output.text_dict
                if base64_str:
                    # 2. 解码Base64字符串为二进制数据
                    # 注意：Base64字符串可能包含前缀（如 'data:image/png;base64,'），需先去除
                    if 'base64,' in base64_str:
                        base64_str = base64_str.split('base64,')[1]  # 提取纯Base64部分
                    image_bytes = base64.b64decode(base64_str)
                    img_filename = f"env_{env_config.env_id}/step_{step_id:04d}.png"
                    save_path = os.path.join(self.visual_save_path, img_filename)
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)

                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)

                if text_content:
                    txt_filename = f"env_{env_config.env_id}.txt"
                    save_path = os.path.join(self.visual_save_path, txt_filename)
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    with open(save_path, "a", encoding="utf-8") as file:
                        file.write(f"step_{step_id:04d}: " + text_content + "\n")

                if text_dict:
                    # 保存字典数据为JSON文件，按环境ID区分，每个步骤追加更新
                    json_filename = f"env_{env_config.env_id}.json"
                    save_path = os.path.join(self.visual_save_path, json_filename)
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 读取已有数据（如果文件存在）
                    existing_data = {}
                    if os.path.exists(save_path):
                        with open(save_path, 'r', encoding='utf-8') as f:
                            try:
                                existing_data = json.load(f)
                            except json.JSONDecodeError:
                                # 处理文件损坏情况，保留损坏文件并新建
                                os.rename(save_path, f"{save_path}.corrupted")
                                existing_data = {}
                    
                    # 将当前步骤的字典数据添加到现有数据中（以步骤ID为键）
                    existing_data[f"step_{step_id:04d}"] = text_dict
                    
                    # 写入更新后的数据
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)

                # 记录交互步骤
                await self.data_manager.record_step(
                    session=session,
                    step_id=step_id,
                    prompt=prompt,
                    response=response,
                    reward=reward,
                    done=done
                )

                # 更新状态
                total_reward += reward
                step_id += 1

            # 4. 完成会话记录
            await self.data_manager.update_session(
                session=session,
                total_reward=total_reward,
                is_completed=True
            )

        except Exception as e:
            print(f"环境 {env_config.env_name}_{env_config.env_id} 出错: {str(e)}")
            await self.data_manager.update_session(
                session=session,
                total_reward=total_reward,
                is_completed=False
            )

        return session, total_reward


    async def run_all_environments(self) -> Dict[str, float]:
        """并行运行所有配置的环境（逻辑不变）"""
        env_configs = await self.data_manager.get_all_environments()
        if not env_configs:
            print("没有找到激活的环境配置")
            return {}

        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def bounded_task(env_config):
            async with semaphore:
                return await self._run_single_environment(env_config)

        tasks = [bounded_task(config) for config in env_configs]
        results = {}
        
        for task in asyncio.as_completed(tasks):
            session, total_reward = await task
            env_key = f"{session.env.env_name}_{session.env.env_id}"
            results[env_key] = total_reward
            print(f"环境 {env_key} 完成，总奖励: {total_reward}")

        return results