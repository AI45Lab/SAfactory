import asyncio
import base64
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Type
from .llm import LLM, BaseURLProvider
from .data_manager.manager import DataManager
from .data_manager.models import EnvironmentConfig, InteractionSession
from .env.env_register import get_env_class

class Interactor:
    def __init__(
        self,
        base_url_provider: BaseURLProvider,
        api_key: str,
        model: str,
        data_manager: DataManager,
        temperature: float = 1.0,
        max_workers: int = 5,
        max_steps: int = 1000,
        visual_save_path: str = None
    ):
        self.base_url_provider = base_url_provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
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
        start_ts = time.time()
        env_key = f"{env_config.env_name}_{env_config.env_id}"
        try:
            print(
                f"[Interactor] start env={env_key} at {datetime.now().isoformat(timespec='seconds')}",
                flush=True,
            )
        except Exception:
            pass
        env = await self._init_environment(env_config)
        session = await self.data_manager.create_session(
            env_config=env_config,
            llm_model=self.model
        )

        # 根据 session 获取 base_url 并创建 LLM
        base_url = self.base_url_provider.get_base_url(session)
        llm = LLM(
            api_key=self.api_key,
            base_url=base_url,
            model=self.model,
            temperature=self.temperature,
        )

        total_reward = 0.0
        step_id = 1
        trajectory = ""  # 内存维护 trajectory，避免 update_session 时查询数据库

        try:

            # 2. 重置环境获取初始状态
            obs, info = env.reset()
            done = False

            # 3. 交互循环
            while not done and step_id <= self.max_steps:
                # Step start log
                try:
                    print(
                        f"[Interactor] step_start env={env_key} step={step_id} at {datetime.now().isoformat(timespec='seconds')}",
                        flush=True,
                    )
                except Exception:
                    pass
                prompt = env.get_task_prompt()

                # LLM 生成响应
                response = await llm.generate(prompt)

                # 环境执行动作（统一接口假设：step返回(state, reward, done, info)）
                step_output = env.step(response)
                reward = step_output.reward
                terminated = step_output.terminated
                truncated = step_output.truncated
                # 基于环境信号 + 最大步数限制综合判断 done
                done = terminated or truncated
                try:
                    # 若达到或超过最大步数，也视为 episode 结束（在 Interactor 层生效）
                    if self.max_steps is not None and step_id >= int(self.max_steps):
                        done = True
                except Exception:
                    pass

                img_filename = f"env_{env_config.env_id}/step_{step_id:04d}.png"
                # 可选渲染（通过环境变量 AIEVOBOX_NO_RENDER 控制，'1' 跳过渲染）
                if os.environ.get("AIEVOBOX_NO_RENDER", "0") != "1":
                    render_output = env.render()
                    base64_str = render_output.image_base64
                    if not base64_str:
                        raise ValueError("RenderOutput中未包含有效的base64图片数据")
                    # 解码Base64字符串为二进制数据
                    if 'base64,' in base64_str:
                        base64_str = base64_str.split('base64,')[1]
                    image_bytes = base64.b64decode(base64_str)
                    save_path = os.path.join(self.visual_save_path, img_filename)
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    with open(save_path, 'wb') as f:
                        f.write(image_bytes)

                # 记录交互步骤
                await self.data_manager.record_step(
                    session=session,
                    step_id=step_id,
                    prompt=prompt,
                    response=response,
                    reward=reward,
                    done=done
                )

                # 累积 trajectory（简化格式：只记录 step 和 response）
                trajectory += f"Step {step_id}:\nResponse: {response}\n\n"

                # 更新状态
                total_reward += reward
                step_id += 1

            # 4. 完成会话记录
            await self.data_manager.update_session(
                session=session,
                trajectory=trajectory,
                total_reward=total_reward,
                is_completed=True
            )
            try:
                elapsed = time.time() - start_ts
                print(
                    f"[Interactor] end env={env_key} steps={step_id-1} total_reward={total_reward:.4f} elapsed={elapsed:.2f}s",
                    flush=True,
                )
            except Exception:
                pass

        except Exception as e:
            print(f"环境 {env_config.env_name}_{env_config.env_id} 出错: {str(e)}")
            await self.data_manager.update_session(
                session=session,
                trajectory=trajectory,
                total_reward=total_reward,
                is_completed=False
            )
            try:
                elapsed = time.time() - start_ts
                print(
                    f"[Interactor] abort env={env_key} steps={step_id-1} total_reward={total_reward:.4f} elapsed={elapsed:.2f}s",
                    flush=True,
                )
            except Exception:
                pass

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
