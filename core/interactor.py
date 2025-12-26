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
        llm_proxy: str = "",
        max_workers: int = 5,
        max_steps: int = 1000,
        visual_save_path: str = None,
        enable_render: bool = True,
        n_episodes: int = 1,
        message_cut: int = 3,
    ):
        self.base_url_provider = base_url_provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.llm_proxy = llm_proxy
        self.data_manager = data_manager
        self.max_workers = max_workers  # 最大并行环境数
        self.max_steps = max_steps      # 每个环境最大交互步数
        self.visual_save_path = visual_save_path
        self.enable_render = enable_render
        self.n_episodes = n_episodes
        self.message_cut = message_cut

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

        # 3. 动态传入所有环境参数（使用线程池避免阻塞）
        def create_env():
            return env_class(env_id=env_id, env_name=env_name, **env_params)

        try:
            return await asyncio.to_thread(create_env)
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
            llm_proxy=self.llm_proxy,
        )

        total_reward = 0.0
        step_id = 1
        trajectory = ""  # 内存维护 trajectory，避免 update_session 时查询数据库

        try:

            # 2. 重置环境获取初始状态
            obs, info = await asyncio.to_thread(env.reset)
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
                
                # 加入message截断逻辑   -> 后续这里可以做智能体的记忆
                if not prompt:
                    raise ValueError("get_task_prompt中未输出有效messages")
                
                messages_cut = []
                start_index = 0
                if prompt[0].get("role") == "system":
                    messages_cut.append(prompt[0])
                    start_index = 1
                    
                remaining_messages = prompt[start_index:]
                keep_count = self.message_cut * 2
                
                if keep_count <= 0:
                    response = await llm.generate(messages=messages_cut)
                else:
                    messages_cut += remaining_messages[-keep_count:]
                    response = await llm.generate(messages=messages_cut)

                # 环境执行动作（统一接口假设：step返回(state, reward, done, info)）
                # 使用线程池执行可能包含同步阻塞调用的 step 方法
                step_output = await asyncio.to_thread(env.step, response)
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

                # 可选渲染
                if self.enable_render:
                    img_filename = f"env_{env_config.env_id}/step_{step_id:04d}.png"
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


    async def _run_env_episodes(self, env_config: EnvironmentConfig) -> Tuple[str, List[float]]:
        """
        运行一个环境的所有 episodes（并发）

        同一环境的 n_episodes 同时启动
        """
        env_key = f"{env_config.env_name}_{env_config.env_id}"

        # 同时启动该环境的所有 episodes
        episode_tasks = [
            self._run_single_environment(env_config)
            for _ in range(self.n_episodes)
        ]
        results = await asyncio.gather(*episode_tasks, return_exceptions=True)

        # 收集奖励
        rewards = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"环境 {env_key} episode {i} 失败: {result}")
                rewards.append(0.0)
            else:
                session, reward = result
                rewards.append(reward)
                print(f"环境 {env_key} episode {i} 完成，奖励: {reward:.4f}")

        return env_key, rewards

    async def run_all_environments(self) -> Dict[str, List[float]]:
        """
        并行运行所有配置的环境

        - 同一环境的 n_episodes 必须同时启动
        - 环境并发数 = max(max_workers // n_episodes, 1)
        - 总并发数 ≈ max_workers
        """
        env_configs = await self.data_manager.get_all_environments()
        if not env_configs:
            print("没有找到激活的环境配置")
            return {}

        # 计算环境并发数
        env_concurrent = max(self.max_workers // self.n_episodes, 1)
        semaphore = asyncio.Semaphore(env_concurrent)
        print(f"并发配置: max_workers={self.max_workers}, n_episodes={self.n_episodes}, 环境并发数={env_concurrent}")

        async def bounded_env_group(env_config):
            async with semaphore:
                return await self._run_env_episodes(env_config)

        tasks = [bounded_env_group(config) for config in env_configs]
        results: Dict[str, List[float]] = {}

        for task in asyncio.as_completed(tasks):
            env_key, rewards = await task
            results[env_key] = rewards
            avg = sum(rewards) / len(rewards) if rewards else 0.0
            print(f"环境 {env_key} 全部完成: {len(rewards)} episodes, 平均奖励: {avg:.4f}")

        return results
