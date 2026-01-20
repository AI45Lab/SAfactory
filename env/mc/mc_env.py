from typing import Tuple
from pathlib import Path
import yaml
import sys

# 添加 MineStudio 到路径（在导入之前）
_current_file = Path(__file__).resolve()
_mc_env_dir = _current_file.parent
_minestudio_path = _mc_env_dir / "MineStudio"
if _minestudio_path.exists() and str(_minestudio_path) not in sys.path:
    sys.path.insert(0, str(_minestudio_path))

# Gym compatibility: Make gymnasium available as 'gym'
try:
    import gymnasium
    sys.modules['gym'] = gymnasium
    if hasattr(gymnasium, 'spaces'):
        sys.modules['gym.spaces'] = gymnasium.spaces
except ImportError:
    pass  # If gymnasium not available, try with gym

from typing import List
from openai.types.chat import ChatCompletionMessageParam
from core.types.base import ResetOutput, StepOutput, RenderOutput, PromptOutput, TextContent, ImageContent, OpenAIMessage, MessageContent
from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from minestudio.simulator.entry import MinecraftSim
from env.mc.utils.action_converter import ActionFromLLMConverter
from env.mc.utils.sim_callbacks_loader import load_simulator_setup_from_yaml
from env.mc.utils.mc_prompt import gen_sys_prompt
from rich import print, console

from minestudio.simulator.callbacks import (
    SpeedTestCallback,
    RecordCallback,
    SummonMobsCallback,
    MaskActionsCallback,
    RewardsCallback,
    CommandsCallback,
    TaskCallback,
    FastResetCallback,
    JudgeResetCallback,
    InitInventoryCallback
)

rich_console = console.Console()

@register_env("mc_gym")
class MCGym(BaseEnv):
    def __init__(
        self,
        env_config: str = "",
        env_id: str = "",
        env_name: str = "",
        dataset=None,
        **kwargs,
    ):
        super().__init__(env_id, env_name)
        self.env_config = env_config
        self.dataset = dataset
        self.instructions = ""  # 初始化 instructions
        self.obs_size = (360, 640)  # 默认值 (height, width)，会在 init_simulator 中更新
        self.simulator: MinecraftSim = self.init_simulator(env_config)
        self.init_fov()
        self.current_step = 0  # 跟踪当前步数
        self.last_obs = None  # 保存最后一次观察
        # 初始化动作转换器
        self.action_converter = ActionFromLLMConverter(
            hfov_deg=self.current_hfov,
            vfov_deg=self.current_vfov,
            return_numpy=True,      # MinecraftSim 需要 numpy 数组格式
            map_camera_to_11=True   # 将 21×21 camera bins 映射到 11×11（VPT 格式）
        )
    
    def step(self, action: str) -> StepOutput:
        self.messages.append(
            {"role": "assistant", "content": action}
        )
        # 转换字符串 action 为字典
        if isinstance(action, str):
            # 优先从 last_obs 获取实际图像形状，否则使用配置的 obs_size
            if self.last_obs and 'image' in self.last_obs:
                img = self.last_obs['image']
                image_shape = img.shape[:2] if hasattr(img, 'shape') else self.obs_size
            else:
                image_shape = self.obs_size
            action = self.action_converter.convert(action, image_shape)
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.simulator.step(action)
        
        # 更新状态
        self.last_obs = obs
        self.current_step += 1
        
        return StepOutput(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        
    def reset(self, seed: int | None = None) -> ResetOutput:
        obs, info = self.simulator.reset()
        self.last_obs = obs
        self.current_step = 0
        return ResetOutput(observation=obs, info=info)
    
    def close(self) -> None:
        self.simulator.close()
    
    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        return self.build_prompt()
    
    def render(self) -> RenderOutput:
        """渲染当前环境状态，返回 POV 图像
        
        Returns:
            RenderOutput: 包含当前 POV 图像的渲染输出
        """
        import io
        import numpy as np
        from PIL import Image
        
        if self.last_obs is None:
            raise RuntimeError("No observation available. Call reset() first.")
        
        # 从观察中获取图像数据
        # MinecraftSim 的 obs 是字典，包含 'image' 键
        if isinstance(self.last_obs, dict) and 'image' in self.last_obs:
            image_array = self.last_obs['image']
        else:
            raise ValueError(f"Unexpected observation format: {type(self.last_obs)}")
        
        # 确保是 numpy 数组并转换为 uint8
        if isinstance(image_array, np.ndarray):
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
        else:
            raise ValueError(f"Expected numpy array, got {type(image_array)}")
        
        # 转换为 PIL Image
        img = Image.fromarray(image_array, 'RGB')
        
        # 转换为字节流
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = buffer.read()
        buffer.close()
        
        return RenderOutput(
            step=self.current_step,
            image_data=image_data
        )
    
    def action_string_to_dict(self, action_input, **kwargs) -> tuple:
        """执行动作

        Args:
            action_input: 支持两种格式
                - str: LLM格式 '[{"action": "attack", "yaw": -25.0, "pitch": 8.0}]'
                - dict: Agent格式 {'buttons': 5, 'camera': 222}
        """
        # 判断action格式并转换
        if isinstance(action_input, str):
            # LLM格式：使用ActionConverter转换
            action = self.action_converter.convert(action_input, self._obs_shape)
        elif isinstance(action_input, dict):
            # Agent格式：将整数转换为numpy数组
            import numpy as np
            action = {}
            for key, value in action_input.items():
                if isinstance(value, int):
                    action[key] = np.array(value)
                else:
                    action[key] = value
        else:
            raise ValueError(f"Unsupported action type: {type(action_input)}")

        return action
    
    def _extract_instructions(self, data: dict) -> str:
        """从配置数据中提取 instructions"""
        task_conf = data.get("task_conf", [])
        if isinstance(task_conf, list):
            texts = [item.get("text", "") for item in task_conf if isinstance(item, dict) and "text" in item]
            return " ".join(texts) if texts else data.get("text", "")
        return str(task_conf) if task_conf else data.get("text", "")
    
    def init_simulator(self, config: str) -> MinecraftSim:
        import time
        start_time = time.perf_counter()
        try:
            # 区分两种配置来源:
            # 1. 字典 -> MVP2 模式
            # 2. 字符串/Path -> 从 YAML 加载
            if isinstance(config, dict):
                sim_callbacks = config.get("sim_callbacks", [])
                env_overrides = {k: v for k, v in config.items() if k != "sim_callbacks"}
                self.instructions = self._extract_instructions(config)
            else:
                # 从 YAML 加载
                _desc = config
                sim_callbacks, env_overrides = load_simulator_setup_from_yaml(_desc) if _desc else ([], {})
                # 简单加载 YAML 获取 instructions
                if _desc:
                    try:
                        with Path(_desc).open("r", encoding="utf-8") as f:
                            yaml_data = yaml.safe_load(f) or {}
                        self.instructions = self._extract_instructions(yaml_data)
                    except Exception:
                        self.instructions = ""

            # rich_console.log(f"[mc_simulator] env_overrides={env_overrides}")
            if 'seed' in env_overrides:
                # rich_console.log(f"[mc_simulator] seed={env_overrides['seed']}")
                ...

            # Ensure default recording callback exists
            if not any(getattr(cb, "__class__", type(None)).__name__ == "RecordCallback" for cb in sim_callbacks):
                sim_callbacks.append(RecordCallback(record_path="./output", fps=30, frame_type="pov"))

            obs_w = int(env_overrides.get('obs_width', env_overrides.get('obs_w', 640)))
            obs_h = int(env_overrides.get('obs_height', env_overrides.get('obs_h', 360)))

            # Support 'resolution' key: [W, H] or {width: W, height: H}
            resolution = env_overrides.get('resolution')
            # rich_console.log(f"[mc_simulator] resolution={resolution}")
            try:
                if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                    obs_w, obs_h = int(resolution[0]), int(resolution[1])
                elif isinstance(resolution, dict):
                    obs_w = int(resolution.get('width', obs_w))
                    obs_h = int(resolution.get('height', obs_h))
            except Exception:
                pass

            # rich_console.log(f"[mc_simulator] resolved obs_size=({obs_w},{obs_h})")
            preferred_biome = env_overrides.get('preferred_spawn_biome', 'plains')
            action_type = env_overrides.get('action_type', 'agent')
            timestep_limit = int(env_overrides.get('timestep_limit', 1000))
            # rich_console.log(obs_w)
            # rich_console.log(obs_h)

            # 添加 FastResetCallback 以加速后续 reset（如果还没有）
            if not any(isinstance(cb, FastResetCallback) for cb in sim_callbacks):
                fast_reset_biomes = env_overrides.get('fast_reset_biomes', ['plains', 'forest', 'mountains'])
                fast_reset_range = int(env_overrides.get('fast_reset_range', 1000))
                fast_reset_time = int(env_overrides.get('fast_reset_time', 1000))
                fast_reset_weather = env_overrides.get('fast_reset_weather', 'clear')
                
                sim_callbacks.append(FastResetCallback(
                    biomes=fast_reset_biomes,
                    random_tp_range=fast_reset_range,
                    start_time=fast_reset_time,
                    start_weather=fast_reset_weather
                ))
                # rich_console.log(f"[mc_simulator] Added FastResetCallback: biomes={fast_reset_biomes}, range={fast_reset_range}")
            # 保存 obs_size 用于 FOV 计算和动作转换
            self.obs_size = (obs_h, obs_w)  # (height, width)
            
            simulator = MinecraftSim(
                obs_size=(obs_w, obs_h),
                preferred_spawn_biome=preferred_biome,
                action_type=action_type,
                timestep_limit=timestep_limit,
                callbacks=sim_callbacks,
            )
        
                    # Debug verification of action mapping in use
            try:
                # rich_console.log(f"[mc_simulator] action_type={self.simulator.action_type}")
                mapper = getattr(self.simulator, "action_mapper", None)
                # rich_console.log(f"[mc_simulator] action_mapper={type(mapper).__name__}")
                cam_tbl = getattr(mapper, "CAMERA_IDX_TO_FACTORED", None)
                if cam_tbl is not None:
                    # rich_console.log(f"[mc_simulator] camera_table_size={len(cam_tbl)}")
                    ...
                else:
                    # rich_console.log("[mc_simulator] camera_table_size=N/A")
                    ...
            except Exception as _e:
                # rich_console.log(f"[mc_simulator] debug print failed: {_e}")
                ...

            self.action_converter = ActionFromLLMConverter(
                map_camera_to_11=True,   # 21×21 → 11×11
                return_numpy=True        # 返回 np.array([int])
            )
            self._obs_shape = (obs_h, obs_w, 3)

        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # logger.info(f"[mc_simulator] __init__ elapsed={elapsed_ms:.2f} ms")
        
        return simulator
    
    def init_fov(self):
        # Camera FOV settings (degrees). Default HFOV=70; VFOV derived from image aspect on use.
        self.hfov_deg = 70.0
        self.vfov_deg = None
        def _derive_vfov_if_needed(img_shape, hfov_deg, vfov_deg):
            try:
                if vfov_deg is not None:
                    return vfov_deg
                if hfov_deg is None:
                    return None
                h, w = img_shape[0], img_shape[1]
                import math as _math
                return 2.0 * _math.degrees(_math.atan(_math.tan(_math.radians(hfov_deg * 0.5)) * (h / max(1.0, float(w)))))
            except Exception:
                return vfov_deg
        # compute current FOVs for prompt/mapping
        self.current_hfov = self.hfov_deg
        # 使用实际的 obs_size 计算 VFOV
        self.current_vfov = _derive_vfov_if_needed(self.obs_size, self.hfov_deg, self.vfov_deg)
        # compute half ranges for explicit guidance
        try:
            self.hfov_half = round((self.current_hfov or 70.0) * 0.5, 2)
        except Exception:
            self.hfov_half = 35.0
        try:
            self.vfov_half = round((self.current_vfov if isinstance(self.current_vfov, (int, float)) else 0.0) * 0.5, 2)
        except Exception:
            self.vfov_half = 0.0
        
    
    def build_prompt(self) -> List[ChatCompletionMessageParam]:
        # 使用 gen_sys_prompt 生成 system prompt
        instructions = self.instructions if self.instructions else "Complete the Minecraft task."
        sys_prompt_text = gen_sys_prompt(
            instructions=instructions,
            current_hfov=self.current_hfov,
            current_vfov=self.current_vfov,
            hfov_half=self.hfov_half,
            vfov_half=self.vfov_half
        )
        
        # 构建 system_message
        self.messages = [
            {"role": "system", "content": sys_prompt_text}
        ]
        
        # 构建 user_message，包含当前 POV 图像
        # 添加当前 POV 图像（如果有观察）
        if self.last_obs is not None:
            render_output = self.render()
            # 确保 base64 图像使用正确的 data URL 格式
            base64_str = render_output.image_base64
            # 统一得到 data URL 格式，避免重复添加前缀导致解析失败
            if base64_str.startswith('data:'):
                base64_url = base64_str
            else:
                base64_url = f"data:image/png;base64,{base64_str}"
            self.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Based on the current view, what action should I take?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_url
                            }
                        }
                    ]
                }
            )
        else:
            self.messages.append(
                {"role": "user", "content":"Based on the current view, what action should I take?"}
            )
        
        return self.messages