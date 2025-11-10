from core.types.base import ResetOutput, StepOutput, RenderOutput, PromptOutput, TextContent, ImageContent, OpenAIMessage, MessageContent
from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from MineStudio.minestudio.simulator.entry import MinecraftSim

@register_env("mc_gym")
class MCGym(BaseEnv):
    def __init__(self, env_config: str = "", env_id: str = "", env_name: str = ""):
        super().__init__(env_id, env_name)
        self.analysis_config(env_config)
        self.simulator = MinecraftSim()
    
    def step(self, action: str) -> StepOutput:
        result = self.simulator.step(action)
        return StepOutput(
            observation=result.obs,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )
        
    def reset(self, seed: int | None = None) -> ResetOutput:
        result = self.simulator.reset()
        return ResetOutput(observation=result.obs, info=result.info)
    
    def close(self) -> None:
        self.simulator.close()
    
    def get_task_prompt(self) -> PromptOutput:
        return super().get_task_prompt()
    
    def render(self) -> RenderOutput:
        return super().render()
    