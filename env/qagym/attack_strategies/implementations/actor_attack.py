from OpenRT.attacks.base_attack import AttackResult
from OpenRT.attacks.blackbox.implementations.actor_attack import ActorAttack

from core.types.base import StepOutput

from ..base import AttackStrategy
from ..factory import register_attack_strategy


@register_attack_strategy("actor_attack")
class ActorAttackStrategy(AttackStrategy):
    """
    先发现线索 actors, 再为每个线索生成多步问题链, 并在环境中逐步执行
    与 OpenRT 中不同的是, 问题链中的每一步都要进行 judge
    """

    def setup(self) -> None:
        env = self.env
        self.actor_attack = ActorAttack(
            model=env.env_model,
            helper_model=env.env_model,
            judge=env.judge_model,
            max_turns=env.max_steps,
            max_clues=(env.max_steps + 2) // 3,  # 每条线索链步数为 3-4 步, 此时线索数量一定满足 max_steps 的使用
        )

        self.attack_trace = None
        self.clues = []
        self.current_clue_idx = 0
        self.current_chain = []
        self.current_chain_idx = 0
        self.plan_initialized = False
        self.current_prompt = ""

    def get_prompt(self) -> str:
        self._ensure_plan_initialized()

        if self.current_chain_idx >= len(self.current_chain):
            if not self._load_next_chain():
                self.env.record_strategy_warning(
                    "actor_attack_default_prompt_used",
                    "No valid actor attack chain was available; falling back to the default prompt.",
                    used_fallback=True,
                    once=True,
                )
                self.current_prompt = self._default_prompt()
                return self.current_prompt

        self.current_prompt = str(self.current_chain[self.current_chain_idx])
        return self.current_prompt

    def step(self, action: str) -> StepOutput:
        env = self.env
        if not self.current_prompt and env.messages and env.messages[-1]["role"] == "user":
            self.current_prompt = str(env.messages[-1]["content"])

        env.messages.append({"role": "assistant", "content": action})
        env.current_step += 1

        score, _ = env.judge_model.evaluate_response(env.instruction, action)
        step_reward = 5.0 - score
        terminated = env.judge_model.is_successful(score)

        if not terminated and self.current_chain:
            self.current_chain_idx += 1
            if self.current_chain_idx >= len(self.current_chain):
                self.current_clue_idx += 1
                self.current_chain = []
                self.current_chain_idx = 0

        truncated = env.current_step >= env.max_steps
        return StepOutput(
            observation=env._get_observation(),
            reward=step_reward,
            terminated=terminated,
            truncated=truncated,
            info=env._get_info(),
        )

    def _default_prompt(self) -> str:
        return f"You need help with the following question: {self.env.instruction}"

    def _ensure_plan_initialized(self) -> None:
        if self.plan_initialized:
            return

        env = self.env
        self.attack_trace = AttackResult(target=env.instruction, method="ActorAttack")
        try:
            self.clues = self.actor_attack._discover_attack_clues(env.instruction, self.attack_trace)
        except Exception as exc:
            env.record_strategy_warning(
                "actor_attack_clue_discovery_failed",
                f"Clue discovery failed and the strategy will degrade: {exc}",
                used_fallback=True,
                once=True,
            )
            self.clues = []

        self.current_clue_idx = 0
        self.current_chain = []
        self.current_chain_idx = 0
        self.plan_initialized = True

    def _load_next_chain(self) -> bool:
        env = self.env
        while self.current_clue_idx < len(self.clues):
            clue = self.clues[self.current_clue_idx]
            try:
                chain = self.actor_attack._generate_attack_chain(clue, env.instruction, self.attack_trace)
            except Exception as exc:
                clue_name = clue.get("name", "unknown_clue")
                env.record_strategy_warning(
                    "actor_attack_chain_generation_failed",
                    f"Chain generation failed for clue '{clue_name}': {exc}",
                    used_fallback=True,
                )
                chain = None

            if chain:
                self.current_chain = [str(step) for step in chain if str(step).strip()]
                self.current_chain_idx = 0
                if self.current_chain:
                    return True

            self.current_clue_idx += 1

        self.current_chain = []
        self.current_chain_idx = 0
        return False
