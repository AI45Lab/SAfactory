from OpenRT.attacks.blackbox.implementations.crescendo_attack import (
    ConversationState,
    ConversationTurn,
    CrescendoAttack,
)

from core.types.base import StepOutput

from ..base import AttackStrategy
from ..factory import register_attack_strategy


@register_attack_strategy("crescendo_attack")
class CrescendoAttackStrategy(AttackStrategy):
    def setup(self) -> None:
        env = self.env
        self.crescendo_attack = CrescendoAttack(
            model=env.env_model,
            attack_model=env.env_model,
            judge=env.judge_model,
            max_turns=env.max_steps,
            max_backtracks=env.max_steps,
            success_threshold=float(env.judge_model.success_threshold),
            verbose=False,
        )
        self.conversation_state = ConversationState(
            turns=[],
            current_turn=0,
            conversation_summary="",
        )
        self.current_prompt = ""
        self.next_prompt_override = None

    def get_prompt(self) -> str:
        env = self.env
        if self.next_prompt_override is not None:
            prompt = self.next_prompt_override
            self.next_prompt_override = None
            self.current_prompt = prompt
            return prompt

        if not self.conversation_state.turns:
            prompt = self.crescendo_attack._generate_first_prompt(env.instruction)
        else:
            prompt = self.crescendo_attack._generate_escalated_prompt(
                env.instruction,
                self.conversation_state,
            )
        self.current_prompt = prompt
        return prompt

    def step(self, action: str) -> StepOutput:
        env = self.env
        if not self.current_prompt and env.messages and env.messages[-1]["role"] == "user":
            self.current_prompt = str(env.messages[-1]["content"])

        env.messages.append({"role": "assistant", "content": action})
        env.current_step += 1

        score, _ = env.judge_model.evaluate_response(env.instruction, action)
        step_reward = 5.0 - score

        turn = ConversationTurn(
            turn_number=self.conversation_state.current_turn + 1,
            user_prompt=self.current_prompt,
            assistant_response=action,
            refusal_score=score,
            objective_score=score,
            success_threshold=self.crescendo_attack.success_threshold,
        )
        self.conversation_state.add_turn(turn)

        terminated = env.judge_model.is_successful(score)
        if (
            not terminated
            and self.crescendo_attack._should_backtrack(score, self.conversation_state)
            and self.crescendo_attack._perform_backtrack(self.conversation_state)
        ):
            self.next_prompt_override = self.crescendo_attack._generate_escalated_prompt(
                env.instruction,
                self.conversation_state,
                self.conversation_state.refused_text,
            )

        truncated = env.current_step >= env.max_steps
        return StepOutput(
            observation=env._get_observation(),
            reward=step_reward,
            terminated=terminated,
            truncated=truncated,
            info=env._get_info(),
        )
