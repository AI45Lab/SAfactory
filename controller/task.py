from typing import Any, Callable, Mapping, Optional, Sequence

from transformers import GenerationConfig

from . import Agent, APIAgent, BaseEnvClient
from .types import ConversationMessage, APIConversationMessage, ExperienceOutput, APIExperienceOutput


class BaseTask:
    env_client_cls: Callable
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
    ) -> None:
        """
        Initializes the Task object.

        Args:
            client_args (Mapping[str, Any]): A mapping of client arguments.
            n_clients (int, optional): The number of clients. Defaults to 1. Larger than 1 for batch generation. Batch generation is not implemented yet.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])

    def _generate_experience_one(
        self,
        agent: Agent | APIAgent,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        if isinstance(agent, Agent):
            tokenizer = agent.tokenizer
            conversation = list(client.conversation_start)
            conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            conversation_tokenized = agent.chat_template.tokenize_conversation(
                conversation, tokenizer, add_generation_prompt=True
            )
        elif isinstance(agent, APIAgent):
            conversation = [APIConversationMessage({"role": "user", "content": client.conversation_start[0]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "assistant", "content": client.conversation_start[1]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "user", "content": state, "reasoning_content": None})]
        else:
            raise NotImplementedError
        rounds = 0

        while not done:
            if isinstance(agent, Agent):
                input_length = len(conversation_tokenized["input_ids"])
                # if input_length exceeds max_length, break
                if input_length >= (generation_config.max_length or 4096):
                    break
                try:
                    generated_tokens = agent.generate(
                        [conversation_tokenized["input_ids"]], generation_config
                    )[0]
                except Exception as e:  # pylint: disable=W0718:broad-exception-caught
                    print(e)
                    break  # break if generate method raises exceptions

                if generated_tokens[-1] != tokenizer.eos_token_id:
                    generated_tokens += [tokenizer.eos_token_id]

                generated_text = tokenizer.decode(generated_tokens)
                conversation_tokenized["text"] += f" {generated_text}"
                conversation_tokenized["input_ids"] += generated_tokens
                conversation_tokenized["action_mask"] += [1] * len(generated_tokens)

                generated_text = generated_text[
                    : -len(tokenizer.eos_token)
                ]  # not endswith eos_token
                conversation.append(
                    ConversationMessage(
                        {"from": "gpt", "loss": True, "value": generated_text}
                    )
                )
            elif isinstance(agent, APIAgent):
                generated_text, generated_reasoning_text = agent.generate(conversation)
                conversation.append(
                    APIConversationMessage(
                        {"role": "assistant", "content": generated_text, "reasoning_content": generated_reasoning_text}
                    )
                )
            else:
                raise NotImplementedError

            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            if isinstance(agent, Agent):
                env_message = ConversationMessage(
                    {"from": "human", "loss": None, "value": state}
                )
                env_message_tokenized = agent.chat_template.tokenize_conversation_one(
                    env_message, tokenizer, add_generation_prompt=True
                )

                conversation.append(env_message)
                conversation_tokenized["text"] += env_message_tokenized["text"]
                conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
                conversation_tokenized["action_mask"] += env_message_tokenized[
                    "action_mask"
                ]
            elif isinstance(agent, APIAgent):
                conversation.append(
                    APIConversationMessage(
                        {"role": "user", "content": state, "reasoning_content": None}
                    )
                )
            else:
                raise NotImplementedError

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        if isinstance(agent, Agent):
            return ExperienceOutput(
                conversation=conversation,
                reward=reward,
                text=conversation_tokenized["text"],
                seq_ids=conversation_tokenized["input_ids"],
                attention_mask=[1] * len(conversation_tokenized["input_ids"]),
                action_mask=conversation_tokenized["action_mask"],
            )
        elif isinstance(agent, APIAgent):
            return APIExperienceOutput(
                conversation=conversation,
                reward=reward,
            )
        else:
            raise NotImplementedError

    def _generate_experience_batch(
        self,
        agent: Agent | APIAgent,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        client = self.clients[0]
        result = [
            self._generate_experience_one(
                agent=agent,
                client=client,
                idx=idx,
                generation_config=generation_config,
                max_rounds=max_rounds,
            )
            for idx in idxs
        ]
        return result

    def generate_experience(
        self,
        agent: Agent | APIAgent,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        if isinstance(idxs, int):
            idxs = [idxs]

        return self._generate_experience_batch(
            agent=agent,
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
