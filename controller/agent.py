import gc
import math
import os
import random
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from .types import ConversationMessage, APIConversationMessage, InferenceEngine, TokenizedConversationOutput

import time
from typing import Tuple
from openai import OpenAI

try:
    import torch_npu
except ImportError:
    torch_npu = None


class BaseChatTemplate(metaclass=ABCMeta):
    @abstractmethod
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        raise NotImplementedError

    def tokenize_conversation(
        self,
        conversation: list[ConversationMessage],
        tokenizer: PreTrainedTokenizerBase,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        text = ""
        input_ids = []
        action_mask = []
        for idx, message in enumerate(conversation):
            res = self.tokenize_conversation_one(
                message, tokenizer, idx, add_generation_prompt and idx == len(conversation) - 1
            )
            text += res["text"]
            input_ids += res["input_ids"]
            action_mask += res["action_mask"]
        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class Agent:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: BaseChatTemplate | None = None,
        inference_engine: InferenceEngine = "default",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template = chat_template or Llama2Template()
        self.inference_engine = InferenceEngine(inference_engine)
        self._vllm = None

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[int],
        generation_config: GenerationConfig,
        refresh_engine: bool = False,
    ) -> torch.Tensor:
        if isinstance(self.model, DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model
        if self.inference_engine == InferenceEngine.VLLM:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            from vllm import LLM, SamplingParams, TokensPrompt

            if not refresh_engine and self._vllm is not None:
                llm = self._vllm
            else:
                print("Initializing vLLM engine.")
                self._vllm = None
                gc.collect()
                if model.device != torch.cpu:
                    model.to("cpu")

                while shm_path := Path(
                    f"/dev/shm/agentgym/inference_model_cache/{str(random.randint(0, 2**32))}"
                ):
                    if not shm_path.exists():
                        break
                model.save_pretrained(shm_path)
                self.tokenizer.save_pretrained(shm_path)

                if torch.cuda.is_available():
                    num_devices = torch.cuda.device_count()
                    torch.cuda.empty_cache()
                elif torch_npu:
                    num_devices = torch_npu.npu.device_count()
                else:
                    num_devices = 1

                try:
                    num_heads = self.model.config.num_attention_heads
                    vocab_size = self.model.config.vocab_size
                    n = math.gcd(num_heads, vocab_size)
                except:
                    n = 1

                for tp_size in range(num_devices, 0, -1):
                    if n % tp_size == 0:
                        break
                print(f"{num_devices=}, {n=}, {tp_size=}.")
                try:
                    llm = LLM(
                        str(shm_path),
                        tensor_parallel_size=tp_size,
                        enable_prefix_caching=bool(not torch_npu),
                        use_v2_block_manager=True,
                        disable_custom_all_reduce=True,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    print(e)
                    print("Fail to create vLLM engine.")
                    exit(-1)

                self._vllm = llm
                shutil.rmtree(shm_path)

            INF = float("inf")
            max_tokens = generation_config.max_new_tokens or INF
            if generation_config.max_length:
                max_length = generation_config.max_length - len(input_ids)
            else:
                max_length = INF
            max_tokens = min(max_tokens, max_length)
            if max_tokens == INF:
                max_tokens = None

            generation_config = {
                "repetition_penalty": generation_config.repetition_penalty,
                "temperature": generation_config.temperature,
                "top_p": generation_config.top_p,
                "top_k": generation_config.top_k,
                "min_p": generation_config.min_p,
                # "length_penalty": generation_config.length_penalty,
                "early_stopping": generation_config.early_stopping,
                "max_tokens": max_tokens,
                "min_new_tokens": generation_config.min_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id],
            }
            generation_config = {k: v for k, v in generation_config.items() if v}
            output = llm.generate(
                # prompts=TokensPrompt(prompt_token_ids=input_ids),
                prompt_token_ids=input_ids,
                sampling_params=SamplingParams.from_optional(
                    **generation_config,
                    detokenize=False,
                ),
                use_tqdm=False,
            )

            generated_tokens = []

            for o in output:
                generated_tokens.append(list(o.outputs[0].token_ids))

        else:
            output = model.generate(
                inputs=torch.tensor(input_ids, device=model.device),
                generation_config=generation_config,
            )
            if isinstance(output, GenerateOutput):
                output = output.sequences
            generated_tokens = [
                o[len(input_ids[0]) :].cpu().numpy().tolist() for o in output
            ]

        return generated_tokens


class APIAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1,
        top_p: float = 1,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # self.role = {"system": "system", "human": "user", "gpt": "assistant"}

    def generate(
        self,
        conversation: list[APIConversationMessage],
    ) -> Tuple[str, str | None]:
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    # messages=[{"role": self.role[c["from"]], "content": c["value"]} for c in conversation],
                    # messages=conversation,
                    messages=[{"role": c["role"], "content": c["content"]} for c in conversation],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                return response.choices[0].message.content, response.choices[0].message.reasoning_content if hasattr(response.choices[0].message, "reasoning_content") else None
            except Exception as e:
                print(e)
                time.sleep(1)


class APIAgentvllm:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 1,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )
        resp = response.choices[0].message.content
        return resp


class Llama2Template(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        """
        This function applied Llama Chat template on the given vicuna-styled conversation message.
        You can provide your own _tokenize_conversation_one to adapt to your own task.
        """
        if message["from"] == "human":
            text = f"<s>[INST] {message['value']} [/INST]"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            text = f"{message['value']}</s>"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            text = f" {text}"
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class ChatMLTemplate(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        """
        This function applied Llama Chat template on the given vicuna-styled conversation message.
        You can provide your own _tokenize_conversation_one to adapt to your own task.
        """
        if idx == 0 and message["from"] != "system":
            text = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
        else:
            text = ""
        if add_generation_prompt:
            if message["from"] == "human":
                text += f"<|im_start|>user\n{message['value']}<|im_end|>\n<|im_start|>assistant\n"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
            else:
                text += f"{message['value']}<|im_end|>"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
                # text = f" {text}"
        else:
            if message["from"] == "human":
                text += f"<|im_start|>user\n{message['value']}<|im_end|>\n"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
            else:
                text += f"<|im_start|>assistant\n{message['value']}<|im_end|>"
                input_ids = tokenizer.encode(text, add_special_tokens=False)
                # text = f" {text}"

        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class Llama3Template(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        val = message["value"]
        while len(val) and val[-1] in [" ", "\n", "\t"]:
            val = val[:-1]
        mfrom = message["from"]
        if add_generation_prompt:
            mfrom = message["from"]
            if idx == 0:
                text = f"<|begin_of_text|>"
            else:
                text = ""
            if mfrom == "human":
                text += f"<|start_header_id|>user<|end_header_id|>\n\n{val}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif mfrom == "gpt":
                text += f"{val}<|eot_id|>"
        else:
            if mfrom == "human":
                mfrom = "user"
            elif mfrom == "gpt":
                mfrom = "assistant"
            if idx == 0:
                text = f"<|begin_of_text|><|start_header_id|>{mfrom}<|end_header_id|>\n\n{val}<|eot_id|>"
            else:
                text = f"<|start_header_id|>{mfrom}<|end_header_id|>\n\n{val}<|eot_id|>"

        input_ids = tokenizer.encode(text, add_special_tokens=False)
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)
        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )


class ChatGLM4Template(BaseChatTemplate):
    def tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
        idx: int = -1,
        add_generation_prompt: bool = False,
    ) -> TokenizedConversationOutput:
        val = message["value"]
        if add_generation_prompt:
            mfrom = message["from"]
            if idx == 0:
                text = "[gMASK]<sop>"
            else:
                text = ""
            if mfrom == "human":
                text += f"<|user|>\n{val}<|assistant|>"
            else:
                text += f"\n{val}"
        else:
            mfrom = message["from"]
            if mfrom == "human":
                mfrom = "user"
            elif mfrom == "gpt":
                mfrom = "assistant"
            if idx == 0:
                text = f"[gMASK]<sop><|{mfrom}|>\n{val}"
            else:
                text = f"<|{mfrom}|>\n{val}"
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)
        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )
