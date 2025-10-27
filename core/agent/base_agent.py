from openai import OpenAI
from core.types.base import PromptOutput


class APIAgent:
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

    def generate(self, prompt_output: PromptOutput) -> str:
        # 将PromtOutput转换为OpenAI API所需要的字典格式
        messages = []
        system_msg = {
            "role": prompt_output.system_message.role,
            "content": [item.root.model_dump() for item in prompt_output.system_message.content]
        }
        messages.append(system_msg)
        user_msg = {
            "role": prompt_output.user_message.role,
            "content": [item.root.model_dump() for item in prompt_output.user_message.content]
        }
        messages.append(user_msg)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        resp = response.choices[0].message.content
        return resp