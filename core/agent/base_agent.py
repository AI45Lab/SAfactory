from openai import OpenAI


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