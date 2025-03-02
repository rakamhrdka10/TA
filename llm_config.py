import requests

class GroqLLM:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    async def invoke(self, prompt: str) -> str:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500,
            },
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(
                f"Error dari Groq: {response.status_code}, {response.text}"
            )


# Konfigurasi LLM
GROQ_API_KEY = "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"  # Ganti dengan API key Anda
GROQ_MODEL = "llama3-70b-8192"  # Model yang digunakan

llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
