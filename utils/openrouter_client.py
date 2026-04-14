import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# RnD model — cheap, fast, smart enough for exploration
RND_MODEL = "deepseek/deepseek-v3.2"
# Production model — swap in when building final agents
PROD_MODEL = "anthropic/claude-sonnet-4-5"


def query_llm(messages: list, model: str = RND_MODEL, max_tokens: int = 1000) -> str:
    """Single entry point for all LLM calls. Swap model here for RnD vs prod."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://olist-ai-club",
        "X-Title": "Olist AI Club"
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def build_analyst_prompt(data_summary: str, question: str) -> list:
    """Standard prompt structure for all agent analysis calls."""
    return [
        {
            "role": "system",
            "content": "You are a business intelligence analyst specialising in e-commerce operations. Be concise, specific, and data-driven. Use bullet points. No fluff."
        },
        {
            "role": "user",
            "content": f"DATA:\n{data_summary}\n\nQUESTION:\n{question}"
        }
    ]
