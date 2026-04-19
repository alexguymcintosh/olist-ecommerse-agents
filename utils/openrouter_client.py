import json
import os
import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> None:
        return None

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


def parse_batch_llm_response(
    raw: str,
    items: list[dict],
    state_key: str = "state",
    category_key: str = "category",
) -> list[dict | None]:
    """Parse a batch LLM response into per-item results aligned with *items*.

    Attempts in order:
    1. JSON array with state+category keys → keyed lookup.
    2. JSON array without keys → positional alignment.
    3. Single JSON object → broadcast to all items (handles simple mocks).
    4. Returns [None, ...] for total failure; caller must apply fallback.
    """
    # --- attempt 1 & 2: JSON array ---
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            arr = json.loads(raw[start : end + 1])
            if isinstance(arr, list) and arr:
                result_map = {
                    (str(entry.get(state_key, "")), str(entry.get(category_key, ""))): entry
                    for entry in arr
                    if isinstance(entry, dict)
                }
                if result_map:
                    return [
                        result_map.get(
                            (str(inp.get(state_key, "")), str(inp.get(category_key, "")))
                        )
                        for inp in items
                    ]
                # positional fallback
                return [arr[i] if i < len(arr) else None for i in range(len(items))]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # --- attempt 3: single JSON object → broadcast ---
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return [obj] * len(items)
    except (json.JSONDecodeError, TypeError):
        pass

    return [None] * len(items)


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
