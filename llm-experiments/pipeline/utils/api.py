from __future__ import annotations

import re
import time
from typing import Any

MAX_RETRIES      = 3
RETRY_BASE_DELAY = 1.0

BLABLADOR_BASE_URL  = "https://api.blablador.fz-juelich.de/v1/"
BLABLADOR_MAX_TOKENS = 6000

# Maps clean display name → full Blablador API ID
BLABLADOR_API_IDS: dict[str, str] = {
    "MiniMax-M2.7": "01 - MiniMax-M2.7 - our best model as of April, 2026",
    "GPT-OSS-120b": "01 - GPT-OSS-120b - an open model released by OpenAI in August 2025",
    "Qwen3.6-35B":  "08 - Qwen3.6-35B-A3B-FP8 - Multimodal model from Apr 2026",
    "Apertus-8B":   "15 - Apertus-8B-Instruct-2509 - A new swiss model from September 2025",
}

SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\Thesis\Ivaxi LLMs\llm-juries-503009-babf6fe232e6.json"
PROJECT_ID       = "llm-juries-503009"
DEFAULT_LOCATION = "global"


def is_blablador_model(model_name: str) -> bool:
    return model_name in BLABLADOR_API_IDS


def is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-")


def model_location_for(model_name: str) -> str:
    if "meta" in model_name.lower():
        return "us-central1"
    return DEFAULT_LOCATION


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_blablador(client: Any, model_name: str, prompt: str) -> tuple[str | None, str | None]:
    api_id = BLABLADOR_API_IDS[model_name]
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=api_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=BLABLADOR_MAX_TOKENS,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                return None, "Empty response after retries"
            return _strip_think(text), None
        except Exception as exc:
            err = str(exc)
            if attempt < MAX_RETRIES - 1 and ("429" in err or "rate_limit" in err.lower()):
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return None, err
    return None, "Max retries exceeded"


def call_openai(client: Any, model_name: str, prompt: str) -> tuple[str | None, str | None]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,
                temperature=0,
            )
            text = response.choices[0].message.content
            if text is None:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                return None, "Empty response after retries"
            return text, None
        except Exception as exc:
            err = str(exc)
            if attempt < MAX_RETRIES - 1 and ("429" in err or "rate_limit" in err.lower()):
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return None, err
    return None, "Max retries exceeded"


def call_vertex(genai: Any, HttpOptions: Any, model_name: str, prompt: str) -> tuple[str | None, str | None]:
    for attempt in range(MAX_RETRIES):
        try:
            client = genai.Client(
                http_options=HttpOptions(api_version="v1"),
                vertexai=True,
                project=PROJECT_ID,
                location=model_location_for(model_name),
            )
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = getattr(response, "text", None)
            if text is None:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                return None, "Empty response after retries"
            return text, None
        except Exception as exc:
            err = str(exc)
            if attempt < MAX_RETRIES - 1 and ("429" in err or "RESOURCE_EXHAUSTED" in err):
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                continue
            return None, err
    return None, "Max retries exceeded"


def init_clients(models: list[str]) -> dict[str, Any]:
    """Init and return only the clients needed for the given model list."""
    import os
    clients: dict[str, Any] = {}

    if any(is_blablador_model(m) for m in models):
        try:
            from openai import OpenAI
            key = os.environ.get("BLABLADOR_API_KEY")
            if key:
                clients["blablador"] = OpenAI(base_url=BLABLADOR_BASE_URL, api_key=key)
        except ImportError:
            pass

    if any(is_openai_model(m) for m in models):
        try:
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY")
            if key:
                clients["openai"] = OpenAI(api_key=key)
        except ImportError:
            pass

    return clients


def dispatch(model_name: str, prompt: str, clients: dict[str, Any], genai: Any = None, HttpOptions: Any = None) -> tuple[str | None, str | None]:
    """Route to the right backend based on model name."""
    if is_blablador_model(model_name):
        if "blablador" not in clients:
            return None, "Blablador client not initialized — check BLABLADOR_API_KEY env var"
        return call_blablador(clients["blablador"], model_name, prompt)
    if is_openai_model(model_name):
        if "openai" not in clients:
            return None, "OpenAI client not initialized — check OPENAI_API_KEY env var"
        return call_openai(clients["openai"], model_name, prompt)
    return call_vertex(genai, HttpOptions, model_name, prompt)
