"""Test connectivity to Blablador API (OpenAI-compatible, Helmholtz/JSC)."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.blablador.fz-juelich.de/v1/",
    api_key=os.environ["BLABLADOR_API_KEY"],
)

# List available models
print("=== Available models ===")
try:
    models = sorted(m.id for m in client.models.list().data)
    for m in models:
        print(f"  {m}")
except Exception as e:
    print(f"  Could not list models: {e}")

# Test a quick call — verify each model gives a response
PROMPT = "Reply with exactly: 'API connection successful.'"

# Keywords to match against the available model list (case-insensitive)
KEYWORDS = [
    "gpt-oss-120b",
    "minimax",
    "qwen3.5",
    "qwen3.6",
    "apertus",
    "eve-instruct",
]

def find_models(available: list[str], keywords: list[str]) -> list[str]:
    matched = []
    for kw in keywords:
        hits = [m for m in available if kw.lower() in m.lower()]
        if hits:
            matched.append(hits[0])
            print(f"  Matched '{kw}' → {hits[0]}")
        else:
            print(f"  No match for '{kw}'")
    return matched

print("\n=== Resolving model names ===")
try:
    available_models = sorted(m.id for m in client.models.list().data)
    TEST_MODELS = find_models(available_models, KEYWORDS)
except Exception as e:
    print(f"Could not list models: {e}")
    TEST_MODELS = []

print("\n=== Connectivity test ===")
results = []
for model in TEST_MODELS:
    print(f"\nTesting {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=2000,
            temperature=0,
        )
        content = response.choices[0].message.content
        if content:
            print(f"  OK: {content.strip()[:100]}")
            results.append((model, "OK"))
        else:
            print(f"  OK (empty content) — finish_reason: {response.choices[0].finish_reason}")
            results.append((model, "OK (empty content)"))
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append((model, f"FAILED"))

print("\n=== Summary ===")
for model, status in results:
    print(f"  {'✓' if status.startswith('OK') else '✗'} {model}: {status}")
