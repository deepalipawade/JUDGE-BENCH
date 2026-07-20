"""Test Vertex AI connectivity using the service account JSON (same auth as judge_memerag.py).

Usage:
    python scripts/test_vertexai_gemini.py
"""

import os

SERVICE_ACCOUNT_PATH = r"C:\Users\Deepali\Downloads\Thesis\Ivaxi LLMs\llm-juries-503009-babf6fe232e6.json"
PROJECT_ID = "llm-juries-503009"
LOCATION = "global"
MODEL = "gemini-2.5-flash"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

try:
    from google import genai
    from google.genai.types import HttpOptions
except ImportError:
    raise ImportError("Run: pip install google-genai")

print(f"Project : {PROJECT_ID}")
print(f"Location: {LOCATION}")
print(f"Model   : {MODEL}")
print(f"Auth    : {SERVICE_ACCOUNT_PATH}\n")

client = genai.Client(
    http_options=HttpOptions(api_version="v1"),
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

print("Sending test prompt...")
try:
    response = client.models.generate_content(
        model=MODEL,
        contents="Reply with exactly: 'Vertex AI connection successful.'",
    )
    print(f"OK: {response.text.strip()}")
except Exception as e:
    print(f"FAILED: {e}")
