"""Run this to see the REAL error behind the Gemini API call.
Usage: python diagnose.py
"""

from google import genai
from config import get_settings

settings = get_settings()
print(f"API key loaded: {settings.gemini_api_key[:8]}... (length {len(settings.gemini_api_key)})")
print(f"Generation model: {settings.generation_model}")
print(f"Embedding model: {settings.embedding_model}")
print()

client = genai.Client(api_key=settings.gemini_api_key)

print("Testing generate_content...")
try:
    response = client.models.generate_content(
        model=settings.generation_model, contents="Say hello in one word."
    )
    print("✅ SUCCESS:", response.text)
except Exception as exc:
    print("❌ FAILED WITH REAL ERROR:")
    print(type(exc).__name__, "-", exc)