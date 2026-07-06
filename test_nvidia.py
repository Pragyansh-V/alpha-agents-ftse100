from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("NVIDIA_API_KEY")
print(f"API key loaded: {api_key[:10]}...")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    timeout=30.0
)

print("Sending request...")
try:
    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
        stream=False    # ← explicit non-streaming
    )
    print(f"✅ Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Error: {e}")