import os
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Шукаю всі моделі для векторів...")
try:
    models = client.models.list()
    for m in models:
        if "embed" in m.name.lower():
            print(f"Доступна модель: {m.name}")
except Exception as e:
    print(f"Помилка доступу до API: {e}")