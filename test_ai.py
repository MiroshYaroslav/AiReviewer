import asyncio
import os

import httpx
from google import genai

API_KEY = str(os.getenv("GEMINI_API_KEY", "")).strip()


# Зменшили одночасну кількість запитів, щоб м'якше проходити ліміти
CONCURRENCY_LIMIT = 2


async def fetch_available_models() -> list[str]:
    """Отримує список усіх доступних моделей."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

    print("📡 [СТАДІЯ 0] Отримання списку моделей...")
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            models = response.json().get("models", [])
            valid_models = [
                m["name"].replace("models/", "")
                for m in models
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            print(f"✅ Знайдено {len(valid_models)} моделей для перевірки.\n")
            return valid_models
        else:
            print(f"❌ Помилка отримання моделей: {response.status_code}")
            return []


async def generate_with_retry(client: genai.Client, model_name: str, contents: str, max_retries: int = 4):
    """Обгортка для запитів з експоненційною затримкою (Exponential Backoff)."""
    for attempt in range(max_retries):
        try:
            # Намагаємося зробити запит
            return await client.aio.models.generate_content(
                model=model_name,
                contents=contents
            )
        except Exception as e:
            error_msg = str(e)
            # Якщо це помилка Rate Limit (429)
            if "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 5 * (2 ** attempt)  # 5, 10, 20 секунд...
                    print(
                        f"   ⏳ [{model_name}] Rate Limit 429! Спроба {attempt + 1}/{max_retries}. Чекаємо {wait_time} сек...")
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception("Здався після всіх спроб обійти Rate Limit.")
            # Якщо інша помилка (404, 403, 400) — не чекаємо, одразу викидаємо
            else:
                raise e


async def test_model(model_name: str, client: genai.Client, semaphore: asyncio.Semaphore) -> tuple[str, bool, str]:
    """Виконує перевірку моделі з детальним логуванням."""
    async with semaphore:
        print(f"▶️ [{model_name}] Початок тестування...")
        try:
            # --- СТАДІЯ 1: Ping ---
            ping_response = await generate_with_retry(
                client, model_name, "Ping. Дай відповідь лише одним словом: Pong."
            )

            if not ping_response.text:
                print(f"   ❌ [{model_name}] Провал Stage 1: Пуста відповідь.")
                return model_name, False, "Пуста відповідь на етапі Ping"

            print(f"   ✔️ [{model_name}] Stage 1 (Ping) пройдено успішно.")

            # --- СТАДІЯ 2: Heavy Load ---
            heavy_prompt = f"""
            Ти — аналізатор систем. Знайди 3 помилки в цьому блоці тексту. 
            Текст для аналізу: {' '.join(['async wait lock loop connection'] * 50)}
            Поверни результат у форматі JSON.
            """

            load_response = await generate_with_retry(client, model_name, heavy_prompt)

            if load_response.text:
                print(f"   🏆 [{model_name}] Stage 2 (Load) пройдено успішно!")
                return model_name, True, "Пройшла всі тести"
            else:
                print(f"   ❌ [{model_name}] Провал Stage 2: Пуста відповідь.")
                return model_name, False, "Не впоралась із навантаженням"

        except Exception as e:
            error_msg = str(e)
            print(f"   🚫 [{model_name}] Відхилено: {error_msg[:60]}...")
            if "404" in error_msg:
                return model_name, False, "404: Недоступна для цього ключа/регіону"
            elif "400" in error_msg:
                return model_name, False, "400: Модель не підтримує такий тип запиту"
            else:
                return model_name, False, f"Помилка: {error_msg[:50]}"


async def main():
    models = await fetch_available_models()
    if not models:
        return

    ai_client = genai.Client(api_key=API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print("🚀 Запуск інтелектуального стрес-тесту з Backoff-алгоритмом...\n")

    tasks = [test_model(model, ai_client, semaphore) for model in models]
    results = await asyncio.gather(*tasks)

    working_models = []
    failed_models = []

    for model_name, success, message in results:
        if success:
            working_models.append(model_name)
        else:
            failed_models.append((model_name, message))

    print("\n" + "=" * 60)
    print("❌ ВІДХИЛЕНІ МОДЕЛІ:")
    for m, reason in failed_models:
        print(f" - {m}: {reason}")

    print("\n" + "=" * 60)
    print("✅ СТАБІЛЬНІ РОБОЧІ МОДЕЛІ:")
    for m in working_models:
        print(f" + {m}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())