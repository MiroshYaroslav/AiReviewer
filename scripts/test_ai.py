import asyncio
import os
import sys
import logging
import httpx
from dotenv import load_dotenv
from google import genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_tester")

load_dotenv(override=True)
API_KEY = str(os.getenv("GEMINI_API_KEY", "")).strip()
CONCURRENCY_LIMIT = 2


async def fetch_available_models() -> list[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    logger.info("Fetching available models...")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                models = response.json().get("models", [])
                valid_models = [
                    m["name"].replace("models/", "")
                    for m in models
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                logger.info(f"Found {len(valid_models)} models for testing.")
                return valid_models
            else:
                logger.error(f"Failed to fetch models: HTTP {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Network error while fetching models: {e}", exc_info=True)
            return []


async def generate_with_retry(
    client: genai.Client, model_name: str, contents: str, max_retries: int = 4
):
    for attempt in range(max_retries):
        try:
            return await client.aio.models.generate_content(
                model=model_name, contents=contents
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 5 * (2**attempt)
                    logger.warning(
                        f"[{model_name}] Rate Limit 429. Attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise Exception("Exhausted retries due to Rate Limiting.")
            else:
                raise e


async def test_model(
    model_name: str, client: genai.Client, semaphore: asyncio.Semaphore
) -> tuple[str, bool, str]:
    async with semaphore:
        logger.info(f"[{model_name}] Starting test sequence...")
        try:
            ping_response = await generate_with_retry(
                client, model_name, "Ping. Reply with a single word: Pong."
            )

            if not ping_response.text:
                logger.warning(f"[{model_name}] Failed Stage 1: Empty response.")
                return model_name, False, "Empty response on Ping"

            heavy_prompt = f"""
            System analyzer. Find 3 errors in this text block.
            Text for analysis: {' '.join(['async wait lock loop connection'] * 50)}
            Return result in JSON format.
            """

            load_response = await generate_with_retry(client, model_name, heavy_prompt)

            if load_response.text:
                logger.info(f"[{model_name}] Passed all stages.")
                return model_name, True, "Passed all tests"
            else:
                logger.warning(f"[{model_name}] Failed Stage 2: Empty response.")
                return model_name, False, "Failed under load"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{model_name}] Rejected: {error_msg[:60]}")
            if "404" in error_msg:
                return model_name, False, "404: Not available for this key/region"
            elif "400" in error_msg:
                return model_name, False, "400: Unsupported request type"
            else:
                return model_name, False, f"Error: {error_msg[:50]}"


async def main():
    if not API_KEY:
        logger.error("API Key is missing.")
        return

    models = await fetch_available_models()
    if not models:
        return

    ai_client = genai.Client(api_key=API_KEY)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    logger.info("Initiating model stress test with backoff algorithm...")

    tasks = [test_model(model, ai_client, semaphore) for model in models]
    results = await asyncio.gather(*tasks)

    working_models = []
    failed_models = []

    for model_name, success, message in results:
        if success:
            working_models.append(model_name)
        else:
            failed_models.append((model_name, message))

    logger.info("--- REJECTED MODELS ---")
    for m, reason in failed_models:
        logger.info(f"{m}: {reason}")

    logger.info("--- STABLE MODELS ---")
    for m in working_models:
        logger.info(m)


if __name__ == "__main__":
    asyncio.run(main())
