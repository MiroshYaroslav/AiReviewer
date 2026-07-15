import asyncio
import os
from dotenv import load_dotenv
from google import genai
from sqlalchemy import select
from database import AsyncSessionLocal
from models import ReviewRule

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY)


async def generate_embedding(text: str) -> list[float]:
    response = await ai_client.aio.models.embed_content(
        model='text-embedding-004',
        contents=text
    )
    return response.embeddings[0].values


async def update_all_rules():
    async with AsyncSessionLocal() as db:
        stmt = select(ReviewRule).where(ReviewRule.embedding == None)
        result = await db.execute(stmt)
        rules = result.scalars().all()

        if not rules:
            print("Усі правила вже мають згенеровані вектори. Нічого оновлювати!")
            return

        print(f"Знайдено {len(rules)} правил без векторів. Починаємо генерацію...")

        for rule in rules:
            rule_text = (
                f"Rule name: {rule.name}. "
                f"Description: {rule.description}. "
                f"Bad example (anti-pattern): {rule.example_bad} "
                f"Good example: {rule.example_good}"
            )

            print(f"Векторизація правила: {rule.name}...")
            rule.embedding = await generate_embedding(rule_text)

            await asyncio.sleep(1)

        await db.commit()
        print("Усі вектори успішно згенеровані та збережені в базу даних")


if __name__ == "__main__":
    asyncio.run(update_all_rules())