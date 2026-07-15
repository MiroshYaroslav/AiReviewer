import asyncio
import json
import os
import re
from dotenv import load_dotenv
from google import genai
from sqlalchemy import select
from database import AsyncSessionLocal
from models import ReviewRule, Technology

load_dotenv(override=True)
GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY", ""))
ai_client = genai.Client(api_key=GEMINI_API_KEY)


async def generate_embedding(text: str) -> list[float]:
    """Generates a vector for the rule text."""
    response = await ai_client.aio.models.embed_content(
        model='gemini-embedding-2',
        contents=text
    )

    if response.embeddings and len(response.embeddings) > 0:
        return response.embeddings[0].values
    return []


def extract_json(text: str) -> str:
    """Extracts clean JSON from AI response, ignoring markdown formatting."""
    match = re.search(r'\[.*]', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


async def ai_seed_database() -> None:
    prompt = """
    You are an Expert Backend Developer. We are building a database of code review rules.
    Search your knowledge base for 5 crucial best practices, common anti-patterns, and architectural rules for Python backend development (specifically focusing on Python, FastAPI, and SQLAlchemy).

    Return the result STRICTLY as a JSON array. 
    Each object must have exactly these keys:
    - "name": string (short rule name)
    - "description": string (detailed explanation)
    - "example_bad": string (code snippet showing anti-pattern)
    - "example_good": string (code snippet showing correct approach)
    - "technologies": array of strings (e.g., ["python", "sqlalchemy"], use lowercase)
    """

    print("AI is searching for best practices and generating rules...")

    try:
        response = await ai_client.aio.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )

        text_content = response.text if response.text else ""
        raw_json = extract_json(text_content)
        rules_data = json.loads(raw_json)

        print(f"Found {len(rules_data)} rules. Starting database write...")

        async with AsyncSessionLocal() as db:
            for rule_data in rules_data:
                tech_objects = []
                for tech_name in rule_data.get("technologies", []):
                    tech_name = str(tech_name).lower().strip()
                    stmt = select(Technology).where(Technology.name == tech_name)
                    tech = (await db.execute(stmt)).scalar_one_or_none()
                    if not tech:
                        tech = Technology(name=tech_name)
                        db.add(tech)
                        await db.commit()
                        await db.refresh(tech)
                    tech_objects.append(tech)

                rule_text = (
                    f"Rule: {rule_data.get('name', '')}. "
                    f"Desc: {rule_data.get('description', '')}. "
                    f"Bad: {rule_data.get('example_bad', '')} "
                    f"Good: {rule_data.get('example_good', '')}"
                )

                print(f"Vectorizing rule: {rule_data.get('name')}...")
                embedding_data = await generate_embedding(rule_text)

                new_rule = ReviewRule(
                    name=str(rule_data.get('name', '')),
                    description=str(rule_data.get('description', '')),
                    example_bad=str(rule_data.get('example_bad', '')),
                    example_good=str(rule_data.get('example_good', '')),
                    is_global=False,
                    embedding=embedding_data if embedding_data else None,
                    technologies=tech_objects
                )
                db.add(new_rule)
                await asyncio.sleep(1)

            await db.commit()
            print("Successfully saved all generated rules and embeddings!")

    except Exception as e:
        print(f"Error during generation or writing: {e}")


if __name__ == "__main__":
    asyncio.run(ai_seed_database())