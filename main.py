import os
import re
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from google import genai
from sqlalchemy import select, or_
from database import AsyncSessionLocal
from models import ReviewRule, Technology


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


load_dotenv(override=True)

GITHUB_TOKEN = str(os.getenv("GITHUB_TOKEN", "")).strip()
GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY", "")).strip()

app = FastAPI(title="AI Code Reviewer", lifespan=lifespan)
ai_client = genai.Client(api_key=GEMINI_API_KEY)


async def get_repo_file(repo_full_name: str, file_path: str, ref: str) -> str | None:
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}?ref={ref}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.raw"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.text if response.status_code == 200 else None


def parse_requirements(req_text: str) -> list[str]:
    if not req_text: return []
    techs = set()
    for line in req_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            match = re.match(r"^([a-zA-Z0-9_\-]+)", line)
            if match: techs.add(match.group(1).lower())
    return list(techs)


async def get_pr_diff(repo_full_name: str, pr_number: int) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3.diff"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.text if response.status_code == 200 else ""


async def analyze_code_with_ai(diff_text: str, rules: list[ReviewRule]) -> str | None:
    rules_text = "\n".join(
        [f"{r.name}: {r.description}\nBad: {r.example_bad}\nGood: {r.example_good}\n" for r in rules]) or "Follow PEP8."
    prompt = f"Review this PR based on these rules:\n{rules_text}\n\nDiff:\n{diff_text}"
    try:
        response = await ai_client.aio.models.generate_content(model='gemini-flash-lite-latest', contents=prompt)
        return response.text
    except Exception as e:
        return f"AI analysis failed: {str(e)}"


async def process_pull_request(payload: dict) -> None:
    action = str(payload.get("action", ""))
    if action not in ["opened", "synchronize"]:
        return

    pr_data = payload.get("pull_request", {})
    repo_data = payload.get("repository", {})
    pr_number = pr_data.get("number")
    repo_full_name = repo_data.get("full_name")
    head_sha = pr_data.get("head", {}).get("sha")

    diff_text = await get_pr_diff(repo_full_name, pr_number)
    if not diff_text:
        print(f"No diff found for PR #{pr_number}")
        return

    diff_vector = await generate_embedding(diff_text)

    async with AsyncSessionLocal() as db:
        req_text = await get_repo_file(repo_full_name, "requirements.txt", head_sha)
        tech_names = parse_requirements(req_text)

        conditions = [ReviewRule.is_global == True]
        if tech_names:
            conditions.append(Technology.name.in_(tech_names))

        stmt = select(ReviewRule).outerjoin(ReviewRule.technologies).where(
            ReviewRule.is_active == True,
            or_(*conditions)
        )

        if diff_vector:
            stmt = stmt.order_by(ReviewRule.embedding.cosine_distance(diff_vector)).limit(3)
        else:
            stmt = stmt.limit(3)

        result = await db.execute(stmt)
        rules = list(result.scalars().all())

        review = await analyze_code_with_ai(diff_text, rules)
        if review:
            await post_comment_to_github(repo_full_name, pr_number, review)


async def post_comment_to_github(repo_full_name: str, pr_number: int, comment: str) -> None:
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json={"body": comment})


async def generate_embedding(text: str) -> list[float] | None | list[Any]:
    try:
        response = await ai_client.aio.models.embed_content(
            model='gemini-embedding-2',
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Помилка генерації вектора: {e}")
        return []

@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    if "pull_request" in payload:
        background_tasks.add_task(process_pull_request, payload)
        return {"status": "accepted"}
    return {"status": "ignored"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)