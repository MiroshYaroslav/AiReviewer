import jwt
import time
import httpx
import requests
import logging
from app.core.config import settings

logger = logging.getLogger("ai_reviewer.services.github")

http_client: httpx.AsyncClient | None = None


async def init_client():
    global http_client
    http_client = httpx.AsyncClient()


async def close_client():
    global http_client
    if http_client:
        await http_client.aclose()


def get_installation_access_token(installation_id: int) -> str:
    with open(settings.github_private_key_path, "rb") as f:
        private_key = f.read()

    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + (10 * 60),
        "iss": settings.github_app_id,
    }
    encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {encoded_jwt}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.post(url, headers=headers)
    response.raise_for_status()
    return response.json()["token"]


async def get_repo_file(
    repo_full_name: str, file_path: str, ref: str, token: str
) -> str | None:
    url = (
        f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}?ref={ref}"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.raw",
    }
    response = await http_client.get(url, headers=headers)
    return response.text if response.status_code == 200 else None


async def get_pr_diff(repo_full_name: str, pr_number: int, token: str) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    response = await http_client.get(url, headers=headers)
    return response.text if response.status_code == 200 else ""


async def get_pr_comments(
    repo_full_name: str, pr_number: int, token: str
) -> list[dict]:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = await http_client.get(url, headers=headers, params={"per_page": 100})
    return response.json() if response.status_code == 200 else []


async def post_review_to_github(
    repo_full_name: str,
    pr_number: int,
    commit_id: str,
    comments: list[dict],
    token: str,
) -> None:
    if not comments:
        return
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    review_comments = [
        {
            "path": c.get("file_path"),
            "line": int(c.get("line_number")),
            "body": c.get("comment"),
        }
        for c in comments
    ]
    payload = {"commit_id": commit_id, "event": "COMMENT", "comments": review_comments}

    response = await http_client.post(url, headers=headers, json=payload)
    if response.status_code not in [200, 201]:
        logger.error(f"GitHub API Error posting review: {response.text}")


async def post_reply_to_github(
    repo_full_name: str, pr_number: int, comment_id: int, body: str, token: str
) -> None:
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/comments/{comment_id}/replies"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = await http_client.post(url, headers=headers, json={"body": body})
    if response.status_code not in [200, 201]:
        logger.error(f"GitHub API Error posting reply: {response.text}")
