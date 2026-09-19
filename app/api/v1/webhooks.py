from fastapi import APIRouter, Request, BackgroundTasks
from app.services.review import process_pull_request, process_review_comment

router = APIRouter()


@router.post("/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    if "pull_request" in payload and payload.get("action") in ["opened", "synchronize"]:
        background_tasks.add_task(process_pull_request, payload)
        return {"status": "accepted_pr"}
    elif (
        "comment" in payload
        and "pull_request" in payload
        and payload.get("action") == "created"
    ):
        background_tasks.add_task(process_review_comment, payload)
        return {"status": "accepted_comment"}
    return {"status": "ignored"}
