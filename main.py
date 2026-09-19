import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.router import api_router
from app.services import github
from app.services.ai import check_ai_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("ai_reviewer.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated.")

    await github.init_client()

    ai_ok = await check_ai_health()
    if not ai_ok:
        logger.error("AI service health check failed. Webhooks may fail to process.")

    yield

    logger.info("Application shutdown initiated.")

    await github.close_client()


app = FastAPI(title="AI Code Reviewer", lifespan=lifespan)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
