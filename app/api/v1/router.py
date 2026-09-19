from fastapi import APIRouter
from app.api.v1.webhooks import router as webhooks_router

v1_router = APIRouter()

v1_router.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])
