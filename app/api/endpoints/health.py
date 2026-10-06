from fastapi import APIRouter

from app.api.config import APP_NAME, app_version

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"service": APP_NAME, "version": app_version(), "status": "ok"}
