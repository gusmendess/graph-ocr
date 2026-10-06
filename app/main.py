from fastapi import FastAPI

from app.api.config import APP_NAME, app_version, configure_logging
from app.api.endpoints import compare, health, search

configure_logging()

app = FastAPI(title=APP_NAME, version=app_version())
app.include_router(health.router)
app.include_router(compare.router)
app.include_router(search.router)
