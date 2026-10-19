from fastapi import FastAPI

from enterprise_ai_workflow_agent.api.routes.data import router as data_router
from enterprise_ai_workflow_agent.api.routes.health import router as health_router
from enterprise_ai_workflow_agent.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.2.0")
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(data_router, prefix="/api/v1")
    return app


app = create_app()
