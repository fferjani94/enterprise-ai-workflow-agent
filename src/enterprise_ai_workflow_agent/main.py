from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from enterprise_ai_workflow_agent import __version__
from enterprise_ai_workflow_agent.api.routes.data import router as data_router
from enterprise_ai_workflow_agent.api.routes.health import router as health_router
from enterprise_ai_workflow_agent.api.routes.investigations import invalid_request_response
from enterprise_ai_workflow_agent.api.routes.investigations import router as investigations_router
from enterprise_ai_workflow_agent.core.config import settings
from enterprise_ai_workflow_agent.web import mount_ui

INVESTIGATIONS_PATH = "/api/v1/investigations"


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=__version__)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(investigations_router, prefix="/api/v1")

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Only the investigation API uses the structured error shape.
        if request.url.path == INVESTIGATIONS_PATH:
            return invalid_request_response(exc)
        return await request_validation_exception_handler(request, exc)

    mount_ui(app)
    return app


app = create_app()
