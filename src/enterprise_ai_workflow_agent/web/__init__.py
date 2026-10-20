"""Serves the static OpsPilot web interface."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).parent

# No inline script or style is used, and nothing is loaded from other origins.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)


def mount_ui(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            WEB_DIR / "index.html",
            headers={
                "Content-Security-Policy": CONTENT_SECURITY_POLICY,
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )
