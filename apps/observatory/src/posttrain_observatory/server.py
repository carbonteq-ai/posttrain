"""In-process Observatory server entry point for framework composition."""

from __future__ import annotations

import uvicorn

from .composition import create_service
from .http import create_http_app
from .settings import ObservatorySettings


def serve(settings: ObservatorySettings) -> None:
    """Serve the configured Observatory HTTP API, MCP endpoint, and frontend."""

    uvicorn.run(
        create_http_app(create_service(settings), settings),
        host=settings.host,
        port=settings.port,
    )


__all__ = ["serve"]
