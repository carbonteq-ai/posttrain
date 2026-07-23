"""ASGI application for production servers."""

from .composition import create_service
from .http import create_http_app
from .settings import ObservatorySettings

settings = ObservatorySettings.from_env()
app = create_http_app(create_service(settings), settings)

__all__ = ["app"]
