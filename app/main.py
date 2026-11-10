"""RelayCore FastAPI Application Entrypoint."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints.health import router as health_root_router
from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.context import set_current_request_id
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager handling startup and shutdown events."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level, app_env=settings.app_env)
    logger.info("Initializing %s v%s in %s mode", settings.app_name, settings.version, settings.app_env)
    
    # Startup tasks (e.g. warming indexes, starting queue workers)
    yield
    
    # Graceful shutdown tasks
    logger.info("Shutting down %s", settings.app_name)


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="A Resilient AI Orchestration and Load Management System",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and Latency Middleware
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        set_current_request_id(req_id)
        start_time = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled server exception processing request %s: %s", req_id, exc)
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "request_id": req_id},
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"
        return response

    # Mount Root Healthcheck for load balancers
    app.include_router(health_root_router, prefix="", tags=["Root Health"])

    # Mount API v1 Routes
    app.include_router(api_v1_router)

    return app


app = create_application()
