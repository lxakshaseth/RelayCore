"""Health check and diagnostic endpoints."""

import time
from fastapi import APIRouter, Depends, status
from app.core.config import Settings, get_settings
from app.schemas.health import ComponentStatus, HealthCheckResponse

router = APIRouter(tags=["Health"])

START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns service availability, uptime, environment, and subsystem health.",
)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthCheckResponse:
    uptime = time.time() - START_TIME

    # Check key configurations without exposing secrets
    groq_configured = bool(settings.groq_api_key and settings.groq_api_key.strip())
    components = {
        "groq_provider": ComponentStatus(
            status="configured" if groq_configured else "missing_key",
            details={
                "model": settings.groq_model,
                "rpm_limit": str(settings.groq_rpm_limit),
                "tpm_limit": str(settings.groq_tpm_limit),
            },
        ),
        "standin_simulator": ComponentStatus(
            status="active" if settings.simulator_enabled else "idle",
            details={"configured_port": str(settings.simulator_port)},
        ),
        "queue_manager": ComponentStatus(
            status="operational",
            details={
                "max_depth": str(settings.max_queue_depth),
                "concurrency": str(settings.worker_concurrency),
            },
        ),
    }

    return HealthCheckResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.version,
        environment=settings.app_env,
        uptime_seconds=round(uptime, 2),
        components=components,
    )
