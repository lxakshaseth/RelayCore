"""Pydantic schemas for health and system status endpoints."""

from datetime import datetime, timezone
from typing import Dict, Optional
from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    """Status details for an individual subsystem."""

    status: str = Field(description="Operational status, e.g., 'healthy', 'degraded', 'unavailable'")
    details: Optional[Dict[str, str]] = Field(default=None, description="Diagnostic metadata")


class HealthCheckResponse(BaseModel):
    """System health check payload."""

    status: str = Field(default="healthy", description="Overall system health status")
    app_name: str = Field(description="Name of the service")
    version: str = Field(description="Application version")
    environment: str = Field(description="Execution environment")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Current server timestamp (UTC)",
    )
    components: Dict[str, ComponentStatus] = Field(
        default_factory=dict,
        description="Health status of downstream components (LLM provider, retrieval index, queue)",
    )
