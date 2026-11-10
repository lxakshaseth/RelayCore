"""API v1 Router aggregating all sub-routes."""

from fastapi import APIRouter
from app.api.v1.endpoints import health

api_v1_router = APIRouter(prefix="/api/v1")

# Mount health routes
api_v1_router.include_router(health.router)
