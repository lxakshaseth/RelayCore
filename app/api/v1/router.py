"""API v1 Router aggregating all sub-routes."""

from fastapi import APIRouter
from app.api.v1.endpoints import health, query

api_v1_router = APIRouter(prefix="/api/v1")

# Mount endpoints
api_v1_router.include_router(health.router)
api_v1_router.include_router(query.router)
