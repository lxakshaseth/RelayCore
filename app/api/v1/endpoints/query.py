"""Query execution endpoints for RelayCore."""

import uuid
from fastapi import APIRouter, Depends, Request, status
from app.core.context import get_current_request_id
from app.schemas.query import QueryRequest, QueryResponse
from app.services.orchestrator import OrchestrationPipeline, get_pipeline

router = APIRouter(tags=["Query Pipeline"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Grounded Question Answering Pipeline",
    description="Processes questions through the 12-stage RelayCore pipeline with citation verification and rate-limiting.",
)
async def execute_query(
    request_body: QueryRequest,
    pipeline: OrchestrationPipeline = Depends(get_pipeline),
) -> QueryResponse:
    request_id = get_current_request_id() or f"req_{uuid.uuid4().hex[:12]}"
    response = await pipeline.process_query(
        request=request_body,
        request_id=request_id,
        scenario="api_call",
    )
    return response
