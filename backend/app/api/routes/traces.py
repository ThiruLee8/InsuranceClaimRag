from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.exceptions import raise_http
from app.schemas import ApiResponse, TraceListOut, TraceOut
from app.services.trace_service import get_trace_service

router = APIRouter(tags=["traces"])


@router.get("/traces", response_model=ApiResponse[TraceListOut])
def list_traces(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List recent complete chat traces for error analysis."""
    svc = get_trace_service()
    items = svc.list_traces(limit=limit, offset=offset)
    return ApiResponse(
        data=TraceListOut(
            total=svc.count(),
            items=[TraceOut.model_validate(t) for t in items],
        )
    )


@router.get("/traces/{trace_id}", response_model=ApiResponse[TraceOut])
def get_trace(trace_id: str):
    trace = get_trace_service().get_trace(trace_id)
    if not trace:
        raise_http("Trace not found", "TRACE_NOT_FOUND", 404)
    return ApiResponse(data=TraceOut.model_validate(trace))
