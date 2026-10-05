"""Production report, support drill, and the failure-to-test loop."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.exceptions import raise_http
from app.schemas import ApiResponse
from app.services.production import (
    DRILL_NOW,
    build_report,
    load_failures,
    promote_failure,
    support_pool,
    support_search,
    traces_as_logs,
)
from app.services.trace_service import get_trace_service

router = APIRouter(prefix="/production", tags=["production"])


class SupportQuery(BaseModel):
    complaint: str = Field(min_length=1, max_length=500)


class PromoteFailureIn(BaseModel):
    traceId: str
    note: str = Field(min_length=1, max_length=1000)
    mustContain: list[str] = Field(default_factory=list)
    mustNotContain: list[str] = Field(default_factory=list)
    fixedAnswer: str = Field(min_length=1, max_length=4000)
    caseId: Optional[str] = None


@router.get("/report", response_model=ApiResponse[dict])
def production_report():
    """Measured cost before and after caching, routing, and fallbacks."""
    return ApiResponse(data=build_report())


@router.post("/support", response_model=ApiResponse[dict])
def production_support(payload: SupportQuery):
    """Find one past answer from a vague complaint."""
    logs = support_pool(now=DRILL_NOW)
    live = traces_as_logs(get_trace_service().list_traces(limit=500))
    # Live rows use their own timestamps. The seeded drill is anchored to DRILL_NOW
    # so "yesterday" stays reproducible in the recorded log.
    result = support_search(logs, payload.complaint, now=DRILL_NOW)
    if live:
        live_result = support_search(live, payload.complaint)
        result["liveScanned"] = live_result["scanned"]
        result["liveHits"] = live_result["hits"]
    else:
        result["liveScanned"] = 0
        result["liveHits"] = []
    return ApiResponse(data=result)


@router.get("/failures", response_model=ApiResponse[dict])
def production_failures():
    return ApiResponse(data={"items": load_failures()})


@router.post("/failures", response_model=ApiResponse[dict])
def production_promote(payload: PromoteFailureIn):
    trace = get_trace_service().get_trace(payload.traceId)
    if not trace:
        raise_http("Trace not found", "TRACE_NOT_FOUND", 404)
    try:
        case = promote_failure(
            trace,
            note=payload.note,
            must_contain=payload.mustContain,
            must_not_contain=payload.mustNotContain,
            fixed_answer=payload.fixedAnswer,
            case_id=payload.caseId,
        )
    except ValueError as exc:
        raise_http(str(exc), "FAILURE_NOT_A_MISS", 400)
    return ApiResponse(data=case)
