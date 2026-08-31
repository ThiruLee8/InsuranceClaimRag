"""Unit tests for TraceService JSONL recording."""

from __future__ import annotations

from pathlib import Path

from app.services.trace_service import TraceService


def test_trace_service_records_replayable_fields(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ENABLE_TRACE_LOGGING", "true")
    from app.core.config import get_settings

    get_settings.cache_clear()
    svc = TraceService(traces_dir=tmp_path)
    # Force enabled regardless of env caching quirks in test process
    object.__setattr__(svc.settings, "enable_trace_logging", True)

    trace = svc.record(
        question="What deductible applies?",
        answer="USD 1,000",
        sources=[
            {
                "document_id": "d1",
                "chunk_id": "c1",
                "file_name": "policy-schedule.txt",
                "page_number": 1,
                "score": 0.9,
                "content": "All Peril Deductible: USD 1,000",
            }
        ],
        conversation_id="conv-1",
        message_id="msg-1",
        agent_id="claims_assistant",
        model="llama3.2",
        original_question="What deductible applies?",
        search_query="policy deductible amount",
        search_mode="hybrid",
    )
    assert trace is not None
    assert (tmp_path / "traces.jsonl").exists()
    loaded = svc.get_trace(trace["traceId"])
    assert loaded is not None
    assert loaded["question"] == "What deductible applies?"
    assert loaded["retrieved"][0]["content"].startswith("All Peril")
    assert loaded["answer"] == "USD 1,000"
    get_settings.cache_clear()
