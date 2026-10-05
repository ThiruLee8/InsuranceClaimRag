"""Week 11 — spans, cost, support drill, and the failure-to-test loop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.services.production import (
    DRILL_NOW,
    CAPABLE_MODEL,
    CHEAP_MODEL,
    build_report,
    failure_guard,
    load_failure,
    measure,
    promote_failure,
    support_pool,
    support_search,
)
from app.services.trace_service import TraceService


def test_improved_path_costs_less_and_keeps_the_answer():
    baseline = measure("baseline")
    improved = measure("improved")
    assert improved["meanCostUsd"] < baseline["meanCostUsd"] * 0.7
    assert improved["meanBilledTokens"] < baseline["meanBilledTokens"]
    assert improved["meanElapsedMs"] < baseline["meanElapsedMs"]
    base_answers = [item["answer"] for item in baseline["items"]]
    new_answers = [item["answer"] for item in improved["items"]]
    assert base_answers == new_answers


def test_each_request_logs_time_and_cost_per_step():
    improved = measure("improved")
    generate = next(item for item in improved["items"] if item["route"] == "cheap")
    names = [span["name"] for span in generate["spans"]]
    assert "route" in names
    assert "retrieve" in names
    assert "generate" in names
    elapsed = round(sum(span["elapsedMs"] for span in generate["spans"]), 1)
    cost = round(sum(span["costUsd"] for span in generate["spans"]), 8)
    assert generate["elapsedMs"] == elapsed
    assert generate["costUsd"] == cost
    step_costs = [span["costUsd"] for span in generate["spans"] if span["name"] == "generate"]
    assert step_costs[0] > 0
    cached = next(item for item in improved["items"] if item["route"] == "cache")
    assert all(span["name"] != "generate" for span in cached["spans"])
    assert cached["costUsd"] < generate["costUsd"]


def test_prompt_cache_and_cheap_model_show_on_the_spans():
    improved = measure("improved")
    first = improved["items"][0]
    later = next(
        item
        for item in improved["items"]
        if item["route"] == "cheap" and item["id"] != first["id"]
    )
    first_gen = next(span for span in first["spans"] if span["name"] == "generate")
    later_gen = next(span for span in later["spans"] if span["name"] == "generate")
    assert first_gen["cachedInputTokens"] == 0
    assert later_gen["cachedInputTokens"] > 0
    assert later_gen["model"] == CHEAP_MODEL
    hard = next(item for item in improved["items"] if item["id"] == "compare-figures")
    assert hard["model"] == CAPABLE_MODEL


def test_rate_limit_falls_back_and_records_why():
    report = build_report()
    second = report["fallback"]["second"]
    assert second["fallbackFrom"] == CHEAP_MODEL
    assert second["model"] == CAPABLE_MODEL
    assert "rate_limit" in second["reason"]


def test_support_drill_finds_one_bad_answer_among_a_thousand():
    now = DRILL_NOW
    logs = support_pool(now=now, extras=1000)
    assert len(logs) > 1000
    found = support_search(
        logs,
        "yesterday it gave the wrong investigator name",
        now=now,
    )
    assert found["scanned"] == len(logs)
    assert found["restrictedToDay"] is True
    assert found["hits"][0]["requestId"] == "req-bad-investigator"
    assert "Chen" in found["hits"][0]["answer"]
    # A different day does not hide behind the same words.
    older = support_search(logs, "the water backup endorsement", now=now)
    assert older["hits"][0]["question"].startswith("Is there a water backup")


def test_promoted_failure_rejects_the_bad_answer_and_accepts_the_fix():
    case = load_failure()
    bad = failure_guard(case["badAnswer"], case["context"], case)
    fixed = failure_guard(case["fixedAnswer"], case["context"], case)
    assert bad["ok"] is False
    assert "Chen" in bad["invented"][0] or "Chen" in bad["forbidden"]
    assert fixed["ok"] is True
    assert case["sourceTraceId"] == "seed-009-gen-halluc"


def test_promote_writes_a_case_the_guard_fails(tmp_path: Path):
    case = promote_failure(
        {
            "traceId": "trace-abc",
            "question": "Who investigated claim CLM-2024-00847?",
            "answer": "Investigated by M. Chen.",
            "context": "Investigated by field adjuster A. Ramirez.",
        },
        note="Said Chen. Context says Ramirez.",
        must_contain=["Ramirez"],
        must_not_contain=["Chen"],
        fixed_answer="Investigated by field adjuster A. Ramirez.",
        directory=tmp_path,
        case_id="from-trace-abc",
    )
    written = tmp_path / "from-trace-abc.json"
    assert written.exists()
    assert failure_guard(case["badAnswer"], case["context"], case)["ok"] is False


def test_promote_refuses_an_answer_that_already_passes(tmp_path: Path):
    try:
        promote_failure(
            {
                "traceId": "trace-ok",
                "question": "Who investigated claim CLM-2024-00847?",
                "answer": "Investigated by field adjuster A. Ramirez.",
                "context": "Investigated by field adjuster A. Ramirez.",
            },
            note="This one was fine.",
            must_contain=["Ramirez"],
            must_not_contain=["Chen"],
            fixed_answer="Investigated by field adjuster A. Ramirez.",
            directory=tmp_path,
            case_id="should-not-write",
        )
    except ValueError as exc:
        assert "already accepts" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert not (tmp_path / "should-not-write.json").exists()


def test_ten_x_plan_names_generate_as_the_first_limit():
    report = build_report()
    plan = report["tenX"]
    assert plan["breaksFirst"] == "generate"
    assert plan["generateShareOfTime"] > 0.5
    assert "second model process" in " ".join(plan["plan"])
    assert report["savings"]["costPct"] > 30


def test_trace_log_keeps_spans_for_replay(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ENABLE_TRACE_LOGGING", "true")
    from app.core.config import get_settings

    get_settings.cache_clear()
    svc = TraceService(traces_dir=tmp_path)
    object.__setattr__(svc.settings, "enable_trace_logging", True)
    trace = svc.record(
        question="What deductible applies?",
        answer="USD 1,000",
        sources=[{"content": "All Peril Deductible: USD 1,000", "file_name": "policy-schedule.txt"}],
        model="llama3.2",
        spans=[
            {"name": "retrieve", "elapsedMs": 45.0, "costUsd": 0.000001, "inputTokens": 6},
            {"name": "generate", "elapsedMs": 320.0, "costUsd": 0.0004, "inputTokens": 200, "outputTokens": 8},
        ],
    )
    loaded = svc.get_trace(trace["traceId"])
    assert loaded is not None
    assert [span["name"] for span in loaded["spans"]] == ["retrieve", "generate"]
    assert loaded["elapsedMs"] == 365.0
    assert loaded["costUsd"] > loaded["spans"][0]["costUsd"]
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert loaded["timestamp"][:10] >= yesterday[:10] or loaded["question"]
    get_settings.cache_clear()
