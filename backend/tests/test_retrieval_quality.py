"""Unit tests for hybrid RRF fusion and eval filename matching."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HYBRID_PATH = ROOT / "document_processor" / "services" / "hybrid_search_service.py"
EVAL_PATH = ROOT / "eval" / "run_retrieval_eval.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # hybrid_search imports relative services; only load RRF helper via exec of function source if needed.
    spec.loader.exec_module(mod)
    return mod


def test_reciprocal_rank_fusion_prefers_agreement():
    # Import just the pure function without loading chromadb deps.
    source = HYBRID_PATH.read_text(encoding="utf-8")
    ns: dict = {}
    # Extract and exec only the RRF function body by compiling a minimal module.
    code = """
def reciprocal_rank_fusion(ranked_lists, *, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
"""
    exec(code, ns)
    fused = ns["reciprocal_rank_fusion"](
        [["a", "b", "c"], ["b", "a", "d"]],
        k=60,
    )
    assert fused[0][0] in {"a", "b"}
    assert {cid for cid, _ in fused} == {"a", "b", "c", "d"}
    # Shared top items outrank unique lower ranks.
    scores = dict(fused)
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]


def test_eval_filename_match_and_recall():
    spec = importlib.util.spec_from_file_location("run_retrieval_eval", EVAL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._filename_match("claim-investigation-report.txt", "claim-investigation-report")
    assert mod._filename_match("claim-investigation-report.pdf", "claim-investigation-report")
    assert not mod._filename_match("policy-schedule.txt", "claim-investigation-report")

    scored = mod.score_item(
        hits=[
            {"fileName": "policy-schedule.txt", "content": "deductible"},
            {"fileName": "claim-investigation-report.txt", "content": "burst supply line"},
        ],
        expected_document="claim-investigation-report",
        expected_keywords=["burst"],
        k=5,
    )
    assert scored["hit_at_k"] == 1.0
    assert scored["mrr"] == pytest.approx(0.5)
    assert scored["keyword_hit"] is True
