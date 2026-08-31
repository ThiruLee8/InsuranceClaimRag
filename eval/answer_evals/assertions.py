"""Rule-based assertion checks for answer evals (free, no LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    stem = Path(str(name)).stem.lower().strip()
    return stem.replace("_", "-").replace(" ", "-")


def filename_match(hit_name: str | None, expected: str | None) -> bool:
    if not expected:
        return True
    hit = normalize_name(hit_name)
    exp = normalize_name(expected)
    if not hit or not exp:
        return False
    return hit == exp or exp in hit or hit in exp


def _contains_any(text: str, needles: list[str] | None) -> bool:
    if not needles:
        return True
    hay = (text or "").lower()
    return any(n.lower() in hay for n in needles)


def _contains_none(text: str, needles: list[str] | None) -> bool:
    if not needles:
        return True
    hay = (text or "").lower()
    return all(n.lower() not in hay for n in needles)


def run_assertions(
    *,
    case: dict[str, Any],
    answer: str,
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Returns {passed: bool, checks: [{name, passed, detail}], score: float}.
    score = fraction of checks that passed (0..1).
    """
    asserts = case.get("assertions") or {}
    checks: list[dict[str, Any]] = []

    if "must_have_sources" in asserts:
        want = bool(asserts["must_have_sources"])
        has = len(retrieved) > 0
        ok = has if want else True  # when false, we don't require empty — refuse case uses answer text
        if want:
            checks.append(
                {
                    "name": "must_have_sources",
                    "passed": has,
                    "detail": f"retrieved={len(retrieved)}",
                }
            )

    expected_doc = asserts.get("retrieved_must_include_document") or case.get(
        "expected_document"
    )
    if asserts.get("retrieved_must_include_document") or (
        case.get("expected_document") and asserts.get("must_have_sources")
    ):
        # Only enforce doc match when explicitly requested via retrieved_must_include_document
        if asserts.get("retrieved_must_include_document"):
            expected_doc = asserts["retrieved_must_include_document"]
            hit = any(filename_match(r.get("fileName"), expected_doc) for r in retrieved)
            checks.append(
                {
                    "name": "retrieved_must_include_document",
                    "passed": hit,
                    "detail": f"expected={expected_doc}; files={[r.get('fileName') for r in retrieved]}",
                }
            )

    if asserts.get("answer_must_contain_any"):
        needles = list(asserts["answer_must_contain_any"])
        ok = _contains_any(answer, needles)
        checks.append(
            {
                "name": "answer_must_contain_any",
                "passed": ok,
                "detail": f"needles={needles}",
            }
        )

    if asserts.get("answer_must_not_contain_any"):
        needles = list(asserts["answer_must_not_contain_any"])
        ok = _contains_none(answer, needles)
        checks.append(
            {
                "name": "answer_must_not_contain_any",
                "passed": ok,
                "detail": f"forbidden={needles}",
            }
        )

    if not checks:
        checks.append(
            {
                "name": "no_assertions",
                "passed": True,
                "detail": "case had no rule assertions",
            }
        )

    passed_n = sum(1 for c in checks if c["passed"])
    score = passed_n / max(len(checks), 1)
    return {
        "passed": all(c["passed"] for c in checks),
        "score": score,
        "checks": checks,
    }


def context_recall(
    *,
    retrieved: list[dict[str, Any]],
    expected_document: str | None,
) -> float | None:
    """RAGAS-inspired context recall proxy: 1 if expected doc appears in retrieved, else 0."""
    if not expected_document:
        return None
    return 1.0 if any(filename_match(r.get("fileName"), expected_document) for r in retrieved) else 0.0


def context_precision(
    *,
    retrieved: list[dict[str, Any]],
    expected_document: str | None,
) -> float | None:
    """
    RAGAS-inspired context precision proxy:
    fraction of retrieved files that match expected document (when expected is set).
    """
    if not expected_document or not retrieved:
        return None
    hits = sum(1 for r in retrieved if filename_match(r.get("fileName"), expected_document))
    return hits / len(retrieved)
