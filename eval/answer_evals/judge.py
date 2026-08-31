"""
LLM-as-judge helpers (RAGAS-inspired faithfulness / answer relevancy).

Uses Ollama HTTP API. Binary pass/fail plus 1–10 score (G-Eval style), parsed from JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FAITHFULNESS_SYSTEM = (
    "You are an evaluation judge for an insurance claims RAG assistant. "
    "Score whether the ANSWER is faithful to the CONTEXT (no invented facts). "
    "Respond with JSON only: "
    '{"score": <int 1-10>, "pass": <true|false>, "reason": "<one sentence>"}'
)

RELEVANCY_SYSTEM = (
    "You are an evaluation judge for an insurance claims RAG assistant. "
    "Score whether the ANSWER is relevant and helpful for the QUESTION. "
    "Respond with JSON only: "
    '{"score": <int 1-10>, "pass": <true|false>, "reason": "<one sentence>"}'
)


def _post_ollama(
    *,
    base_url: str,
    model: str,
    system: str,
    prompt: str,
    timeout: float = 180.0,
) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
        "format": "json",
    }
    raw = json.dumps(payload).encode("utf-8")
    req = Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip()


def _parse_judge_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"score": 1, "pass": False, "reason": f"unparseable judge output: {text[:120]}"}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"score": 1, "pass": False, "reason": f"unparseable judge output: {text[:120]}"}

    score = data.get("score", 1)
    try:
        score_i = int(score)
    except (TypeError, ValueError):
        score_i = 1
    score_i = max(1, min(10, score_i))
    passed = data.get("pass")
    if passed is None:
        passed = score_i >= 7
    return {
        "score": score_i,
        "pass": bool(passed),
        "reason": str(data.get("reason") or "").strip(),
    }


def judge_faithfulness(
    *,
    question: str,
    answer: str,
    context: str,
    ollama_base: str,
    model: str,
) -> dict[str, Any]:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Is the answer faithful to the context? JSON only."
    )
    raw = _post_ollama(
        base_url=ollama_base, model=model, system=FAITHFULNESS_SYSTEM, prompt=prompt
    )
    result = _parse_judge_json(raw)
    result["criterion"] = "faithfulness"
    result["raw"] = raw
    return result


def judge_answer_relevancy(
    *,
    question: str,
    answer: str,
    ollama_base: str,
    model: str,
) -> dict[str, Any]:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Is the answer relevant and helpful? JSON only."
    )
    raw = _post_ollama(
        base_url=ollama_base, model=model, system=RELEVANCY_SYSTEM, prompt=prompt
    )
    result = _parse_judge_json(raw)
    result["criterion"] = "answer_relevancy"
    result["raw"] = raw
    return result


def run_judges(
    *,
    case: dict[str, Any],
    question: str,
    answer: str,
    retrieved: list[dict[str, Any]],
    ollama_base: str,
    model: str,
) -> dict[str, Any]:
    cfg = case.get("judge") or {}
    if not cfg.get("enabled"):
        return {"enabled": False, "passed": True, "score": None, "criteria": []}

    context = "\n\n".join(
        f"[{i}] {r.get('fileName')}: {r.get('content') or ''}"
        for i, r in enumerate(retrieved, start=1)
    ) or "(no context)"

    criteria = list(cfg.get("criteria") or [])
    threshold = float(cfg.get("pass_threshold") or 0.7)
    results: list[dict[str, Any]] = []

    for name in criteria:
        try:
            if name == "faithfulness":
                results.append(
                    judge_faithfulness(
                        question=question,
                        answer=answer,
                        context=context,
                        ollama_base=ollama_base,
                        model=model,
                    )
                )
            elif name in {"answer_relevancy", "relevancy"}:
                results.append(
                    judge_answer_relevancy(
                        question=question,
                        answer=answer,
                        ollama_base=ollama_base,
                        model=model,
                    )
                )
            else:
                results.append(
                    {
                        "criterion": name,
                        "score": 1,
                        "pass": False,
                        "reason": f"unknown criterion: {name}",
                    }
                )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            results.append(
                {
                    "criterion": name,
                    "score": 1,
                    "pass": False,
                    "reason": f"judge call failed: {exc}",
                }
            )

    if not results:
        return {"enabled": True, "passed": True, "score": 1.0, "criteria": []}

    # Normalize 1–10 → 0–1 for aggregation
    norm_scores = [r["score"] / 10.0 for r in results]
    avg = sum(norm_scores) / len(norm_scores)
    passed = avg >= threshold and all(r.get("pass") for r in results)
    return {
        "enabled": True,
        "passed": passed,
        "score": avg,
        "threshold": threshold,
        "criteria": results,
    }
