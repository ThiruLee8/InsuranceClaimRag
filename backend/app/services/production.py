"""Production observability, cost, and the failure-to-test loop.

A request is a trace. Each step is a span with its own time and cost.
LangSmith, Phoenix, and an OpenTelemetry collector display this same shape;
this module keeps the record locally so a past answer can still be found
when those accounts are not configured.

Cost uses hosted-equivalent prices. Local Ollama does not send a token bill.
The prices make caching and model choice visible in dollars. Billed tokens
are recorded beside the dollars so the saving does not depend on the price card.

The cheap model is a stand-in for a LiteLLM router entry: a model name, a
price, a rate limit, and a fallback. The decision is written on the route span.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.services.agents import CLAIMS_ASSISTANT_PROMPT

# USD per 1,000,000 tokens. Hosted-equivalent, not an Ollama invoice.
PRICES: dict[str, dict[str, float]] = {
    "llama3.2": {"input": 0.80, "output": 2.40},
    "llama3.2:1b": {"input": 0.10, "output": 0.30},
}
CAPABLE_MODEL = "llama3.2"
CHEAP_MODEL = "llama3.2:1b"
EMBED_USD_PER_MTOK = 0.10
# Prompt-cache reads are billed at this fraction of the input price.
CACHE_READ_FRACTION = 0.10
SEMANTIC_THRESHOLD = 0.66

# Scripted step times. The measured run does not call Ollama.
ROUTE_MS = 1.0
RETRIEVE_MS = 45.0
CACHE_LOOKUP_MS = 2.0
CAPABLE_MS_PER_OUTPUT_TOKEN = 8.0
CHEAP_MS_PER_OUTPUT_TOKEN = 3.0

_STOP = frozenset(
    "a an the of to for in on at is are was were be it its this that and or "
    "what who how when which with from by about me my your you they them "
    "can u tell give gave just really actually thing hey please".split()
)
_TIME_WORDS = frozenset({"yesterday", "today", "tonight"})
_NAME_RE = re.compile(
    r"\b([A-Z]\.\s*[A-Z][a-z]{2,}|[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b"
)
_HARD_MARKERS = (
    "compare",
    "why",
    "difference",
    "versus",
    " vs ",
    "which document",
    "and which",
)

_prompt_cache_keys: set[str] = set()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def failures_dir() -> Path:
    return repo_root() / "eval" / "production" / "failures"


def results_path() -> Path:
    return repo_root() / "eval" / "production" / "results" / "cost.json"


def estimate_tokens(text: str) -> int:
    """Same character rule as the rest of the app: 4 characters ≈ 1 token."""
    raw = text or ""
    if not raw:
        return 0
    return max(1, len(raw) // 4)


def price_tokens(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
) -> float:
    card = PRICES.get(model) or PRICES[CAPABLE_MODEL]
    cached = max(0, min(cached_input_tokens, input_tokens))
    fresh = max(0, input_tokens - cached)
    usd = (
        fresh * card["input"]
        + cached * card["input"] * CACHE_READ_FRACTION
        + output_tokens * card["output"]
    ) / 1_000_000
    return round(usd, 8)


def make_span(
    name: str,
    *,
    elapsed_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    cost_usd: float = 0.0,
    model: str | None = None,
    status: str = "ok",
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "spanId": uuid.uuid4().hex[:16],
        "name": name,
        "elapsedMs": round(float(elapsed_ms), 1),
        "inputTokens": int(input_tokens),
        "outputTokens": int(output_tokens),
        "cachedInputTokens": int(cached_input_tokens),
        "costUsd": round(float(cost_usd), 8),
        "model": model,
        "status": status,
        "attributes": attributes or {},
    }


def span_totals(spans: list[dict[str, Any]]) -> dict[str, float]:
    elapsed = round(sum(float(s.get("elapsedMs") or 0) for s in spans), 1)
    cost = round(sum(float(s.get("costUsd") or 0) for s in spans), 8)
    billed = 0.0
    for span in spans:
        cached = int(span.get("cachedInputTokens") or 0)
        incoming = int(span.get("inputTokens") or 0)
        fresh = max(0, incoming - cached)
        billed += fresh + cached * CACHE_READ_FRACTION + int(span.get("outputTokens") or 0)
    return {"elapsedMs": elapsed, "costUsd": cost, "billedTokens": round(billed, 1)}


def note_system_prompt(system: str) -> bool:
    """Return True when this system prompt was already billed in-process."""
    key = str(hash(system or ""))
    if key in _prompt_cache_keys:
        return True
    _prompt_cache_keys.add(key)
    return False


def clear_prompt_cache() -> None:
    _prompt_cache_keys.clear()


def normalize_words(text: str) -> list[str]:
    words = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()
    return [w for w in words if w not in _STOP and len(w) > 1]


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_easy_question(question: str) -> bool:
    q = f" {(question or '').lower()} "
    if any(marker in q for marker in _HARD_MARKERS):
        return False
    if q.count("?") > 1:
        return False
    if len(q.split()) > 22:
        return False
    return True


class RateLimiter:
    """Per-run stand-in for a per-minute limit on a LiteLLM deployment."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used: dict[str, int] = {}

    def allow(self, model: str) -> bool:
        count = self.used.get(model, 0)
        if count >= self.limit:
            return False
        self.used[model] = count + 1
        return True


def route_model(
    question: str,
    *,
    pinned: str | None = None,
    limiter: RateLimiter | None = None,
    failed: set[str] | None = None,
) -> dict[str, Any]:
    """Pick a model the way a LiteLLM router would: cheap, capable, or fallback."""
    failed = failed or set()
    if pinned:
        chosen = pinned
        reason = "caller pinned a model"
        preferred = pinned
    elif is_easy_question(question):
        preferred = CHEAP_MODEL
        chosen = CHEAP_MODEL
        reason = "single-fact question uses the cheaper model"
    else:
        preferred = CAPABLE_MODEL
        chosen = CAPABLE_MODEL
        reason = "multi-part question stays on the capable model"

    fallback_from = None
    fallback_reason = None
    if chosen in failed or (limiter is not None and not limiter.allow(chosen)):
        fallback_from = chosen
        fallback_reason = "model_failed" if chosen in failed else "rate_limit"
        chosen = CAPABLE_MODEL if fallback_from != CAPABLE_MODEL else CHEAP_MODEL
        if chosen in failed or (limiter is not None and not limiter.allow(chosen)):
            reason = f"{fallback_reason} on {fallback_from}; fallback {chosen} also unavailable"
        else:
            reason = f"{fallback_reason} on {fallback_from}; fell back to {chosen}"
    return {
        "model": chosen,
        "preferred": preferred,
        "reason": reason,
        "fallbackFrom": fallback_from,
        "fallbackReason": fallback_reason,
    }


class SemanticCache:
    def __init__(self, threshold: float = SEMANTIC_THRESHOLD) -> None:
        self.threshold = threshold
        self.rows: list[dict[str, Any]] = []

    def lookup(self, question: str) -> dict[str, Any] | None:
        words = normalize_words(question)
        best: dict[str, Any] | None = None
        best_score = 0.0
        for row in self.rows:
            score = jaccard(words, row["words"])
            if score > best_score:
                best_score = score
                best = row
        if best is not None and best_score >= self.threshold:
            return {"answer": best["answer"], "score": round(best_score, 3), "question": best["question"]}
        return None

    def store(self, question: str, answer: str) -> None:
        if not answer:
            return
        self.rows.append({"question": question, "answer": answer, "words": normalize_words(question)})


def _generate_ms(model: str, output_tokens: int) -> float:
    per = CHEAP_MS_PER_OUTPUT_TOKEN if model == CHEAP_MODEL else CAPABLE_MS_PER_OUTPUT_TOKEN
    floor = 40.0 if model == CHEAP_MODEL else 80.0
    return max(floor, output_tokens * per)


def run_request(
    item: dict[str, Any],
    *,
    mode: str,
    semantic: SemanticCache | None,
    prompt_cache: set[str],
    limiter: RateLimiter | None,
    failed: set[str] | None = None,
    system: str = CLAIMS_ASSISTANT_PROMPT,
) -> dict[str, Any]:
    """One measured request. Answers are scripted so cost is the thing that changes."""
    question = item["question"]
    context = item["context"]
    answer = item["answer"]
    spans: list[dict[str, Any]] = []

    if mode == "baseline":
        decision = {
            "model": CAPABLE_MODEL,
            "preferred": CAPABLE_MODEL,
            "reason": "baseline always uses the capable model",
            "fallbackFrom": None,
            "fallbackReason": None,
        }
    else:
        decision = route_model(question, limiter=limiter, failed=failed)

    spans.append(
        make_span(
            "route",
            elapsed_ms=ROUTE_MS,
            model=decision["model"],
            attributes={
                "reason": decision["reason"],
                "preferred": decision["preferred"],
                "fallbackFrom": decision["fallbackFrom"],
                "fallbackReason": decision["fallbackReason"],
                "router": "litellm-shaped",
            },
        )
    )

    cache_hit = None
    if semantic is not None:
        cache_hit = semantic.lookup(question)
        spans.append(
            make_span(
                "semantic_cache",
                elapsed_ms=CACHE_LOOKUP_MS,
                attributes={
                    "hit": cache_hit is not None,
                    "score": None if cache_hit is None else cache_hit["score"],
                    "matchedQuestion": None if cache_hit is None else cache_hit["question"],
                },
            )
        )

    if cache_hit is not None:
        answer = str(cache_hit["answer"])
        route = "cache"
    else:
        user_prompt = f"Document context:\n{context}\n\nUser question: {question}"
        query_tokens = estimate_tokens(question)
        embed_cost = round(query_tokens * EMBED_USD_PER_MTOK / 1_000_000, 8)
        spans.append(
            make_span(
                "retrieve",
                elapsed_ms=RETRIEVE_MS,
                input_tokens=query_tokens,
                cost_usd=embed_cost,
                attributes={"searchMode": "hybrid"},
            )
        )
        system_tokens = estimate_tokens(system)
        user_tokens = estimate_tokens(user_prompt)
        output_tokens = estimate_tokens(answer)
        system_key = str(hash(system))
        cached_input = 0
        if mode == "improved" and system_key in prompt_cache:
            cached_input = system_tokens
        elif mode == "improved":
            prompt_cache.add(system_key)
        input_tokens = system_tokens + user_tokens
        model = str(decision["model"])
        spans.append(
            make_span(
                "generate",
                elapsed_ms=_generate_ms(model, output_tokens),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input,
                cost_usd=price_tokens(
                    model,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens=cached_input,
                ),
                model=model,
                attributes={"promptCache": cached_input > 0},
            )
        )
        if semantic is not None:
            semantic.store(question, answer)
        route = "fallback" if decision["fallbackFrom"] else (
            "cheap" if model == CHEAP_MODEL else "capable"
        )

    totals = span_totals(spans)
    return {
        "id": item["id"],
        "question": question,
        "answer": answer,
        "model": decision["model"] if route != "cache" else None,
        "route": route,
        "elapsedMs": totals["elapsedMs"],
        "costUsd": totals["costUsd"],
        "billedTokens": totals["billedTokens"],
        "spans": spans,
    }


WORKLOAD: list[dict[str, str]] = [
    {
        "id": "deductible",
        "question": "What deductible applies?",
        "context": "Policy schedule, page 1. All Peril Deductible: USD 1,000.",
        "answer": "The All Peril Deductible is USD 1,000.",
    },
    {
        "id": "deductible-again",
        "question": "What deductible applies?",
        "context": "Policy schedule, page 1. All Peril Deductible: USD 1,000.",
        "answer": "The All Peril Deductible is USD 1,000.",
    },
    {
        "id": "coverage",
        "question": "What is the Coverage A dwelling limit?",
        "context": "Policy schedule. Coverage A (Dwelling): USD 425,000.",
        "answer": "Coverage A (Dwelling) is USD 425,000.",
    },
    {
        "id": "coverage-paraphrase",
        "question": "What is Coverage A dwelling limit?",
        "context": "Policy schedule. Coverage A (Dwelling): USD 425,000.",
        "answer": "Coverage A (Dwelling) is USD 425,000.",
    },
    {
        "id": "investigator",
        "question": "Who investigated claim CLM-2024-00847?",
        "context": "Claim CLM-2024-00847 was investigated by field adjuster A. Ramirez.",
        "answer": "The claim was investigated by field adjuster A. Ramirez.",
    },
    {
        "id": "cause",
        "question": "What was the cause of the loss?",
        "context": "The cause of loss was a burst flexible supply line on the upstairs bathroom faucet.",
        "answer": "The cause of the loss was a burst flexible supply line on the upstairs bathroom faucet.",
    },
    {
        "id": "compare-figures",
        "question": (
            "Compare the damage-assessment range with the contractor estimate "
            "and say which document each figure comes from."
        ),
        "context": (
            "Damage assessment: preliminary repair range USD 16,800–19,200. "
            "Investigation report: contractor estimate EST-4412 is USD 18,450."
        ),
        "answer": (
            "The damage assessment states USD 16,800–19,200. "
            "The investigation report states contractor estimate EST-4412 at USD 18,450."
        ),
    },
    {
        "id": "why-settlement",
        "question": (
            "Why did the settlement differ from the preliminary damage range, "
            "and which document does each number come from?"
        ),
        "context": (
            "Damage assessment range USD 16,800–19,200. "
            "Settlement letter net USD 19,550 after the USD 1,000 deductible."
        ),
        "answer": (
            "The damage assessment gives a preliminary range of USD 16,800–19,200. "
            "The settlement letter states a net of USD 19,550 after the deductible."
        ),
    },
]


def measure(mode: str, *, cheap_limit: int = 100) -> dict[str, Any]:
    semantic = SemanticCache() if mode == "improved" else None
    prompt_cache: set[str] = set()
    limiter = RateLimiter(cheap_limit) if mode == "improved" else None
    items = [
        run_request(
            item,
            mode=mode,
            semantic=semantic,
            prompt_cache=prompt_cache,
            limiter=limiter,
        )
        for item in WORKLOAD
    ]
    return _summarize(mode, items)


def _summarize(mode: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(items))
    total_cost = sum(float(item["costUsd"]) for item in items)
    total_ms = sum(float(item["elapsedMs"]) for item in items)
    total_tokens = sum(float(item["billedTokens"]) for item in items)
    return {
        "mode": mode,
        "requests": len(items),
        "meanCostUsd": round(total_cost / n, 8),
        "costPerThousandUsd": round(total_cost / n * 1000, 4),
        "meanElapsedMs": round(total_ms / n, 1),
        "meanBilledTokens": round(total_tokens / n, 1),
        "items": items,
    }


def fallback_example() -> dict[str, Any]:
    """Second easy question hits the cheap-model rate limit and falls back."""
    limiter = RateLimiter(1)
    semantic = SemanticCache()
    first = run_request(
        WORKLOAD[0],
        mode="improved",
        semantic=semantic,
        prompt_cache=set(),
        limiter=limiter,
    )
    second = run_request(
        WORKLOAD[4],
        mode="improved",
        semantic=semantic,
        prompt_cache=set(),
        limiter=limiter,
    )
    route = next(span for span in second["spans"] if span["name"] == "route")
    return {
        "limit": 1,
        "first": {"id": first["id"], "model": first["model"], "route": first["route"]},
        "second": {
            "id": second["id"],
            "model": second["model"],
            "route": second["route"],
            "reason": route["attributes"].get("reason"),
            "fallbackFrom": route["attributes"].get("fallbackFrom"),
        },
    }


def _span_time_share(items: list[dict[str, Any]], name: str) -> float:
    total = sum(float(item["elapsedMs"]) for item in items) or 1.0
    step = 0.0
    for item in items:
        for span in item["spans"]:
            if span["name"] == name:
                step += float(span["elapsedMs"])
    return round(step / total, 3)


def ten_x_plan(improved: dict[str, Any]) -> dict[str, Any]:
    items = improved["items"]
    mean_ms = float(improved["meanElapsedMs"]) or 1.0
    generate_share = _span_time_share(items, "generate")
    per_minute = round(60_000 / mean_ms, 1)
    return {
        "breaksFirst": "generate",
        "generateShareOfTime": generate_share,
        "meanElapsedMs": mean_ms,
        "oneWorkerRequestsPerMinute": per_minute,
        "atTenX": (
            f"One local model process finishes about {per_minute:.0f} requests a minute "
            f"at this mean. Ten times that rate queues on generate "
            f"({generate_share:.0%} of the time) before retrieval or the API process fills up."
        ),
        "plan": [
            "Keep the semantic cache so a repeated question never calls the model.",
            "Keep single-fact questions on the cheap model. The capable model only sees multi-part questions.",
            "When the generate queue grows, add a second model process. Retrieval is not the first limit.",
            "On a rate limit or a dead model, fall back and record it on the route span. Do not drop the request.",
            "Fine-tune only after the same wrong answer survives the failure test. Fine-tuning does not add capacity.",
        ],
    }


def build_report() -> dict[str, Any]:
    baseline = measure("baseline")
    improved = measure("improved")
    base_cost = float(baseline["meanCostUsd"]) or 1.0
    base_ms = float(baseline["meanElapsedMs"]) or 1.0
    base_tokens = float(baseline["meanBilledTokens"]) or 1.0
    failure = load_failure()
    return {
        "label": "scripted",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "method": {
            "note": (
                "Both sides answer with the same scripted text, so quality stays even. "
                "Times are scripted per step (the capable model waits longer per output token). "
                "Dollars use hosted-equivalent prices because local Ollama has no token invoice. "
                "Billed tokens count cached prompt tokens at 10%."
            ),
            "pricesPerMillion": PRICES,
            "cacheReadFraction": CACHE_READ_FRACTION,
            "semanticThreshold": SEMANTIC_THRESHOLD,
            "capableModel": CAPABLE_MODEL,
            "cheapModel": CHEAP_MODEL,
            "tokenRule": "characters ÷ 4",
        },
        "baseline": {k: v for k, v in baseline.items() if k != "items"},
        "improved": {k: v for k, v in improved.items() if k != "items"},
        "baselineItems": baseline["items"],
        "improvedItems": improved["items"],
        "savings": {
            "costPct": round((1 - float(improved["meanCostUsd"]) / base_cost) * 100, 1),
            "latencyPct": round((1 - float(improved["meanElapsedMs"]) / base_ms) * 100, 1),
            "billedTokenPct": round((1 - float(improved["meanBilledTokens"]) / base_tokens) * 100, 1),
        },
        "fallback": fallback_example(),
        "tenX": ten_x_plan(improved),
        "failure": {
            "id": failure["id"],
            "sourceTraceId": failure["sourceTraceId"],
            "question": failure["question"],
            "badAnswer": failure["badAnswer"],
            "fixedAnswer": failure["fixedAnswer"],
            "note": failure["note"],
        },
        "fineTuning": (
            "Fine-tuning is the last resort. It does not make generate faster, "
            "and it cannot stop a hallucination that a test already catches. "
            "Use it only when the same format error remains after caching, routing, "
            "and a permanent test."
        ),
        "observability": {
            "whatToLog": [
                "request id and time",
                "question, retrieved context, and answer",
                "model and why it was chosen",
                "one span per step with time, tokens, and cost",
            ],
            "openTelemetry": (
                "Each request is a trace. Each step is a span with duration, status, and attributes. "
                "An OTLP exporter ships this shape to a collector."
            ),
            "phoenix": "Arize Phoenix shows these spans beside the retrieved chunks and the answer.",
            "langsmith": "LangSmith is a hosted view of the same fields: inputs, outputs, tokens, and latency per step.",
        },
    }


def write_report(path: Path | None = None) -> dict[str, Any]:
    report = build_report()
    target = path or results_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def person_names(text: str) -> list[str]:
    found: list[str] = []
    for match in _NAME_RE.findall(text or ""):
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned not in found:
            found.append(cleaned)
    return found


def invented_people(answer: str, context: str) -> list[str]:
    """People named in the answer whose last name is absent from the context."""
    ctx = (context or "").lower()
    invented: list[str] = []
    for name in person_names(answer):
        last = name.replace(".", " ").split()[-1].lower()
        if last not in ctx:
            invented.append(name)
    return invented


def failure_guard(answer: str, context: str, case: dict[str, Any]) -> dict[str, Any]:
    missing = [s for s in case.get("mustContain") or [] if s.lower() not in (answer or "").lower()]
    forbidden = [s for s in case.get("mustNotContain") or [] if s.lower() in (answer or "").lower()]
    invented = invented_people(answer, context)
    return {
        "ok": not missing and not forbidden and not invented,
        "missing": missing,
        "forbidden": forbidden,
        "invented": invented,
    }


def load_failures(directory: Path | None = None) -> list[dict[str, Any]]:
    folder = directory or failures_dir()
    if not folder.exists():
        return []
    cases: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def load_failure(directory: Path | None = None) -> dict[str, Any]:
    cases = load_failures(directory)
    if not cases:
        raise FileNotFoundError("No promoted failures in eval/production/failures")
    return cases[0]


def promote_failure(
    trace: dict[str, Any],
    *,
    note: str,
    must_contain: list[str],
    must_not_contain: list[str],
    fixed_answer: str,
    directory: Path | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Write a production miss as a permanent case. The bad answer must fail the guard."""
    context = trace.get("context") or ""
    if not context:
        chunks = trace.get("retrieved") or []
        context = "\n".join(str(chunk.get("content") or "") for chunk in chunks)
    case = {
        "id": case_id or f"failure-{str(trace.get('traceId') or uuid.uuid4())[:8]}",
        "sourceTraceId": trace.get("traceId"),
        "question": trace.get("question") or "",
        "context": context,
        "badAnswer": trace.get("answer") or "",
        "fixedAnswer": fixed_answer,
        "note": note,
        "mustContain": must_contain,
        "mustNotContain": must_not_contain,
        "promotedAt": datetime.now(timezone.utc).isoformat(),
    }
    verdict = failure_guard(case["badAnswer"], case["context"], case)
    if verdict["ok"]:
        raise ValueError("Refusing to promote an answer the guard already accepts")
    folder = directory or failures_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{case['id']}.json"
    target.write_text(json.dumps(case, indent=2), encoding="utf-8")
    case["path"] = str(target)
    return case


def _word_hit(keyword: str, words: list[str]) -> bool:
    for word in words:
        if keyword == word:
            return True
        if len(keyword) >= 6 and (word.startswith(keyword[:6]) or keyword.startswith(word[:6])):
            return True
    return False


def parse_complaint(complaint: str, *, now: datetime) -> dict[str, Any]:
    words = re.sub(r"[^a-z0-9\s]", " ", (complaint or "").lower()).split()
    when = None
    if "yesterday" in words:
        when = (now - timedelta(days=1)).date()
    elif "today" in words or "tonight" in words:
        when = now.date()
    keywords = [w for w in words if w not in _STOP and w not in _TIME_WORDS and len(w) > 2]
    return {"when": when, "keywords": keywords}


def support_search(
    logs: list[dict[str, Any]],
    complaint: str,
    *,
    now: datetime | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Find a past answer from a vague complaint. Date first, then words in the question and answer."""
    clock = now or datetime.now(timezone.utc)
    parsed = parse_complaint(complaint, now=clock)
    when = parsed["when"]
    keywords: list[str] = parsed["keywords"]

    def on_day(row: dict[str, Any]) -> bool:
        if when is None:
            return True
        stamp = datetime.fromisoformat(row["timestamp"])
        return stamp.date() == when

    pool = [row for row in logs if on_day(row)]
    used_day = when is not None and bool(pool)
    if when is not None and not pool:
        pool = list(logs)
        used_day = False

    ranked: list[dict[str, Any]] = []
    for row in pool:
        hay = normalize_words(f"{row.get('question') or ''} {row.get('answer') or ''}")
        raw_words = re.sub(
            r"[^a-z0-9\s]",
            " ",
            f"{row.get('question') or ''} {row.get('answer') or ''}".lower(),
        ).split()
        hits = [kw for kw in keywords if _word_hit(kw, raw_words) or _word_hit(kw, hay)]
        if keywords and not hits:
            continue
        ranked.append(
            {
                "requestId": row.get("requestId"),
                "timestamp": row.get("timestamp"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "model": row.get("model"),
                "matched": hits,
                "score": len(hits),
            }
        )
    ranked.sort(key=lambda row: (-int(row["score"]), str(row["timestamp"])))
    return {
        "complaint": complaint,
        "when": None if when is None else when.isoformat(),
        "restrictedToDay": used_day,
        "scanned": len(logs),
        "candidates": len(pool),
        "hits": ranked[:limit],
    }


INVESTIGATOR_FAILURE = {
    "requestId": "req-bad-investigator",
    "hoursAgo": 20,
    "question": "Who investigated claim CLM-2024-00847?",
    "answer": "The claim was investigated by field surveyor M. Chen.",
    "model": CAPABLE_MODEL,
}


def support_pool(*, now: datetime, extras: int = 0) -> list[dict[str, Any]]:
    """A yesterday-sized log, plus optional filler, with one bad investigator answer buried in it."""
    samples = [
        (22, "What deductible applies?", "The All Peril Deductible is USD 1,000."),
        (18, "What is the Coverage A dwelling limit?", "Coverage A (Dwelling) is USD 425,000."),
        (6, "What was the cause of the loss?", "A burst flexible supply line on the upstairs bathroom faucet."),
        (30, "What is the net settlement amount?", "The net settlement is USD 19,550."),
        (50, "Is there a water backup endorsement?", "The Water Backup and Sump Overflow limit is USD 10,000."),
        (4, "What is the property address?", "The loss location is 412 Maple Avenue."),
        (21, "What is the preliminary repair range?", "The damage assessment range is USD 16,800–19,200."),
        (26, "What deductible applies?", "The All Peril Deductible is USD 1,000."),
    ]
    rows: list[dict[str, Any]] = []
    for index, (hours, question, answer) in enumerate(samples):
        rows.append(
            {
                "requestId": f"req-seed-{index}",
                "timestamp": (now - timedelta(hours=hours)).isoformat(),
                "question": question,
                "answer": answer,
                "model": CAPABLE_MODEL,
            }
        )
    rows.append(
        {
            "requestId": INVESTIGATOR_FAILURE["requestId"],
            "timestamp": (now - timedelta(hours=int(INVESTIGATOR_FAILURE["hoursAgo"]))).isoformat(),
            "question": INVESTIGATOR_FAILURE["question"],
            "answer": INVESTIGATOR_FAILURE["answer"],
            "model": INVESTIGATOR_FAILURE["model"],
        }
    )
    for index in range(extras):
        hours = 5 + (index % 70)
        rows.append(
            {
                "requestId": f"req-fill-{index}",
                "timestamp": (now - timedelta(hours=hours)).isoformat(),
                "question": "What deductible applies?",
                "answer": "The All Peril Deductible is USD 1,000.",
                "model": CAPABLE_MODEL,
            }
        )
    return rows


DRILL_NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def traces_as_logs(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for trace in traces:
        logs.append(
            {
                "requestId": trace.get("traceId") or trace.get("requestId"),
                "timestamp": trace.get("timestamp"),
                "question": trace.get("question"),
                "answer": trace.get("answer"),
                "model": trace.get("model"),
            }
        )
    return logs
