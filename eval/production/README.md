# Production — observability, cost, and the failure→test loop

Week 11. The app records every answer so a bad one can be found, prices each step so cost can be cut, and turns one real miss into a test that stays red if that miss comes back.

## Mentor checklist

| Check | Result |
|-------|--------|
| Find a past answer from a vague complaint | “yesterday it gave the wrong investigator name” returns `req-bad-investigator` (M. Chen). The test buries that row among 1,000 others. |
| Time and cost per step | Each request has spans (`route`, `retrieve`, `semantic_cache`, `generate`). The request total is the sum of the spans. |
| Cost measured, then improved | Same 8 answers. **$0.1654 → $0.0414 per 1,000 requests (75% lower).** Billed tokens 184 → 80. Time 186 ms → 125 ms. |
| A real failure is now a test | `seed-009-gen-halluc` (invented investigator M. Chen) is `eval/production/failures/hallucinated_investigator.json`. The bad answer fails the guard. “A. Ramirez” passes. |

## What a request logs

Enough to replay it, and enough to see which step was slow or expensive:

- request id and time
- question, retrieved text, answer, model
- why that model was chosen
- one span per step: time, input tokens, output tokens, cached prompt tokens, cost

That span is the same shape OpenTelemetry uses. Phoenix would show it next to the retrieved chunks. LangSmith is a hosted view of the same fields. This app keeps the JSONL locally (`eval/error_analysis/traces/traces.jsonl` for live chat, plus the measured run in `results/cost.json`) so the lookup still works without those accounts.

Live chat writes the spans on the trace. The measured run below does not call Ollama: both sides use the same scripted answer, and each step has a scripted duration. Dollars are hosted-equivalent prices (`llama3.2` input $0.80 / output $2.40 per million tokens, `llama3.2:1b` at one eighth of that). Local Ollama does not send a token bill. Billed tokens are recorded beside the dollars. Cached system-prompt tokens count at 10%.

## Cost, before and after

Eight questions. Baseline always calls the capable model and pays the system prompt in full every time. The improved path:

1. **Semantic cache** — a repeated or near-copy question returns the stored answer. No retrieve, no generate.
2. **Prompt cache** — the system prompt is billed once, then at 10%.
3. **Model routing** — a single-fact question uses `llama3.2:1b`. A compare / why question stays on `llama3.2`. This is the decision a LiteLLM router would make; the route span records the reason.
4. **Fallback** — if the cheap model is over its limit, the next request uses the capable model and the span says `rate_limit`.

| | Baseline | Improved |
|--|--------:|---------:|
| Cost / 1,000 requests | $0.1654 | $0.0414 |
| Cost / request | $0.000165 | $0.000041 |
| Billed tokens (mean) | 183.6 | 79.6 |
| Time (mean) | 186 ms | 125 ms |

Two of the eight improved requests are cache hits (3 ms, $0). The two multi-part questions stay on the capable model. Quality is the same text on both sides; this run measures cost, not a second model’s writing.

Regenerate with:

```bash
cd backend
python -c "from app.services.production import write_report; write_report()"
```

## Support drill

Complaint: “yesterday it gave the wrong investigator name.”

The search drops filler words, keeps yesterday’s date, and ranks what is left. It does not need the trace id. Page: `/production`. API: `POST /api/production/support`.

## Failure → test

From the open-coded sample (`eval/error_analysis`, trace `seed-009-gen-halluc`):

- Question: who investigated claim CLM-2024-00847?
- Context names field adjuster **A. Ramirez**
- The answer said **M. Chen**

`failure_guard` rejects that answer (missing Ramirez, names Chen, and Chen is not in the context). The corrected answer passes. `pytest tests/test_production.py` loads every JSON file in `failures/`, so a newly promoted miss is guarded without a new test function.

`POST /api/production/failures` writes a case from a stored trace. It refuses an answer the guard already accepts.

## 10×

Generate is **71%** of the improved run’s time. One model process finishes about **480 requests a minute** at this mean. Ten times the traffic queues there first. Retrieval does not fill up before that.

Plan, in order: keep the semantic cache, keep easy questions on the cheap model, add a second model process when the generate queue grows, fall back on rate limit instead of dropping the request. Fine-tuning is last. It does not make generate faster, and it is not how this hallucination is stopped — the test is.

## Where to look

| Piece | Path |
|-------|------|
| Spans, prices, cache, router, drill, guard | `backend/app/services/production.py` |
| API | `GET /api/production/report`, `POST /api/production/support` |
| Live chat spans | `backend/app/services/rag_service.py` (rewrite, retrieve, generate) |
| Permanent case | `eval/production/failures/hallucinated_investigator.json` |
| Numbers | `eval/production/results/cost.json` |
| Page | `/production` |
