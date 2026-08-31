# Error Analysis — Reading Traces

Turn "it fails sometimes" into a **ranked list of real problems** by reading complete answer traces.

This is not an automated benchmark (MMLU / HumanEval). You read **your** app’s answers.

## Mentor checklist

| Check | How this folder supports it |
|-------|-----------------------------|
| Fair sample, not cherry-picked | `sample_traces.py --seed …` draws a random subset |
| Honest note per failure **before** grouping | Worksheet fields: fill `open_code_note` first, then `problem_type` |
| Clear problem-group names | You invent names after coding; seed example uses readable labels |
| Ranked problems + next fix target | `rank_problems.py` scores frequency × severity and writes a prediction stub |

## What is a complete trace?

Enough to **replay** one answer later:

- question (and rewritten search query if any)
- retrieved chunks **with text**
- final answer
- model / search mode / ids

Live chats append to `traces/traces.jsonl` when the backend runs with `ENABLE_TRACE_LOGGING=true` (default). API: `GET /api/traces`.

## Workflow

### 1. Collect a batch (fair coverage)

Either chat normally in the UI, **or** drive golden questions through the API:

```bash
python eval/error_analysis/collect_batch.py
```

### 2. Draw a random sample

```bash
python eval/error_analysis/sample_traces.py --n 20 --seed 42
# or from the running API:
python eval/error_analysis/sample_traces.py --from-api --n 20 --seed 42
```

Opens `samples/sample_*.json` + `.md`.

### 3. Open coding (by hand)

For each failure in the sample JSON:

1. Read question → retrieved → answer
2. Write **one honest sentence** in `open_code_note` (**before** naming a category)
3. Set `ok: false`, `severity` 1–3
4. Only then set `problem_type` (short stranger-readable name)

Skip detailed notes when `ok: true`.

### 4. Rank and pick a target

```bash
python eval/error_analysis/rank_problems.py --sample eval/error_analysis/samples/coded_sample_seed.json
```

Produces:

- `*_ranked.json` — problems sorted by **frequency × mean severity**
- `*_prediction.md` — write what you will change and what you expect

### 5. One fix + re-measure

Change **one** thing aimed at the top problem. Re-sample or re-run retrieval eval. Note which failures the change did **not** fix.

## Seed demo (already coded)

| File | Role |
|------|------|
| [`traces/traces.jsonl`](traces/traces.jsonl) | 10 replayable seed traces |
| [`samples/coded_sample_seed.json`](samples/coded_sample_seed.json) | Open notes written before taxonomy |
| Run `rank_problems.py` on that sample | See ranked target |

Suggested starter taxonomy (only after notes exist):

- `wrong_document_retrieved`
- `messy_query_hurts_retrieval`
- `omits_key_number_from_context`
- `vague_or_imprecise_answer`
- `hallucinated_despite_context`

## vs retrieval eval (`eval/run_retrieval_eval.py`)

| Retrieval eval | Error analysis |
|----------------|----------------|
| Automated Recall@k / MRR | Human reading of full answers |
| Proves document showed up | Finds generation bugs, vague answers, hallucinations |
| Great for hybrid/rerank | Great for “what should we fix next?” |

Use both.
