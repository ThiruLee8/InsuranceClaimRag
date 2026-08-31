# Answer evals — measuring whether a change helped

Automatic tests that score every change: **rule assertions first** (free), then optional **LLM-as-judge** (after you validate it against your own grades).

## Mentor checklist

| Check | How |
|-------|-----|
| Single command | `python eval/answer_evals/run_evals.py --label after --fixtures ... --compare-to before` |
| Week 5 failures as tests | `cases.json` ids prefixed `reg-*` from error-analysis problem types |
| AI judge checked vs human | `validate_judge.py` + `human_labels.json` |
| Before/after score | `--compare-to` prints pass_rate / mean_score deltas **per problem_type** |

## One command (offline demo — no live stack)

```bash
python eval/answer_evals/run_evals.py --label before --fixtures eval/answer_evals/fixtures/before.json --no-judge
python eval/answer_evals/run_evals.py --label after  --fixtures eval/answer_evals/fixtures/after.json  --no-judge --compare-to before
```

## Live against your app

```bash
# Assertions only (recommended first)
python eval/answer_evals/run_evals.py --label live_before --no-judge

# After a change
python eval/answer_evals/run_evals.py --label live_after --no-judge --compare-to live_before

# With LLM judge (only after validation passes)
python eval/answer_evals/validate_judge.py
python eval/answer_evals/run_evals.py --label live_judged --judge --compare-to live_before
```

## What’s in a case

From [`cases.json`](cases.json):

- **question** + **problem_type** (ties to Week 5 taxonomy)
- **assertions** — rule checks: sources present, expected document retrieved, must/must-not substrings, refuse language
- **judge** (optional) — faithfulness / answer relevancy (RAGAS-inspired; G-Eval style 1–10 + pass)

Regression cases from Week 5:

| Case id | Problem type |
|---------|----------------|
| `reg-property-address` | wrong_document_retrieved |
| `reg-claim-about` | wrong_document_retrieved |
| `reg-damage-cost-range` | wrong_document_retrieved |
| `reg-messy-payout` | messy_query_hurts_retrieval |
| `reg-net-settlement-exact` | vague_or_imprecise_answer |
| `reg-deductible-amount` | omits_key_number_from_context |
| `reg-investigator-no-hallucination` | hallucinated_despite_context |

## Judge validation

1. Humans grade examples in [`human_labels.json`](human_labels.json) (`human_pass` + note).
2. Run the judge and measure agreement:

```bash
# With Ollama up:
python eval/answer_evals/validate_judge.py

# CI / offline (uses expected_judge_pass aligned with human labels):
python eval/answer_evals/validate_judge.py --dry-run-offline
```

Do **not** enable `--judge` on the main suite until agreement ≥ ~0.7.

## Metrics reported

| Metric | Meaning |
|--------|---------|
| `pass_rate` | Fraction of cases that passed overall |
| `mean_score` | Average combined score (0–1) |
| `assertion_pass_rate` | Rule checks only |
| `context_recall` / `context_precision` | Proxy RAGAS retrieval metrics vs `expected_document` |
| `by_problem_type` | Same scores sliced by Week 5 problem type — see what the change fixed |

## Study map

| Concept | Where |
|---------|--------|
| Eval sets | `cases.json` |
| Regression from failures | `reg-*` cases |
| Assertion checks | `assertions.py` |
| LLM-as-judge / G-Eval-style 1–10 | `judge.py` |
| Judge validation | `validate_judge.py` |
| RAGAS-inspired faithfulness / relevancy | judge criteria |
| RAGAS-inspired context precision & recall | assertion helpers |
| Before/after deltas | `--compare-to` |
