# Fix-target prediction (seed demo)

Write this **before** you implement the fix.

## Chosen target
`wrong_document_retrieved`

## Why this one (from ranking)
Highest impact in `coded_sample_seed_ranked.json`: frequency 3 × mean severity 2.67 → impact 8.0 — ahead of generation issues at impact 3.0.

## What I will change
(one concrete change — not five)

Enable / keep **hybrid search (BM25 + RRF)** so exact addresses and claim IDs are less likely to miss the investigation report. Do **not** also change the system prompt in the same experiment.

## What I expect to happen
- Failures of type `wrong_document_retrieved` should drop from **3** in this sample to about **1 or 0**.
- Other problem types I do **not** expect to fix: `hallucinated_despite_context`, `omits_key_number_from_context`, `vague_or_imprecise_answer` (those need prompt/model work, not retrieval).

## How I will measure
Re-run `eval/run_retrieval_eval.py --label hybrid --compare-to baseline`, and/or re-sample traces and re-code the same question set.
