# Retrieval evaluation

Measure whether the **right document** shows up in the top-k retrieved chunks (Recall@k / MRR), so you can separate retrieval failures from generation failures.

## Prerequisites

1. Sample docs indexed (upload via UI **or** use the helper below).
2. Document-processor at `http://localhost:7071`.
3. For `--rewrite`, backend at `http://localhost:8000/api` with Ollama up.

### Index sample-data into Chroma (eval helper)

```bash
docker cp sample-data/. insurance-rag-document-processor:/tmp/sample-data
docker cp eval/index_sample_for_eval.py insurance-rag-document-processor:/tmp/index_sample_for_eval.py
docker exec -w /home/site/wwwroot insurance-rag-document-processor \
  python /tmp/index_sample_for_eval.py --docs /tmp/sample-data
```

## Golden set

[`golden_qa.json`](golden_qa.json) — questions with `expected_document` filename stems matching the sample claim artifacts.

## Measured results (sample-data, 18 questions, k=5)

| Label | Mode | Recall@5 | MRR |
|-------|------|----------|-----|
| baseline | semantic, no rerank | 0.944 | 0.755 |
| hybrid | hybrid (BM25+RRF), no rerank | **1.000** | 0.824 |
| rerank | hybrid + cross-encoder | **1.000** | **0.917** |
| full | hybrid + rerank + query rewrite | **1.000** | 0.880 |

Hybrid recovered the baseline miss (`property-address` / `412 Maple`). Rerank further improved MRR (best chunk more often rank 1). Full stack keeps perfect Recall@5; rewrite helps messy questions with a small MRR tradeoff vs rerank-only on this set.

## Run baseline (dense / semantic only, no rewrite)

```bash
python eval/run_retrieval_eval.py --label baseline --search-mode semantic --no-rerank --no-rewrite
```

## After hybrid search

```bash
python eval/run_retrieval_eval.py --label hybrid --search-mode hybrid --no-rerank --no-rewrite --compare-to baseline
```

## After rerank

```bash
python eval/run_retrieval_eval.py --label rerank --search-mode hybrid --rerank --no-rewrite --compare-to baseline
```

## Full stack (hybrid + rerank + query rewrite)

```bash
python eval/run_retrieval_eval.py --label full --search-mode hybrid --rerank --rewrite --compare-to baseline
```

Results are written to `eval/results/<label>.json`.

## Reading failures

- **MISS / low Recall@k** → retrieval problem (hybrid, rerank, rewrite, indexing).
- **Hit but bad chat answer** → generation problem (prompt / model / context packing). Use Chat **Inspect** to confirm the right chunk text was fetched.

For a full **hand-read error analysis** (random sample → open coding → ranked problems → fix prediction), see [`error_analysis/README.md`](error_analysis/README.md).

For **automatic answer evals** (assertions + optional LLM judge + before/after), see [`answer_evals/README.md`](answer_evals/README.md).
