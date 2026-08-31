#!/usr/bin/env python3
"""
Offline retrieval eval: Recall@k and MRR against eval/golden_qa.json.

Examples:
  python eval/run_retrieval_eval.py --label baseline --search-mode semantic --no-rerank --no-rewrite
  python eval/run_retrieval_eval.py --label hybrid --search-mode hybrid --no-rerank --no-rewrite
  python eval/run_retrieval_eval.py --label rerank --search-mode hybrid --rerank --no-rewrite
  python eval/run_retrieval_eval.py --label full --search-mode hybrid --rerank --rewrite
  python eval/run_retrieval_eval.py --label full --compare-to baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "eval" / "golden_qa.json"
DEFAULT_RESULTS = ROOT / "eval" / "results"


def _normalize_name(name: str | None) -> str:
    if not name:
        return ""
    stem = Path(str(name)).stem.lower().strip()
    return stem.replace("_", "-").replace(" ", "-")


def _filename_match(hit_name: str | None, expected: str) -> bool:
    hit = _normalize_name(hit_name)
    exp = _normalize_name(expected)
    if not hit or not exp:
        return False
    return hit == exp or exp in hit or hit in exp


def _post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def retrieve_via_backend(
    *,
    base_url: str,
    question: str,
    top_k: int,
    search_mode: str,
    rewrite: bool,
    rerank: bool,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/retrieve"
    data = _post_json(
        url,
        {
            "question": question,
            "topK": top_k,
            "searchMode": search_mode,
            "rewrite": rewrite,
            "rerank": rerank,
            "debug": True,
        },
    )
    # ApiResponse envelope
    body = data.get("data") if isinstance(data.get("data"), dict) else data
    sources = body.get("sources") or []
    return {
        "originalQuestion": body.get("originalQuestion") or question,
        "searchQuery": body.get("searchQuery") or question,
        "searchMode": body.get("searchMode"),
        "hits": [
            {
                "fileName": s.get("fileName"),
                "pageNumber": s.get("pageNumber"),
                "score": s.get("relevanceScore"),
                "content": s.get("content") or "",
            }
            for s in sources
        ],
    }


def retrieve_via_processor(
    *,
    base_url: str,
    question: str,
    top_k: int,
    search_mode: str,
    rerank: bool,
    similarity_threshold: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/vectors/search"
    data = _post_json(
        url,
        {
            "question": question,
            "topK": top_k,
            "similarityThreshold": similarity_threshold,
            "searchMode": search_mode,
            "rerank": rerank,
        },
    )
    return {
        "originalQuestion": question,
        "searchQuery": question,
        "searchMode": data.get("searchMode") or search_mode,
        "hits": [
            {
                "fileName": h.get("fileName"),
                "pageNumber": h.get("pageNumber"),
                "score": h.get("score"),
                "content": h.get("content") or "",
            }
            for h in (data.get("hits") or [])
        ],
    }


def score_item(
    *,
    hits: list[dict[str, Any]],
    expected_document: str,
    expected_keywords: list[str] | None,
    k: int,
) -> dict[str, Any]:
    top = hits[:k]
    ranks = [
        i + 1
        for i, hit in enumerate(top)
        if _filename_match(hit.get("fileName"), expected_document)
    ]
    hit_at_k = 1.0 if ranks else 0.0
    mrr = 1.0 / ranks[0] if ranks else 0.0

    keyword_hit = None
    if expected_keywords:
        blob = "\n".join(str(h.get("content") or "") for h in top).lower()
        keyword_hit = all(kw.lower() in blob for kw in expected_keywords)

    return {
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "rank": ranks[0] if ranks else None,
        "keyword_hit": keyword_hit,
        "retrieved_files": [h.get("fileName") for h in top],
    }


def load_compare(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval Recall@k / MRR eval")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--label", default="run", help="Result file stem, e.g. baseline")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000/api",
        help="Backend API base (for rewrite path). Empty to call processor only.",
    )
    parser.add_argument(
        "--processor-url",
        default="http://localhost:7071",
        help="Document-processor base URL (used when --no-rewrite)",
    )
    parser.add_argument("--search-mode", default="semantic", choices=["semantic", "keyword", "hybrid"])
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Label of a prior results file to diff (e.g. baseline)",
    )
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    items = golden.get("items") or []
    k = int(args.top_k or golden.get("k") or 5)

    per_question: list[dict[str, Any]] = []
    hits_sum = 0.0
    mrr_sum = 0.0
    keyword_hits = 0
    keyword_total = 0

    for item in items:
        qid = item.get("id") or item.get("question")
        question = item["question"]
        expected = item["expected_document"]
        expected_keywords = item.get("expected_keywords") or []

        try:
            if args.rewrite and args.backend_url:
                retrieved = retrieve_via_backend(
                    base_url=args.backend_url,
                    question=question,
                    top_k=k,
                    search_mode=args.search_mode,
                    rewrite=True,
                    rerank=args.rerank,
                )
            else:
                retrieved = retrieve_via_processor(
                    base_url=args.processor_url,
                    question=question,
                    top_k=k,
                    search_mode=args.search_mode,
                    rerank=args.rerank,
                    similarity_threshold=args.similarity_threshold,
                )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"ERROR retrieving for {qid}: {exc}", file=sys.stderr)
            return 2

        scored = score_item(
            hits=retrieved["hits"],
            expected_document=expected,
            expected_keywords=expected_keywords,
            k=k,
        )
        hits_sum += scored["hit_at_k"]
        mrr_sum += scored["mrr"]
        if scored["keyword_hit"] is not None:
            keyword_total += 1
            if scored["keyword_hit"]:
                keyword_hits += 1

        per_question.append(
            {
                "id": qid,
                "question": question,
                "expected_document": expected,
                "search_query": retrieved.get("searchQuery"),
                "search_mode": retrieved.get("searchMode"),
                **scored,
            }
        )
        mark = "OK" if scored["hit_at_k"] else "MISS"
        print(f"[{mark}] {qid}: rank={scored['rank']} files={scored['retrieved_files']}")

    n = max(len(items), 1)
    summary = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "search_mode": args.search_mode,
        "rerank": args.rerank,
        "rewrite": args.rewrite,
        "n": len(items),
        "recall_at_k": hits_sum / n,
        "mrr": mrr_sum / n,
        "keyword_recall": (keyword_hits / keyword_total) if keyword_total else None,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"{args.label}.json"
    payload = {"summary": summary, "questions": per_question}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(f"label:       {summary['label']}")
    print(f"Recall@{k}:  {summary['recall_at_k']:.3f}")
    print(f"MRR:         {summary['mrr']:.3f}")
    if summary["keyword_recall"] is not None:
        print(f"KeywordHit:  {summary['keyword_recall']:.3f}")
    print(f"wrote:       {out_path}")

    if args.compare_to:
        prior_path = args.results_dir / f"{args.compare_to}.json"
        prior = load_compare(prior_path)
        if not prior:
            print(f"No prior results at {prior_path}", file=sys.stderr)
            return 1
        ps = prior["summary"]
        print()
        print(f"=== vs {args.compare_to} ===")
        print(
            f"Recall@{k}:  {ps['recall_at_k']:.3f} -> {summary['recall_at_k']:.3f} "
            f"({summary['recall_at_k'] - ps['recall_at_k']:+.3f})"
        )
        print(
            f"MRR:         {ps['mrr']:.3f} -> {summary['mrr']:.3f} "
            f"({summary['mrr'] - ps['mrr']:+.3f})"
        )
        prior_by_id = {q["id"]: q for q in prior.get("questions") or []}
        regressions = []
        improvements = []
        for q in per_question:
            old = prior_by_id.get(q["id"])
            if not old:
                continue
            if old["hit_at_k"] and not q["hit_at_k"]:
                regressions.append(q["id"])
            if (not old["hit_at_k"]) and q["hit_at_k"]:
                improvements.append(q["id"])
        if improvements:
            print(f"improved:    {', '.join(improvements)}")
        if regressions:
            print(f"regressed:   {', '.join(regressions)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
