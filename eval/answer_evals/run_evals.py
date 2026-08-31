#!/usr/bin/env python3
"""
Single-command answer evals: assertions (+ optional LLM judge) with before/after compare.

  # Live against your running stack
  python eval/answer_evals/run_evals.py --label before
  python eval/answer_evals/run_evals.py --label after --compare-to before

  # Offline fixture mode (no API) — scores fixed Q/A/retrieved bundles
  python eval/answer_evals/run_evals.py --label before --fixtures eval/answer_evals/fixtures/before.json
  python eval/answer_evals/run_evals.py --label after --fixtures eval/answer_evals/fixtures/after.json --compare-to before

  # Assertions only (skip LLM judge)
  python eval/answer_evals/run_evals.py --label before --no-judge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CASES = HERE / "cases.json"
DEFAULT_RESULTS = HERE / "results"

# Local imports
sys.path.insert(0, str(HERE))
from assertions import context_precision, context_recall, run_assertions  # noqa: E402
from judge import run_judges  # noqa: E402


def _post_json(url: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_live(api_base: str, question: str) -> dict[str, Any]:
    data = _post_json(
        f"{api_base.rstrip('/')}/chat",
        {"question": question, "debug": True, "agentId": "claims_assistant"},
    )
    body = data.get("data") if isinstance(data.get("data"), dict) else data
    sources = body.get("sources") or []
    retrieved = [
        {
            "fileName": s.get("fileName"),
            "pageNumber": s.get("pageNumber"),
            "relevanceScore": s.get("relevanceScore"),
            "content": s.get("content") or "",
        }
        for s in sources
    ]
    return {
        "answer": body.get("answer") or "",
        "retrieved": retrieved,
        "searchQuery": body.get("searchQuery"),
        "searchMode": body.get("searchMode"),
        "model": body.get("model"),
    }


def load_fixtures(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    items = data.get("items") if isinstance(data, dict) else data
    return {item["id"]: item for item in items}


def score_case(
    case: dict[str, Any],
    *,
    answer: str,
    retrieved: list[dict[str, Any]],
    use_judge: bool,
    ollama_base: str,
    judge_model: str,
) -> dict[str, Any]:
    assertion = run_assertions(case=case, answer=answer, retrieved=retrieved)
    ctx_rec = context_recall(
        retrieved=retrieved, expected_document=case.get("expected_document")
    )
    ctx_prec = context_precision(
        retrieved=retrieved, expected_document=case.get("expected_document")
    )

    judge_result: dict[str, Any] = {"enabled": False, "passed": True, "score": None}
    if use_judge and (case.get("judge") or {}).get("enabled"):
        judge_result = run_judges(
            case=case,
            question=case["question"],
            answer=answer,
            retrieved=retrieved,
            ollama_base=ollama_base,
            model=judge_model,
        )

    # Overall pass: assertions must pass; judge must pass when enabled
    overall = assertion["passed"] and (
        not judge_result.get("enabled") or judge_result.get("passed")
    )

    # Combined score: assertion score + optional judge (equal weight when judge on)
    if judge_result.get("enabled") and judge_result.get("score") is not None:
        combined = 0.5 * assertion["score"] + 0.5 * float(judge_result["score"])
    else:
        combined = assertion["score"]

    return {
        "id": case["id"],
        "problem_type": case.get("problem_type"),
        "source": case.get("source"),
        "passed": overall,
        "score": combined,
        "assertions": assertion,
        "judge": judge_result,
        "context_recall": ctx_rec,
        "context_precision": ctx_prec,
        "answer": answer,
        "retrieved_files": [r.get("fileName") for r in retrieved],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("problem_type") or "unknown")].append(row)

    per_type = {}
    for ptype, items in by_type.items():
        m = max(len(items), 1)
        per_type[ptype] = {
            "n": len(items),
            "pass_rate": sum(1 for i in items if i["passed"]) / m,
            "mean_score": sum(float(i["score"]) for i in items) / m,
        }

    ctx_rec = [r["context_recall"] for r in rows if r.get("context_recall") is not None]
    ctx_prec = [r["context_precision"] for r in rows if r.get("context_precision") is not None]

    return {
        "n": len(rows),
        "pass_rate": sum(1 for r in rows if r["passed"]) / n,
        "mean_score": sum(float(r["score"]) for r in rows) / n,
        "assertion_pass_rate": sum(1 for r in rows if r["assertions"]["passed"]) / n,
        "mean_context_recall": (sum(ctx_rec) / len(ctx_rec)) if ctx_rec else None,
        "mean_context_precision": (sum(ctx_prec) / len(ctx_prec)) if ctx_prec else None,
        "by_problem_type": per_type,
    }


def compare(current: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    cs = current["summary"]
    ps = prior["summary"]
    delta = {
        "pass_rate": cs["pass_rate"] - ps["pass_rate"],
        "mean_score": cs["mean_score"] - ps["mean_score"],
        "assertion_pass_rate": cs["assertion_pass_rate"] - ps["assertion_pass_rate"],
        "by_problem_type": {},
    }
    for ptype, cur in (cs.get("by_problem_type") or {}).items():
        old = (ps.get("by_problem_type") or {}).get(ptype) or {
            "pass_rate": 0.0,
            "mean_score": 0.0,
        }
        delta["by_problem_type"][ptype] = {
            "pass_rate": cur["pass_rate"] - old["pass_rate"],
            "mean_score": cur["mean_score"] - old["mean_score"],
        }

    prior_by_id = {c["id"]: c for c in prior.get("cases") or []}
    improved = []
    regressed = []
    for row in current.get("cases") or []:
        old = prior_by_id.get(row["id"])
        if not old:
            continue
        if (not old["passed"]) and row["passed"]:
            improved.append(row["id"])
        if old["passed"] and (not row["passed"]):
            regressed.append(row["id"])
    delta["improved_case_ids"] = improved
    delta["regressed_case_ids"] = regressed
    return delta


def main() -> int:
    parser = argparse.ArgumentParser(description="Run answer evals (assertions + optional judge)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--label", default="run")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--api-base", default="http://localhost:8000/api")
    parser.add_argument("--fixtures", type=Path, default=None, help="Offline fixture JSON")
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ollama-base", default="http://localhost:11434")
    parser.add_argument("--judge-model", default="llama3.2")
    parser.add_argument("--compare-to", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    suite = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = list(suite.get("cases") or [])
    if args.limit:
        cases = cases[: args.limit]

    fixtures = load_fixtures(args.fixtures) if args.fixtures else None
    rows: list[dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        cid = case["id"]
        print(f"[{i}/{len(cases)}] {cid}")
        try:
            if fixtures is not None:
                if cid not in fixtures:
                    print(f"  SKIP: no fixture for {cid}", file=sys.stderr)
                    continue
                fx = fixtures[cid]
                answer = fx.get("answer") or ""
                retrieved = fx.get("retrieved") or []
            else:
                live = chat_live(args.api_base, case["question"])
                answer = live["answer"]
                retrieved = live["retrieved"]
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            rows.append(
                {
                    "id": cid,
                    "problem_type": case.get("problem_type"),
                    "source": case.get("source"),
                    "passed": False,
                    "score": 0.0,
                    "assertions": {"passed": False, "score": 0.0, "checks": []},
                    "judge": {"enabled": False},
                    "context_recall": None,
                    "context_precision": None,
                    "error": str(exc),
                    "answer": "",
                    "retrieved_files": [],
                }
            )
            continue

        scored = score_case(
            case,
            answer=answer,
            retrieved=retrieved,
            use_judge=args.judge,
            ollama_base=args.ollama_base,
            judge_model=args.judge_model,
        )
        mark = "PASS" if scored["passed"] else "FAIL"
        print(
            f"  {mark} score={scored['score']:.2f} "
            f"assert={scored['assertions']['passed']} "
            f"files={scored['retrieved_files']}"
        )
        rows.append(scored)
        if fixtures is None:
            time.sleep(max(0.0, args.sleep))

    summary = summarize(rows)
    payload = {
        "summary": {
            "label": args.label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "fixtures" if fixtures is not None else "live",
            "judge": args.judge,
            **summary,
        },
        "cases": rows,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{args.label}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(f"label:              {args.label}")
    print(f"pass_rate:          {summary['pass_rate']:.3f}")
    print(f"mean_score:         {summary['mean_score']:.3f}")
    print(f"assertion_pass_rate:{summary['assertion_pass_rate']:.3f}")
    if summary["mean_context_recall"] is not None:
        print(f"context_recall:     {summary['mean_context_recall']:.3f}")
    if summary["mean_context_precision"] is not None:
        print(f"context_precision:  {summary['mean_context_precision']:.3f}")
    print("by problem_type:")
    for ptype, stats in sorted(summary["by_problem_type"].items()):
        print(
            f"  - {ptype}: pass_rate={stats['pass_rate']:.3f} "
            f"mean_score={stats['mean_score']:.3f} n={stats['n']}"
        )
    print(f"wrote: {out}")

    if args.compare_to:
        prior_path = args.results_dir / f"{args.compare_to}.json"
        if not prior_path.exists():
            print(f"No prior results at {prior_path}", file=sys.stderr)
            return 1
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        delta = compare(payload, prior)
        print()
        print(f"=== vs {args.compare_to} ===")
        print(
            f"pass_rate:  {prior['summary']['pass_rate']:.3f} -> {summary['pass_rate']:.3f} "
            f"({delta['pass_rate']:+.3f})"
        )
        print(
            f"mean_score: {prior['summary']['mean_score']:.3f} -> {summary['mean_score']:.3f} "
            f"({delta['mean_score']:+.3f})"
        )
        print("per problem_type deltas:")
        for ptype, d in sorted(delta["by_problem_type"].items()):
            print(
                f"  - {ptype}: pass_rate {d['pass_rate']:+.3f}, "
                f"mean_score {d['mean_score']:+.3f}"
            )
        if delta["improved_case_ids"]:
            print(f"improved:  {', '.join(delta['improved_case_ids'])}")
        if delta["regressed_case_ids"]:
            print(f"regressed: {', '.join(delta['regressed_case_ids'])}")
        payload["delta_vs"] = {"label": args.compare_to, **delta}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
