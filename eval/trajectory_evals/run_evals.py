#!/usr/bin/env python3
"""
Trajectory evals: judge the path, not just the answer.

  python eval/trajectory_evals/run_evals.py --label before --fixtures eval/trajectory_evals/fixtures/before.json
  python eval/trajectory_evals/run_evals.py --label after  --fixtures eval/trajectory_evals/fixtures/after.json --compare-to before
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CASES = HERE / "cases.json"
DEFAULT_RESULTS = HERE / "results"

sys.path.insert(0, str(HERE))
from score import compare, score_trajectory, summarize  # noqa: E402


def _post_json(url: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def agent_live(api_base: str, task: str) -> dict[str, Any]:
    data = _post_json(
        f"{api_base.rstrip('/')}/agent-loop/run",
        {"task": task, "mode": "agent", "persistMemory": False, "maxSteps": 8},
    )
    body = data.get("data") if isinstance(data.get("data"), dict) else data
    return {
        "answer": body.get("answer") or "",
        "steps": body.get("steps") or [],
        "metrics": body.get("metrics") or {},
        "verdict": body.get("verdict") or {},
    }


def load_fixtures(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    items = data.get("items") if isinstance(data, dict) else data
    return {item["id"]: item for item in items}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent trajectory evals")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--label", default="run")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--api-base", default="http://localhost:8000/api")
    parser.add_argument("--fixtures", type=Path, default=None)
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
                run = fixtures[cid]
            else:
                run = agent_live(args.api_base, case["task"])
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            rows.append(
                {
                    "id": cid,
                    "problem_type": case.get("problem_type"),
                    "passed": False,
                    "score": 0.0,
                    "outcome_pass": False,
                    "trajectory_pass": False,
                    "gap": False,
                    "error": str(exc),
                }
            )
            continue

        scored = score_trajectory(case, run)
        mark = "PASS" if scored["passed"] else "FAIL"
        gap = " GAP" if scored["gap"] else ""
        print(
            f"  {mark}{gap} outcome={scored['outcome_pass']} "
            f"traj={scored['trajectory_pass']} tools={scored['tools']}"
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
            "top_failure": suite.get("top_failure"),
            **summary,
        },
        "cases": rows,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"{args.label}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=== Summary ===")
    print(f"label:                   {args.label}")
    print(f"pass_rate:               {summary['pass_rate']:.3f}")
    print(f"outcome_pass_rate:       {summary['outcome_pass_rate']:.3f}")
    print(f"trajectory_pass_rate:    {summary['trajectory_pass_rate']:.3f}")
    print(f"gap_rate (right≠path):   {summary['gap_rate']:.3f}")
    print(f"injection_followed_rate: {summary['injection_followed_rate']:.3f}")
    print(f"cost_per_task mean:      {summary['cost_per_task_mean']:.3f}")
    print(f"cost_per_task p99:       {summary['cost_per_task_p99']:.3f}")
    if summary["mean_tool_choice_accuracy"] is not None:
        print(f"tool_choice_accuracy:    {summary['mean_tool_choice_accuracy']:.3f}")
    print("by problem_type:")
    for ptype, stats in sorted(summary["by_problem_type"].items()):
        print(
            f"  - {ptype}: pass={stats['pass_rate']:.3f} "
            f"hijack={stats['injection_followed_rate']:.3f} n={stats['n']}"
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
            f"injection_followed_rate: {prior['summary']['injection_followed_rate']:.3f} -> "
            f"{summary['injection_followed_rate']:.3f} ({delta['injection_followed_rate']:+.3f})"
        )
        print(
            f"pass_rate:               {prior['summary']['pass_rate']:.3f} -> "
            f"{summary['pass_rate']:.3f} ({delta['pass_rate']:+.3f})"
        )
        print("per problem_type:")
        for ptype, d in sorted(delta["by_problem_type"].items()):
            print(
                f"  - {ptype}: pass {d['pass_rate']:+.3f}, "
                f"hijack {d['injection_followed_rate']:+.3f}"
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
