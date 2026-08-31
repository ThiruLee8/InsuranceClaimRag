#!/usr/bin/env python3
"""
Rank error taxonomy from a coded sample worksheet.

Input: sample JSON after you filled open_code_note, severity, problem_type.
Output: ranked problem list by frequency × mean severity, plus fix-target stub.

  python eval/error_analysis/rank_problems.py --sample samples/sample_....json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank error types by frequency × severity")
    parser.add_argument("--sample", type=Path, required=True, help="Coded sample JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write ranked report JSON (default: alongside sample)",
    )
    parser.add_argument(
        "--prediction-out",
        type=Path,
        default=None,
        help="Write fix-target prediction stub markdown",
    )
    args = parser.parse_args()

    data = json.loads(args.sample.read_text(encoding="utf-8"))
    items = list(data.get("items") or [])
    if not items:
        print("Sample has no items", file=sys.stderr)
        return 1

    coded = []
    uncoded = []
    for item in items:
        note = (item.get("open_code_note") or "").strip()
        ptype = (item.get("problem_type") or "").strip()
        ok = item.get("ok")
        if ok is True:
            continue
        if not note:
            uncoded.append(item.get("traceId"))
            continue
        severity = item.get("severity")
        try:
            severity_n = int(severity) if severity is not None else 2
        except (TypeError, ValueError):
            severity_n = 2
        severity_n = max(1, min(3, severity_n))
        coded.append(
            {
                "traceId": item.get("traceId"),
                "question": item.get("question"),
                "open_code_note": note,
                "severity": severity_n,
                "problem_type": ptype or "UNCATEGORIZED",
            }
        )

    if uncoded:
        print(
            f"Warning: {len(uncoded)} failures still missing open_code_note "
            f"(code notes BEFORE grouping).",
            file=sys.stderr,
        )

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in coded:
        by_type[row["problem_type"]].append(row)

    ranked = []
    for name, rows in by_type.items():
        freq = len(rows)
        mean_sev = sum(r["severity"] for r in rows) / freq
        score = freq * mean_sev
        ranked.append(
            {
                "problem_type": name,
                "frequency": freq,
                "mean_severity": round(mean_sev, 2),
                "impact_score": round(score, 2),
                "example_notes": [r["open_code_note"] for r in rows[:3]],
                "trace_ids": [r["traceId"] for r in rows],
            }
        )
    ranked.sort(key=lambda r: (-r["impact_score"], -r["frequency"], r["problem_type"]))

    ok_count = sum(1 for i in items if i.get("ok") is True)
    fail_count = len(items) - ok_count
    report = {
        "meta": {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "samplePath": str(args.sample),
            "sampleSize": len(items),
            "okCount": ok_count,
            "failCount": fail_count,
            "codedFailures": len(coded),
            "uncodedFailures": len(uncoded),
        },
        "ranked_problems": ranked,
        "chosen_fix_target": ranked[0]["problem_type"] if ranked else None,
    }

    out = args.out or args.sample.with_name(args.sample.stem + "_ranked.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print()
    print("=== Ranked problems (frequency × mean severity) ===")
    if not ranked:
        print("(no coded failures)")
    for i, row in enumerate(ranked, start=1):
        print(
            f"{i}. {row['problem_type']}  "
            f"freq={row['frequency']}  sev={row['mean_severity']}  "
            f"impact={row['impact_score']}"
        )
        for note in row["example_notes"][:2]:
            print(f"   - {note}")

    pred_path = args.prediction_out or args.sample.with_name(
        args.sample.stem + "_prediction.md"
    )
    target = report["chosen_fix_target"] or "TBD"
    pred = f"""# Fix-target prediction

Write this **before** you implement the fix.

## Chosen target
`{target}`

## Why this one (from ranking)
Impact score = frequency × mean severity. See `{out.name}`.

## What I will change
(one concrete change — not five)

## What I expect to happen
- Failures of type `{target}` should drop from **{next((r['frequency'] for r in ranked if r['problem_type']==target), 0)}** in this sample to about **___**.
- Other problem types I do **not** expect to fix: _______________

## How I will measure
Re-sample / re-code N traces after the change, or re-run retrieval eval if the target is retrieval-related.
"""
    pred_path.write_text(pred, encoding="utf-8")
    print(f"Wrote prediction stub {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
