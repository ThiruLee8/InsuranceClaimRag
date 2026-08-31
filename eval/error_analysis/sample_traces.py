#!/usr/bin/env python3
"""
Draw a fair random sample of complete traces for open coding.

  python eval/error_analysis/sample_traces.py --n 20 --seed 42
  python eval/error_analysis/sample_traces.py --from-api --n 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = Path(__file__).resolve().parent / "traces" / "traces.jsonl"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "samples"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_from_api(base_url: str, limit: int = 500) -> list[dict]:
    url = f"{base_url.rstrip('/')}/traces?limit={limit}&offset=0"
    with urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    body = data.get("data") if isinstance(data.get("data"), dict) else data
    return list(body.get("items") or [])


def compact_for_worksheet(trace: dict) -> dict:
    """Human-readable worksheet row — full enough to replay."""
    retrieved = []
    for i, chunk in enumerate(trace.get("retrieved") or [], start=1):
        content = (chunk.get("content") or "").strip()
        if len(content) > 600:
            content = content[:600] + "…"
        retrieved.append(
            {
                "rank": i,
                "fileName": chunk.get("fileName"),
                "pageNumber": chunk.get("pageNumber"),
                "relevanceScore": chunk.get("relevanceScore"),
                "content": content,
            }
        )
    return {
        "traceId": trace.get("traceId"),
        "timestamp": trace.get("timestamp"),
        "question": trace.get("question") or trace.get("originalQuestion"),
        "searchQuery": trace.get("searchQuery"),
        "searchMode": trace.get("searchMode"),
        "model": trace.get("model"),
        "retrieved": retrieved,
        "answer": trace.get("answer"),
        # Open coding fields — fill BEFORE assigning problem_type
        "ok": None,
        "open_code_note": "",
        "severity": None,
        "problem_type": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Random-sample traces for error analysis")
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--from-api", action="store_true")
    parser.add_argument("--api-base", default="http://localhost:8000/api")
    parser.add_argument("--n", type=int, default=20, help="Sample size")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.from_api:
        pool = load_from_api(args.api_base)
        source = f"api:{args.api_base}"
    else:
        pool = load_jsonl(args.traces)
        source = str(args.traces)

    if not pool:
        print(
            f"No traces found ({source}). Chat with the app first, or run collect_batch.py.",
            file=sys.stderr,
        )
        return 1

    n = min(args.n, len(pool))
    rng = random.Random(args.seed)
    sample = rng.sample(pool, n)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"sample_{stamp}_seed{args.seed}.json"
    out_md = args.out_dir / f"sample_{stamp}_seed{args.seed}.md"

    worksheet = {
        "meta": {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "poolSize": len(pool),
            "sampleSize": n,
            "seed": args.seed,
            "instructions": (
                "1) Read each trace. 2) Write open_code_note (one honest sentence) BEFORE "
                "setting problem_type. 3) Set ok=true/false and severity 1-3 (3=worst). "
                "4) Only then assign problem_type from your emerging taxonomy."
            ),
        },
        "items": [compact_for_worksheet(t) for t in sample],
    }
    out_json.write_text(json.dumps(worksheet, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Error analysis sample (seed={args.seed}, n={n}/{len(pool)})",
        "",
        f"Source: `{source}`",
        "",
        "For each item: write **open_code_note** first, then severity, then problem_type.",
        "",
    ]
    for i, item in enumerate(worksheet["items"], start=1):
        lines.append(f"## {i}. `{item['traceId']}`")
        lines.append("")
        lines.append(f"**Question:** {item['question']}")
        if item.get("searchQuery") and item["searchQuery"] != item["question"]:
            lines.append(f"**Search query:** {item['searchQuery']}")
        lines.append("")
        lines.append("**Retrieved:**")
        for chunk in item["retrieved"]:
            score = chunk.get("relevanceScore")
            score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
            lines.append(
                f"- [{chunk['rank']}] {chunk.get('fileName')} "
                f"(p.{chunk.get('pageNumber')}, score={score_s})"
            )
            lines.append(f"  > {chunk.get('content', '')[:240]}")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(item.get("answer") or "(empty)")
        lines.append("")
        lines.append("- ok: ")
        lines.append("- open_code_note: ")
        lines.append("- severity (1-3): ")
        lines.append("- problem_type: ")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Sampled {n} of {len(pool)} traces (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
