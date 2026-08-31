#!/usr/bin/env python3
"""
Collect a batch of complete traces by asking the chat API a list of questions.

Uses eval/golden_qa.json by default (fair coverage of claim topics — not cherry-picked wins).

  python eval/error_analysis/collect_batch.py
  python eval/error_analysis/collect_batch.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN = ROOT / "eval" / "golden_qa.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "traces" / "traces.jsonl"


def post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    req = Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: float = 60.0) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect chat traces for error analysis")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--api-base", default="http://localhost:8000/api")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument(
        "--export",
        type=Path,
        default=DEFAULT_OUT,
        help="Also mirror API-recorded traces into this JSONL (from GET /traces)",
    )
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    items = list(golden.get("items") or [])
    if args.limit:
        items = items[: args.limit]

    chat_url = f"{args.api_base.rstrip('/')}/chat"
    ok = 0
    for i, item in enumerate(items, start=1):
        q = item["question"]
        qid = item.get("id") or f"q{i}"
        print(f"[{i}/{len(items)}] {qid}: {q[:70]}…")
        try:
            data = post_json(
                chat_url,
                {"question": q, "debug": True, "agentId": "claims_assistant"},
            )
            body = data.get("data") if isinstance(data.get("data"), dict) else data
            print(f"  -> messageId={body.get('messageId')} sources={len(body.get('sources') or [])}")
            ok += 1
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
        time.sleep(max(0.0, args.sleep))

    # Pull whatever the backend recorded and mirror locally for offline sampling.
    try:
        listed = get_json(f"{args.api_base.rstrip('/')}/traces?limit=500")
        body = listed.get("data") if isinstance(listed.get("data"), dict) else listed
        traces = list(body.get("items") or [])
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("w", encoding="utf-8") as fh:
            for t in reversed(traces):  # chronological
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"Exported {len(traces)} traces -> {args.export}")
    except Exception as exc:  # noqa: BLE001
        print(f"Export skipped: {exc}", file=sys.stderr)

    print(f"Asked {ok}/{len(items)} questions (traces auto-saved by backend).")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
