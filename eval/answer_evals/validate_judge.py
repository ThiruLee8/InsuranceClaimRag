#!/usr/bin/env python3
"""
Validate LLM-as-judge against human labels before trusting it.

  python eval/answer_evals/validate_judge.py
  python eval/answer_evals/validate_judge.py --labels eval/answer_evals/human_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import run_judges  # noqa: E402


def agreement(y_true: list[bool], y_pred: list[bool]) -> dict:
    n = max(len(y_true), 1)
    matches = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    tn = sum(1 for a, b in zip(y_true, y_pred) if (not a) and (not b))
    fp = sum(1 for a, b in zip(y_true, y_pred) if (not a) and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and (not b))
    return {
        "n": len(y_true),
        "accuracy": matches / n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LLM judge vs human labels")
    parser.add_argument("--labels", type=Path, default=HERE / "human_labels.json")
    parser.add_argument("--ollama-base", default="http://localhost:11434")
    parser.add_argument("--judge-model", default="llama3.2")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "judge_validation.json")
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.7,
        help="Exit non-zero if accuracy below this (default 0.7)",
    )
    parser.add_argument(
        "--dry-run-offline",
        action="store_true",
        help="Skip Ollama; compare human labels to recorded expected_judge_pass only",
    )
    args = parser.parse_args()

    data = json.loads(args.labels.read_text(encoding="utf-8-sig"))
    items = list(data.get("items") or [])
    if not items:
        print("No labeled items", file=sys.stderr)
        return 1

    y_true: list[bool] = []
    y_pred: list[bool] = []
    details = []

    for item in items:
        human_pass = bool(item["human_pass"])
        y_true.append(human_pass)

        if args.dry_run_offline:
            # Use embedded expected_judge_pass from the label file (for CI without Ollama)
            pred = bool(item.get("expected_judge_pass", human_pass))
            judge_payload = {
                "enabled": True,
                "passed": pred,
                "score": 1.0 if pred else 0.2,
                "criteria": [{"criterion": "offline", "pass": pred, "score": 10 if pred else 2}],
            }
        else:
            case = {
                "judge": {
                    "enabled": True,
                    "criteria": item.get("criteria") or ["faithfulness", "answer_relevancy"],
                    "pass_threshold": item.get("pass_threshold", 0.7),
                }
            }
            judge_payload = run_judges(
                case=case,
                question=item["question"],
                answer=item["answer"],
                retrieved=item.get("retrieved") or [],
                ollama_base=args.ollama_base,
                model=args.judge_model,
            )
            pred = bool(judge_payload.get("passed"))

        y_pred.append(pred)
        details.append(
            {
                "id": item.get("id"),
                "human_pass": human_pass,
                "judge_pass": pred,
                "agree": human_pass == pred,
                "human_note": item.get("human_note"),
                "judge": judge_payload,
            }
        )
        mark = "OK" if human_pass == pred else "DISAGREE"
        print(f"[{mark}] {item.get('id')}: human={human_pass} judge={pred}")

    stats = agreement(y_true, y_pred)
    report = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.judge_model,
            "offline": args.dry_run_offline,
            "min_agreement": args.min_agreement,
        },
        "agreement": stats,
        "items": details,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=== Judge vs human ===")
    print(f"accuracy: {stats['accuracy']:.3f} ({stats['n']} items)")
    print(f"tp={stats['tp']} tn={stats['tn']} fp={stats['fp']} fn={stats['fn']}")
    print(f"wrote: {args.out}")

    if stats["accuracy"] < args.min_agreement:
        print(
            f"FAIL: agreement {stats['accuracy']:.3f} < {args.min_agreement}",
            file=sys.stderr,
        )
        return 2
    print("PASS: judge agreement meets threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
