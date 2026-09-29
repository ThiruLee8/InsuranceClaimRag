"""Score a single-agent vs specialist-team race.

The decision rule is fixed before looking at a run:

1. Quality gap of 0.05 or more wins (same assertions on both answers).
2. Else a cost gap of 5% or more wins (lower cost).
3. Else a speed gap of 10% or more wins (lower elapsed time).
4. Else keep the single agent — same result, fewer hand-offs.
"""

from __future__ import annotations

from typing import Any


DECISION_RULE = (
    "Quality first (0.05 gap on the same assertions). "
    "If quality ties, lower cost wins (5% gap). "
    "If quality and cost tie, lower latency wins (10% gap). "
    "If all three tie, keep the single agent."
)

QUALITY_GAP = 0.05
COST_RATIO = 0.95
SPEED_RATIO = 0.90


def quality_score(answer: str, groups: list[list[str]] | None) -> float | None:
    """Fraction of fact-groups found. Each group passes if any alias is present."""
    if not groups:
        return None
    text = (answer or "").lower()
    hits = 0
    for group in groups:
        if any(str(alias).lower() in text for alias in group):
            hits += 1
    return hits / len(groups)


def compare_sides(single: dict[str, Any], team: dict[str, Any]) -> dict[str, Any]:
    """Pick a winner from one pair of summaries. Delta is team minus single."""
    qs = single.get("quality")
    qt = team.get("quality")
    scored = qs is not None and qt is not None

    def row(key: str) -> dict[str, float | None]:
        av = single.get(key)
        bv = team.get(key)
        delta = None
        if av is not None and bv is not None:
            delta = round(float(bv) - float(av), 4)
        return {"single": av, "team": bv, "delta": delta}

    comparison = {
        "quality": row("quality"),
        "elapsedMs": row("elapsedMs"),
        "estimatedTokens": row("estimatedTokens"),
        "costUnits": row("costUnits"),
    }

    ship = "single"
    why = "quality"
    if scored and float(qt) >= float(qs) + QUALITY_GAP:
        ship, why = "team", "quality"
    elif scored and float(qs) >= float(qt) + QUALITY_GAP:
        ship, why = "single", "quality"
    elif float(team["costUnits"]) < float(single["costUnits"]) * COST_RATIO:
        ship, why = "team", "cost"
    elif float(single["costUnits"]) < float(team["costUnits"]) * COST_RATIO:
        ship, why = "single", "cost"
    elif float(team["elapsedMs"]) < float(single["elapsedMs"]) * SPEED_RATIO:
        ship, why = "team", "speed"
    elif float(single["elapsedMs"]) < float(team["elapsedMs"]) * SPEED_RATIO:
        ship, why = "single", "speed"
    else:
        ship, why = "single", "tie"

    rationale = _rationale(ship, why, single, team, scored)
    return {
        "ship": ship,
        "winner": ship,
        "decidedBy": why,
        "rule": DECISION_RULE,
        "rationale": rationale,
        "comparison": comparison,
    }


def _fmt_quality(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0%}"


def _rationale(
    ship: str,
    why: str,
    single: dict[str, Any],
    team: dict[str, Any],
    scored: bool,
) -> str:
    kept = "specialist team" if ship == "team" else "single agent"
    numbers = (
        f"Quality {_fmt_quality(single.get('quality'))} vs {_fmt_quality(team.get('quality'))}. "
        f"Speed {float(single['elapsedMs']):.0f} ms vs {float(team['elapsedMs']):.0f} ms. "
        f"Tokens {int(single['estimatedTokens'])} vs {int(team['estimatedTokens'])}. "
        f"Cost {float(single['costUnits']):.3f} vs {float(team['costUnits']):.3f}."
    )
    if not scored:
        lead = "This wording has no assertion list, so quality was not scored. "
    elif why == "quality":
        lead = "The quality gap on the same tests decides it. "
    elif why == "cost":
        lead = "Quality is tied, so the lower token bill decides it. "
    elif why == "speed":
        lead = "Quality and cost are tied, so waiting time decides it. "
    else:
        lead = "Quality, cost, and speed are too close to call. "
    if ship == "single" and why in {"cost", "tie", "speed"}:
        tail = (
            " Keep the single agent. Every hand-off re-sends the task, "
            "and this suite does not pay that back."
        )
    elif ship == "team" and why == "speed":
        tail = (
            " Keep the team for this task: the specialists ran at the same time "
            "and the token bill did not go up."
        )
    elif ship == "team" and why == "cost":
        tail = (
            " Keep the team. The specialists carry a smaller prompt than the "
            "single agent's tool list and growing scratchpad, and that saving "
            "beat the cost of pasting the task again."
        )
    elif ship == "team":
        tail = " Keep the team. The quality gap on these tests covers the extra hand-offs."
    else:
        tail = " Keep the single agent."
    return f"{lead}{numbers} Ship the {kept}.{tail}"


def summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of the four metrics, then the same decision rule on those means."""
    if not case_results:
        raise ValueError("no case results")

    def avg(side: str, key: str) -> float:
        values = [float(item[side][key]) for item in case_results]
        return sum(values) / len(values)

    single = {
        "quality": round(avg("single", "quality"), 4),
        "elapsedMs": round(avg("single", "elapsedMs"), 1),
        "estimatedTokens": round(avg("single", "estimatedTokens"), 1),
        "costUnits": round(avg("single", "costUnits"), 3),
    }
    team = {
        "quality": round(avg("team", "quality"), 4),
        "elapsedMs": round(avg("team", "elapsedMs"), 1),
        "estimatedTokens": round(avg("team", "estimatedTokens"), 1),
        "costUnits": round(avg("team", "costUnits"), 3),
    }
    resent = [float(item.get("contextResendTokens") or 0) for item in case_results]
    parallel_faster = [
        item["id"]
        for item in case_results
        if item.get("parallel")
        and float(item["team"]["elapsedMs"]) < float(item["single"]["elapsedMs"]) * SPEED_RATIO
        and float(item["team"]["quality"]) + 1e-9 >= float(item["single"]["quality"]) - QUALITY_GAP
    ]
    return {
        "cases": len(case_results),
        "single": single,
        "team": team,
        "contextResendTokensMean": round(sum(resent) / len(resent), 1),
        "verdict": compare_sides(single, team),
        "worthIt": {
            "when": (
                "When each specialist can hold a smaller prompt than the single "
                "agent's full scratchpad and tool list, or when independent slices "
                "run together and waiting time is the limit."
            ),
            "whenNot": (
                "When the manager calls specialists the question does not need, "
                "or a planning model re-sends the task before anyone works. "
                "A short single prompt, such as chat RAG, is also cheaper than a team. "
                "Hand-offs always re-send the task; that copy has to stay smaller "
                "than what the single agent was already repeating."
            ),
            "parallelFasterOn": parallel_faster,
        },
    }
