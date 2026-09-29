# Single agent vs specialist team — Week 10

Same four questions. Both sides. Four numbers. The rule below was fixed before the run.

## Mentor checklist

| Check | Result |
|-------|--------|
| Same tests for both | `cases.json` — one fact, three independent facts, coverage only, cause-then-coverage-then-money |
| Quality, speed, tokens, cost | Table below, from `results/race.json` |
| Verdict follows the numbers | **Ship the specialist team.** Quality tied. The team used fewer tokens, so it cost less. |
| When a team is worth it | When each specialist's prompt is smaller than the single agent's tool list and growing scratchpad, or when independent slices can run at the same time |
| When it is not | When the manager calls specialists the question does not need, or a planning model re-sends the task before anyone works. Chat RAG is one short prompt — a team would be waste there |

## Decision rule

1. Quality gap of 0.05 or more wins (same fact checks on both answers).
2. Else a cost gap of 5% or more wins (lower cost).
3. Else a speed gap of 10% or more wins.
4. Else keep the single agent.

Cost is `llm_calls + tokens/4000`, the same formula as the existing agent. Tokens are characters ÷ 4 of the real prompts, including every copy of the user task.

## What was raced

- **Single agent** — the existing ReAct loop (`ClaimAgentRunner.run_agent`). Every turn re-sends the tool list and the scratchpad.
- **Team** — a manager picks specialists from Agent Cards (loss, coverage, settlement). Independent facts run together. Dependent steps run in order. Each hand-off pastes the original task again. The manager writes the final answer. No extra planning model call (that would re-send the task once more; the page shows how many tokens that would add).

This recorded run does **not** call Ollama. Both sides answer with the text they retrieved, and each model call waits 0.05s so a parallel team can overlap that wait. That holds retrieval and generation quality constant and measures the orchestration. Re-run live when Ollama and the document index are up:

```bash
python eval/multi_agent/run_race.py --scripted --delay 0.05
python eval/multi_agent/run_race.py --live
```

## Means (4 questions)

| | Single agent | Specialist team |
|--|-------------:|----------------:|
| Quality | 100% | 100% |
| Speed | 189 ms | 154 ms |
| Tokens | 2,383 | 1,369 |
| Cost | 3.596 | 3.342 |

**Shipped: the team**, decided by cost. The token gap is the stable part (2,383 vs 1,369). Cost follows it (3.596 vs 3.342, about 7% lower, past the 5% line). Speed favors the team on average mostly because the three-fact question overlaps; the 0.05s wait jitters by a few milliseconds on the other questions.

The team re-sent the original task on every hand-off: **97 tokens on average**. That copy is real. It was still smaller than the tool catalog and scratchpad the single loop pastes on every turn.

## Same questions, one by one

Cells are single / team.

| Test | Quality | Speed (ms) | Tokens | Cost | Hand-off |
|------|---------|------------|--------|------|----------|
| One fact | 100% / 100% | 133 / 124 | 1,330 / 644 | 2.333 / 2.161 | 1 specialist, task pasted twice (14 tokens) |
| Three independent facts | 100% / 100% | 248 / 122 | 3,335 / 1,831 | 4.834 / 4.458 | 3 specialists at once. Wall clock is about half. Sum of specialist time is higher than the wall clock |
| Coverage only | 100% / 100% | 123 / 124 | 1,392 / 726 | 2.348 / 2.181 | Coverage only — no fan-out |
| Cause, then coverage, then money | 100% / 100% | 250 / 246 | 3,475 / 2,274 | 4.869 / 4.569 | In order, not parallel. Task re-sent 4 times (240 tokens) plus earlier findings |

## What this does not say

- It does not say a team is always cheaper. If the manager had called all three specialists for "what was the cause?", the extra calls would have lost. The router does not do that.
- It does not say chat should become a team. Chat is one short RAG prompt. This race is against the ReAct investigation loop.
- CrewAI and AutoGen use the same manager-and-workers shape. They hide the hand-off, which hides this token line. The hand-off here is a normal function, and `POST /api/multi-agent/tasks` is the same task another agent can call.
- MCP is how a specialist searches documents. A2A is how the manager hands that specialist a task (`submitted → working → completed`). The model stays on this host either way.

## Where to look

| Piece | Path |
|-------|------|
| Agent Cards and task lifecycle | `backend/app/services/a2a.py` |
| Manager, specialists, race | `backend/app/services/multi_agent.py` |
| Score rule | `eval/multi_agent/score.py` |
| Questions | `eval/multi_agent/cases.json` |
| Numbers | `eval/multi_agent/results/race.json` |
| Page | `/team` |
