from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "this",
    "that",
    "was",
    "were",
    "is",
    "are",
    "be",
    "it",
    "as",
    "by",
    "at",
}


@dataclass
class MemoryEntry:
    id: str
    session_id: str
    created_at: str
    task: str
    summary: str
    facts: list[str]


def _tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def summarise_run(task: str, answer: str, facts: list[str] | None = None) -> str:
    """Extractive summary — no extra LLM call. Stand-in for mem0-style compression."""
    compact_answer = re.sub(r"\s+", " ", (answer or "").strip())
    if len(compact_answer) > 500:
        compact_answer = compact_answer[:497].rsplit(" ", 1)[0] + "…"
    fact_line = ""
    if facts:
        fact_line = " Key facts: " + "; ".join(facts[:4])
    task_bit = (task or "").strip().replace("\n", " ")
    if len(task_bit) > 140:
        task_bit = task_bit[:137] + "…"
    return f"Task: {task_bit}. Finding: {compact_answer}.{fact_line}".strip()


class AgentMemoryStore:
    """
    Long-term memory for the claims agent.

    Short-term memory lives in the ReAct scratchpad (current run only).
    This store keeps summarised findings across runs in a session —
    a teaching stand-in for vector memory / mem0 (keyword Jaccard, not embeddings).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        settings = get_settings()
        raw = path or settings.agent_memory_path
        if raw:
            self.path = Path(raw)
        else:
            self.path = Path(__file__).resolve().parents[2] / "data" / "agent_memory.json"
        self._lock = threading.Lock()

    def _load(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("agent_memory_load_failed", error=str(exc))
            return []
        entries: list[MemoryEntry] = []
        for item in data.get("entries") or []:
            entries.append(
                MemoryEntry(
                    id=str(item.get("id") or uuid.uuid4()),
                    session_id=str(item.get("session_id") or item.get("sessionId") or ""),
                    created_at=str(item.get("created_at") or item.get("createdAt") or ""),
                    task=str(item.get("task") or ""),
                    summary=str(item.get("summary") or ""),
                    facts=list(item.get("facts") or []),
                )
            )
        return entries

    def _save(self, entries: list[MemoryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [asdict(e) for e in entries]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_entries(self, session_id: str | None = None, limit: int = 40) -> list[MemoryEntry]:
        with self._lock:
            entries = self._load()
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]

    def add(
        self,
        *,
        session_id: str,
        task: str,
        summary: str,
        facts: list[str] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            task=task.strip(),
            summary=summary.strip(),
            facts=[f.strip() for f in (facts or []) if f and f.strip()],
        )
        with self._lock:
            entries = self._load()
            entries.append(entry)
            self._save(entries)
        logger.info("agent_memory_saved", session_id=session_id, entry_id=entry.id)
        return entry

    def add_fact(self, *, session_id: str, fact: str, task: str = "") -> MemoryEntry:
        return self.add(session_id=session_id, task=task or "(fact)", summary=fact, facts=[fact])

    def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        limit: int = 5,
        min_score: float = 0.04,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve by token overlap (Jaccard). Same job embeddings would do in mem0."""
        q = _tokens(query)
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in self.list_entries(session_id=session_id, limit=200):
            blob = " ".join([entry.task, entry.summary, " ".join(entry.facts)])
            score = _jaccard(q, _tokens(blob))
            if session_id and entry.session_id == session_id:
                score += 0.05
            if score >= min_score:
                scored.append((entry, round(score, 3)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def clear(self, session_id: str | None = None) -> int:
        with self._lock:
            entries = self._load()
            if session_id:
                kept = [e for e in entries if e.session_id != session_id]
                removed = len(entries) - len(kept)
                self._save(kept)
            else:
                removed = len(entries)
                self._save([])
        logger.info("agent_memory_cleared", session_id=session_id, removed=removed)
        return removed
