from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


def _default_traces_dir() -> Path:
    settings = get_settings()
    raw = (settings.traces_dir or "").strip()
    if raw:
        return Path(raw)
    # Prefer repo-local path when running outside Docker.
    repo_eval = Path(__file__).resolve().parents[3] / "eval" / "error_analysis" / "traces"
    if repo_eval.parent.exists() or Path(__file__).resolve().parents[3].name == "InsuranceClaimRag":
        return repo_eval
    return Path("/app/data/traces")


class TraceService:
    """Append-only JSONL store of complete chat traces for error analysis."""

    def __init__(self, traces_dir: Path | None = None) -> None:
        self.settings = get_settings()
        self.traces_dir = traces_dir or _default_traces_dir()
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.traces_dir / "traces.jsonl"

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enable_trace_logging)

    def record(
        self,
        *,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        conversation_id: str | None = None,
        message_id: str | None = None,
        agent_id: str | None = None,
        model: str | None = None,
        original_question: str | None = None,
        search_query: str | None = None,
        search_mode: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        trace_id = str(uuid.uuid4())
        # Always persist full chunk text for replay — independent of ChatRequest.debug.
        retrieved = []
        for src in sources:
            retrieved.append(
                {
                    "documentId": src.get("documentId") or src.get("document_id"),
                    "chunkId": src.get("chunkId") or src.get("chunk_id"),
                    "fileName": src.get("fileName") or src.get("file_name"),
                    "pageNumber": src.get("pageNumber") or src.get("page_number"),
                    "relevanceScore": src.get("relevanceScore")
                    if src.get("relevanceScore") is not None
                    else src.get("score"),
                    "content": src.get("content") or "",
                }
            )

        trace: dict[str, Any] = {
            "traceId": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversationId": conversation_id,
            "messageId": message_id,
            "question": question,
            "originalQuestion": original_question or question,
            "searchQuery": search_query or question,
            "searchMode": search_mode,
            "agentId": agent_id,
            "model": model,
            "retrieved": retrieved,
            "answer": answer,
        }
        if extra:
            trace["extra"] = extra

        line = json.dumps(trace, default=str, ensure_ascii=False)
        with _lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        logger.info(
            "trace_recorded",
            trace_id=trace_id,
            retrieved=len(retrieved),
            path=str(self.log_path),
        )
        return trace

    def list_traces(self, *, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        rows = self._read_all()
        rows.reverse()  # newest first
        return rows[offset : offset + max(1, limit)]

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        for row in self._read_all():
            if row.get("traceId") == trace_id:
                return row
        return None

    def count(self) -> int:
        return len(self._read_all())

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


_trace_service: TraceService | None = None


def get_trace_service() -> TraceService:
    global _trace_service
    if _trace_service is None:
        _trace_service = TraceService()
    return _trace_service
