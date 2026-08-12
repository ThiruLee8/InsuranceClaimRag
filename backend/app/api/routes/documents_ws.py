from __future__ import annotations

import asyncio
import hashlib
import json
from uuid import UUID

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db.models import DocumentStatus
from app.services.document_service import DocumentService

router = APIRouter(tags=["documents-ws"])


def _fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_list_payload(
    db: Session,
    *,
    search: str | None,
    status: DocumentStatus | None,
    content_type: str | None,
) -> dict:
    service = DocumentService(db)
    items, total = service.list_documents(
        search=search,
        status=status,
        content_type=content_type,
        skip=0,
        limit=200,
    )
    return {
        "type": "snapshot",
        "total": total,
        "documents": [item.model_dump(mode="json") for item in items],
    }


def _build_document_payload(db: Session, document_id: UUID) -> dict | None:
    service = DocumentService(db)
    try:
        item = service.get_document(document_id)
    except LookupError:
        return {"type": "document", "document": None, "deleted": True}
    return {
        "type": "document",
        "document": item.model_dump(mode="json"),
        "deleted": False,
    }


@router.websocket("/ws/documents")
async def documents_websocket(
    websocket: WebSocket,
    documentId: str | None = Query(default=None),
    search: str | None = Query(default=None),
    status: DocumentStatus | None = Query(default=None),
    contentType: str | None = Query(default=None),
):
    """
    Push document list/detail updates to the UI.

    The browser opens one socket instead of HTTP polling.
    The API watches SQL for changes (Functions still write status there).
    """
    await websocket.accept()
    last_hash: str | None = None
    watch_document_id: UUID | None = None
    if documentId:
        try:
            watch_document_id = UUID(documentId)
        except ValueError:
            await websocket.send_json({"type": "error", "message": "Invalid documentId"})
            await websocket.close(code=1008)
            return

    try:
        while True:
            db = SessionLocal()
            try:
                if watch_document_id:
                    payload = _build_document_payload(db, watch_document_id)
                else:
                    payload = _build_list_payload(
                        db,
                        search=search,
                        status=status,
                        content_type=contentType,
                    )
            finally:
                db.close()

            if payload is not None:
                digest = _fingerprint(payload)
                if digest != last_hash:
                    await websocket.send_json(payload)
                    last_hash = digest

                    # Stop watching a single document once terminal.
                    if watch_document_id and payload.get("type") == "document":
                        doc = payload.get("document") or {}
                        if payload.get("deleted") or doc.get("status") in {
                            "Completed",
                            "Processed",
                            "Failed",
                            "Deleted",
                        }:
                            # Keep connection open but slow down; client may reprocess.
                            await asyncio.sleep(2.0)
                            continue

            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
