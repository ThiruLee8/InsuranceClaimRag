from __future__ import annotations

import json
import uuid
from uuid import UUID

import azure.functions as func

from models.document_models import DocumentStatus
from models.queue_message_models import DocumentQueueMessage, DocumentQueueOperation
from repositories.document_repository import DocumentRepository
from services.document_queue_service import DocumentQueueService
from shared.configuration import get_settings
from shared.database import SessionLocal
from shared.logging_utils import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


def _json_response(payload: dict, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(payload),
        status_code=status_code,
        mimetype="application/json",
    )


@app.blob_trigger(
    arg_name="blob",
    path="%DOCUMENT_CONTAINER_NAME%/{name}",
    connection="AzureWebJobsStorage",
)
def blob_document_trigger(blob: func.InputStream) -> None:
    """Detect new blobs and enqueue processing. Never processes documents directly."""
    # blob.name is typically "<container>/<path>"; use the path after the container.
    blob_path = ""
    if blob.name and "/" in blob.name:
        blob_path = blob.name.split("/", 1)[1]
    elif blob.name:
        blob_path = blob.name

    correlation_id = str(uuid.uuid4())
    logger.info(
        "blob_trigger_received",
        blob_name=blob_path,
        correlation_id=correlation_id,
        size=blob.length,
    )

    db = SessionLocal()
    try:
        repo = DocumentRepository(db)
        document = repo.get_by_blob_name(blob_path)
        if not document:
            logger.info(
                "blob_trigger_document_not_found",
                blob_name=blob_path,
                correlation_id=correlation_id,
            )
            return
        if document.IsDeleted:
            logger.info(
                "blob_trigger_document_deleted",
                document_id=str(document.Id),
                blob_name=blob_path,
                correlation_id=correlation_id,
            )
            return

        message = DocumentQueueService().enqueue_document(
            document.Id,
            operation=DocumentQueueOperation.PROCESS,
            correlation_id=correlation_id,
            requested_by="blob-trigger",
        )
        repo.mark_queued(document, correlation_id=message.correlationId or correlation_id)
        logger.info(
            "blob_trigger_queued",
            document_id=str(document.Id),
            correlation_id=message.correlationId,
            blob_name=blob_path,
        )
    finally:
        db.close()


@app.queue_trigger(
    arg_name="msg",
    queue_name="%DOCUMENT_PROCESSING_QUEUE_NAME%",
    connection="AzureWebJobsStorage",
)
def queue_document_processor(msg: func.QueueMessage) -> None:
    """SINGLE entry point for actual document processing."""
    # Lazy import keeps function indexing alive even if heavy deps fail at cold start.
    from services.document_processor_service import DocumentProcessorService

    raw = msg.get_body().decode("utf-8")
    payload = DocumentQueueMessage.from_json(raw)
    document_id = UUID(payload.documentId)
    correlation_id = payload.correlationId or str(uuid.uuid4())

    logger.info(
        "queue_trigger_received",
        document_id=str(document_id),
        operation=payload.operation,
        correlation_id=correlation_id,
        requested_by=payload.requestedBy,
        dequeue_count=msg.dequeue_count,
    )

    DocumentProcessorService().process(
        document_id,
        correlation_id=correlation_id,
        operation=payload.operation,
    )


@app.route(route="documents/{documentId}/process", methods=["POST"])
def process_document_http(req: func.HttpRequest) -> func.HttpResponse:
    """Queue a single document for processing. Does not process synchronously."""
    document_id_raw = req.route_params.get("documentId")
    try:
        document_id = UUID(document_id_raw)
    except Exception:
        return _json_response({"message": "Invalid documentId"}, 400)

    correlation_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        repo = DocumentRepository(db)
        document = repo.get(document_id, include_deleted=True)
        if not document or document.IsDeleted:
            return _json_response({"message": "Document not found"}, 404)

        message = DocumentQueueService().enqueue_document(
            document.Id,
            operation=DocumentQueueOperation.REPROCESS,
            correlation_id=correlation_id,
            requested_by="functions-http-process",
        )
        repo.mark_queued(document, correlation_id=message.correlationId or correlation_id)
        return _json_response(
            {
                "documentId": str(document.Id),
                "status": "Queued",
                "message": "Document processing has been queued.",
                "correlationId": message.correlationId,
            },
            202,
        )
    finally:
        db.close()


@app.route(route="documents/process-all", methods=["POST"])
def process_all_documents_http(req: func.HttpRequest) -> func.HttpResponse:
    """Queue all non-deleted documents. Does not process synchronously."""
    force = str(req.params.get("force", "false")).lower() in {"1", "true", "yes"}
    correlation_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        repo = DocumentRepository(db)
        all_active = repo.list_active(force=True)
        total_found = len(all_active)
        to_queue = repo.list_active(force=force)
        queued = 0
        for document in to_queue:
            op = (
                DocumentQueueOperation.REPROCESS
                if force
                or document.Status
                in {
                    DocumentStatus.Failed,
                    DocumentStatus.Completed,
                    DocumentStatus.Processed,
                }
                else DocumentQueueOperation.PROCESS
            )
            DocumentQueueService().enqueue_document(
                document.Id,
                operation=op,
                correlation_id=correlation_id,
                requested_by="functions-http-process-all",
            )
            repo.mark_queued(document, correlation_id=correlation_id)
            queued += 1

        return _json_response(
            {
                "status": "Queued",
                "totalDocumentsFound": total_found,
                "documentsQueued": queued,
                "documentsSkipped": total_found - queued,
                "correlationId": correlation_id,
            },
            202,
        )
    finally:
        db.close()


@app.route(route="vectors/search", methods=["POST"])
def vectors_search_http(req: func.HttpRequest) -> func.HttpResponse:
    """Embed query + hybrid/semantic/keyword search. Only Functions may touch the vector DB."""
    from services.embedding_service import get_embedding_service
    from services.hybrid_search_service import HybridSearchService

    try:
        body = req.get_json()
    except ValueError:
        body = {}
    question = str((body or {}).get("question") or "").strip()
    if not question:
        return _json_response({"message": "question is required"}, 400)

    top_k = int((body or {}).get("topK") or 5)
    similarity_threshold = float((body or {}).get("similarityThreshold") or 0.3)
    search_mode = (body or {}).get("searchMode")
    rerank = (body or {}).get("rerank")
    if isinstance(rerank, str):
        rerank = rerank.lower() in {"1", "true", "yes"}

    embedding = get_embedding_service().embed_query(question)
    result = HybridSearchService().search(
        question=question,
        query_embedding=embedding,
        top_k=max(1, top_k),
        similarity_threshold=similarity_threshold,
        search_mode=str(search_mode) if search_mode else None,
        rerank=rerank if isinstance(rerank, bool) else None,
    )
    return _json_response(
        {
            "searchMode": result.search_mode,
            "hits": [
                {
                    "chunkId": hit.chunk_id,
                    "documentId": hit.document_id,
                    "fileName": hit.file_name,
                    "pageNumber": hit.page_number,
                    "chunkIndex": hit.chunk_index,
                    "content": hit.content,
                    "score": hit.score,
                    "scoreSemantic": result.score_semantic.get(hit.chunk_id),
                    "scoreKeyword": result.score_keyword.get(hit.chunk_id),
                }
                for hit in result.hits
            ]
        }
    )


@app.route(route="vectors/documents/{documentId}", methods=["DELETE"])
def vectors_delete_document_http(req: func.HttpRequest) -> func.HttpResponse:
    from services.vector_store_service import VectorStoreService

    document_id_raw = req.route_params.get("documentId")
    try:
        document_id = UUID(document_id_raw)
    except Exception:
        return _json_response({"message": "Invalid documentId"}, 400)

    VectorStoreService().delete_document(document_id)
    return _json_response({"documentId": str(document_id), "deleted": True})


@app.route(route="vectors/clear", methods=["POST"])
def vectors_clear_http(req: func.HttpRequest) -> func.HttpResponse:
    from services.vector_store_service import VectorStoreService

    VectorStoreService().clear_collection()
    return _json_response({"cleared": True})


@app.route(route="vectors/health", methods=["GET"])
def vectors_health_http(req: func.HttpRequest) -> func.HttpResponse:
    from services.vector_store_service import VectorStoreService

    status = VectorStoreService().health_check()
    code = 200 if status == "ok" else 503
    return _json_response({"status": status}, code)
