from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import Response

from app.api.dependencies import get_document_service
from app.core.exceptions import raise_http
from app.db.models import DocumentStatus
from app.schemas import (
    ApiResponse,
    DocumentListOut,
    DocumentOut,
    DocumentQueueAcceptedOut,
    DocumentStatusOut,
    ProcessAllAcceptedOut,
    VectorStoreActionOut,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=ApiResponse[DocumentListOut])
def list_documents(
    search: str | None = None,
    status: DocumentStatus | None = None,
    contentType: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    service: DocumentService = Depends(get_document_service),
):
    items, total = service.list_documents(
        search=search, status=status, content_type=contentType, skip=skip, limit=limit
    )
    return ApiResponse(data=DocumentListOut(items=items, total=total))


@router.post("/vector-store/clear", response_model=ApiResponse[VectorStoreActionOut])
def clear_vector_store(service: DocumentService = Depends(get_document_service)):
    try:
        result = service.clear_vector_store()
        return ApiResponse(message="Vector store cleared", data=result)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Failed to clear vector store: {exc}", "VECTOR_CLEAR_ERROR", 500)


@router.post(
    "/process-all",
    response_model=ApiResponse[ProcessAllAcceptedOut],
    status_code=status.HTTP_202_ACCEPTED,
)
def process_all_documents(
    force: bool = Query(False, description="When true, also queue Completed documents for reprocess"),
    service: DocumentService = Depends(get_document_service),
):
    try:
        result = service.process_all(force=force)
        return ApiResponse(message="Documents queued for processing", data=result)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Failed to queue process-all: {exc}", "PROCESS_ALL_ERROR", 500)


@router.post(
    "/reprocess-all",
    response_model=ApiResponse[ProcessAllAcceptedOut],
    status_code=status.HTTP_202_ACCEPTED,
)
def reprocess_all_documents(service: DocumentService = Depends(get_document_service)):
    """Compatibility alias for process-all?force=true."""
    try:
        result = service.process_all(force=True, requested_by="api-reprocess-all")
        return ApiResponse(message="Documents queued for reprocessing", data=result)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Failed to queue reprocess: {exc}", "REPROCESS_ALL_ERROR", 500)


@router.post(
    "/vector-store/clear-and-reprocess",
    response_model=ApiResponse[ProcessAllAcceptedOut],
    status_code=status.HTTP_202_ACCEPTED,
)
def clear_and_reprocess_all(service: DocumentService = Depends(get_document_service)):
    try:
        result = service.clear_and_reprocess_all()
        return ApiResponse(
            message="Vector store cleared and documents queued for reprocessing",
            data=result,
        )
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Failed to clear and reprocess: {exc}", "VECTOR_REPROCESS_ERROR", 500)


@router.get("/{document_id}", response_model=ApiResponse[DocumentOut])
def get_document(document_id: UUID, service: DocumentService = Depends(get_document_service)):
    try:
        return ApiResponse(data=service.get_document(document_id))
    except LookupError:
        raise_http("Document not found", "DOCUMENT_NOT_FOUND", 404)


@router.post("/upload", response_model=ApiResponse[DocumentOut], status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    chunkSize: int | None = Form(None),
    chunkOverlap: int | None = Form(None),
    service: DocumentService = Depends(get_document_service),
):
    try:
        document = await service.upload(
            file,
            chunk_size=chunkSize,
            chunk_overlap=chunkOverlap,
        )
        return ApiResponse(message="Document uploaded and queued for processing", data=document)
    except FileExistsError as exc:
        raise_http(str(exc), "DOCUMENT_CONFLICT", 409)
    except ValueError as exc:
        raise_http(str(exc), "DOCUMENT_VALIDATION_ERROR", 400)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Upload failed: {exc}", "DOCUMENT_UPLOAD_ERROR", 500)


@router.delete("/{document_id}", response_model=ApiResponse[None])
def delete_document(document_id: UUID, service: DocumentService = Depends(get_document_service)):
    try:
        service.delete(document_id)
        return ApiResponse(message="Document soft-deleted; blob and vector data removed")
    except LookupError:
        raise_http("Document not found", "DOCUMENT_NOT_FOUND", 404)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Delete failed: {exc}", "DOCUMENT_DELETE_ERROR", 500)


@router.post(
    "/{document_id}/process",
    response_model=ApiResponse[DocumentQueueAcceptedOut],
    status_code=status.HTTP_202_ACCEPTED,
)
def process_document(document_id: UUID, service: DocumentService = Depends(get_document_service)):
    try:
        return ApiResponse(
            message="Document processing has been queued.",
            data=service.enqueue_process(document_id, requested_by="api-process"),
        )
    except LookupError:
        raise_http("Document not found", "DOCUMENT_NOT_FOUND", 404)
    except Exception as exc:  # noqa: BLE001
        raise_http(f"Failed to queue document: {exc}", "DOCUMENT_PROCESS_ERROR", 500)


@router.post(
    "/{document_id}/reprocess",
    response_model=ApiResponse[DocumentQueueAcceptedOut],
    status_code=status.HTTP_202_ACCEPTED,
)
def reprocess_document(document_id: UUID, service: DocumentService = Depends(get_document_service)):
    try:
        return ApiResponse(message="Reprocessing queued", data=service.reprocess(document_id))
    except LookupError:
        raise_http("Document not found", "DOCUMENT_NOT_FOUND", 404)


@router.get("/{document_id}/status", response_model=ApiResponse[DocumentStatusOut])
def document_status(document_id: UUID, service: DocumentService = Depends(get_document_service)):
    try:
        return ApiResponse(data=service.get_status(document_id))
    except LookupError:
        raise_http("Document not found", "DOCUMENT_NOT_FOUND", 404)


@router.get("/{document_id}/download")
def download_document(document_id: UUID, service: DocumentService = Depends(get_document_service)):
    try:
        document, data = service.download(document_id)
        headers = {
            "Content-Disposition": f'attachment; filename="{document.OriginalFileName}"'
        }
        return Response(content=data, media_type=document.ContentType, headers=headers)
    except LookupError:
        raise_http("Document not found", "DOCUMENT_NOT_FOUND", 404)
