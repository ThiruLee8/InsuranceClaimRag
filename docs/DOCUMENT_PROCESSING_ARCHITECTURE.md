# Document Processing Architecture (Azure Functions + Queue)

## Overview

Long-running document processing no longer runs inside the FastAPI process.

```text
Angular → Python Web API → Blob + SQL metadata
                │
                ├─ Upload / Process / Process-All → Azure Queue (Azurite locally)
                │
Blob Trigger ───┘
                │
                ▼
         Queue Trigger  (ONLY processor entry point)
                │
                ▼
     DocumentProcessorService
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      Blob    SQL     Chroma
```

## Projects

| Path | Role |
|------|------|
| `backend/` | FastAPI: upload, metadata, chat/RAG, enqueue-only process APIs |
| `document_processor/` | Azure Functions: Blob trigger, Queue trigger, HTTP process helpers |
| `frontend/` | Angular UI (talks only to Web API) |

## Queue contract

Queue name: `document-processing`

```json
{
  "documentId": "uuid",
  "operation": "PROCESS|REPROCESS",
  "correlationId": "uuid",
  "requestedBy": "api-upload|blob-trigger|api-process|..."
}
```

## Vector database ownership

ChromaDB is **only** accessed by Azure Functions (`document_processor`).

The Web API never opens a Chroma client. It calls Functions HTTP:

| API need | Functions endpoint |
|----------|--------------------|
| RAG retrieval | `POST /api/vectors/search` |
| Delete doc vectors | `DELETE /api/vectors/documents/{id}` |
| Clear index | `POST /api/vectors/clear` |
| Health | `GET /api/vectors/health` |

Config: `DOCUMENT_PROCESSOR_BASE_URL` (Docker: `http://document-processor:80`).

## Local Docker

```bash
docker compose up --build
```

Services:

- Angular: http://localhost:4200
- Web API: http://localhost:8000/api
- Functions: http://localhost:7071/api
- Azurite blob `10000` / queue `10001`
- SQL Server `1433`, Chroma `8001`, Ollama `11434`

## Web API examples

```http
POST /api/documents/upload
POST /api/documents/{id}/process          → 202 Queued
POST /api/documents/process-all?force=false
POST /api/documents/process-all?force=true
GET  /api/documents/{id}/status
```

## Functions HTTP examples

```http
POST http://localhost:7071/api/documents/{id}/process
POST http://localhost:7071/api/documents/process-all?force=true
```

## Processing statuses

`Uploaded → Queued → Processing → ExtractingText → Chunking → GeneratingEmbeddings → Indexing → Completed`

Failures set `Failed` and increment `RetryCount`. Queue poison handling uses host.json `maxDequeueCount`.

## Safety

- Soft-deleted documents are skipped by the queue worker
- Concurrent claim uses conditional SQL update
- Reprocessing deletes existing Chroma vectors for `document_id` before re-index
- Angular never talks to SQL, Chroma, or Queue directly
