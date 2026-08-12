import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.blob_service import BlobService
from app.services.document_processor import DocumentProcessor, ExtractedDocument, ExtractedPage
from app.services.document_service import DocumentService


def test_chunking_strategy():
    processor = DocumentProcessor()
    extracted = ExtractedDocument(
        pages=[ExtractedPage(page_number=1, text="A" * 1000)],
        page_count=1,
    )
    chunks = processor.chunk_pages(extracted, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_txt_extraction():
    processor = DocumentProcessor()
    content = b"Claim number CLM-1001\nCause of loss: burst pipe"
    extracted = processor.extract_text(content, "note.txt", "text/plain")
    assert extracted.page_count == 1
    assert "burst pipe" in extracted.pages[0].text


def test_sanitize_filename():
    assert ".." not in BlobService.sanitize_filename("../evil.pdf")
    assert BlobService.sanitize_filename("claim report.pdf").endswith(".pdf")


def test_file_hash_stable():
    data = b"insurance-claim"
    assert BlobService.compute_hash(data) == BlobService.compute_hash(data)


def test_upload_rejects_unsupported_extension():
    service = DocumentService(db=MagicMock())
    upload = MagicMock()
    upload.filename = "malware.exe"
    upload.content_type = "application/octet-stream"
    upload.read = AsyncMock(return_value=b"abc")

    with pytest.raises(ValueError):
        asyncio.run(service.upload(upload, MagicMock()))


def test_upload_rejects_empty_file():
    service = DocumentService(db=MagicMock())
    upload = MagicMock()
    upload.filename = "empty.txt"
    upload.content_type = "text/plain"
    upload.read = AsyncMock(return_value=b"")

    with pytest.raises(ValueError):
        asyncio.run(service.upload(upload, MagicMock()))


def test_rag_prompt_constant():
    from app.services.llm_service import RAG_SYSTEM_PROMPT

    assert "insurance claims" in RAG_SYSTEM_PROMPT.lower()
    assert "I could not find sufficient information" in RAG_SYSTEM_PROMPT
