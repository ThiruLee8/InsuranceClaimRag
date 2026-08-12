"""
Additional unit tests for document processing helpers.
"""

from app.services.document_processor import DocumentProcessor, ExtractedDocument, ExtractedPage


def test_chunk_overlap_preserves_continuity():
    processor = DocumentProcessor()
    text = ("water damage caused by burst pipe. " * 40).strip()
    extracted = ExtractedDocument(pages=[ExtractedPage(1, text)], page_count=1)
    chunks = processor.chunk_pages(extracted, chunk_size=120, chunk_overlap=30)
    assert len(chunks) >= 2
    assert "burst pipe" in chunks[0].content


def test_empty_pages_produce_no_chunks():
    processor = DocumentProcessor()
    extracted = ExtractedDocument(pages=[ExtractedPage(1, "   ")], page_count=1)
    chunks = processor.chunk_pages(extracted, chunk_size=100, chunk_overlap=20)
    assert chunks == []
