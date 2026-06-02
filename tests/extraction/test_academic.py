"""Unit tests for academic content extraction (REQ-04)."""

from __future__ import annotations

from research_to_dev.extraction.academic import extract_academic_content
from research_to_dev.extraction.types import AcademicContent
from research_to_dev.retriever.protocol import SearchResult


class TestExtractAcademicContent:
    """Verify SearchResult → AcademicContent mapping."""

    def test_full_fields(self) -> None:
        """All SearchResult fields map to AcademicContent (REQ-04: full)."""
        result = SearchResult(
            source="arxiv",
            title="Quantum Computing Advances",
            url="https://arxiv.org/abs/1234.5678",
            abstract="A comprehensive study.",
            authors=["Alice Smith", "Bob Jones"],
            published_date="2025-01-15",
            metadata={"pdf_url": "https://arxiv.org/pdf/1234.5678"},
        )
        content = extract_academic_content(result)
        assert isinstance(content, AcademicContent)
        assert content.source == "arxiv"
        assert content.title == "Quantum Computing Advances"
        assert content.url == "https://arxiv.org/abs/1234.5678"
        assert content.abstract == "A comprehensive study."
        assert content.authors == ["Alice Smith", "Bob Jones"]
        assert content.published_date == "2025-01-15"
        assert content.metadata == {"pdf_url": "https://arxiv.org/pdf/1234.5678"}

    def test_minimal_fields(self) -> None:
        """Only required fields present — others are None (REQ-04: minimal)."""
        result = SearchResult(source="semantic_scholar", title="A Paper")
        content = extract_academic_content(result)
        assert isinstance(content, AcademicContent)
        assert content.source == "semantic_scholar"
        assert content.title == "A Paper"
        assert content.url is None
        assert content.abstract is None
        assert content.authors is None
        assert content.published_date is None
        assert content.metadata == {}

    def test_abstract_becomes_body(self) -> None:
        """Abstract field serves as content."""
        result = SearchResult(
            source="arxiv",
            title="Paper",
            abstract="This is the abstract text.",
        )
        content = extract_academic_content(result)
        assert content.abstract == "This is the abstract text."

    def test_metadata_preserved(self) -> None:
        """Metadata dictionary is carried through unchanged."""
        result = SearchResult(
            source="arxiv",
            title="Paper",
            metadata={"citation_count": 42, "open_access_pdf": "http://..."},
        )
        content = extract_academic_content(result)
        assert content.metadata == {
            "citation_count": 42,
            "open_access_pdf": "http://...",
        }

    def test_semantic_scholar_source(self) -> None:
        """Semantic Scholar results also become AcademicContent."""
        result = SearchResult(
            source="semantic_scholar",
            title="S2 Paper",
            url="https://www.semanticscholar.org/paper/abc123",
        )
        content = extract_academic_content(result)
        assert content.source == "semantic_scholar"
        assert content.title == "S2 Paper"
