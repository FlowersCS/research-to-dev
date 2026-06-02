"""Unit tests for extraction data types."""

from __future__ import annotations

from research_to_dev.extraction.types import (
    AcademicContent,
    ExtractionError,
    ExtractionResult,
    GeneralContent,
)


# -------------------------------------------------------------------
# AcademicContent
# -------------------------------------------------------------------


class TestAcademicContent:
    """Verify AcademicContent dataclass creation and defaults."""

    def test_full_fields(self) -> None:
        content = AcademicContent(
            source="arxiv",
            title="Quantum Computing Advances",
            url="https://arxiv.org/abs/1234.5678",
            abstract="A comprehensive study of quantum computing.",
            authors=["Alice Smith", "Bob Jones"],
            published_date="2025-01-15",
            metadata={"pdf_url": "https://arxiv.org/pdf/1234.5678"},
        )
        assert content.source == "arxiv"
        assert content.title == "Quantum Computing Advances"
        assert content.url == "https://arxiv.org/abs/1234.5678"
        assert content.abstract == "A comprehensive study of quantum computing."
        assert content.authors == ["Alice Smith", "Bob Jones"]
        assert content.published_date == "2025-01-15"
        assert content.metadata == {"pdf_url": "https://arxiv.org/pdf/1234.5678"}

    def test_minimal_fields(self) -> None:
        content = AcademicContent(source="semantic_scholar", title="S2 Paper")
        assert content.source == "semantic_scholar"
        assert content.title == "S2 Paper"
        assert content.url is None
        assert content.abstract is None
        assert content.authors is None
        assert content.published_date is None
        assert content.metadata == {}

    def test_default_metadata_is_mutable_copy(self) -> None:
        c1 = AcademicContent(source="arxiv", title="A")
        c2 = AcademicContent(source="arxiv", title="B")
        c1.metadata["key"] = "val"
        # Default factory creates independent dicts
        assert c2.metadata == {}


# -------------------------------------------------------------------
# GeneralContent
# -------------------------------------------------------------------


class TestGeneralContent:
    """Verify GeneralContent dataclass creation and defaults."""

    def test_full_fields(self) -> None:
        content = GeneralContent(
            source="tavily",
            title="Deep Learning Overview",
            url="https://example.com/dl",
            body="Deep learning is a subset of machine learning...",
            metadata={"score": 0.95, "raw_content": "<html>...</html>"},
        )
        assert content.source == "tavily"
        assert content.title == "Deep Learning Overview"
        assert content.url == "https://example.com/dl"
        assert content.body == "Deep learning is a subset of machine learning..."
        assert content.metadata == {"score": 0.95, "raw_content": "<html>...</html>"}

    def test_default_body_is_empty_string(self) -> None:
        content = GeneralContent(source="tavily", title="No Body")
        assert content.body == ""
        assert content.metadata == {}


# -------------------------------------------------------------------
# ExtractionError
# -------------------------------------------------------------------


class TestExtractionError:
    """Verify ExtractionError dataclass creation and defaults."""

    def test_full_fields(self) -> None:
        error = ExtractionError(
            source="tavily",
            url="https://example.com",
            message="Connection timeout",
            exception_type="TimeoutException",
        )
        assert error.source == "tavily"
        assert error.url == "https://example.com"
        assert error.message == "Connection timeout"
        assert error.exception_type == "TimeoutException"

    def test_minimal_fields(self) -> None:
        error = ExtractionError(source="tavily")
        assert error.source == "tavily"
        assert error.url is None
        assert error.message == ""
        assert error.exception_type is None


# -------------------------------------------------------------------
# ExtractionResult
# -------------------------------------------------------------------


class TestExtractionResult:
    """Verify ExtractionResult aggregation dataclass."""

    def test_populated_result(self) -> None:
        academic = AcademicContent(source="arxiv", title="Paper A")
        general = GeneralContent(source="tavily", title="Web Result", body="text")
        error = ExtractionError(source="tavily", message="Failed")

        result = ExtractionResult(
            contents=[academic, general],
            errors=[error],
        )
        assert len(result.contents) == 2
        assert isinstance(result.contents[0], AcademicContent)
        assert isinstance(result.contents[1], GeneralContent)
        assert len(result.errors) == 1
        assert result.errors[0].message == "Failed"

    def test_empty_result(self) -> None:
        result = ExtractionResult(contents=[], errors=[])
        assert result.contents == []
        assert result.errors == []
