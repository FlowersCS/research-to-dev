"""Retriever system — async multi-source research retrieval layer.

Public API
----------
- **Protocol**: ``Retriever`` — structural protocol for retriever adapters
- **Data**: ``SearchResult``, ``RetrievalResult``, ``RetrievalError``
- **Adapters**: ``ArxivRetriever``, ``SemanticScholarRetriever``, ``TavilyRetriever``
- **Orchestrator**: ``RetrieverOrchestrator``
"""

from research_to_dev.retriever.arxiv import ArxivRetriever
from research_to_dev.retriever.orchestrator import RetrieverOrchestrator
from research_to_dev.retriever.protocol import (
    RetrievalError,
    RetrievalResult,
    Retriever,
    SearchResult,
)
from research_to_dev.retriever.semantic_scholar import SemanticScholarRetriever
from research_to_dev.retriever.tavily import TavilyRetriever

__all__ = [
    "ArxivRetriever",
    "RetrievalError",
    "RetrievalResult",
    "Retriever",
    "RetrieverOrchestrator",
    "SearchResult",
    "SemanticScholarRetriever",
    "TavilyRetriever",
]
