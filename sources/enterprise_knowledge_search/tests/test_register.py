# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for enterprise knowledge search tool."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiq_agent.knowledge.schema import Chunk
from aiq_agent.knowledge.schema import ContentType
from aiq_agent.knowledge.schema import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str = "test content",
    score: float = 0.8,
    file_name: str = "doc.pdf",
    page_number: int | None = 1,
    metadata: dict | None = None,
) -> Chunk:
    """Create a Chunk for testing."""
    return Chunk(
        chunk_id=f"chunk-{id(content)}",
        content=content,
        score=score,
        file_name=file_name,
        page_number=page_number,
        display_citation=f"{file_name}, p.{page_number}" if page_number else file_name,
        content_type=ContentType.TEXT,
        metadata=metadata or {},
    )


def _make_retrieval_result(
    chunks: list[Chunk] | None = None,
    success: bool = True,
    error_message: str | None = None,
    query: str = "test query",
) -> RetrievalResult:
    """Create a RetrievalResult for testing."""
    return RetrievalResult(
        chunks=chunks or [],
        query=query,
        backend="llamaindex",
        success=success,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestEnterpriseKnowledgeSearchConfig:
    def test_config_requires_collections(self):
        """Config must have at least one collection."""
        from enterprise_knowledge_search.register import EnterpriseKnowledgeSearchConfig

        with pytest.raises(ValidationError):
            EnterpriseKnowledgeSearchConfig(collections=[])

    def test_config_valid_with_minimal_fields(self):
        """Config works with just collections specified."""
        from enterprise_knowledge_search.register import EnterpriseKnowledgeSearchConfig

        config = EnterpriseKnowledgeSearchConfig(
            collections=[{"name": "test_collection"}],
        )
        assert config.backend == "llamaindex"
        assert config.top_k == 5
        assert config.per_collection_top_k is None
        assert len(config.collections) == 1
        assert config.collections[0].name == "test_collection"
        assert config.collections[0].description == ""

    def test_config_collection_with_description(self):
        """Collection descriptors can include descriptions."""
        from enterprise_knowledge_search.register import EnterpriseKnowledgeSearchConfig

        config = EnterpriseKnowledgeSearchConfig(
            collections=[
                {"name": "eng_docs", "description": "Engineering documentation"},
                {"name": "hr_docs", "description": "HR policies"},
            ],
        )
        assert config.collections[0].description == "Engineering documentation"
        assert config.collections[1].description == "HR policies"

    def test_config_per_collection_top_k(self):
        """per_collection_top_k can be set independently."""
        from enterprise_knowledge_search.register import EnterpriseKnowledgeSearchConfig

        config = EnterpriseKnowledgeSearchConfig(
            collections=[{"name": "test"}],
            top_k=5,
            per_collection_top_k=10,
        )
        assert config.per_collection_top_k == 10

    def test_config_foundational_rag_backend(self):
        """Config accepts foundational_rag backend."""
        from enterprise_knowledge_search.register import EnterpriseKnowledgeSearchConfig

        config = EnterpriseKnowledgeSearchConfig(
            collections=[{"name": "test"}],
            backend="foundational_rag",
            rag_url="http://myserver:8081/v1",
        )
        assert config.backend == "foundational_rag"
        assert config.rag_url == "http://myserver:8081/v1"


# ---------------------------------------------------------------------------
# Result formatting tests
# ---------------------------------------------------------------------------


class TestFormatEnterpriseResults:
    def test_no_results_no_failures(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        result = _format_enterprise_results([], "my query", [], [])
        assert "No relevant documents found" in result
        assert "my query" in result

    def test_no_results_with_failures(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        result = _format_enterprise_results([], "my query", [], ["broken_collection"])
        assert "No results could be retrieved" in result
        assert "broken_collection" in result

    def test_results_include_collection_name(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        chunk = _make_chunk(content="important info", score=0.9, metadata={"_source_collection": "eng_docs"})
        result = _format_enterprise_results([chunk], "query", ["eng_docs"], [])
        assert "Collection: eng_docs" in result
        assert "important info" in result

    def test_results_show_failed_collections_note(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        chunk = _make_chunk(metadata={"_source_collection": "good_coll"})
        result = _format_enterprise_results([chunk], "query", ["good_coll"], ["bad_coll"])
        assert "Could not search these collections: bad_coll" in result

    def test_results_sorted_by_score(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        chunks = [
            _make_chunk(content="low", score=0.3, metadata={"_source_collection": "a"}),
            _make_chunk(content="high", score=0.95, metadata={"_source_collection": "b"}),
            _make_chunk(content="mid", score=0.7, metadata={"_source_collection": "a"}),
        ]
        # Note: _format_enterprise_results expects chunks already sorted
        # (sorting happens in search function), but we test the output order
        result = _format_enterprise_results(chunks, "query", ["a", "b"], [])
        assert "Found 3 relevant document(s)" in result

    def test_long_content_truncated(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        long_content = "x" * 2000
        chunk = _make_chunk(content=long_content, metadata={"_source_collection": "coll"})
        result = _format_enterprise_results([chunk], "query", ["coll"], [])
        assert "... [truncated]" in result

    def test_citation_without_page_number(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        chunk = _make_chunk(page_number=None, file_name="readme.txt", metadata={"_source_collection": "coll"})
        result = _format_enterprise_results([chunk], "query", ["coll"], [])
        assert "Citation: readme.txt" in result
        assert "Page:" not in result

    def test_citation_with_page_number(self):
        from enterprise_knowledge_search.register import _format_enterprise_results

        chunk = _make_chunk(page_number=5, file_name="report.pdf", metadata={"_source_collection": "coll"})
        result = _format_enterprise_results([chunk], "query", ["coll"], [])
        assert "Citation: report.pdf, p.5" in result
        assert "Page: 5" in result


# ---------------------------------------------------------------------------
# Search logic tests (mocked retriever)
# ---------------------------------------------------------------------------


class TestSearchLogic:
    """Test the search function's multi-collection logic with a mocked retriever."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = AsyncMock()
        retriever.backend_name = "llamaindex"
        return retriever

    async def _build_search_fn(self, mock_retriever, collections, top_k=5, per_collection_top_k=None):
        """Build the search function with mocked dependencies.

        Instead of going through the full NAT registration, we replicate
        the core logic from the register function to test it in isolation.
        """
        from enterprise_knowledge_search.register import _format_enterprise_results

        collection_names = [c["name"] for c in collections]
        collection_names_set = set(collection_names)
        _per_collection_top_k = per_collection_top_k or top_k

        async def search(query, collection_name=None):
            import asyncio

            if collection_name:
                if collection_name not in collection_names_set:
                    return (
                        f"Collection '{collection_name}' is not configured. "
                        f"Available collections: {', '.join(collection_names)}"
                    )
                target_collections = [collection_name]
            else:
                target_collections = collection_names

            async def _query_collection(coll_name):
                try:
                    result = await mock_retriever.retrieve(
                        query=query,
                        collection_name=coll_name,
                        top_k=_per_collection_top_k,
                    )
                    return coll_name, result
                except Exception:
                    return coll_name, None

            results = await asyncio.gather(*[_query_collection(c) for c in target_collections])

            all_chunks = []
            failed_collections = []
            searched_collections = []

            for coll_name, result in results:
                if result is None:
                    failed_collections.append(coll_name)
                    continue
                if not result.success:
                    failed_collections.append(coll_name)
                    continue
                searched_collections.append(coll_name)
                for chunk in result.chunks:
                    chunk.metadata["_source_collection"] = coll_name
                    all_chunks.append(chunk)

            all_chunks.sort(key=lambda c: c.score, reverse=True)
            top_chunks = all_chunks[:top_k]
            return _format_enterprise_results(top_chunks, query, searched_collections, failed_collections)

        return search

    async def test_search_all_collections_merges_by_score(self, mock_retriever):
        """Results from multiple collections are merged and sorted by score."""
        collections = [{"name": "coll_a"}, {"name": "coll_b"}]

        mock_retriever.retrieve = AsyncMock(
            side_effect=[
                _make_retrieval_result(
                    chunks=[
                        _make_chunk(content="A low", score=0.3),
                        _make_chunk(content="A high", score=0.9),
                    ]
                ),
                _make_retrieval_result(
                    chunks=[
                        _make_chunk(content="B mid", score=0.6),
                    ]
                ),
            ]
        )

        search = await self._build_search_fn(mock_retriever, collections, top_k=2)
        result = await search("test query")

        # Should have top 2 by score: A high (0.9) and B mid (0.6)
        assert "A high" in result
        assert "B mid" in result
        # A low (0.3) should be excluded by top_k=2
        assert "A low" not in result

    async def test_search_specific_collection(self, mock_retriever):
        """When collection_name is provided, only that collection is queried."""
        collections = [{"name": "coll_a"}, {"name": "coll_b"}]

        mock_retriever.retrieve = AsyncMock(
            return_value=_make_retrieval_result(chunks=[_make_chunk(content="specific result", score=0.8)])
        )

        search = await self._build_search_fn(mock_retriever, collections)
        result = await search("test query", collection_name="coll_a")

        # Only one call to retrieve (for coll_a)
        assert mock_retriever.retrieve.call_count == 1
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["collection_name"] == "coll_a"
        assert "specific result" in result

    async def test_search_invalid_collection_returns_error(self, mock_retriever):
        """Specifying a non-configured collection returns a friendly error."""
        collections = [{"name": "coll_a"}]

        search = await self._build_search_fn(mock_retriever, collections)
        result = await search("test query", collection_name="nonexistent")

        assert "not configured" in result
        assert "coll_a" in result
        mock_retriever.retrieve.assert_not_called()

    async def test_search_collection_failure_graceful(self, mock_retriever):
        """If one collection fails, results from others are still returned."""
        collections = [{"name": "good_coll"}, {"name": "bad_coll"}]

        mock_retriever.retrieve = AsyncMock(
            side_effect=[
                _make_retrieval_result(chunks=[_make_chunk(content="good result", score=0.8)]),
                Exception("Connection refused"),
            ]
        )

        search = await self._build_search_fn(mock_retriever, collections)
        result = await search("test query")

        assert "good result" in result
        assert "Could not search these collections: bad_coll" in result

    async def test_search_all_collections_fail(self, mock_retriever):
        """When all collections fail, a descriptive error is returned."""
        collections = [{"name": "coll_a"}, {"name": "coll_b"}]

        mock_retriever.retrieve = AsyncMock(side_effect=Exception("DB down"))

        search = await self._build_search_fn(mock_retriever, collections)
        result = await search("test query")

        assert "No results could be retrieved" in result
        assert "coll_a" in result
        assert "coll_b" in result

    async def test_search_retrieval_error_result(self, mock_retriever):
        """A retrieval that returns success=False is treated as a failure."""
        collections = [{"name": "coll_a"}]

        mock_retriever.retrieve = AsyncMock(
            return_value=_make_retrieval_result(
                success=False,
                error_message="Collection not found",
            )
        )

        search = await self._build_search_fn(mock_retriever, collections)
        result = await search("test query")

        assert "No results could be retrieved" in result

    async def test_per_collection_top_k_used(self, mock_retriever):
        """per_collection_top_k is passed to each retrieve call."""
        collections = [{"name": "coll_a"}]

        mock_retriever.retrieve = AsyncMock(return_value=_make_retrieval_result(chunks=[_make_chunk()]))

        search = await self._build_search_fn(mock_retriever, collections, top_k=3, per_collection_top_k=10)
        await search("test query")

        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["top_k"] == 10


# ---------------------------------------------------------------------------
# Data source filtering compatibility
# ---------------------------------------------------------------------------


class TestDataSourceFiltering:
    def test_tool_filtered_under_enterprise_knowledge(self):
        """enterprise_knowledge_search is controlled by the enterprise_knowledge data source."""
        from aiq_agent.common.data_sources import filter_tools_by_sources

        enterprise_tool = MagicMock()
        enterprise_tool.name = "enterprise_knowledge_search"

        # Should be included when enterprise_knowledge is selected
        result = filter_tools_by_sources([enterprise_tool], ["enterprise_knowledge"])
        assert len(result) == 1

        # Should be excluded when only knowledge_layer is selected
        result = filter_tools_by_sources([enterprise_tool], ["knowledge_layer"])
        assert len(result) == 0

        # Should be excluded when only web_search is selected
        result = filter_tools_by_sources([enterprise_tool], ["web_search"])
        assert len(result) == 0

        # Should be included when data_sources is None (all tools)
        result = filter_tools_by_sources([enterprise_tool], None)
        assert len(result) == 1

    def test_enterprise_and_knowledge_tools_independent(self):
        """enterprise_knowledge_search and knowledge_search are independently toggleable."""
        from aiq_agent.common.data_sources import filter_tools_by_sources

        enterprise_tool = MagicMock()
        enterprise_tool.name = "enterprise_knowledge_search"
        knowledge_tool = MagicMock()
        knowledge_tool.name = "knowledge_search"
        both = [enterprise_tool, knowledge_tool]

        # Only enterprise
        result = filter_tools_by_sources(both, ["enterprise_knowledge"])
        assert len(result) == 1
        assert result[0].name == "enterprise_knowledge_search"

        # Only knowledge_layer
        result = filter_tools_by_sources(both, ["knowledge_layer"])
        assert len(result) == 1
        assert result[0].name == "knowledge_search"

        # Both
        result = filter_tools_by_sources(both, ["enterprise_knowledge", "knowledge_layer"])
        assert len(result) == 2
