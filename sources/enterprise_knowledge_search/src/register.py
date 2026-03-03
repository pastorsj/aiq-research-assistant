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

"""NAT function for enterprise knowledge search.

This function provides retrieval-only access to pre-existing vector database
collections that are managed outside of this application. It reuses the
existing knowledge layer retriever adapters (LlamaIndex, Foundational RAG)
but does NOT initialize an ingestor or participate in session-based
collection routing.

Use this when you have an external ingestion pipeline populating your
vector database and want agents to search those collections directly.
"""

import asyncio
import logging
import os
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

BackendType = Literal["llamaindex", "foundational_rag"]


class CollectionDescriptor(BaseModel):
    """Describes a single pre-existing collection in the vector database."""

    name: str = Field(..., description="Collection name in the vector DB")
    description: str = Field(default="", description="What this collection contains (shown to the agent)")


class EnterpriseKnowledgeSearchConfig(FunctionBaseConfig, name="enterprise_knowledge_search"):
    """Configuration for enterprise knowledge search function.

    This tool searches pre-existing vector database collections without
    requiring file ingestion through this application.
    """

    backend: BackendType = Field(default="llamaindex", description="Knowledge backend to use")
    collections: list[CollectionDescriptor] = Field(
        ...,
        min_length=1,
        description="List of pre-existing collections to search",
    )
    top_k: int = Field(default=5, description="Total number of results to return across all collections")
    per_collection_top_k: int | None = Field(
        default=None,
        description=(
            "Number of results to retrieve per collection before merging. "
            "Defaults to top_k if not set. Set higher than top_k for better "
            "cross-collection result quality."
        ),
    )
    # LlamaIndex-specific options
    chroma_dir: str = Field(
        default="/tmp/chroma_data",
        description="Directory for ChromaDB persistence (LlamaIndex only)",
    )
    # Foundational RAG-specific options
    rag_url: str = Field(
        default="http://localhost:8081/v1",
        description="RAG query server URL (foundational_rag only)",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds (foundational_rag only)")
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates (foundational_rag only)",
    )


def _setup_retriever_backend(config: EnterpriseKnowledgeSearchConfig) -> tuple[str, dict[str, Any]]:
    """Import the backend adapter and build retriever-only configuration.

    Importing the adapter module triggers the @register_retriever decorator,
    which registers the adapter class with the factory.

    Unlike the knowledge_retrieval tool's _setup_backend, this function:
    - Does NOT set KNOWLEDGE_INGESTOR_BACKEND
    - Does NOT include summary configuration
    - Only configures what the retriever needs
    """
    backend = config.backend.lower()

    if backend == "llamaindex":
        import knowledge_layer.llamaindex.adapter  # noqa: F401

        os.environ.setdefault("AIQ_CHROMA_DIR", config.chroma_dir)
        backend_config = {"persist_dir": config.chroma_dir}

    elif backend == "foundational_rag":
        import knowledge_layer.foundational_rag.adapter  # noqa: F401

        backend_config = {
            "rag_url": config.rag_url,
            "timeout": config.timeout,
            "verify_ssl": config.verify_ssl,
        }

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'llamaindex' or 'foundational_rag'.")

    os.environ.setdefault("KNOWLEDGE_RETRIEVER_BACKEND", backend)

    return backend, backend_config


def _format_enterprise_results(
    chunks: list[Any],
    query: str,
    searched_collections: list[str],
    failed_collections: list[str],
) -> str:
    """Format multi-collection retrieval results for LLM consumption.

    Similar to knowledge_layer's _format_results but includes the source
    collection for each result.
    """
    if not chunks and not failed_collections:
        return f"No relevant documents found for query: '{query}'"

    if not chunks and failed_collections:
        return f"No results could be retrieved. Failed collections: {', '.join(failed_collections)}. Query: '{query}'"

    lines = [f"Found {len(chunks)} relevant document(s) from {len(searched_collections)} collection(s):\n"]

    if failed_collections:
        lines.append(f"Note: Could not search these collections: {', '.join(failed_collections)}\n")

    for i, chunk in enumerate(chunks, 1):
        collection = chunk.metadata.get("_source_collection", "unknown")

        if chunk.page_number and chunk.page_number > 0:
            citation = f"{chunk.file_name}, p.{chunk.page_number}"
        else:
            citation = chunk.file_name

        lines.append(f"--- Result {i} ---")
        lines.append(f"Collection: {collection}")
        lines.append(f"Source: {chunk.file_name}")
        if chunk.page_number and chunk.page_number > 0:
            lines.append(f"Page: {chunk.page_number}")
        lines.append(f"Citation: {citation}")
        lines.append(f"Content Type: {chunk.content_type.value}")
        lines.append(f"Relevance Score: {chunk.score:.2f}")
        lines.append("")

        content = chunk.content
        if len(content) > 1500:
            content = content[:1500] + "... [truncated]"
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


@register_function(config_type=EnterpriseKnowledgeSearchConfig)
async def enterprise_knowledge_search(config: EnterpriseKnowledgeSearchConfig, _builder: Builder):
    """Enterprise knowledge search over pre-existing vector DB collections.

    This function provides semantic search over collections that have been
    externally populated (e.g., by a separate ingestion pipeline). It does
    NOT initialize an ingestor or use session-based collection routing.

    The retriever is initialized once and reused for all queries.
    """
    from aiq_agent.knowledge.factory import get_retriever

    backend, backend_config = _setup_retriever_backend(config)
    retriever = get_retriever(backend, backend_config)

    collection_names = [c.name for c in config.collections]
    collection_names_set = set(collection_names)
    top_k = config.top_k
    per_collection_top_k = config.per_collection_top_k or top_k

    # Build collection descriptions for the tool docstring
    collection_info_lines = []
    for c in config.collections:
        if c.description:
            collection_info_lines.append(f"  - {c.name}: {c.description}")
        else:
            collection_info_lines.append(f"  - {c.name}")
    collections_desc = "\n".join(collection_info_lines)

    logger.info(
        "Enterprise knowledge search initialized: backend=%s, collections=%s, top_k=%d",
        backend,
        collection_names,
        top_k,
    )

    async def search(query: str, collection_name: str | None = None) -> str:
        """Search enterprise knowledge bases for documents relevant to the query.

        This tool searches pre-existing enterprise document collections that have
        been externally indexed. Use it for finding information in internal company
        documents, policies, specifications, and other enterprise knowledge.

        Args:
            query (str): Natural language query describing what information you need.
            collection_name (str | None): Optional. Search a specific collection by
                name. If not provided, searches all available collections.

        Returns:
            str: Formatted results with relevant document excerpts and citations.
        """
        # Validate and determine target collections
        if collection_name:
            if collection_name not in collection_names_set:
                return (
                    f"Collection '{collection_name}' is not configured. "
                    f"Available collections: {', '.join(collection_names)}"
                )
            target_collections = [collection_name]
        else:
            target_collections = collection_names

        logger.info(
            "Enterprise search: query='%s' collections=%s",
            query[:100],
            target_collections,
        )

        # Query each collection concurrently
        async def _query_collection(coll_name: str):
            try:
                result = await retriever.retrieve(
                    query=query,
                    collection_name=coll_name,
                    top_k=per_collection_top_k,
                )
                return coll_name, result
            except Exception as e:
                logger.warning("Failed to search collection '%s': %s", coll_name, e)
                return coll_name, None

        results = await asyncio.gather(*[_query_collection(c) for c in target_collections])

        # Collect chunks, handling failures gracefully
        all_chunks = []
        failed_collections = []
        searched_collections = []

        for coll_name, result in results:
            if result is None:
                failed_collections.append(coll_name)
                continue
            if not result.success:
                logger.warning("Collection '%s' returned error: %s", coll_name, result.error_message)
                failed_collections.append(coll_name)
                continue
            searched_collections.append(coll_name)
            for chunk in result.chunks:
                chunk.metadata["_source_collection"] = coll_name
                all_chunks.append(chunk)

        # Sort by score descending and take top_k
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        top_chunks = all_chunks[:top_k]

        formatted = _format_enterprise_results(top_chunks, query, searched_collections, failed_collections)
        logger.info(
            "Enterprise search: %d results from %d collection(s)",
            len(top_chunks),
            len(searched_collections),
        )
        return formatted

    # Build dynamic tool description with collection info
    tool_description = (
        "Search enterprise knowledge bases for relevant documents. "
        "Use this to find information from internal company documents, "
        "policies, specifications, and other externally managed knowledge.\n\n"
        f"Available collections:\n{collections_desc}\n\n"
        f"Returns up to {top_k} relevant excerpts with citations. "
        "You can optionally specify a collection_name to search a specific collection."
    )

    yield FunctionInfo.from_fn(
        search,
        description=tool_description,
    )
