# Enterprise Knowledge Search

Search pre-existing vector database collections without uploading or ingesting files through the application. This tool connects to collections that are populated and managed by an external pipeline (or by the included `populate_collection.py` script) and exposes them as a searchable data source to the research agents.

## How It Works

```
External Pipeline ──▶ Vector DB (ChromaDB / Milvus)
                              │
                    enterprise_knowledge_search
                              │
                    ┌─────────┴─────────┐
                    │  Collection A     │  Collection B ...
                    └─────────┬─────────┘
                         Retriever Adapter
                    (LlamaIndex or Foundational RAG)
                              │
                    Merge by relevance score ──▶ Agent
```

The tool reuses the existing knowledge layer retriever adapters (`LlamaIndexRetriever`, `FoundationalRagRetriever`) but **does not** initialize an ingestor, manage sessions, or require file uploads. It is retrieval-only.

When the agent calls the tool:
1. All configured collections are queried concurrently.
2. Results are merged and ranked by relevance score.
3. The top-K results (across all collections) are formatted and returned to the agent.
4. If one collection fails, results from the others are still returned.

The agent can also target a specific collection by name if the query is domain-specific.

## Setup

### 1. Install the package

The package is already part of the uv workspace. After cloning the repo:

```bash
uv pip install -e ".[dev]"
```

This installs `enterprise-knowledge-search` along with all other workspace packages. It registers itself as a NAT plugin via the `nat.plugins` entry point.

### 2. Populate a collection

Use the included script to ingest documents into a named collection:

```bash
# LlamaIndex / ChromaDB (default)
uv run python scripts/populate_collection.py <document_directory> <collection_name>

# Example
uv run python scripts/populate_collection.py data/cybersecurity cybersecurity

# With optional features
uv run python scripts/populate_collection.py data/cybersecurity cybersecurity \
    --chroma-dir /data/chroma \
    --chunk-size 1024 \
    --chunk-overlap 128 \
    --extract-tables \
    --extract-images

# Foundational RAG / Milvus
uv run python scripts/populate_collection.py data/cybersecurity cybersecurity \
    --backend foundational_rag \
    --rag-ingest-url http://localhost:8082/v1
```

Supported file types: `.pdf`, `.txt`, `.md`, `.docx`, `.csv`, `.json`, `.html`, `.htm`

Requires `NVIDIA_API_KEY` to be set (used by the embedding model).

### 3. Configure the tool in your YAML config

Add the function definition and wire it into the agents that should use it:

```yaml
functions:
  enterprise_knowledge_search:
    _type: enterprise_knowledge_search
    backend: llamaindex                              # or "foundational_rag"
    collections:
      - name: cybersecurity
        description: "NIST SP 800-53, NIST CSF, ISO 27001, and GDPR frameworks"
      - name: engineering_docs
        description: "Internal engineering design documents and architecture specs"
    top_k: 5                                         # Total results returned to agent
    per_collection_top_k: 10                         # Results fetched per collection before merging
    chroma_dir: ${AIQ_CHROMA_DIR:-/tmp/chroma_data}  # LlamaIndex only
    # rag_url: http://localhost:8081/v1              # Foundational RAG only

  intent_classifier:
    _type: intent_classifier
    tools:
      - web_search_tool
      - enterprise_knowledge_search      # Add here

  shallow_research_agent:
    _type: shallow_research_agent
    tools:
      - web_search_tool
      - enterprise_knowledge_search      # Add here

  deep_research_agent:
    _type: deep_research_agent
    tools:
      - advanced_web_search_tool
      - enterprise_knowledge_search      # Add here
```

### 4. Run the application

```bash
# Backend + UI
./scripts/start_e2e.sh

# Backend only
./scripts/start_server_in_debug_mode.sh
```

The "Enterprise Knowledge" toggle will appear in the Data Sources panel in the UI. It is enabled by default and does **not** require user authentication.

## Collection Structure Requirements

For citations and metadata to work correctly, your vector database documents must have the right metadata fields. The retriever adapter reads these fields from each stored chunk and maps them to the `Chunk` model used by the citation formatter.

### LlamaIndex (ChromaDB)

Each document node stored in ChromaDB has an embedding, text content, and a metadata dict. The retriever reads the metadata dict to build citations. Here is what a stored node looks like:

```
ChromaDB Node
├── id:        "node_abc123"              (auto-generated or provided)
├── embedding: [0.012, -0.034, ...]       (NVIDIA embedding vector)
├── document:  "The NIST Cybersecurity…"  (chunk text content)
└── metadata:  {                          (key-value pairs — this is what matters)
      "file_name":    "nist_sp_800-53.pdf",
      "page_label":   "42",
      "content_type": "text"
    }
```

#### Required metadata fields

| Metadata Key | Type | Required | Fallback | Used For |
| --- | --- | --- | --- | --- |
| `file_name` | string | **Yes** | `"unknown"` | Source filename in citations (`Source: nist_sp_800-53.pdf`) |
| `page_label` | string | No | `None` | Page number in citations (`Page: 42`). Stored as a string, converted to int at retrieval time |
| `content_type` | string | No | `"text"` | Content type label. One of: `"text"`, `"table"`, `"image"`, `"chart"` |

#### Additional metadata for multimodal content

These are only needed if your collection contains tables or images:

| Metadata Key | Type | When Required | Used For |
| --- | --- | --- | --- |
| `table_index` | int | Tables | Disambiguates tables on the same page (`Table 3 on p.12`) |
| `image_index` | int | Images/Charts | Disambiguates images on the same page (`Image 1 on p.5`) |
| `image_format` | string | Images | Passed through in metadata (e.g., `"jpeg"`) |
| `image_width` | int | Images | Passed through in metadata |
| `image_height` | int | Images | Passed through in metadata |
| `rows` | int | Tables | Passed through in metadata |
| `cols` | int | Tables | Passed through in metadata |

#### How citations are generated

The retriever reads the metadata and builds a `display_citation` string:

| Content Type | Has Page | Citation Format |
| --- | --- | --- |
| Text | Yes | `report.pdf, p.15 ('The NIST framework defines...')` |
| Text | No | `readme.txt` |
| Table | Yes | `report.pdf, p.15, Table 3` |
| Image | Yes | `report.pdf, p.5, Image 1` |
| Chart | Yes | `report.pdf, p.8, Chart 2` |

The enterprise knowledge search formatter then outputs each result as:

```text
--- Result 1 ---
Collection: cybersecurity
Source: nist_sp_800-53.pdf
Page: 42
Citation: nist_sp_800-53.pdf, p.42
Content Type: text
Relevance Score: 0.89

The NIST Cybersecurity Framework provides a policy framework...
```

#### Minimum viable metadata

If your external pipeline only produces text chunks from documents, the absolute minimum metadata for working citations is:

```python
metadata = {
    "file_name": "your_document.pdf",   # Required — shows up in every citation
    "page_label": "15",                 # Recommended — string, enables page references
}
```

Without `file_name`, citations will show `"unknown"`. Without `page_label`, citations will omit the page number (which is fine for non-paged formats like `.txt` or `.md`).

#### Example: Populating a collection from an external pipeline

If you are writing your own ingestion pipeline (not using `populate_collection.py`), here is how to create a ChromaDB collection that works with the LlamaIndex retriever:

```python
import chromadb
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.nvidia import NVIDIAEmbedding

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="/tmp/chroma_data")
chroma_collection = chroma_client.get_or_create_collection("my_collection")

# Create nodes with the required metadata
nodes = [
    TextNode(
        text="Your chunk text content here...",
        metadata={
            "file_name": "document.pdf",
            "page_label": "1",           # String, not int
            "content_type": "text",      # "text", "table", "image", or "chart"
        },
    ),
    # ... more nodes
]

# Build index (this embeds and stores the nodes)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5")

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)
```

### Foundational RAG (Milvus)

For the Foundational RAG backend, the RAG server's `/search` endpoint returns results. The retriever expects these fields in each search result:

| Response Field | Type | Required | Fallback | Used For |
| --- | --- | --- | --- | --- |
| `document_name` | string | **Yes** | `"unknown"` | Source filename in citations |
| `content` | string | **Yes** | `""` | Chunk text |
| `document_type` | string | No | `"text"` | Content type (`"text"`, `"table"`, `"image"`, `"chart"`) |
| `score` | float | No | `0.0` | Relevance ranking |
| `page_number` | int | No | `None` | Page reference. Can also be in `metadata.page_number` or `metadata.content_metadata.page_number` |
| `chunk_id` | string | No | Auto-generated | Unique chunk identifier |

The adapter automatically strips temporary file prefixes (e.g., `tmp8a3b1c2d_report.pdf` → `report.pdf`) from `document_name` for display.

## Configuration Reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | `"llamaindex"` \| `"foundational_rag"` | `"llamaindex"` | Which retriever backend to use |
| `collections` | list of `{name, description}` | *required* | At least one collection must be specified |
| `top_k` | int | `5` | Total number of results returned to the agent |
| `per_collection_top_k` | int \| null | same as `top_k` | Results fetched per collection before merging. Set higher than `top_k` for better cross-collection ranking |
| `chroma_dir` | string | `/tmp/chroma_data` | ChromaDB persistence directory (LlamaIndex only) |
| `rag_url` | string | `http://localhost:8081/v1` | RAG query server URL (Foundational RAG only) |
| `timeout` | int | `120` | Request timeout in seconds (Foundational RAG only) |
| `verify_ssl` | bool | `true` | Verify SSL certificates (Foundational RAG only) |

## UI Data Source Toggle

Enterprise Knowledge appears as its own independent toggle in the Data Sources panel, separate from "Web Search" and "Knowledge Base" (upload-based).

- **Does not require authentication** — available even when `REQUIRE_AUTH` is not enabled.
- **Enabled by default** — alongside Web Search.
- **Independent from Knowledge Base** — toggling one does not affect the other.

The backend reports it as a distinct data source via `GET /v1/data_sources`:

```json
[
  {"id": "web_search", "name": "Web Search", "description": "..."},
  {"id": "enterprise_knowledge", "name": "Enterprise Knowledge", "description": "..."},
  {"id": "knowledge_layer", "name": "Knowledge Base", "description": "..."}
]
```

The `data_sources` field sent from the UI in WebSocket messages controls which tools the agent can use. When `enterprise_knowledge` is included, the `enterprise_knowledge_search` tool is available. When it is not, the tool is filtered out. This filtering happens in `src/aiq_agent/common/data_sources.py`.

## Files Changed

### New files

| File | Purpose |
| --- | --- |
| `sources/enterprise_knowledge_search/pyproject.toml` | Package definition with NAT plugin entry point |
| `sources/enterprise_knowledge_search/src/__init__.py` | Package exports |
| `sources/enterprise_knowledge_search/src/register.py` | Tool implementation: config, retriever setup, search logic, result formatting |
| `sources/enterprise_knowledge_search/tests/__init__.py` | Test package |
| `sources/enterprise_knowledge_search/tests/test_register.py` | 22 unit tests covering config validation, result formatting, search logic, and data source filtering |
| `scripts/populate_collection.py` | CLI script to populate collections from a directory of documents |

### Modified files

| File | Change |
| --- | --- |
| `pyproject.toml` (root) | Added `enterprise-knowledge-search` to workspace sources and dev dependencies |
| `configs/config_web_default_llamaindex.yml` | Added `enterprise_knowledge_search` function and wired into all agents |
| `src/aiq_agent/common/data_sources.py` | Added `enterprise_knowledge` as a recognized data source ID in `filter_tools_by_sources()` and `format_data_source_tools()` |
| `frontends/aiq_api/src/aiq_api/routes/jobs.py` | Added `enterprise_knowledge` detection in `GET /v1/data_sources` endpoint |
| `frontends/ui/src/features/layout/data-sources.ts` | Added `ENTERPRISE_KNOWLEDGE_SOURCE_ID` constant and `doesDataSourceNeedAuthentication()` helper |
| `frontends/ui/src/features/layout/index.ts` | Re-exported new constants |
| `frontends/ui/src/features/layout/store.ts` | Updated default-enabled logic and `disableNonWebSources` to include enterprise knowledge |
| `frontends/ui/src/features/layout/components/DataSourcesPanel.tsx` | Updated auth gating to allow enterprise knowledge without authentication |
| `frontends/ui/src/features/chat/store.ts` | Updated `getDefaultEnabledDataSourceIds` to include enterprise knowledge |
| `frontends/ui/src/features/chat/hooks/use-websocket-chat.ts` | Updated send-time filter to include enterprise knowledge without auth token |

## Testing

Run the enterprise knowledge search tests:

```bash
uv run pytest sources/enterprise_knowledge_search/tests/ -v
```

Run the full test suite:

```bash
uv run pytest
```

## Adding a New No-Auth Data Source

If you add another data source that should be available without user authentication:

1. Add the source ID constant to `frontends/ui/src/features/layout/data-sources.ts`.
2. Add it to the `NO_AUTH_SOURCE_IDS` set in the same file.
3. That's it — the `doesDataSourceNeedAuthentication()` helper propagates the change to all UI components, stores, and the WebSocket chat hook.
