#!/usr/bin/env python3
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

"""Populate a vector database collection from a directory of documents.

This script uses the existing knowledge layer ingestors to ingest all supported
files from a directory into a named collection. Supports both the LlamaIndex
(ChromaDB) and Foundational RAG (Milvus) backends. The collection can then be
queried by the enterprise_knowledge_search tool.

Usage:
    uv run python scripts/populate_collection.py <directory> <collection_name>
    uv run python scripts/populate_collection.py data/cybersecurity cybersecurity
    uv run python scripts/populate_collection.py data/cybersecurity cybersecurity --backend foundational_rag
    uv run python scripts/populate_collection.py data/cybersecurity cybersecurity --chroma-dir /tmp/my_chroma
    uv run python scripts/populate_collection.py data/cybersecurity cybersecurity --extract-tables

Requires:
    NVIDIA_API_KEY environment variable to be set.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Supported file extensions (matching LlamaIndex SimpleDirectoryReader)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".html", ".htm"}

BACKENDS = ("llamaindex", "foundational_rag")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate a vector database collection from a directory of documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/cybersecurity cybersecurity
  %(prog)s data/cybersecurity cybersecurity --backend foundational_rag --rag-ingest-url http://localhost:8082/v1
  %(prog)s data/cybersecurity cybersecurity --chroma-dir /data/chroma
  %(prog)s data/cybersecurity cybersecurity --extract-tables --extract-images
        """,
    )
    parser.add_argument("directory", type=str, help="Path to directory containing documents to ingest")
    parser.add_argument("collection_name", type=str, help="Name for the collection")

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=BACKENDS,
        default="llamaindex",
        help="Ingestion backend to use (default: llamaindex)",
    )

    # LlamaIndex-specific options
    llama_group = parser.add_argument_group("LlamaIndex backend options")
    llama_group.add_argument(
        "--chroma-dir",
        type=str,
        default=os.environ.get("AIQ_CHROMA_DIR", "/tmp/chroma_data"),
        help="ChromaDB persistence directory (default: $AIQ_CHROMA_DIR or /tmp/chroma_data)",
    )
    llama_group.add_argument("--chunk-size", type=int, default=1024, help="Chunk size in tokens (default: 1024)")
    llama_group.add_argument("--chunk-overlap", type=int, default=128, help="Chunk overlap in tokens (default: 128)")
    llama_group.add_argument("--extract-tables", action="store_true", help="Extract tables from PDFs using pdfplumber")
    llama_group.add_argument("--extract-images", action="store_true", help="Extract and caption images using VLM")
    llama_group.add_argument("--extract-charts", action="store_true", help="Extract and analyze charts using VLM")

    # Foundational RAG-specific options
    frag_group = parser.add_argument_group("Foundational RAG backend options")
    frag_group.add_argument(
        "--rag-ingest-url",
        type=str,
        default=os.environ.get("RAG_INGEST_URL", "http://localhost:8082/v1"),
        help="RAG ingestion server URL (default: $RAG_INGEST_URL or http://localhost:8082/v1)",
    )
    frag_group.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds for foundational_rag (default: 300)",
    )
    frag_group.add_argument(
        "--no-verify-ssl", action="store_true", help="Disable SSL verification for foundational_rag"
    )

    # General options
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Status poll interval in seconds (default: 2)")
    return parser.parse_args()


def find_documents(directory: Path) -> list[Path]:
    """Find all supported document files in the directory."""
    files = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def _import_backend(backend: str):
    """Import the backend adapter module to trigger registration decorators."""
    if backend == "llamaindex":
        import knowledge_layer.llamaindex.adapter  # noqa: F401
    elif backend == "foundational_rag":
        import knowledge_layer.foundational_rag.adapter  # noqa: F401
    else:
        print(f"Error: Unknown backend '{backend}'. Choose from: {', '.join(BACKENDS)}", file=sys.stderr)
        sys.exit(1)


def _build_config(args) -> dict:
    """Build backend-specific config dict from CLI args."""
    if args.backend == "llamaindex":
        return {
            "persist_dir": args.chroma_dir,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "extract_tables": args.extract_tables,
            "extract_charts": args.extract_charts,
            "extract_images": args.extract_images,
        }
    elif args.backend == "foundational_rag":
        return {
            "ingest_url": args.rag_ingest_url,
            "timeout": args.timeout,
            "verify_ssl": not args.no_verify_ssl,
        }
    return {}


def _print_config(args):
    """Print configuration summary."""
    print(f"Initializing {args.backend} ingestor...")
    if args.backend == "llamaindex":
        print(f"  ChromaDB dir: {args.chroma_dir}")
        print(f"  Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
        if args.extract_tables:
            print("  Table extraction: enabled")
        if args.extract_images:
            print("  Image extraction: enabled")
        if args.extract_charts:
            print("  Chart extraction: enabled")
    elif args.backend == "foundational_rag":
        print(f"  Ingest URL: {args.rag_ingest_url}")
        print(f"  Timeout: {args.timeout}s")
        print(f"  SSL verification: {'disabled' if args.no_verify_ssl else 'enabled'}")
    print()


def main():
    args = parse_args()

    # Validate NVIDIA_API_KEY
    if not os.environ.get("NVIDIA_API_KEY"):
        print("Error: NVIDIA_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with: export NVIDIA_API_KEY=<your-key>", file=sys.stderr)
        sys.exit(1)

    # Validate directory
    doc_dir = Path(args.directory).resolve()
    if not doc_dir.is_dir():
        print(f"Error: '{args.directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Find documents
    files = find_documents(doc_dir)
    if not files:
        print(f"Error: No supported files found in '{doc_dir}'.", file=sys.stderr)
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} document(s) in {doc_dir}:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.0f} KB)")
    print()

    # Import backend and initialize ingestor
    _import_backend(args.backend)

    from aiq_agent.knowledge import get_ingestor

    config = _build_config(args)
    _print_config(args)

    ingestor = get_ingestor(args.backend, config)

    # Create collection
    collection_name = args.collection_name
    print(f"Creating collection '{collection_name}'...")
    try:
        ingestor.create_collection(name=collection_name)
        print(f"  Collection '{collection_name}' created.")
    except Exception as e:
        # Collection may already exist — that's OK for get_or_create semantics
        print(f"  Collection '{collection_name}' already exists or created: {e}")
    print()

    # Submit ingestion job
    file_paths = [str(f) for f in files]
    print(f"Submitting ingestion job for {len(file_paths)} file(s)...")
    job_id = ingestor.submit_job(
        file_paths=file_paths,
        collection_name=collection_name,
    )
    print(f"  Job ID: {job_id}")
    print()

    # Poll until complete
    print("Ingesting documents...")
    last_processed = -1
    while True:
        status = ingestor.get_job_status(job_id)

        # Print progress when it changes
        if status.processed_files != last_processed:
            last_processed = status.processed_files
            print(f"  [{status.status.value}] {status.processed_files}/{status.total_files} files processed")
            for detail in status.file_details:
                chunks_info = f", {detail.chunks_created} chunks" if detail.chunks_created > 0 else ""
                print(f"    - {detail.file_name}: {detail.status.value}{chunks_info}")

        if status.is_terminal:
            break

        time.sleep(args.poll_interval)

    print()

    # Report results
    if status.is_success:
        metadata = status.metadata or {}
        print("Ingestion completed successfully!")
        print(f"  Backend: {args.backend}")
        print(f"  Collection: {collection_name}")
        print(f"  Files processed: {status.processed_files}/{status.total_files}")
        print(f"  Total chunks: {metadata.get('total_chunks', 'N/A')}")
        if metadata.get("tables_extracted"):
            print(f"  Tables extracted: {metadata['tables_extracted']}")
        if metadata.get("charts_extracted"):
            print(f"  Charts extracted: {metadata['charts_extracted']}")
        if metadata.get("images_captioned"):
            print(f"  Images captioned: {metadata['images_captioned']}")
        if args.backend == "llamaindex":
            print(f"  ChromaDB dir: {args.chroma_dir}")
        elif args.backend == "foundational_rag":
            print(f"  Ingest URL: {args.rag_ingest_url}")
        print()
        print("You can now use this collection with enterprise_knowledge_search in your YAML config:")
        print("  collections:")
        print(f"    - name: {collection_name}")
        print('      description: "Your collection description here"')
    else:
        print(f"Ingestion failed: {status.error_message}", file=sys.stderr)
        for detail in status.file_details:
            if detail.error_message:
                print(f"  - {detail.file_name}: {detail.error_message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
