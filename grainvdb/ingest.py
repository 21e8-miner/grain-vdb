"""
GrainVDB Document Ingestion & Chunking Engine
Extracts, splits, and vectorizes local documents (PDFs, Markdown, Code, Text)
directly into GrainVDB Apple Silicon shared memory.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from .engine import GrainVDB
from .embeddings import BaseEmbeddingProvider, FastLocalEmbedding


class DocumentChunker:
    """Configurable text chunker with sliding window overlap."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping character/token windows."""
        text = text.strip()
        if not text:
            return []
        
        # Clean multiple whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += step
            if start + self.chunk_overlap >= len(text):
                # Add final trailing slice if significant
                last_slice = text[start:]
                if last_slice and last_slice not in chunks[-1]:
                    chunks.append(last_slice)
                break

        return chunks


class LocalIngestPipeline:
    """
    High-throughput local file ingest pipeline for GrainVDB.
    Scans directories, parses files, generates embeddings, and writes to GrainVDB.
    """
    SUPPORTED_EXTENSIONS = {
        ".md", ".txt", ".py", ".swift", ".metal", ".cpp", ".h", ".c",
        ".js", ".ts", ".jsx", ".tsx", ".json", ".csv", ".log", ".yaml", ".yml"
    }

    def __init__(
        self,
        vdb: GrainVDB,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.vdb = vdb
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = embedding_provider or FastLocalEmbedding(dimension=vdb.dimension)

    def ingest_text(self, text: str, title: str, category: str = "General", extra_meta: Optional[Dict[str, Any]] = None) -> int:
        """Chunk a text string and ingest into GrainVDB."""
        chunks = self.chunker.split_text(text)
        if not chunks:
            return 0

        vectors = self.embedder.embed_documents(chunks)
        metas = []
        for i, chunk in enumerate(chunks):
            meta = {
                "title": title,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk
            }
            if extra_meta:
                meta.update(extra_meta)
            metas.append(meta)

        self.vdb.add_vectors(vectors, metadata=metas)
        return len(chunks)

    def ingest_file(self, file_path: Union[str, Path], category: Optional[str] = None) -> int:
        """Read a single local file, chunk, and ingest."""
        path = Path(file_path)
        if not path.is_file():
            return 0

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            title = path.name
            cat = category or path.parent.name or "Documents"
            return self.ingest_text(content, title=title, category=cat, extra_meta={"path": str(path.resolve())})
        except Exception as e:
            print(f"Failed to ingest {path}: {e}")
            return 0

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
        category: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Recursively scan and ingest all supported files in a directory.
        Returns (files_ingested, total_chunks_created).
        """
        root = Path(dir_path)
        if not root.is_dir():
            return 0, 0

        files_count = 0
        chunks_count = 0

        iterator = root.rglob("*") if recursive else root.glob("*")
        for item in iterator:
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                c = self.ingest_file(item, category=category)
                if c > 0:
                    files_count += 1
                    chunks_count += c

        return files_count, chunks_count
