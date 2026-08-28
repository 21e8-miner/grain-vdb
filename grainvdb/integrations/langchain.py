"""
GrainVDB LangChain VectorStore Integration
Enables drop-in local vector retrieval for LangChain on Apple Silicon.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np

from ..engine import GrainVDB, SearchMode, DistanceMetric, EngineType

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    LANGCHAIN_INSTALLED = True
except ImportError:
    # Minimal fallback mock classes if langchain is not installed
    LANGCHAIN_INSTALLED = False

    class Document:  # type: ignore
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class Embeddings:  # type: ignore
        pass

    class VectorStore:  # type: ignore
        pass


class GrainVDBVectorStore(VectorStore):
    """
    Apple Silicon-Native VectorStore for LangChain.
    
    Example:
    --------
    >>> from langchain_community.embeddings import FakeEmbeddings
    >>> from grainvdb.integrations import GrainVDBVectorStore
    >>> embeddings = FakeEmbeddings(size=128)
    >>> vs = GrainVDBVectorStore(embedding=embeddings, dim=128)
    >>> vs.add_texts(["Hello world", "Apple Silicon RAG"])
    >>> docs = vs.similarity_search("M2 Ultra bandwidth", k=2)
    """

    def __init__(
        self,
        embedding: Any,
        dim: int = 128,
        mode: SearchMode = SearchMode.EXACT,
        engine: EngineType = EngineType.AUTO,
        distance: DistanceMetric = DistanceMetric.COSINE,
        db: Optional[GrainVDB] = None,
    ):
        self.embedding = embedding
        self.dim = dim
        self._db = db or GrainVDB(dim=dim, mode=mode, engine=engine, distance=distance)
        self._doc_store: Dict[int, Document] = {}

    @property
    def embeddings(self) -> Any:
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> List[int]:
        """Add texts to the GrainVDB vector store."""
        text_list = list(texts)
        if not text_list:
            return []

        # Generate embeddings
        if hasattr(self.embedding, "embed_documents"):
            embeddings_matrix = self.embedding.embed_documents(text_list)
        elif callable(self.embedding):
            embeddings_matrix = [self.embedding(t) for t in text_list]
        else:
            raise ValueError("Embedding model must have embed_documents or be callable")

        vecs = np.asarray(embeddings_matrix, dtype=np.float32)
        n = len(text_list)

        if ids is None:
            curr_count = self._db.vector_count
            assigned_ids = [curr_count + i for i in range(n)]
        else:
            assigned_ids = [int(i) for i in ids]

        meta_list = metadatas if metadatas is not None else [{} for _ in range(n)]

        for i, text in enumerate(text_list):
            doc_id = assigned_ids[i]
            meta = meta_list[i].copy() if i < len(meta_list) else {}
            meta["text"] = text
            self._doc_store[doc_id] = Document(page_content=text, metadata=meta)

        self._db.add_vectors(
            vectors=vecs,
            ids=assigned_ids,
            metadata=[self._doc_store[vid].metadata for vid in assigned_ids],
        )

        return assigned_ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Callable[[int, Dict[str, Any]], bool]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to query along with similarity scores."""
        if hasattr(self.embedding, "embed_query"):
            query_vec = np.asarray(self.embedding.embed_query(query), dtype=np.float32)
        elif callable(self.embedding):
            query_vec = np.asarray(self.embedding(query), dtype=np.float32)
        else:
            raise ValueError("Embedding model must have embed_query or be callable")

        res = self._db.search(query=query_vec, k=k, filter=filter)

        output: List[Tuple[Document, float]] = []
        for idx, score in zip(res.indices, res.scores):
            doc = self._doc_store.get(int(idx))
            if doc is not None:
                output.append((doc, float(score)))
            else:
                meta = res.metadata[int(idx)] if res.metadata else {}
                text = meta.get("text", "")
                output.append((Document(page_content=text, metadata=meta), float(score)))

        return output

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Callable[[int, Dict[str, Any]], bool]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to query."""
        results = self.similarity_search_with_score(query=query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results]

    @classmethod
    def from_texts(
        cls: Type["GrainVDBVectorStore"],
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        dim: int = 128,
        mode: SearchMode = SearchMode.EXACT,
        engine: EngineType = EngineType.AUTO,
        **kwargs: Any,
    ) -> "GrainVDBVectorStore":
        """Construct GrainVDBVectorStore wrapper from raw texts."""
        store = cls(embedding=embedding, dim=dim, mode=mode, engine=engine, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store
