"""Knowledge RAG module — thin wrapper retrieval-augmented generation.

Dev:  chromadb local persistence
Prod: pgvector (fits existing Postgres stack via db/models.py)

Embeddings:
  Offline: all-MiniLM-L6-v2 (sentence-transformers)
  Online:  text-embedding-3-small (OpenAI fallback adapter coexists with Groq)

Ingest sources: past anomaly reports + spacecraft manuals.
Retrieval at event-time → injected into hypothesis generation.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Optional dependency imports
try:
    import chromadb  # type: ignore[import-untyped]
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
CHROMA_DIR = KNOWLEDGE_DIR / "chroma_db"


@dataclass
class Document:
    """A document in the knowledge base."""
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class RetrievalResult:
    """Result of a knowledge retrieval query."""
    documents: list[Document]
    scores: list[float]
    query: str


class _TFIDFEmbedder:
    """Lightweight TF-IDF-based embedder for environments without sentence-transformers.

    Not a real embedding model — produces bag-of-words cosine-comparable vectors.
    Sufficient for CI/offline testing.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            # Simple hash-based pseudo-embedding
            words = text.lower().split()
            vec = np.zeros(self.dim)
            for word in words:
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors)


class KnowledgeBase:
    """Thin-wrapper RAG knowledge base.

    Supports:
    - chromadb for persistent vector storage (dev/prod)
    - In-memory fallback with TF-IDF embeddings for CI
    - OpenAI embedding adapter for online use
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = "spacethink_knowledge",
        use_openai: bool = False,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.use_openai = use_openai

        # Initialize embedder
        if _HAS_SENTENCE_TRANSFORMERS:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._embed_fn = self._embedder.encode
        else:
            self._fallback = _TFIDFEmbedder()
            self._embed_fn = self._fallback.encode

        # Initialize store
        if _HAS_CHROMA:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        else:
            self._documents: list[Document] = []
            self._embeddings: list[np.ndarray] = []
            self._use_chroma = False

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        if self.use_openai:
            return self._get_openai_embedding(text)
        vec = self._embed_fn([text])
        return vec[0].tolist() if hasattr(vec[0], 'tolist') else list(vec[0])

    def _get_openai_embedding(self, text: str) -> list[float]:
        """Get embedding from OpenAI API (fallback adapter)."""
        try:
            import openai  # type: ignore[import-untyped]
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            warnings.warn(f"OpenAI embedding failed ({e}), falling back to local.", stacklevel=2)
            vec = self._embed_fn([text])
            return vec[0].tolist() if hasattr(vec[0], 'tolist') else list(vec[0])

    def ingest(self, doc: Document) -> None:
        """Add a document to the knowledge base."""
        embedding = self._get_embedding(doc.text)

        if self._use_chroma:
            self._collection.upsert(
                ids=[doc.id],
                documents=[doc.text],
                embeddings=[embedding],
                metadatas=[{**doc.metadata, "source": doc.source}],
            )
        else:
            doc_with_emb = Document(
                id=doc.id,
                text=doc.text,
                source=doc.source,
                metadata=doc.metadata,
                embedding=embedding,
            )
            # Replace if exists
            self._documents = [d for d in self._documents if d.id != doc.id]
            self._documents.append(doc_with_emb)
            self._embeddings = [
                np.array(d.embedding) for d in self._documents
            ]

    def ingest_batch(self, docs: list[Document]) -> int:
        """Ingest a batch of documents."""
        for doc in docs:
            self.ingest(doc)
        return len(docs)

    def retrieve(self, query: str, n_results: int = 5) -> RetrievalResult:
        """Retrieve the most relevant documents for a query."""
        if self._use_chroma:
            query_embedding = self._get_embedding(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self._collection.count() or 1),
            )

            docs = []
            scores = []
            if results["documents"] and results["documents"][0]:
                for i, (doc_text, doc_id, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["ids"][0],
                    results["metadatas"][0] if results["metadatas"] else [{}] * len(results["ids"][0]),
                    results["distances"][0] if results["distances"] else [0.0] * len(results["ids"][0]),
                )):
                    docs.append(Document(
                        id=doc_id,
                        text=doc_text,
                        source=metadata.get("source", "unknown"),
                        metadata=metadata,
                    ))
                    scores.append(1.0 - distance)  # Convert distance to similarity

            return RetrievalResult(documents=docs, scores=scores, query=query)
        else:
            # In-memory cosine similarity search
            if not self._documents:
                return RetrievalResult(documents=[], scores=[], query=query)

            query_vec = np.array(self._get_embedding(query))
            embeddings = np.array(self._embeddings)

            # Cosine similarity
            dot = embeddings @ query_vec
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
            norms[norms < 1e-10] = 1e-10
            similarities = dot / norms

            # Top-k
            top_indices = np.argsort(similarities)[::-1][:n_results]

            docs = [self._documents[i] for i in top_indices]
            scores = [float(similarities[i]) for i in top_indices]

            return RetrievalResult(documents=docs, scores=scores, query=query)

    def count(self) -> int:
        """Return the number of documents in the knowledge base."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._documents)


def ingest_anomaly_reports(kb: KnowledgeBase, reports_dir: Path) -> int:
    """Ingest past anomaly reports from a directory of JSON files."""
    count = 0
    if not reports_dir.exists():
        return 0

    for json_file in reports_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, list):
                for item in data:
                    doc = Document(
                        id=f"report_{json_file.stem}_{count}",
                        text=json.dumps(item, default=str),
                        source=f"anomaly_report:{json_file.name}",
                        metadata={"file": json_file.name},
                    )
                    kb.ingest(doc)
                    count += 1
            else:
                doc = Document(
                    id=f"report_{json_file.stem}",
                    text=json.dumps(data, default=str),
                    source=f"anomaly_report:{json_file.name}",
                    metadata={"file": json_file.name},
                )
                kb.ingest(doc)
                count += 1
        except Exception as e:
            warnings.warn(f"Failed to ingest {json_file}: {e}", stacklevel=2)

    return count
