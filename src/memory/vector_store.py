"""Optional semantic vector search for memories using sentence-transformers.

Provides a VectorStore that wraps a sentence-transformer model for generating
dense embeddings.  When available, search becomes: FTS5 (recall) → vector
cosine similarity (precision) → LLM semantic rerank (top-3).

When sentence-transformers is not installed, the vector store degrades
gracefully — search falls through to the existing FTS5 → LLM rerank pipeline.

Usage:
    from src.memory.vector_store import get_vector_store
    vs = get_vector_store()
    if vs.available:
        results = vs.search(query_text, candidates, top_n=10)
"""

from __future__ import annotations

import logging
import struct
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Lightweight Chinese-friendly model (~118 MB, supports 50+ languages)
_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_EMBEDDING_DIM = 384  # Output dimension for this model

_vector_store: VectorStore | None = None
_lock = threading.Lock()


class VectorStore:
    """Optional dense-vector search layer for memory recall.

    Embeds memory body text into 384-dim float32 vectors.  Similarity search
    uses cosine distance over the inner-product space.

    Model loading is deferred to first use — construction is fast and safe
    to call at import time.  When sentence-transformers is unavailable or
    the model download times out, all public methods become no-ops.
    """

    def __init__(self) -> None:
        self._model = None
        self._available: bool | None = None  # None = not yet attempted
        self._load_error: str | None = None
        self._load_lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if the embedding model loaded successfully. Triggers lazy load."""
        self._ensure_loaded()
        return self._available is True

    @property
    def load_error(self) -> str | None:
        """Human-readable error message if model failed to load."""
        self._ensure_loaded()
        return self._load_error

    def encode(self, text: str) -> bytes | None:
        """Encode a single text into a 384-dim float32 BLOB.

        Returns None if the model is unavailable or text is empty.
        """
        if not self.available or not text.strip():
            return None
        try:
            vec = self._model.encode([text], normalize_embeddings=True)[0]
            # Pack as float32 little-endian: 384 floats × 4 bytes = 1536 bytes
            return struct.pack(f"<{_EMBEDDING_DIM}f", *vec)
        except Exception as e:
            logger.debug("Embedding encode failed: %s", e)
            return None

    def encode_batch(self, texts: list[str]) -> list[bytes | None]:
        """Encode multiple texts. Returns list of BLOBs (None per text on error)."""
        if not self.available:
            return [None] * len(texts)
        results: list[bytes | None] = []
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                valid_indices.append(i)
                valid_texts.append(t.strip())
            else:
                results.append(None)

        if not valid_texts:
            return results

        try:
            vectors = self._model.encode(
                valid_texts, normalize_embeddings=True, show_progress_bar=False,
            )
            # Interleave results back in original order
            vec_idx = 0
            final: list[bytes | None] = []
            for i in range(len(texts)):
                if i in valid_indices:
                    vec = vectors[vec_idx]
                    final.append(struct.pack(f"<{_EMBEDDING_DIM}f", *vec))
                    vec_idx += 1
                else:
                    final.append(None)
            return final
        except Exception as e:
            logger.debug("Batch embedding failed: %s", e)
            return [None] * len(texts)

    def similarity(
        self,
        query_blob: bytes,
        candidate_blobs: list[tuple[int, bytes | None]],
    ) -> list[tuple[int, float]]:
        """Compute cosine similarity between query and candidates.

        Args:
            query_blob: Packed float32 vector for the query (from encode()).
            candidate_blobs: List of (memory_id, blob_or_none) tuples.

        Returns:
            List of (memory_id, similarity_score) sorted by descending score.
            Only candidates with valid blobs are returned.
        """
        if not candidate_blobs:
            return []
        try:
            q_vec = list(struct.unpack(f"<{_EMBEDDING_DIM}f", query_blob))
        except (struct.error, TypeError):
            return []

        scored: list[tuple[int, float]] = []
        for mid, blob in candidate_blobs:
            if blob is None:
                continue
            try:
                c_vec = list(struct.unpack(f"<{_EMBEDDING_DIM}f", blob))
            except (struct.error, TypeError):
                continue
            # Dot product (vectors are already normalized → cosine similarity)
            sim = sum(a * b for a, b in zip(q_vec, c_vec))
            scored.append((mid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def index_all_pending(self, memory_db: Any, game: str = "arknights") -> int:
        """Index all memories that don't yet have an embedding.

        Called during startup maintenance.  Returns count of indexed memories.
        """
        if not self.available:
            return 0
        try:
            rows = memory_db.conn.execute(
                """SELECT id, body FROM memories_data
                   WHERE game = ? AND embedding IS NULL""",
                (game,),
            ).fetchall()
            if not rows:
                return 0

            ids = [r["id"] for r in rows]
            texts = [r["body"] or "" for r in rows]
            blobs = self.encode_batch(texts)

            count = 0
            for mid, blob in zip(ids, blobs):
                if blob is not None:
                    memory_db.conn.execute(
                        "UPDATE memories_data SET embedding = ? WHERE id = ?",
                        (blob, mid),
                    )
                    count += 1
            memory_db.conn.commit()
            if count:
                logger.info("Indexed %d memory embeddings (game=%s)", count, game)
            return count
        except Exception as e:
            logger.debug("Embedding index maintenance failed: %s", e)
            return 0

    def search(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Re-rank FTS5 candidates by semantic similarity.

        Computes the query embedding, loads stored embeddings for each
        candidate, and returns the top-N by cosine similarity.  Candidates
        without stored embeddings keep their original FTS5 rank.

        Args:
            query: Natural-language query (OCR texts + task description).
            candidates: FTS5 results (must have "id" key).
            top_n: Max results to return.

        Returns:
            Re-ranked candidate list.
        """
        if not self.available or not candidates:
            return candidates[:top_n]

        query_blob = self.encode(query)
        if query_blob is None:
            return candidates[:top_n]

        # Load embeddings for all candidates
        from src.memory.memory_db import memory_db
        candidate_blobs: list[tuple[int, bytes | None]] = []
        for m in candidates:
            mid = m.get("id")
            if mid is None:
                candidate_blobs.append((-1, None))
                continue
            row = memory_db.conn.execute(
                "SELECT embedding FROM memories_data WHERE id = ?", (mid,)
            ).fetchone()
            blob = row["embedding"] if row else None
            candidate_blobs.append((mid, blob))

        # If fewer than half of candidates have embeddings, fall back to FTS5 order
        with_blobs = sum(1 for _, b in candidate_blobs if b is not None)
        if with_blobs < len(candidates) * 0.4:
            return candidates[:top_n]

        scored = self.similarity(query_blob, candidate_blobs)
        id_to_candidate = {m["id"]: m for m in candidates}
        ranked: list[dict] = []
        seen: set[int] = set()
        for mid, sim in scored:
            m = id_to_candidate.get(mid)
            if m and mid not in seen:
                m["_embedding_score"] = round(sim, 4)
                ranked.append(m)
                seen.add(mid)

        # Append candidates without embeddings at the end (preserve FTS5 order)
        for m in candidates:
            if m["id"] not in seen:
                ranked.append(m)

        return ranked[:top_n]

    def _ensure_loaded(self) -> None:
        """Lazy-load the sentence-transformer model on first use.

        Thread-safe via double-check locking.  Model download has a 15-second
        timeout to prevent blocking when HuggingFace is unreachable (common
        behind firewalls in China).
        """
        if self._available is not None:
            return  # Already attempted (success or failure)

        with self._load_lock:
            if self._available is not None:
                return

            import concurrent.futures

            def _load():
                from sentence_transformers import SentenceTransformer
                import os
                # Prefer HF mirror for China accessibility.
                # If the model is already cached, use it offline.
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                try:
                    model = SentenceTransformer(
                        _EMBEDDING_MODEL,
                        local_files_only=True,
                    )
                except Exception:
                    # Fallback: allow online if local cache is missing
                    model = SentenceTransformer(_EMBEDDING_MODEL)
                # Verify with a quick test encoding
                model.encode(["test"], normalize_embeddings=True)
                return model

            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401
            except ImportError:
                self._available = False
                self._load_error = (
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
                logger.info("VectorStore unavailable: %s", self._load_error)
                return

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_load)
                    self._model = future.result(timeout=15.0)
                self._available = True
                logger.info(
                    "VectorStore ready: model=%s dim=%d",
                    _EMBEDDING_MODEL, _EMBEDDING_DIM,
                )
            except concurrent.futures.TimeoutError:
                self._available = False
                self._load_error = (
                    "Model download timed out (HuggingFace unreachable). "
                    "Vector search disabled — using FTS5 search only."
                )
                logger.warning("VectorStore: %s", self._load_error)
            except Exception as e:
                self._available = False
                self._load_error = f"Model load failed: {e}"
                logger.warning("VectorStore unavailable: %s", self._load_error)


def get_vector_store() -> VectorStore:
    """Get or create the global VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                _vector_store = VectorStore()
    return _vector_store
