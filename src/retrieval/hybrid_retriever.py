from loguru import logger
from .base import BaseRetriever
from src.embedding.embedder import Embedder
from src.embedding.vector_store import VectorStore
from src.embedding.bm25_index import BM25Index


class HybridRetriever(BaseRetriever):
    """Reciprocal Rank Fusion over BM25 + vector results."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, filter_dict: dict = None) -> list[dict]:
        # Vector results
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.query(query_embedding, top_k=top_k * 3, filter_dict=filter_dict)

        # BM25 results
        bm25_results = self.bm25_index.query(query, top_k=top_k * 3)

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}

        for rank, r in enumerate(vector_results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0) + self.vector_weight * (1 / (self.rrf_k + rank + 1))
            meta[cid] = r

        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0) + self.bm25_weight * (1 / (self.rrf_k + rank + 1))
            if cid not in meta:
                meta[cid] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for cid, score in ranked:
            entry = dict(meta[cid])
            entry["score"] = score
            entry["retrieval_method"] = "hybrid"
            results.append(entry)

        logger.debug(f"HybridRetriever: '{query[:60]}' -> {len(results)} results")
        return results
