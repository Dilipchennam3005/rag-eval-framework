from loguru import logger
from .base import BaseRetriever
from src.embedding.embedder import Embedder
from src.embedding.vector_store import VectorStore
from src.embedding.sparse_encoder import SparseEncoder

_SPARSE_SCALE = 3.0


class HybridRetriever(BaseRetriever):
    """
    Pinecone native sparse-dense hybrid retrieval.
    Dense (text-embedding-3-small) + sparse (TF-IDF) vectors are stored
    together in a dotproduct-metric index; Pinecone combines the scores
    in a single query call — no local BM25 or RRF needed.

    sparse_weight scales the TF-IDF values so sparse signals compete
    with the typically higher dense dot-product scores.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        sparse_encoder: SparseEncoder,
        sparse_weight: float = 0.3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_encoder = sparse_encoder
        self.sparse_weight = sparse_weight

    def retrieve(self, query: str, top_k: int = 5, filter_dict: dict = None) -> list[dict]:
        dense = self.embedder.embed_query(query)
        raw_sparse = self.sparse_encoder.encode_query(query)

        # Scale sparse values so they're weighted relative to dense scores
        sparse = {
            "indices": raw_sparse["indices"],
            "values": [v * self.sparse_weight * _SPARSE_SCALE for v in raw_sparse["values"]],
        }

        results = self.vector_store.query(
            dense, top_k=top_k, filter_dict=filter_dict, sparse_vector=sparse
        )
        for r in results:
            r["retrieval_method"] = "hybrid_pinecone"

        logger.debug(f"HybridRetriever (Pinecone): '{query[:60]}' -> {len(results)} results")
        return results
