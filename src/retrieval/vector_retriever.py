from loguru import logger
from .base import BaseRetriever
from src.embedding.embedder import Embedder
from src.embedding.vector_store import VectorStore


class VectorRetriever(BaseRetriever):

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, filter_dict: dict = None) -> list[dict]:
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.query(query_embedding, top_k=top_k, filter_dict=filter_dict)
        for r in results:
            r["retrieval_method"] = "vector"
        logger.debug(f"VectorRetriever: '{query[:60]}' -> {len(results)} results")
        return results
