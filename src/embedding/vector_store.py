import os
import time
from loguru import logger
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class VectorStore:
    """
    Pinecone-backed dense vector store.
    One index, one namespace per chunking strategy.
    """

    def __init__(
        self,
        index_name: str = "rag-eval-framework",
        namespace: str = "default",
        dimension: int = 1536,
    ):
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension

        existing = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing:
            logger.info(f"Creating Pinecone index '{index_name}' (dim={dimension})...")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait until ready
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
            logger.info(f"Index '{index_name}' ready")

        self.index = self.pc.Index(index_name)
        logger.info(
            f"Pinecone: index='{index_name}', namespace='{namespace}', "
            f"vectors={self.count()}"
        )

    def add_chunks(
        self,
        chunks: list,
        embeddings: list[list[float]],
        sparse_vectors: list[dict] | None = None,
    ) -> int:
        if not chunks:
            return 0

        vectors = []
        for i, (c, emb) in enumerate(zip(chunks, embeddings)):
            vec: dict = {
                "id": c.chunk_id,
                "values": emb,
                "metadata": {
                    "ticker": c.ticker,
                    "filing_date": c.filing_date,
                    "section_name": c.section_name,
                    "accession_number": c.accession_number,
                    "strategy": c.strategy,
                    "chunk_index": c.chunk_index,
                    "char_count": c.char_count,
                    "token_estimate": c.token_estimate,
                    "text": c.text,
                },
            }
            if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i]["indices"]:
                vec["sparse_values"] = sparse_vectors[i]
            vectors.append(vec)

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i : i + batch_size], namespace=self.namespace)

        logger.info(f"Upserted {len(chunks)} vectors → namespace '{self.namespace}'")
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_dict: dict = None,
        sparse_vector: dict | None = None,
    ) -> list[dict]:
        kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "namespace": self.namespace,
            "include_metadata": True,
        }
        if filter_dict:
            kwargs["filter"] = filter_dict
        if sparse_vector and sparse_vector.get("indices"):
            kwargs["sparse_vector"] = sparse_vector

        response = self.index.query(**kwargs)

        results = []
        for match in response.matches:
            meta = dict(match.metadata or {})
            text = meta.pop("text", "")
            results.append({
                "chunk_id": match.id,
                "text": text,
                "metadata": meta,
                "distance": 1 - match.score,
                "score": match.score,
            })

        return results

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        ns = stats.namespaces.get(self.namespace)
        return ns.vector_count if ns else 0

    def reset(self):
        self.index.delete(delete_all=True, namespace=self.namespace)
        logger.info(f"Deleted all vectors in namespace '{self.namespace}'")
