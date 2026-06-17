from loguru import logger


class CrossEncoderReranker:
    """
    Re-scores retrieved chunks with a cross-encoder for higher precision.
    Drop-in wrapper: pass any retriever's results through rerank().
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking: pip install sentence-transformers"
            ) from exc
        self.model_name = model
        self.model = CrossEncoder(model)
        logger.info(f"CrossEncoderReranker loaded: {model}")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        if not chunks:
            return chunks

        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for chunk, score in ranked:
            entry = dict(chunk)
            entry["rerank_score"] = float(score)
            entry["score"] = float(score)
            results.append(entry)

        logger.debug(f"Reranked {len(chunks)} → {len(results)} chunks for '{query[:60]}'")
        return results
