import pickle
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi


class BM25Index:

    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = Path(index_path)
        self.bm25 = None
        self.chunks = []

    def build(self, chunks: list) -> None:
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} chunks")
        self._save()

    def _save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"BM25 index saved to {self.index_path}")

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        logger.info(f"BM25 index loaded: {len(self.chunks)} chunks")
        return True

    def query(self, query_text: str, top_k: int = 20) -> list[dict]:
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build() first.")

        tokens = query_text.lower().split()
        scores = self.bm25.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "ticker": chunk.ticker,
                        "filing_date": chunk.filing_date,
                        "section_name": chunk.section_name,
                        "accession_number": chunk.accession_number,
                        "strategy": chunk.strategy,
                    },
                    "bm25_score": float(scores[idx]),
                })

        return results