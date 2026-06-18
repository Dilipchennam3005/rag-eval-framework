import pickle
from pathlib import Path
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer


class SparseEncoder:
    """
    Fits a TF-IDF vocabulary on the corpus and converts text to Pinecone
    sparse-vector format: {"indices": [...], "values": [...]}.

    The fitted vectorizer is serialized as a small pkl (~500 KB for 30K vocab)
    with no chunk text, so it can be committed to git and baked into Docker.
    """

    def __init__(self, params_path: str = "data/sparse_encoder.pkl"):
        self.params_path = Path(params_path)
        self.vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: list[str]) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=30_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z0-9]+\b",
        )
        self.vectorizer.fit(texts)
        self._save()
        logger.info(f"SparseEncoder fitted: vocab={len(self.vectorizer.vocabulary_)} terms")

    def _save(self) -> None:
        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.params_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"SparseEncoder saved to {self.params_path}")

    def load(self) -> bool:
        if not self.params_path.exists():
            return False
        with open(self.params_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        logger.info(f"SparseEncoder loaded from {self.params_path}")
        return True

    def encode_documents(self, texts: list[str]) -> list[dict]:
        matrix = self.vectorizer.transform(texts)
        results = []
        for i in range(matrix.shape[0]):
            row = matrix[i].tocsr()
            results.append({
                "indices": row.indices.tolist(),
                "values": [float(v) for v in row.data],
            })
        return results

    def encode_query(self, text: str) -> dict:
        row = self.vectorizer.transform([text]).tocsr()[0]
        return {
            "indices": row.indices.tolist(),
            "values": [float(v) for v in row.data],
        }
