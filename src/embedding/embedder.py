import os
import time
from loguru import logger
from openai import OpenAI, InternalServerError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

_MAX_RETRIES = 3
_RETRY_DELAY = 5


class Embedder:
    """
    Wraps OpenAI's text-embedding-3-small model.
    Handles batching and retries on transient 5xx errors.
    """

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info(f"Embedder initialized: model={model}, batch_size={batch_size}")

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                return [item.embedding for item in response.data]
            except (InternalServerError, APIStatusError) as e:
                if attempt == _MAX_RETRIES:
                    raise
                wait = _RETRY_DELAY * attempt
                logger.warning(f"OpenAI error (attempt {attempt}/{_MAX_RETRIES}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")
            batch = [t.strip() if t.strip() else "empty" for t in batch]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)
            if batch_num < total_batches:
                time.sleep(0.5)

        logger.info(f"Embedded {len(texts)} texts → vectors of dim {len(all_embeddings[0])}")
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[query.strip()],
        )
        return response.data[0].embedding


if __name__ == "__main__":
    embedder = Embedder()
    test_texts = [
        "Apple reported revenue of $394 billion in fiscal year 2024.",
        "Risk factors include supply chain disruptions and geopolitical tensions.",
        "The company repurchased $90 billion of its common stock.",
    ]
    embeddings = embedder.embed_texts(test_texts)
    print(f"Embedded {len(embeddings)} texts")
    print(f"Vector dimensions: {len(embeddings[0])}")
    print(f"First 5 values of embedding 1: {embeddings[0][:5]}")