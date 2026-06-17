from loguru import logger
from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """
    Strategy A: Split text into fixed-size chunks with overlap.
    Simple, fast, but breaks mid-sentence and loses section context.
    chunk_size: number of characters per chunk
    overlap: number of characters to overlap between chunks
    """

    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        super().__init__("fixed_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, sections: list[dict]) -> list[Chunk]:
        all_chunks = []

        for section in sections:
            text = section["text"]
            accession = section["accession_number"]
            ticker = section["ticker"]
            filing_date = section["filing_date"]
            section_name = section["section_name"]

            # Slide a window across the text
            start = 0
            section_chunks = []

            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]

                if len(chunk_text.strip()) < 50:
                    break

                section_chunks.append(chunk_text)
                start = end - self.overlap  # step back by overlap amount

            # Build Chunk objects
            for i, chunk_text in enumerate(section_chunks):
                chunk_id = self.make_chunk_id(accession, section_name, i)
                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    ticker=ticker,
                    filing_date=filing_date,
                    section_name=section_name,
                    accession_number=accession,
                    strategy=self.strategy_name,
                    chunk_index=i,
                    total_chunks=len(section_chunks),
                    char_count=len(chunk_text),
                    token_estimate=self.estimate_tokens(chunk_text),
                ))

        logger.info(
            f"FixedSizeChunker: {len(sections)} sections → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.overlap})"
        )
        return all_chunks