from loguru import logger
from .base import BaseChunker, Chunk


class DocumentAwareChunker(BaseChunker):
    """
    Strategy B: Respect document structure.
    Splits by paragraphs first, then merges small ones and splits large ones.
    Each chunk stays within a section — never crosses section boundaries.
    Preserves semantic meaning much better than fixed-size.
    """

    def __init__(self, max_chunk_size: int = 3000, min_chunk_size: int = 200):
        super().__init__("document_aware")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, sections: list[dict]) -> list[Chunk]:
        all_chunks = []

        for section in sections:
            text = section["text"]
            accession = section["accession_number"]
            ticker = section["ticker"]
            filing_date = section["filing_date"]
            section_name = section["section_name"]

            # Split by paragraph breaks
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            # Merge short paragraphs, split long ones
            merged = self._merge_paragraphs(paragraphs)

            # Build Chunk objects
            for i, chunk_text in enumerate(merged):
                if len(chunk_text.strip()) < 50:
                    continue

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
                    total_chunks=len(merged),
                    char_count=len(chunk_text),
                    token_estimate=self.estimate_tokens(chunk_text),
                ))

        logger.info(
            f"DocumentAwareChunker: {len(sections)} sections → {len(all_chunks)} chunks "
            f"(max={self.max_chunk_size}, min={self.min_chunk_size})"
        )
        return all_chunks

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """
        Merge short paragraphs together until we hit max_chunk_size.
        Split paragraphs that exceed max_chunk_size.
        """
        result = []
        buffer = ""

        for para in paragraphs:
            # If this paragraph alone exceeds max, split it
            if len(para) > self.max_chunk_size:
                if buffer:
                    result.append(buffer.strip())
                    buffer = ""
                # Split large paragraph into fixed windows
                start = 0
                while start < len(para):
                    result.append(para[start:start + self.max_chunk_size].strip())
                    start += self.max_chunk_size
                continue

            # If adding this paragraph would exceed max, flush buffer
            if len(buffer) + len(para) > self.max_chunk_size:
                if buffer:
                    result.append(buffer.strip())
                buffer = para
            else:
                buffer = f"{buffer}\n\n{para}".strip() if buffer else para

        if buffer:
            result.append(buffer.strip())

        return result