from loguru import logger
from .base import BaseChunker, Chunk


class DocumentAwareChunker(BaseChunker):

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
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            merged = self._merge_paragraphs(paragraphs)
            for i, chunk_text in enumerate(merged):
                if len(chunk_text.strip()) < 50:
                    continue
                chunk_id = self.make_chunk_id(accession, section_name, i, chunk_text)
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
        logger.info(f"DocumentAwareChunker: {len(sections)} sections -> {len(all_chunks)} chunks")
        return all_chunks

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        result = []
        buffer = ""
        for para in paragraphs:
            if len(para) > self.max_chunk_size:
                if buffer:
                    result.append(buffer.strip())
                    buffer = ""
                start = 0
                while start < len(para):
                    result.append(para[start:start + self.max_chunk_size].strip())
                    start += self.max_chunk_size
                continue
            if len(buffer) + len(para) > self.max_chunk_size:
                if buffer:
                    result.append(buffer.strip())
                buffer = para
            else:
                buffer = f"{buffer}\n\n{para}".strip() if buffer else para
        if buffer:
            result.append(buffer.strip())
        return result
