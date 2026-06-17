import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    ticker: str
    filing_date: str
    section_name: str
    accession_number: str
    strategy: str
    chunk_index: int
    total_chunks: int
    char_count: int
    token_estimate: int


class BaseChunker(ABC):

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def chunk(self, sections: list[dict]) -> list[Chunk]:
        pass

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def make_chunk_id(self, accession: str, section: str, index: int, text: str = "") -> str:
        section_clean = section.replace(" ", "_")[:15]
        # Use full 12-char hash to make collisions virtually impossible
        text_hash = hashlib.md5(f"{accession}{section}{index}{text}".encode()).hexdigest()[:12]
        return f"{self.strategy_name[:5]}_{accession}_{section_clean}_{index:04d}_{text_hash}"
