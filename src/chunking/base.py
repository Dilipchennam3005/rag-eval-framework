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
    """
    Abstract base class for all chunking strategies.
    Every chunker must implement the chunk() method.
    This is the interface that makes strategies swappable.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def chunk(self, sections: list[dict]) -> list[Chunk]:
        """
        Takes a list of parsed sections and returns a list of Chunks.
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimate: 1 token ≈ 4 characters for English text.
        """
        return len(text) // 4

    def make_chunk_id(self, accession: str, section: str, index: int) -> str:
        section_clean = section.replace(" ", "_")[:20]
        return f"{accession}_{section_clean}_{index:04d}"