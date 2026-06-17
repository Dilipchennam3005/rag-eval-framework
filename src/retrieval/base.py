from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filter_dict: dict = None) -> list[dict]:
        pass
