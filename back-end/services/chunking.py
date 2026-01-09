
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from dataclasses import dataclass



@dataclass(frozen=True)
class ChunkingStrategy:
    name: str
    chunk_size: int
    chunk_overlap: int


class Chunking:
    """
    Handles text chunking using predefined chunking strategies.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, ChunkingStrategy] = self._load_strategies()

    @staticmethod
    def _load_strategies() -> Dict[str, ChunkingStrategy]:

        return {
            "small": ChunkingStrategy(
                name="small",
                chunk_size=256,
                chunk_overlap=40,
            ),
            "medium": ChunkingStrategy(
                name="medium",
                chunk_size=512,
                chunk_overlap=80,
            ),
            "large": ChunkingStrategy(
                name="large",
                chunk_size=1024,
                chunk_overlap=200,
            ),
            "extra_large": ChunkingStrategy(
                name="extra_large",
                chunk_size=2048,
                chunk_overlap=400,
            ),
        }

    def get_available_strategies(self) -> List[str]:
        """
        Return a list of available strategy names.
        """
        return list(self._strategies.keys())

    def chunk_text(
        self,
        text: str,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        return splitter.split_text(text)

    def chunk_text_with_strategy(
        self,
        text: str,
        strategy_name: str,
    ) -> List[str]:

        if strategy_name not in self._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Valid options: {self.get_available_strategies()}"
            )

        strategy = self._strategies[strategy_name]

        return self.chunk_text(
            text,
            chunk_size=strategy.chunk_size,
            chunk_overlap=strategy.chunk_overlap,
        )
