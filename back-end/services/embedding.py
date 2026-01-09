
from dataclasses import dataclass
from typing import Dict, List

from langchain_community.embeddings import OllamaEmbeddings


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    model_id: str
    dimensions: int


class Embedding:

    def __init__(self, model_name: str) -> None:
        self._models: Dict[str, EmbeddingModel] = self._load_models()

        if model_name not in self._models:
            raise ValueError(
                f"Unknown embedding model '{model_name}'. "
                f"Valid options: {list(self._models.keys())}"
            )

        self.model = self._models[model_name]

        self.embeddings = OllamaEmbeddings(
            model=self.model.model_id
        )

    @staticmethod
    def _load_models() -> Dict[str, EmbeddingModel]:

        return {
            "nomic": EmbeddingModel(
                name="nomic",
                model_id="nomic-embed-text",
                dimensions=768,
            ),
            "mxbai": EmbeddingModel(
                name="mxbai",
                model_id="mxbai-embed-large",
                dimensions=1024,
            ),
            "bge": EmbeddingModel(
                name="bge",
                model_id="bge-large-en-v1.5",
                dimensions=1024,
            ),
        }

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:

        return self.embeddings.embed_documents(chunks)

    def embed_query(self, query: str) -> List[float]:

        return self.embeddings.embed_query(query)
