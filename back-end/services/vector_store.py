
from typing import List, Optional, Dict, Any
import uuid


class VectorStore:

    def __init__(self, collection) -> None:
        self.collection = collection

    def add_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

            if metadatas is None:
                metadatas = [{} for _ in range(len(chunks))]

            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return True

        except Exception as e:
            raise ValueError(f"Failed to store chunks: {e}")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Dict[str, Any]:
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            return result

        except Exception as e:
            raise ValueError(f"Query failed: {e}")

    def count(self) -> int:
        return self.collection.count()

    def delete_all(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)




