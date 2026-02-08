from dataclasses import dataclass
from typing import List
from time import time

from chromadb.types import Collection
from langchain_community.llms import Ollama

from .chunking import Chunking
from .embedding import Embedding
from .vector_store import VectorStore


@dataclass
class PipelineConfig:
    name: str
    chunk_strategy: str
    embedding_model: str
    collection: Collection
    llm_model: str = "llama3.2:3b"
    n_results: int = 5

@dataclass
class QueryResult:
    answer : str
    retrieved_chunks : List[str]
    distances : List[float]
    latency_ms : float
    pipeline_name : str



class RAGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.embedding = Embedding(model_name=self.config.embedding_model)
        self.chunking = Chunking()
        self.vector_store = VectorStore(collection=self.config.collection)
        self.llm = Ollama(model=self.config.llm_model)

    def index_document(self, text: str) -> None:
        chunks = self.chunking.chunk_text_with_strategy(
            text=text,
            strategy_name=self.config.chunk_strategy
        )
        embeddings = self.embedding.embed_chunks(chunks=chunks)
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    def query(self, question: str) -> str:
        start_time = time.time()
        query_embedding = self.embedding.embed_query(query=question)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=self.config.n_results
        )

        context_chunks = results["documents"][0] if results["documents"] else []

        prompt = self._build_prompt(context=context_chunks, question=question)
        answer = self._generate(prompt=prompt)
        return QueryResult(
            answer=answer,
            retrieved_chunks=context_chunks,
            distances=results["distances"][0] if results["distances"] else [],
            latency_ms=(time.time() - start_time) * 1000,
            pipeline_name=self.config.name
        )

    def _build_prompt(self, context: List[str], question: str) -> str:
        context_text = "\n\n".join(context)

        prompt = f"""You are a helpful assistant answering questions based on company documents.
        Use ONLY the following context to answer the question.
        If the context doesn't contain the answer, say "I don't have enough information."
        Context:
        {context_text}

        Question: {question}

        Answer:"""
        return prompt

    def _generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response
