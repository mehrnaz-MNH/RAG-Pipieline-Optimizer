# Purpose: Orchestrate the full RAG pipeline (retrieval + generation)
# Input: User query + pipeline configuration
# Output: Generated answer from LLM with retrieved context

# =============================================================================
# WHAT THIS FILE DOES IN THE RAG PIPELINE:
# =============================================================================
# This is the ORCHESTRATOR - it combines everything:
#
# User asks: "How many PTO days do I get?"
#     ↓
# [rag_pipeline.py] ← YOU ARE HERE
#     ↓
# 1. Embeds the query (using embedding.py)
# 2. Retrieves similar chunks (using vector_store.py)
# 3. Builds a prompt with context
# 4. Calls Ollama LLM for generation
# 5. Returns the answer
# =============================================================================

# =============================================================================
# THE RAG FLOW (Retrieval-Augmented Generation):
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Query: "How many PTO days do I get?"                                   │
# └─────────────────────────────┬───────────────────────────────────────────┘
#                               ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  1. EMBED QUERY                                                         │
# │     embedding.embed_query("How many PTO days...")                       │
# │     → [0.23, -0.45, 0.12, ...]                                         │
# └─────────────────────────────┬───────────────────────────────────────────┘
#                               ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  2. RETRIEVE SIMILAR CHUNKS                                             │
# │     vector_store.query(query_embedding, n_results=3)                   │
# │     → ["Employees are entitled to 15 days PTO...",                     │
# │        "PTO can be carried over to next year...",                      │
# │        "Request PTO through the HR portal..."]                         │
# └─────────────────────────────┬───────────────────────────────────────────┘
#                               ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  3. BUILD PROMPT WITH CONTEXT                                           │
# │     "Based on the following context, answer the question.              │
# │      Context: {retrieved_chunks}                                       │
# │      Question: How many PTO days do I get?                             │
# │      Answer:"                                                          │
# └─────────────────────────────┬───────────────────────────────────────────┘
#                               ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  4. GENERATE ANSWER (Ollama LLM)                                        │
# │     ollama.chat(model="llama3.2", messages=[...])                      │
# │     → "You are entitled to 15 days of PTO per year."                   │
# └─────────────────────────────────────────────────────────────────────────┘
#
# =============================================================================

# =============================================================================
# OLLAMA FOR GENERATION (LLM):
# =============================================================================
# You'll use Ollama to run LLMs locally for generating answers.
#
# Available models (run `ollama list`):
#   - llama3.2 (3B)  - Fast, good for simple Q&A
#   - llama3.1 (8B)  - Better quality, slower
#   - mistral (7B)   - Good balance
#
# Using ollama Python library:
#
#   import ollama
#
#   response = ollama.chat(
#       model="llama3.2",
#       messages=[
#           {"role": "system", "content": "You are a helpful assistant..."},
#           {"role": "user", "content": "Based on context: ... Question: ..."}
#       ]
#   )
#   answer = response["message"]["content"]
#
# Or using LangChain:
#
#   from langchain_community.llms import Ollama
#
#   llm = Ollama(model="llama3.2")
#   answer = llm.invoke(prompt)
# =============================================================================

# =============================================================================
# PROMPT TEMPLATE:
# =============================================================================
# A good RAG prompt tells the LLM:
#   1. Its role
#   2. The context to use
#   3. How to behave if context doesn't have the answer
#
# RAG_PROMPT_TEMPLATE = """
# You are a helpful assistant answering questions based on company documents.
#
# Use ONLY the following context to answer the question.
# If the context doesn't contain the answer, say "I don't have enough information."
#
# Context:
# {context}
#
# Question: {question}
#
# Answer:
# """
# =============================================================================

# =============================================================================
# CLASS DESIGN (following your pattern):
# =============================================================================
# @dataclass
# class PipelineConfig:
#     name: str                    # e.g., "pipeline_medium_nomic"
#     chunk_strategy: str          # e.g., "medium"
#     embedding_model: str         # e.g., "nomic"
#     llm_model: str = "llama3.2"  # Generation model
#     n_results: int = 5           # Number of chunks to retrieve
#
#
# class RAGPipeline:
#     def __init__(self, config: PipelineConfig) -> None:
#         # Initialize embedding, vector_store based on config
#
#     def index_document(self, text: str) -> None:
#         # Process document: chunk → embed → store
#         # Called once when user uploads a document
#
#     def query(self, question: str) -> str:
#         # Full RAG flow: embed query → retrieve → generate
#         # Called each time user asks a question
#
#     def _build_prompt(self, context: List[str], question: str) -> str:
#         # Combine retrieved chunks + question into prompt
#
#     def _generate(self, prompt: str) -> str:
#         # Call Ollama LLM to generate answer
# =============================================================================

# =============================================================================
# PIPELINE CONFIGURATIONS FOR YOUR PROJECT:
# =============================================================================
# Your project runs multiple pipelines to compare performance:
#
# PIPELINE_CONFIGS = [
#     PipelineConfig(
#         name="small_nomic",
#         chunk_strategy="small",
#         embedding_model="nomic"
#     ),
#     PipelineConfig(
#         name="medium_nomic",
#         chunk_strategy="medium",
#         embedding_model="nomic"
#     ),
#     PipelineConfig(
#         name="large_nomic",
#         chunk_strategy="large",
#         embedding_model="nomic"
#     ),
#     PipelineConfig(
#         name="medium_mxbai",
#         chunk_strategy="medium",
#         embedding_model="mxbai"
#     ),
# ]
#
# Each pipeline will:
#   1. Process the same document differently
#   2. Answer the same questions
#   3. Get scored by evaluator.py
# =============================================================================

# =============================================================================
# EXAMPLE USAGE (after implementation):
# =============================================================================
# from services.rag_pipeline import RAGPipeline, PipelineConfig
#
# # Create a pipeline
# config = PipelineConfig(
#     name="medium_nomic",
#     chunk_strategy="medium",
#     embedding_model="nomic"
# )
# pipeline = RAGPipeline(config)
#
# # Index a document (one time)
# document_text = "Chapter 1: Leave Policy. Employees get 15 days PTO..."
# pipeline.index_document(document_text)
#
# # Query the pipeline (many times)
# answer = pipeline.query("How many PTO days do I get?")
# print(answer)  # "Based on the document, employees get 15 days PTO per year."
# =============================================================================

# =============================================================================
# TRACKING METRICS (for evaluation):
# =============================================================================
# Each query should track:
#   - latency: Time taken for full RAG flow
#   - tokens_used: Approximate token count (for cost estimation)
#   - chunks_retrieved: The actual chunks used as context
#   - answer: The generated answer
#
# @dataclass
# class QueryResult:
#     answer: str
#     chunks_retrieved: List[str]
#     latency_ms: float
#     tokens_used: int
#
# These metrics go to evaluator.py for scoring and comparison.
# =============================================================================

# =============================================================================
# ERROR HANDLING TO CONSIDER:
# =============================================================================
# - Ollama not running
# - Model not installed
# - Vector store empty (no documents indexed)
# - No relevant chunks found
# - LLM timeout / rate limit
# =============================================================================


class RAGPipeline :
    def __init__(self):
        pass
