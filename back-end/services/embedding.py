# Purpose: Convert text chunks to vector embeddings using Ollama
# Input: List of text chunks from chunking.py
# Output: List of vector embeddings (arrays of floats)

# =============================================================================
# WHAT THIS FILE DOES IN THE RAG PIPELINE:
# =============================================================================
# Text chunks from chunking.py
#     ↓
# [embedding.py] ← YOU ARE HERE
#     ↓
# Converts to vectors: "What is PTO?" → [0.23, -0.45, 0.12, ...] (768-1536 dims)
#     ↓
# Passes vectors to vector_store.py for storage/retrieval
# =============================================================================

# =============================================================================
# WHAT ARE EMBEDDINGS? (Key concept for RAG)
# =============================================================================
# Embeddings are numerical representations of text that capture MEANING.
#
# Similar meanings → similar vectors:
#   "What is parental leave?"  → [0.82, 0.15, -0.33, ...]
#   "Tell me about maternity leave" → [0.79, 0.18, -0.31, ...]  (very similar!)
#   "What's the weather today?" → [-0.12, 0.91, 0.44, ...]  (very different!)
#
# This is how RAG finds relevant chunks:
#   1. Embed the user's question
#   2. Find chunks with similar embeddings (cosine similarity)
#   3. Return those chunks as context for the LLM
# =============================================================================

# =============================================================================
# OLLAMA EMBEDDING MODELS (What you'll test):
# =============================================================================
# Ollama runs embedding models locally (free, no API costs!)
#
# Available models (run `ollama list` to see installed):
#
# | Model              | Dimensions | Size  | Quality | Speed |
# |--------------------|------------|-------|---------|-------|
# | nomic-embed-text   | 768        | 274MB | Good    | Fast  |
# | mxbai-embed-large  | 1024       | 670MB | Better  | Medium|
# | bge-large-en-v1.5  | 1024       | 1.3GB | Better  | Slow  |
# | all-minilm         | 384        | 46MB  | Basic   | Fastest|
#
# To install a model:
#   ollama pull nomic-embed-text
#   ollama pull mxbai-embed-large
#
# Your project will compare these to find which works best for user's data!
# =============================================================================

# =============================================================================
# HOW TO USE OLLAMA FOR EMBEDDINGS:
# =============================================================================
# Option 1: Using ollama Python library (simple)
#
#   import ollama
#
#   response = ollama.embeddings(
#       model="nomic-embed-text",
#       prompt="What is parental leave?"
#   )
#   vector = response["embedding"]  # List[float] with 768 dimensions
#
# -----------------------------------------------------------------------------
# Option 2: Using LangChain's OllamaEmbeddings (integrates with LangChain)
#
#   from langchain_community.embeddings import OllamaEmbeddings
#
#   embeddings = OllamaEmbeddings(model="nomic-embed-text")
#
#   # Single text
#   vector = embeddings.embed_query("What is parental leave?")
#
#   # Multiple texts (batch - more efficient)
#   vectors = embeddings.embed_documents(["chunk 1", "chunk 2", "chunk 3"])
#
# RECOMMENDATION: Use Option 2 (LangChain) because:
#   - embed_documents() batches requests efficiently
#   - Integrates seamlessly with ChromaDB in vector_store.py
# =============================================================================

# =============================================================================
# FUNCTIONS TO IMPLEMENT:
# =============================================================================
# 1. get_embedding_models() -> List[dict]
#    - Returns list of embedding models to test
#    - Example: [
#        {"name": "nomic", "model_id": "nomic-embed-text", "dimensions": 768},
#        {"name": "mxbai", "model_id": "mxbai-embed-large", "dimensions": 1024},
#      ]
#
# 2. create_embeddings(model_name: str) -> OllamaEmbeddings
#    - Creates and returns an OllamaEmbeddings instance for the given model
#    - Lookup model_id from get_embedding_models()
#
# 3. embed_chunks(chunks: List[str], model_name: str) -> List[List[float]]
#    - Takes list of text chunks
#    - Creates embeddings using specified model
#    - Returns list of vectors (each vector is List[float])
#
# 4. embed_query(query: str, model_name: str) -> List[float]
#    - Embeds a single query/question
#    - Used at query time to find similar chunks
# =============================================================================

# =============================================================================
# EXAMPLE USAGE (after implementation):
# =============================================================================
# from services.chunking import chunk_text
# from services.embedding import embed_chunks, embed_query, get_embedding_models
#
# # Embed document chunks
# chunks = ["Chapter 1: Leave Policy...", "Employees get 15 days PTO..."]
# vectors = embed_chunks(chunks, model_name="nomic")
# print(f"Created {len(vectors)} embeddings of {len(vectors[0])} dimensions")
#
# # Embed a query (for retrieval)
# query_vector = embed_query("How many PTO days do I get?", model_name="nomic")
#
# # Test different models (for your pipeline comparison)
# for model in get_embedding_models():
#     vectors = embed_chunks(chunks, model["name"])
#     print(f"{model['name']}: {len(vectors[0])} dimensions")
# =============================================================================

# =============================================================================
# IMPORTANT: SAME MODEL FOR INDEXING AND QUERYING
# =============================================================================
# You MUST use the same embedding model for:
#   1. Embedding chunks (when storing in vector DB)
#   2. Embedding queries (when searching)
#
# Why? Different models produce different vector spaces.
# nomic's vector for "PTO" won't match bge's vector for "PTO"!
#
# This is why each pipeline in your project uses ONE embedding model throughout.
# =============================================================================

# =============================================================================
# ERROR HANDLING TO CONSIDER:
# =============================================================================
# - Ollama not running (connection refused)
# - Model not installed (404 error)
# - Empty text input
# - Rate limiting / timeouts on large batches
# =============================================================================

