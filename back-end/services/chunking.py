# Purpose: Split text into smaller pieces with different strategies
# Input: Clean text string from document_processor.py
# Output: List of text chunks ready for embedding

# =============================================================================
# WHAT THIS FILE DOES IN THE RAG PIPELINE:
# =============================================================================
# Clean text from document_processor.py
#     ↓
# [chunking.py] ← YOU ARE HERE
#     ↓
# Splits into chunks: ["Chapter 1: Leave Policy...", "...entitled to 15 days..."]
#     ↓
# Passes chunks to embedding.py
# =============================================================================

# =============================================================================
# WHY CHUNKING MATTERS (THE CORE OF YOUR PROJECT):
# =============================================================================
# Your project tests different chunk sizes to find what works best!
#
# Small chunks (256 tokens):
#   ✅ More precise retrieval (finds exact answer)
#   ❌ May lose context (sentence cut in half)
#   ❌ More chunks = more embeddings = slower
#
# Large chunks (1024 tokens):
#   ✅ More context preserved
#   ❌ May retrieve irrelevant info along with relevant
#   ❌ Fewer chunks = might miss specific details
#
# The "best" size depends on:
#   - Document type (legal docs need more context, FAQs need less)
#   - Question type (specific fact vs. summary)
#   - Embedding model capabilities
# =============================================================================

# =============================================================================
# KEY CONCEPT: OVERLAP
# =============================================================================
# Without overlap:
#   Chunk 1: "Employees are entitled to 15 days"
#   Chunk 2: "PTO per year. Unused days carry over."
#   Problem: Question "How many PTO days?" might miss context!
#
# With overlap (e.g., 50 chars):
#   Chunk 1: "Employees are entitled to 15 days PTO per year."
#   Chunk 2: "to 15 days PTO per year. Unused days carry over."
#   Better: Both chunks contain the full answer!
#
# Typical overlap: 10-20% of chunk size
# =============================================================================

# =============================================================================
# LANGCHAIN TEXT SPLITTERS (What you'll use):
# =============================================================================
# LangChain provides ready-made splitters. Main ones:
#
# 1. RecursiveCharacterTextSplitter (RECOMMENDED for most cases)
#    - Tries to split on natural boundaries: "\n\n" → "\n" → " " → ""
#    - Keeps paragraphs/sentences together when possible
#    - Parameters:
#        chunk_size: Max characters per chunk (e.g., 512)
#        chunk_overlap: Characters to repeat between chunks (e.g., 50)
#
#    from langchain.text_splitter import RecursiveCharacterTextSplitter
#
#    splitter = RecursiveCharacterTextSplitter(
#        chunk_size=512,
#        chunk_overlap=50,
#        separators=["\n\n", "\n", ". ", " ", ""]  # Priority order
#    )
#    chunks = splitter.split_text(text)  # Returns List[str]
#
# 2. CharacterTextSplitter (Simple, less smart)
#    - Splits on a single separator only
#    - Use when you have consistent formatting
#
# 3. TokenTextSplitter (For token-based models)
#    - Splits by token count, not character count
#    - More accurate for LLM context limits
# =============================================================================

# =============================================================================
# FUNCTIONS TO IMPLEMENT:
# =============================================================================
# 1. chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]
#    - Create a RecursiveCharacterTextSplitter with given params
#    - Split the text
#    - Return list of chunk strings
#
# 2. get_chunking_strategies() -> List[dict]
#    - Returns predefined strategies for your pipeline comparison
#    - Example: [
#        {"name": "small", "chunk_size": 256, "chunk_overlap": 25},
#        {"name": "medium", "chunk_size": 512, "chunk_overlap": 50},
#        {"name": "large", "chunk_size": 1024, "chunk_overlap": 100},
#      ]
#
# 3. chunk_text_with_strategy(text: str, strategy_name: str) -> List[str]
#    - Convenience function that looks up strategy by name
#    - Calls chunk_text() with the strategy's parameters
# =============================================================================

# =============================================================================
# EXAMPLE USAGE (after implementation):
# =============================================================================
# from services.document_processor import process_document
# from services.chunking import chunk_text, get_chunking_strategies
#
# # Get clean text
# text = process_document("hr_policy.pdf")
#
# # Chunk with specific size
# chunks = chunk_text(text, chunk_size=512, chunk_overlap=50)
# print(f"Created {len(chunks)} chunks")
# print(chunks[0][:100])  # Preview first chunk
#
# # Or use predefined strategies (for your pipeline comparison)
# for strategy in get_chunking_strategies():
#     chunks = chunk_text(text, strategy["chunk_size"], strategy["chunk_overlap"])
#     print(f"{strategy['name']}: {len(chunks)} chunks")
# =============================================================================

# =============================================================================
# WHAT YOUR PROJECT WILL TEST:
# =============================================================================
# Pipeline 1: chunk_size=256,  embed=nomic,  → Score: ?
# Pipeline 2: chunk_size=512,  embed=nomic,  → Score: ?
# Pipeline 3: chunk_size=1024, embed=nomic,  → Score: ?
# Pipeline 4: chunk_size=512,  embed=bge,    → Score: ?
#
# The evaluator.py will compare these and tell the user:
# "For YOUR data, 512 chunks + nomic embeddings scored 20% higher!"
# =============================================================================

from typing import *

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) ->List[str]:
#    - Create a RecursiveCharacterTextSplitter with given params
#    - Split the text
#    - Return list of chunk strings
   pass

def get_chunking_strategies() -> List[dict]:
#    - Returns predefined strategies for your pipeline comparison
#    - Example: [
#        {"name": "small", "chunk_size": 256, "chunk_overlap": 25},
#        {"name": "medium", "chunk_size": 512, "chunk_overlap": 50},
#        {"name": "large", "chunk_size": 1024, "chunk_overlap": 100},
#      ]
   pass

def chunk_text_with_strategy(text: str, strategy_name: str) -> List[str] :
#    - Convenience function that looks up strategy by name
#    - Calls chunk_text() with the strategy's parameters
    pass
