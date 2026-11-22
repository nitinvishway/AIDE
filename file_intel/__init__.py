"""
file_intel package entrypoint
"""
from .config import DB_PATH, LLAMA_CLI_PATH, GEMMA_MODEL_PATH
from .embeddings import get_embedder, TextChunk
from .semantic_index import SemanticFileIndex
from .retriever_ranker import HybridFileRetriever, RankedResult
from .file_agent import FileAgent

__all__ = [
    "DB_PATH",
    "LLAMA_CLI_PATH",
    "GEMMA_MODEL_PATH",
    "get_embedder",
    "TextChunk",
    "SemanticFileIndex",
    "HybridFileRetriever",
    "RankedResult",
    "FileAgent",
]
