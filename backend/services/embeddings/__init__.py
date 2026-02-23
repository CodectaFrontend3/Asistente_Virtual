"""
Módulo de embeddings del Backend.

Incluye:
- Cliente HTTP para Embedding Service
- Cache local LRU
- Cliente integrado con cache automático
"""
from .http_client import (
    EmbeddingHTTPClient,
    get_embedding_client,
    get_http_client,
    close_http_client,
)

from .embeddings_cache import (
    BackendEmbeddingsCache,
    get_embeddings_cache,
)


__all__ = [
    # HTTP Client
    "EmbeddingHTTPClient",
    "get_embedding_client",
    "get_http_client",
    "close_http_client",
    
    # Cache
    "BackendEmbeddingsCache",
    "get_embeddings_cache",
]