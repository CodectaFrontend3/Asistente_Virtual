"""
Módulo de servicios del Embedding Service.

Contiene:
- cpu_manager: Gestión de recursos de CPU
- embedder: Servicio de generación de embeddings
- embeddings_cache: Cache LRU de embeddings
- models: Helpers y utilidades para modelos
"""
from .cpu_manager import (
    CPUManager,
    get_cpu_manager,
    ResourceStats,
)

from .embedder import (
    EmbeddingService,
    get_embedding_service,
)

from .embeddings_cache import (
    EmbeddingsCache,
    get_embeddings_cache,
)

from .models import (
    get_model_info,
    list_recommended_models,
    validate_model_name,
    download_model,
    verify_model_dimension,
    get_best_model_for_cpu,
    get_best_model_for_quality,
    compare_models,
    RECOMMENDED_MODELS,
)

__all__ = [
    # CPU Manager
    "CPUManager",
    "get_cpu_manager",
    "ResourceStats",

    # Embedding Service
    "EmbeddingService",
    "get_embedding_service",

    # Embeddings Cache
    "EmbeddingsCache",
    "get_embeddings_cache",

    # Model Helpers
    "get_model_info",
    "list_recommended_models",
    "validate_model_name",
    "download_model",
    "verify_model_dimension",
    "get_best_model_for_cpu",
    "get_best_model_for_quality",
    "compare_models",
    "RECOMMENDED_MODELS",
]