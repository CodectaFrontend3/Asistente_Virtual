"""
Módulo core del Backend Principal.

Contiene:
- Circuit breakers para resiliencia
- Custom exceptions jerárquicas
- Utilities de error handling
"""
from .circuit_breaker import (
    CircuitBreakerLogListener,
    create_circuit_breaker,
    embedding_service_breaker,
    ollama_breaker,
    redis_breaker,
    chroma_breaker,
    get_breaker_stats,
    reset_breaker,
    get_all_breakers,
)

from .exceptions import (
    # Base
    BackendBaseException,
    
    # Validation (400)
    ValidationError,
    InvalidQueryError,
    InvalidParametersError,
    
    # Service (502, 503, 504)
    ServiceError,
    EmbeddingServiceError,
    OllamaServiceError,
    RedisServiceError,
    ChromaDBServiceError,
    
    # Circuit Breaker (503)
    CircuitBreakerOpenError,
    
    # Timeout (504)
    TimeoutError,
    
    # Search/RAG (500, 404)
    SearchError,
    NoResultsFoundError,
    RAGPipelineError,
    
    # Cache (500)
    CacheError,
    
    # Index (500)
    IndexError,
    IndexNotFoundError,
    IndexBuildError,
    
    # Configuration (500)
    ConfigurationError,
    
    # Business (no cuentan como fallos)
    BusinessException,
    InsufficientContextError,
    UnsupportedQueryTypeError,
    
    # Lista para circuit breaker exclude
    BUSINESS_EXCEPTIONS,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreakerLogListener",
    "create_circuit_breaker",
    "embedding_service_breaker",
    "ollama_breaker",
    "redis_breaker",
    "chroma_breaker",
    "get_breaker_stats",
    "reset_breaker",
    "get_all_breakers",
    
    # Exceptions - Base
    "BackendBaseException",
    
    # Exceptions - Validation
    "ValidationError",
    "InvalidQueryError",
    "InvalidParametersError",
    
    # Exceptions - Service
    "ServiceError",
    "EmbeddingServiceError",
    "OllamaServiceError",
    "RedisServiceError",
    "ChromaDBServiceError",
    
    # Exceptions - Circuit Breaker
    "CircuitBreakerOpenError",
    
    # Exceptions - Timeout
    "TimeoutError",
    
    # Exceptions - Search/RAG
    "SearchError",
    "NoResultsFoundError",
    "RAGPipelineError",
    
    # Exceptions - Cache
    "CacheError",
    
    # Exceptions - Index
    "IndexError",
    "IndexNotFoundError",
    "IndexBuildError",
    
    # Exceptions - Configuration
    "ConfigurationError",
    
    # Exceptions - Business
    "BusinessException",
    "InsufficientContextError",
    "UnsupportedQueryTypeError",
    
    # Business exceptions list
    "BUSINESS_EXCEPTIONS",
]