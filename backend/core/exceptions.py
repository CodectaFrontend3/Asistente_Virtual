"""
Custom Exceptions para el Backend Principal.

Jerarquía de excepciones personalizada para:
- Mejor manejo de errores
- Mensajes claros para usuarios
- Códigos de error HTTP apropiados
- Facilitar debugging

Todas las excepciones derivan de BackendBaseException.
"""
from typing import Any, Dict, Optional


# ===================================
# BASE EXCEPTION
# ===================================

class BackendBaseException(Exception):
    """
    Excepción base para todas las excepciones del backend.
    
    Attributes:
        message: Mensaje descriptivo del error
        status_code: Código HTTP asociado
        error_code: Código interno de error
        details: Información adicional (opcional)
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la excepción a diccionario para JSON response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details
        }
    
    def __str__(self) -> str:
        return f"{self.error_code}: {self.message}"


# ===================================
# VALIDATION ERRORS (400)
# ===================================

class ValidationError(BackendBaseException):
    """Error de validación de datos de entrada."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )


class InvalidQueryError(ValidationError):
    """Query/pregunta inválida o vacía."""
    
    def __init__(self, message: str = "Query inválida o vacía"):
        super().__init__(message, details={"field": "query"})


class InvalidParametersError(ValidationError):
    """Parámetros de request inválidos."""
    
    def __init__(self, message: str, invalid_params: Optional[Dict] = None):
        super().__init__(
            message,
            details={"invalid_parameters": invalid_params or {}}
        )


# ===================================
# SERVICE ERRORS (502, 503, 504)
# ===================================

class ServiceError(BackendBaseException):
    """Error relacionado con servicios externos."""
    
    def __init__(self, message: str, service_name: str, status_code: int = 502):
        super().__init__(
            message=message,
            status_code=status_code,
            error_code="SERVICE_ERROR",
            details={"service": service_name}
        )


class EmbeddingServiceError(ServiceError):
    """Error al comunicarse con Embedding Service."""
    
    def __init__(self, message: str = "Error en Embedding Service"):
        super().__init__(
            message=message,
            service_name="embedding_service",
            status_code=502
        )


class OllamaServiceError(ServiceError):
    """Error al comunicarse con Ollama/LLM."""
    
    def __init__(self, message: str = "Error en Ollama LLM"):
        super().__init__(
            message=message,
            service_name="ollama",
            status_code=502
        )


class RedisServiceError(ServiceError):
    """Error al comunicarse con Redis."""
    
    def __init__(self, message: str = "Error en Redis"):
        super().__init__(
            message=message,
            service_name="redis",
            status_code=502
        )


class ChromaDBServiceError(ServiceError):
    """Error al comunicarse con ChromaDB."""
    
    def __init__(self, message: str = "Error en ChromaDB"):
        super().__init__(
            message=message,
            service_name="chromadb",
            status_code=502
        )


# ===================================
# CIRCUIT BREAKER ERRORS (503)
# ===================================

class CircuitBreakerOpenError(BackendBaseException):
    """Circuit breaker está abierto - servicio temporalmente no disponible."""
    
    def __init__(self, service_name: str):
        super().__init__(
            message=f"Servicio {service_name} temporalmente no disponible (circuit breaker abierto)",
            status_code=503,
            error_code="CIRCUIT_BREAKER_OPEN",
            details={
                "service": service_name,
                "retry_after": "60s"
            }
        )


# ===================================
# TIMEOUT ERRORS (504)
# ===================================

class TimeoutError(BackendBaseException):
    """Timeout al esperar respuesta de servicio."""
    
    def __init__(self, service_name: str, timeout_seconds: int):
        super().__init__(
            message=f"Timeout esperando respuesta de {service_name}",
            status_code=504,
            error_code="TIMEOUT_ERROR",
            details={
                "service": service_name,
                "timeout_seconds": timeout_seconds
            }
        )


# ===================================
# SEARCH/RAG ERRORS (500, 404)
# ===================================

class SearchError(BackendBaseException):
    """Error durante búsqueda híbrida."""
    
    def __init__(self, message: str, search_type: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="SEARCH_ERROR",
            details={"search_type": search_type} if search_type else {}
        )


class NoResultsFoundError(BackendBaseException):
    """No se encontraron resultados relevantes."""
    
    def __init__(self, query: str):
        super().__init__(
            message="No se encontraron resultados relevantes para la consulta",
            status_code=404,
            error_code="NO_RESULTS_FOUND",
            details={"query": query}
        )


class RAGPipelineError(BackendBaseException):
    """Error en el pipeline RAG completo."""
    
    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="RAG_PIPELINE_ERROR",
            details={"stage": stage} if stage else {}
        )


# ===================================
# CACHE ERRORS (500)
# ===================================

class CacheError(BackendBaseException):
    """Error relacionado con cache (Redis o local)."""
    
    def __init__(self, message: str, cache_type: str = "unknown"):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CACHE_ERROR",
            details={"cache_type": cache_type}
        )


# ===================================
# INDEX ERRORS (500)
# ===================================

class IndexError(BackendBaseException):
    """Error relacionado con índices (FAISS, ChromaDB, BM25)."""
    
    def __init__(self, message: str, index_type: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="INDEX_ERROR",
            details={"index_type": index_type} if index_type else {}
        )


class IndexNotFoundError(IndexError):
    """Índice no encontrado o no inicializado."""
    
    def __init__(self, index_type: str):
        super().__init__(
            message=f"Índice {index_type} no encontrado o no inicializado",
            index_type=index_type
        )


class IndexBuildError(IndexError):
    """Error al construir o actualizar índice."""
    
    def __init__(self, message: str, index_type: str):
        super().__init__(
            message=f"Error al construir índice {index_type}: {message}",
            index_type=index_type
        )


# ===================================
# CONFIGURATION ERRORS (500)
# ===================================

class ConfigurationError(BackendBaseException):
    """Error de configuración del backend."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key} if config_key else {}
        )


# ===================================
# BUSINESS EXCEPTIONS (NO CUENTAN COMO FALLOS EN CIRCUIT BREAKER)
# ===================================
# Estas excepciones se usan en exclude= del circuit breaker
# porque NO indican problemas de sistema, sino lógica de negocio.

class BusinessException(BackendBaseException):
    """
    Excepción base para errores de lógica de negocio.
    
    NO deben contar como fallos en circuit breaker.
    """
    pass


class InsufficientContextError(BusinessException):
    """No hay suficiente contexto para responder la pregunta."""
    
    def __init__(self, query: str):
        super().__init__(
            message="No hay suficiente información para responder esta pregunta",
            status_code=200,  # 200 porque es respuesta válida
            error_code="INSUFFICIENT_CONTEXT",
            details={"query": query}
        )


class UnsupportedQueryTypeError(BusinessException):
    """Tipo de consulta no soportado."""
    
    def __init__(self, query_type: str):
        super().__init__(
            message=f"Tipo de consulta '{query_type}' no soportado",
            status_code=400,
            error_code="UNSUPPORTED_QUERY_TYPE",
            details={"query_type": query_type}
        )


# ===================================
# LISTA DE EXCEPCIONES DE NEGOCIO
# ===================================
# Para usar en circuit breaker exclude=

BUSINESS_EXCEPTIONS = [
    BusinessException,
    InsufficientContextError,
    UnsupportedQueryTypeError,
    ValidationError,
    InvalidQueryError,
    InvalidParametersError,
    NoResultsFoundError,
]