"""
Circuit Breaker para resiliencia del Backend.

Implementa el patrón Circuit Breaker usando pybreaker 1.2.0 para:
- Prevenir cascading failures
- Detectar servicios caídos automáticamente
- Permitir recuperación gradual (half-open state)
- Logging de cambios de estado
- Storage en Redis opcional (compartido entre workers)

Basado en: Michael T. Nygard's "Release It!" pattern
"""
import logging
from typing import Callable, List, Optional, Type

import pybreaker
from redis import Redis

from config import settings


logger = logging.getLogger(__name__)


# ===================================
# LISTENER PARA LOGGING
# ===================================

class CircuitBreakerLogListener(pybreaker.CircuitBreakerListener):
    """
    Listener que registra todos los eventos del circuit breaker.
    
    Útil para debugging y monitoreo en desarrollo.
    En producción se reemplaza con metrics (Prometheus, Datadog).
    """
    
    def before_call(self, cb: pybreaker.CircuitBreaker, func: Callable, *args, **kwargs) -> None:
        """Se ejecuta antes de cada llamada protegida."""
        logger.debug(
            f"🔵 CircuitBreaker [{cb.name}] - Llamando a {func.__name__}"
        )
    
    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state: pybreaker.CircuitBreakerState,
        new_state: pybreaker.CircuitBreakerState
    ) -> None:
        """Se ejecuta cuando cambia el estado del circuit breaker."""
        emoji_map = {
            "closed": "🟢",
            "open": "🔴",
            "half-open": "🟡"
        }
        
        old_emoji = emoji_map.get(old_state.name.lower(), "⚪")
        new_emoji = emoji_map.get(new_state.name.lower(), "⚪")
        
        logger.warning(
            f"{old_emoji} → {new_emoji} CircuitBreaker [{cb.name}] - "
            f"Estado cambió: {old_state.name} → {new_state.name}"
        )
    
    def failure(self, cb: pybreaker.CircuitBreaker, exc: Exception) -> None:
        """Se ejecuta cuando una llamada falla."""
        logger.error(
            f"❌ CircuitBreaker [{cb.name}] - Fallo detectado: {type(exc).__name__}: {exc}"
        )
    
    def success(self, cb: pybreaker.CircuitBreaker) -> None:
        """Se ejecuta cuando una llamada tiene éxito."""
        logger.debug(
            f"✅ CircuitBreaker [{cb.name}] - Llamada exitosa"
        )


# ===================================
# FACTORY PARA CIRCUIT BREAKERS
# ===================================

def create_circuit_breaker(
    name: str,
    fail_max: Optional[int] = None,
    reset_timeout: Optional[int] = None,
    excluded_exceptions: Optional[List[Type[Exception]]] = None,
    use_redis: bool = False,
    redis_client: Optional[Redis] = None,
) -> pybreaker.CircuitBreaker:
    """
    Crea un circuit breaker configurado.
    
    Args:
        name: Nombre identificador único
        fail_max: Número de fallos antes de abrir (default: settings.CB_FAIL_MAX)
        reset_timeout: Segundos en open antes de half-open (default: settings.CB_RESET_TIMEOUT)
        excluded_exceptions: Excepciones de negocio a ignorar
        use_redis: Si True, comparte estado via Redis entre workers
        redis_client: Cliente Redis (requerido si use_redis=True)
    
    Returns:
        CircuitBreaker configurado
    
    Notes:
        pybreaker 1.2.0 no tiene success_threshold.
        En estado half-open, un solo éxito cierra el circuito.
    
    Example:
        ```python
        # Circuit breaker básico
        cb = create_circuit_breaker("embedding_service")
        
        @cb
        def call_embedding_service():
            return requests.post(...)
        
        # Con Redis (para Celery workers)
        cb = create_circuit_breaker(
            "ollama",
            use_redis=True,
            redis_client=redis_conn
        )
        ```
    """
    # Valores por defecto desde settings
    fail_max = fail_max or settings.CB_FAIL_MAX
    reset_timeout = reset_timeout or settings.CB_RESET_TIMEOUT
    
    # Configurar storage
    state_storage = None
    if use_redis:
        if not redis_client:
            raise ValueError("redis_client es requerido cuando use_redis=True")
        
        # Redis storage con namespace único por circuit breaker
        state_storage = pybreaker.CircuitRedisStorage(
            state=pybreaker.STATE_CLOSED,
            redis_object=redis_client,
            namespace=f"circuit_breaker:{name}"
        )
        logger.info(f"📦 CircuitBreaker [{name}] - Usando Redis storage")
    
    # Crear circuit breaker
    cb = pybreaker.CircuitBreaker(
        name=name,
        fail_max=fail_max,
        reset_timeout=reset_timeout,
        exclude=excluded_exceptions or [],
        state_storage=state_storage,
        listeners=[CircuitBreakerLogListener()],
    )
    
    logger.info(
        f"🔧 CircuitBreaker [{name}] creado - "
        f"fail_max={fail_max}, reset_timeout={reset_timeout}s"
    )
    
    return cb


# ===================================
# CIRCUIT BREAKERS GLOBALES
# ===================================
# Estos se instancian una sola vez y viven durante toda
# la ejecución de la aplicación.

# Circuit breaker para Embedding Service
embedding_service_breaker = create_circuit_breaker(
    name="embedding_service",
    fail_max=5,
    reset_timeout=60,
)

# Circuit breaker para Ollama/LLM
ollama_breaker = create_circuit_breaker(
    name="ollama",
    fail_max=3,  # Más sensible que embedding service
    reset_timeout=90,  # Más tiempo de recuperación
)

# Circuit breaker para Redis
redis_breaker = create_circuit_breaker(
    name="redis",
    fail_max=10,  # Redis suele ser muy estable
    reset_timeout=30,  # Recuperación rápida
)

# Circuit breaker para ChromaDB
chroma_breaker = create_circuit_breaker(
    name="chromadb",
    fail_max=5,
    reset_timeout=60,
)


# ===================================
# HELPERS
# ===================================

def get_breaker_stats(breaker: pybreaker.CircuitBreaker) -> dict:
    """
    Obtiene estadísticas de un circuit breaker.
    
    Args:
        breaker: CircuitBreaker a inspeccionar
    
    Returns:
        Dict con estado actual y contadores
    
    Example:
        ```python
        stats = get_breaker_stats(embedding_service_breaker)
        print(f"Estado: {stats['state']}")
        print(f"Fallos: {stats['fail_counter']}")
        ```
    """
    # current_state puede ser string o objeto dependiendo de la versión
    current_state = breaker.current_state
    state_name = current_state.name if hasattr(current_state, 'name') else str(current_state)
    
    return {
        "name": breaker.name,
        "state": state_name,
        "fail_counter": breaker.fail_counter,
        "fail_max": breaker.fail_max,
        "reset_timeout": breaker.reset_timeout,
        "is_closed": breaker.current_state == pybreaker.STATE_CLOSED,
        "is_open": breaker.current_state == pybreaker.STATE_OPEN,
        "is_half_open": breaker.current_state == pybreaker.STATE_HALF_OPEN,
    }


def reset_breaker(breaker: pybreaker.CircuitBreaker) -> None:
    """
    Resetea manualmente un circuit breaker a estado cerrado.
    
    Útil para forzar recuperación después de mantenimiento.
    
    Args:
        breaker: CircuitBreaker a resetear
    
    Example:
        ```python
        # Después de arreglar embedding service manualmente
        reset_breaker(embedding_service_breaker)
        ```
    """
    breaker._state = pybreaker.STATE_CLOSED
    breaker.fail_counter = 0
    logger.warning(
        f"🔄 CircuitBreaker [{breaker.name}] reseteado manualmente a CLOSED"
    )


def get_all_breakers() -> List[pybreaker.CircuitBreaker]:
    """
    Retorna lista de todos los circuit breakers globales.
    
    Returns:
        Lista de CircuitBreakers
    
    Example:
        ```python
        for breaker in get_all_breakers():
            stats = get_breaker_stats(breaker)
            print(f"{stats['name']}: {stats['state']}")
        ```
    """
    return [
        embedding_service_breaker,
        ollama_breaker,
        redis_breaker,
        chroma_breaker,
    ]