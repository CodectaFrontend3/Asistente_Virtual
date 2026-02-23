"""
Módulo Cache: Redis async para respuestas Q&A.

Exports principales:
    - RedisCache: Cliente Redis async con TTL y circuit breaker
    - get_redis_cache: Singleton del cliente
    - setup_cache: Inicialización en startup de FastAPI

Usage:
    ```python
    from services.cache import get_redis_cache, setup_cache

    # En lifespan de FastAPI (Parte 8):
    await setup_cache()

    # En QAService (Parte 8):
    cache = get_redis_cache()

    # Verificar cache antes de procesar
    cached = await cache.get_qa_response(query)
    if cached:
        return QAResponse(**cached, from_cache=True)

    # Guardar resultado después de procesar
    await cache.set_qa_response(query, response.model_dump())
    ```
"""
import logging

from .redis_client import RedisCache, get_redis_cache

logger = logging.getLogger(__name__)


async def setup_cache() -> dict:
    """
    Inicializa Redis en startup de FastAPI.

    Se llama desde el lifespan de main.py (Parte 8).
    Si Redis no está disponible, el sistema continúa en degraded mode.

    Returns:
        Dict con status de la conexión:
        {
            "status": "healthy" | "unhealthy" | "disconnected",
            "connected": bool,
            "degraded_mode": bool,
        }
    """
    cache = get_redis_cache()
    connected = await cache.connect()

    if connected:
        health = await cache.health_check()
        logger.info(f"✅ Redis cache activo: {health}")
        return {**health, "degraded_mode": False}
    else:
        logger.warning(
            "⚠️ Redis no disponible → degraded mode "
            "(sistema funciona sin cache)"
        )
        return {
            "status": "disconnected",
            "connected": False,
            "degraded_mode": True,
        }


__all__ = [
    "RedisCache",
    "get_redis_cache",
    "setup_cache",
]