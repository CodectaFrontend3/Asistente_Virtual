"""
Cache LRU de embeddings para el Embedding Service.

Almacena en memoria los embeddings generados recientemente para
evitar re-computarlos cuando el mismo texto se consulta múltiples veces.

Implementado con cachetools 5.5.0 LRUCache con thread-safety.
Cada embedding de 384 dims ocupa ~1.5KB en RAM.
1000 embeddings ≈ 1.5MB de RAM.
"""
import hashlib
import logging
import threading
from typing import Dict, Optional

import numpy as np
from cachetools import LRUCache


logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """
    Cache LRU thread-safe para embeddings.

    Usa texto como clave (hasheado) y numpy array como valor.
    Thread-safe con threading.Lock para acceso concurrente seguro.

    Example:
        ```python
        from services.embeddings_cache import EmbeddingsCache

        cache = EmbeddingsCache(max_size=1000)

        # Guardar embedding
        embedding = np.array([0.1, 0.2, ...])
        cache.set("¿Cuánto tarda el envío?", embedding)

        # Recuperar embedding
        cached = cache.get("¿Cuánto tarda el envío?")
        if cached is not None:
            print("Cache hit!")
        ```
    """

    def __init__(self, max_size: int = 1000):
        """
        Inicializa el cache LRU.

        Args:
            max_size: Máximo de embeddings a almacenar.
                      Cuando se llena, elimina el menos usado recientemente.
        """
        self.max_size = max_size
        self._cache: LRUCache = LRUCache(maxsize=max_size)
        self._lock = threading.Lock()

        # Contadores para métricas
        self._hits = 0
        self._misses = 0

        logger.info(
            f"📦 EmbeddingsCache inicializado | "
            f"max_size={max_size} | "
            f"RAM estimada: ~{max_size * 1.5 / 1024:.1f} MB"
        )

    def _make_key(self, text: str, normalize: bool = True) -> str:
        """
        Genera clave única para un texto.

        Usa SHA256 para evitar colisiones y manejar
        textos largos de forma eficiente.

        Args:
            text: Texto a hashear
            normalize: Si el embedding fue normalizado

        Returns:
            Hash SHA256 del texto + flag normalize
        """
        # Incluir normalize en la clave para diferenciar
        key_input = f"{text.strip()}|normalize={normalize}"
        return hashlib.sha256(key_input.encode("utf-8")).hexdigest()

    def get(
        self,
        text: str,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Recupera embedding del cache.

        Args:
            text: Texto cuyo embedding se busca
            normalize: Si el embedding fue normalizado

        Returns:
            Embedding como numpy array, o None si no está en cache

        Example:
            ```python
            cached = cache.get("¿Cuánto tarda el envío?")
            if cached is not None:
                # Cache hit - usar directamente
                return cached
            else:
                # Cache miss - generar embedding
                embedding = model.encode(text)
                cache.set(text, embedding)
                return embedding
            ```
        """
        key = self._make_key(text, normalize)

        with self._lock:
            embedding = self._cache.get(key)

            if embedding is not None:
                self._hits += 1
                logger.debug(
                    f"🎯 Cache HIT | "
                    f"hits={self._hits}, misses={self._misses}"
                )
                return embedding
            else:
                self._misses += 1
                logger.debug(
                    f"❌ Cache MISS | "
                    f"hits={self._hits}, misses={self._misses}"
                )
                return None

    def set(
        self,
        text: str,
        embedding: np.ndarray,
        normalize: bool = True
    ) -> None:
        """
        Almacena embedding en el cache.

        Args:
            text: Texto del embedding
            embedding: Embedding como numpy array
            normalize: Si el embedding fue normalizado

        Example:
            ```python
            embedding = model.encode("Hola mundo")
            cache.set("Hola mundo", embedding)
            ```
        """
        key = self._make_key(text, normalize)

        with self._lock:
            self._cache[key] = embedding
            logger.debug(
                f"💾 Cache SET | "
                f"size={len(self._cache)}/{self.max_size}"
            )

    def get_or_none(
        self,
        text: str,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """Alias de get() para mayor claridad semántica."""
        return self.get(text, normalize)

    def clear(self) -> None:
        """
        Vacía el cache completamente.

        Resetea también los contadores de hits/misses.
        """
        with self._lock:
            size_before = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0

        logger.info(
            f"🧹 Cache limpiado | "
            f"{size_before} embeddings eliminados"
        )

    def get_stats(self) -> Dict[str, any]:
        """
        Retorna estadísticas del cache.

        Returns:
            Dict con métricas del cache

        Example:
            ```python
            stats = cache.get_stats()
            print(f"Hit rate: {stats['hit_rate_pct']}%")
            print(f"Embeddings en cache: {stats['current_size']}")
            ```
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (
                round(self._hits / total_requests * 100, 1)
                if total_requests > 0
                else 0.0
            )

            return {
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "hit_rate_pct": hit_rate,
                "ram_used_mb": round(len(self._cache) * 1.5 / 1024, 2),
            }

    def __len__(self) -> int:
        """Retorna número de embeddings en cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, text: str) -> bool:
        """Verifica si un texto está en cache."""
        key = self._make_key(text)
        with self._lock:
            return key in self._cache


# ===================================
# SINGLETON GLOBAL
# ===================================

_cache_instance: Optional[EmbeddingsCache] = None


def get_embeddings_cache(max_size: Optional[int] = None) -> EmbeddingsCache:
    """
    Obtiene instancia única del cache de embeddings.

    Args:
        max_size: Tamaño máximo del cache (None = usar settings)

    Returns:
        Instancia singleton de EmbeddingsCache

    Example:
        ```python
        from services.embeddings_cache import get_embeddings_cache

        cache = get_embeddings_cache()
        cached_emb = cache.get("¿Cuánto tarda el envío?")
        ```
    """
    global _cache_instance

    if _cache_instance is None:
        from config import settings
        size = max_size or settings.CACHE_MAX_SIZE
        _cache_instance = EmbeddingsCache(max_size=size)

    return _cache_instance