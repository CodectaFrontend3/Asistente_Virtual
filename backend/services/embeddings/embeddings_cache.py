"""
Cache local de embeddings para el Backend.

Almacena embeddings obtenidos del Embedding Service
para evitar requests HTTP repetidas.

Usa cachetools 7.0.1 LRUCache thread-safe.
"""
import hashlib
import logging
import threading
from typing import Dict, List, Optional

from cachetools import LRUCache

from config import settings


logger = logging.getLogger(__name__)


class BackendEmbeddingsCache:
    """
    Cache LRU thread-safe para embeddings en el backend.
    
    Diferencias con el cache del Embedding Service:
    - Este cache está en el backend (evita HTTP calls)
    - Más pequeño (500 vs 1000) porque hay menos memoria disponible
    - TTL implícito via LRU (no hay TTL explícito)
    
    Thread-safe para uso con FastAPI async/Celery workers.
    """
    
    def __init__(self, max_size: int = 500):
        """
        Inicializa el cache LRU.
        
        Args:
            max_size: Máximo de embeddings a almacenar
        """
        self.max_size = max_size
        self._cache: LRUCache = LRUCache(maxsize=max_size)
        self._lock = threading.Lock()
        
        # Contadores para métricas
        self._hits = 0
        self._misses = 0
        
        logger.info(
            f"📦 BackendEmbeddingsCache inicializado | "
            f"max_size={max_size}"
        )
    
    def _make_key(self, text: str, normalize: bool = True) -> str:
        """
        Genera clave única para un texto.
        
        Args:
            text: Texto a hashear
            normalize: Si el embedding fue normalizado
        
        Returns:
            Hash SHA256 del texto
        """
        key_input = f"{text.strip()}|normalize={normalize}"
        return hashlib.sha256(key_input.encode("utf-8")).hexdigest()
    
    def get(
        self,
        text: str,
        normalize: bool = True
    ) -> Optional[List[float]]:
        """
        Recupera embedding del cache.
        
        Args:
            text: Texto cuyo embedding se busca
            normalize: Si el embedding fue normalizado
        
        Returns:
            Embedding como lista de floats, o None si no está
        """
        key = self._make_key(text, normalize)
        
        with self._lock:
            embedding = self._cache.get(key)
            
            if embedding is not None:
                self._hits += 1
                logger.debug(f"🎯 Cache HIT - {self._hits} hits total")
                return embedding
            else:
                self._misses += 1
                logger.debug(f"❌ Cache MISS - {self._misses} misses total")
                return None
    
    def set(
        self,
        text: str,
        embedding: List[float],
        normalize: bool = True
    ) -> None:
        """
        Almacena embedding en el cache.
        
        Args:
            text: Texto del embedding
            embedding: Embedding como lista de floats
            normalize: Si el embedding fue normalizado
        """
        key = self._make_key(text, normalize)
        
        with self._lock:
            self._cache[key] = embedding
            logger.debug(
                f"💾 Cache SET - size={len(self._cache)}/{self.max_size}"
            )
    
    def get_many(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> Dict[str, Optional[List[float]]]:
        """
        Recupera múltiples embeddings del cache.
        
        Args:
            texts: Lista de textos
            normalize: Si los embeddings fueron normalizados
        
        Returns:
            Dict {texto: embedding} donde embedding es None si no está en cache
        
        Example:
            ```python
            cache = BackendEmbeddingsCache()
            results = cache.get_many([
                "texto 1",
                "texto 2",
                "texto 3"
            ])
            
            # results = {
            #     "texto 1": [0.1, 0.2, ...],  # hit
            #     "texto 2": None,              # miss
            #     "texto 3": [0.3, 0.4, ...]   # hit
            # }
            
            # Identificar misses
            missing_texts = [t for t, emb in results.items() if emb is None]
            ```
        """
        results = {}
        for text in texts:
            results[text] = self.get(text, normalize)
        return results
    
    def set_many(
        self,
        text_embedding_pairs: List[tuple[str, List[float]]],
        normalize: bool = True
    ) -> None:
        """
        Almacena múltiples embeddings en el cache.
        
        Args:
            text_embedding_pairs: Lista de tuplas (texto, embedding)
            normalize: Si los embeddings fueron normalizados
        
        Example:
            ```python
            cache = BackendEmbeddingsCache()
            cache.set_many([
                ("texto 1", [0.1, 0.2, ...]),
                ("texto 2", [0.3, 0.4, ...])
            ])
            ```
        """
        for text, embedding in text_embedding_pairs:
            self.set(text, embedding, normalize)
    
    def clear(self) -> None:
        """Vacía el cache completamente."""
        with self._lock:
            size_before = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
        
        logger.info(f"🧹 Cache limpiado - {size_before} embeddings eliminados")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Retorna estadísticas del cache.
        
        Returns:
            Dict con métricas
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
            }
    
    def __len__(self) -> int:
        """Retorna número de embeddings en cache."""
        with self._lock:
            return len(self._cache)


# ===================================
# SINGLETON GLOBAL
# ===================================

_cache_instance: Optional[BackendEmbeddingsCache] = None


def get_embeddings_cache() -> BackendEmbeddingsCache:
    """
    Obtiene instancia singleton del cache de embeddings.
    
    Returns:
        BackendEmbeddingsCache
    
    Example:
        ```python
        from services.embeddings import get_embeddings_cache
        
        cache = get_embeddings_cache()
        
        # Intentar obtener del cache
        cached_emb = cache.get("¿Cuánto tarda el envío?")
        if cached_emb is None:
            # Cache miss - obtener del servicio
            emb = await client.generate_embedding("...")
            cache.set("...", emb)
        ```
    """
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = BackendEmbeddingsCache(
            max_size=settings.CACHE_MAX_SIZE
        )
    
    return _cache_instance