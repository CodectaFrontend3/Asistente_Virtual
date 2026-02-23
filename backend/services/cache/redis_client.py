"""
Cliente Redis async con cache semántico de 2 niveles.

NIVEL 1 — Normalización textual:
    "¿Cuál es el horario?" == "cual es el horario" → mismo MD5 → HIT

NIVEL 2 — Cache semántico en memoria:
    "¿a qué hora atienden?" ≈ "cual es el horario" 
    → similitud coseno 0.94 > threshold 0.88 → HIT

El índice semántico vive en RAM (dict) y se reconstruye al reiniciar.
No requiere cambios en Redis — solo guarda respuestas y embeddings.
"""
import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as aioredis

from config import settings
from core import RedisServiceError, redis_breaker
from .query_normalizer import normalize_query

logger = logging.getLogger(__name__)

# Threshold de similitud coseno para cache semántico
# 0.88 = muy similar (misma pregunta con palabras distintas)
# 0.95 = casi idéntica
SEMANTIC_SIMILARITY_THRESHOLD = 0.88


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_cache_key(prefix: str, text: str) -> str:
    """Genera key MD5 desde texto normalizado."""
    normalized = normalize_query(text)  # ← Nivel 1: normalizar antes de hashear
    text_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{text_hash}"


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Similitud coseno entre dos vectores."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _deserialize(raw: str) -> Any:
    return json.loads(raw)


# ── Redis Cache ────────────────────────────────────────────────────────────────

class RedisCache:
    """
    Cache Redis con 2 niveles de matching:

    1. Normalización textual → mismo MD5 para variantes superficiales
    2. Similitud semántica en memoria → mismo significado, distintas palabras

    Índice semántico (en RAM):
        {cache_key: embedding_vector}
        Al hacer GET: si no hay MD5 match → busca el embedding más similar
        Si sim > 0.88 → retorna la respuesta cacheada de esa key
    """

    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None
        self._connected: bool = False
        self._hits: int = 0
        self._misses: int = 0
        self._semantic_hits: int = 0
        self._errors: int = 0
        self._qa_ttl: int = settings.REDIS_CACHE_TTL
        self._search_ttl: int = settings.REDIS_CACHE_TTL // 2

        # Índice semántico en memoria: {cache_key → embedding}
        self._semantic_index: Dict[str, List[float]] = {}

        logger.info(
            f"🔴 RedisCache inicializado "
            f"(qa_ttl={self._qa_ttl}s, search_ttl={self._search_ttl}s, "
            f"semantic_threshold={SEMANTIC_SIMILARITY_THRESHOLD})"
        )

    # ── Conexión ───────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        try:
            self._client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                socket_timeout=settings.REDIS_TIMEOUT,
                retry_on_timeout=False,
                max_connections=20,
            )
            await self._client.ping()
            self._connected = True
            logger.info(f"✅ Redis conectado: {settings.redis_url}")
            return True
        except Exception as e:
            self._connected = False
            logger.warning(f"⚠️ Redis no disponible (degraded mode): {e}")
            return False

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            self._connected = False
            self._semantic_index.clear()
            logger.info("🔴 Redis desconectado")

    # ── Cache Q&A ──────────────────────────────────────────────────────────────

    async def get_qa_response(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[dict]:
        """
        Recupera respuesta Q&A con 2 niveles de matching.

        Nivel 1: MD5 del texto normalizado (rápido, O(1))
        Nivel 2: Similitud coseno con embeddings en memoria (O(n))

        Args:
            query: Pregunta del usuario
            query_embedding: Embedding de la query (para cache semántico)
                             Si es None, solo se usa nivel 1

        Returns:
            Dict con respuesta cacheada, o None si no hay match
        """
        if not self._connected or not self._client:
            return None

        # ── NIVEL 1: MD5 normalizado ──────────────────────────────────
        key = _make_cache_key("qa:response", query)
        try:
            raw = await self._get_with_breaker(key)
            if raw:
                self._hits += 1
                norm = normalize_query(query)
                logger.info(f"🎯 Cache HIT (textual): '{query[:50]}'")
                logger.debug(f"   Normalizado: '{norm}'")
                return _deserialize(raw)
        except Exception as e:
            self._errors += 1
            logger.warning(f"⚠️ Redis get error: {e}")
            return None

        # ── NIVEL 2: Similitud semántica ──────────────────────────────
        if query_embedding and self._semantic_index:
            best_key, best_sim = self._find_semantic_match(query_embedding)

            if best_sim >= SEMANTIC_SIMILARITY_THRESHOLD:
                try:
                    raw = await self._get_with_breaker(best_key)
                    if raw:
                        self._hits += 1
                        self._semantic_hits += 1
                        logger.info(
                            f"🧠 Cache HIT (semántico): '{query[:50]}' "
                            f"→ similitud={best_sim:.3f}"
                        )
                        return _deserialize(raw)
                except Exception as e:
                    logger.warning(f"⚠️ Redis semantic get error: {e}")

        # ── MISS ──────────────────────────────────────────────────────
        self._misses += 1
        logger.debug(f"❌ Cache MISS: '{query[:50]}'")
        return None

    async def set_qa_response(
        self,
        query: str,
        response: dict,
        query_embedding: Optional[List[float]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Guarda respuesta Q&A y registra embedding en índice semántico.

        Args:
            query: Pregunta del usuario
            response: Dict con la respuesta
            query_embedding: Embedding para el índice semántico
            ttl: TTL en segundos
        """
        if not self._connected or not self._client:
            return False

        key = _make_cache_key("qa:response", query)
        effective_ttl = ttl or self._qa_ttl

        try:
            serialized = _serialize(response)
            await self._set_with_breaker(key, serialized, effective_ttl)

            # Registrar embedding en índice semántico
            if query_embedding:
                self._semantic_index[key] = query_embedding
                logger.debug(
                    f"🔍 Embedding indexado: {len(self._semantic_index)} "
                    f"entradas en índice semántico"
                )

            logger.debug(f"💾 Cache SET: '{query[:50]}' TTL={effective_ttl}s")
            return True

        except Exception as e:
            self._errors += 1
            logger.warning(f"⚠️ Redis set error: {e}")
            return False

    # ── Búsqueda semántica ─────────────────────────────────────────────────────

    def _find_semantic_match(
        self,
        query_embedding: List[float],
    ) -> Tuple[str, float]:
        """
        Encuentra la entrada más similar en el índice semántico.

        Args:
            query_embedding: Embedding de la query actual

        Returns:
            (cache_key, similitud) del mejor match
        """
        best_key = ""
        best_sim = 0.0

        for key, stored_embedding in self._semantic_index.items():
            sim = _cosine_similarity(query_embedding, stored_embedding)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        return best_key, best_sim

    # ── Cache búsqueda ─────────────────────────────────────────────────────────

    async def get_search_results(self, query: str) -> Optional[list]:
        if not self._connected or not self._client:
            return None
        key = _make_cache_key("qa:search", query)
        try:
            raw = await self._get_with_breaker(key)
            if raw:
                self._hits += 1
                return _deserialize(raw)
            self._misses += 1
            return None
        except Exception as e:
            self._errors += 1
            logger.warning(f"⚠️ Redis get_search error: {e}")
            return None

    async def set_search_results(
        self, query: str, results: list, ttl: Optional[int] = None
    ) -> bool:
        if not self._connected or not self._client:
            return False
        key = _make_cache_key("qa:search", query)
        try:
            await self._set_with_breaker(
                key, _serialize(results), ttl or self._search_ttl
            )
            return True
        except Exception as e:
            self._errors += 1
            logger.warning(f"⚠️ Redis set_search error: {e}")
            return False

    # ── Operaciones genéricas ──────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        if not self._connected or not self._client:
            return None
        try:
            raw = await self._get_with_breaker(key)
            return _deserialize(raw) if raw else None
        except Exception as e:
            logger.warning(f"⚠️ Redis get({key}) error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        if not self._connected or not self._client:
            return False
        try:
            await self._set_with_breaker(key, _serialize(value), ttl)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis set({key}) error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        if not self._connected or not self._client:
            return False
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis delete({key}) error: {e}")
            return False

    async def delete_qa_response(self, query: str) -> bool:
        key = _make_cache_key("qa:response", query)
        self._semantic_index.pop(key, None)
        return await self.delete(key)

    # ── Circuit breaker wrappers ───────────────────────────────────────────────

    @redis_breaker
    async def _get_with_breaker(self, key: str) -> Optional[str]:
        return await self._client.get(key)

    @redis_breaker
    async def _set_with_breaker(self, key: str, value: str, ttl: int) -> None:
        await self._client.setex(key, ttl, value)

    # ── Health check ───────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        if not self._client:
            return {"status": "disconnected", "connected": False,
                    "latency_ms": 0, **self._get_stats_dict()}
        try:
            start = time.perf_counter()
            await self._client.ping()
            latency = (time.perf_counter() - start) * 1000
            return {"status": "healthy", "connected": True,
                    "latency_ms": round(latency, 2), **self._get_stats_dict()}
        except Exception as e:
            return {"status": "unhealthy", "connected": False,
                    "latency_ms": 0, "error": str(e), **self._get_stats_dict()}

    def _get_stats_dict(self) -> dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "semantic_hits": self._semantic_hits,
            "errors": self._errors,
            "hit_rate_pct": round(hit_rate, 1),
            "semantic_index_size": len(self._semantic_index),
        }

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Singleton ──────────────────────────────────────────────────────────────────

_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
        logger.info("🔴 RedisCache singleton creado")
    return _redis_cache