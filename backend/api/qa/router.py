"""
Router Q&A: Endpoint principal del asistente virtual.

Endpoints:
    POST /qa/ask          → Hacer una pregunta (pipeline RAG completo)
    GET  /qa/health       → Estado del servicio Q&A
    GET  /qa/stats        → Estadísticas de índices y cache
    POST /qa/invalidate   → Invalidar cache de una query específica
"""
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from core import (
    InvalidQueryError,
    NoResultsFoundError,
    RAGPipelineError,
    EmbeddingServiceError,
    OllamaServiceError,
    CircuitBreakerOpenError,
)
from services.cache import get_redis_cache
from services.qa import get_qa_service, QARequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qa", tags=["Q&A"])


# ---------------------------------------------------------------------------
# SCHEMAS DE REQUEST / RESPONSE
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request para hacer una pregunta al asistente."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Pregunta del usuario",
        examples=["¿Cuánto tarda el envío?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de fragmentos a recuperar (1-20)",
    )
    include_sources: bool = Field(
        default=False,
        description="Incluir fuentes/chunks usados en la respuesta",
    )
    use_cache: bool = Field(
        default=True,
        description="Usar cache Redis (False = forzar nueva búsqueda)",
    )


class AskResponse(BaseModel):
    """Respuesta del asistente virtual."""

    query: str
    answer: str
    confidence: float = Field(description="Confianza 0.0-1.0 basada en relevancia")
    from_cache: bool = Field(description="True si la respuesta viene del cache")
    sources: list = Field(default_factory=list, description="Fuentes usadas (si include_sources=True)")

    # Métricas de latencia
    total_latency_ms: float
    embedding_latency_ms: float
    search_latency_ms: float
    llm_latency_ms: float


class HealthResponse(BaseModel):
    """Estado del servicio Q&A."""
    status: str
    qa_ready: bool
    llm_available: bool
    cache_connected: bool
    indexes: dict


class StatsResponse(BaseModel):
    """Estadísticas del servicio."""
    qa_stats: dict
    cache_stats: dict


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Hacer una pregunta al asistente",
    description="Pipeline RAG completo: embedding → búsqueda híbrida → LLM → respuesta",
)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Endpoint principal del asistente virtual.

    Flujo:
    1. Verificar cache Redis (si use_cache=True)
    2. Si cache miss → pipeline RAG completo
    3. Guardar en cache
    4. Retornar respuesta

    Errores posibles:
    - 400: Query vacía o inválida
    - 404: No se encontraron resultados relevantes
    - 503: Servicio de embeddings o LLM no disponible
    - 500: Error interno del pipeline RAG
    """
    start_time = time.perf_counter()

    qa_service = get_qa_service()
    cache = get_redis_cache()

    # ── 1. VERIFICAR CACHE ─────────────────────────────────────────────
    if request.use_cache:
        cached = await cache.get_qa_response(request.query)
        if cached:
            logger.info(f"🎯 Cache HIT: '{request.query[:50]}'")
            return AskResponse(
                query=cached["query"],
                answer=cached["answer"],
                confidence=cached["confidence"],
                from_cache=True,
                sources=cached.get("sources", []),
                total_latency_ms=(time.perf_counter() - start_time) * 1000,
                embedding_latency_ms=0,
                search_latency_ms=0,
                llm_latency_ms=0,
            )

    # ── 2. PIPELINE RAG ────────────────────────────────────────────────
    try:
        qa_request = QARequest(
            query=request.query,
            top_k=request.top_k,
            include_metadata=request.include_sources,
        )
        response = await qa_service.answer(qa_request)

    except InvalidQueryError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_QUERY", "message": str(e)},
        )
    except NoResultsFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "NO_RESULTS", "message": str(e)},
        )
    except (EmbeddingServiceError, CircuitBreakerOpenError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "EMBEDDING_SERVICE_UNAVAILABLE", "message": str(e)},
        )
    except OllamaServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "LLM_SERVICE_UNAVAILABLE", "message": str(e)},
        )
    except RAGPipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PIPELINE_ERROR", "message": str(e)},
        )
    except Exception as e:
        logger.error(f"❌ Error inesperado en /qa/ask: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "INTERNAL_ERROR", "message": "Error interno del servidor"},
        )

    # ── 3. GUARDAR EN CACHE ────────────────────────────────────────────
    if request.use_cache:
        await cache.set_qa_response(
            query=request.query,
            response={
                "query": response.query,
                "answer": response.answer,
                "confidence": response.confidence,
                "sources": response.sources,
            },
        )

    # ── 4. RESPUESTA ───────────────────────────────────────────────────
    return AskResponse(
        query=response.query,
        answer=response.answer,
        confidence=response.confidence,
        from_cache=False,
        sources=response.sources,
        total_latency_ms=response.total_latency_ms,
        embedding_latency_ms=response.embedding_latency_ms,
        search_latency_ms=response.search_latency_ms,
        llm_latency_ms=response.llm_latency_ms,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio Q&A",
)
async def health_check() -> HealthResponse:
    """Verifica el estado de todos los componentes del servicio Q&A."""
    qa_service = get_qa_service()
    cache = get_redis_cache()

    qa_stats = qa_service.get_stats()
    cache_health = await cache.health_check()

    overall_status = "healthy"
    if not qa_stats["is_ready"]:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        qa_ready=qa_stats["is_ready"],
        llm_available=qa_stats["llm_available"],
        cache_connected=cache_health.get("connected", False),
        indexes=qa_stats.get("index_stats", {}),
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Estadísticas de índices y cache",
)
async def get_stats() -> StatsResponse:
    """Retorna estadísticas detalladas del servicio, índices y cache Redis."""
    qa_service = get_qa_service()
    cache = get_redis_cache()

    return StatsResponse(
        qa_stats=qa_service.get_stats(),
        cache_stats=cache._get_stats_dict(),
    )


@router.post(
    "/invalidate",
    summary="Invalidar cache de una query",
    status_code=status.HTTP_200_OK,
)
async def invalidate_cache(query: str) -> dict:
    """
    Invalida la respuesta cacheada para una query específica.

    Útil cuando el contenido de la knowledge base cambia
    y necesitas forzar una nueva búsqueda.
    """
    cache = get_redis_cache()
    deleted = await cache.delete_qa_response(query)
    return {
        "query": query,
        "invalidated": deleted,
        "message": "Cache invalidado" if deleted else "No había cache para esta query",
    }