"""
Q&A Service con cache semántico de 2 niveles integrado.

Cambios respecto a la versión anterior:
- Pasa query_embedding al cache → activa nivel 2 semántico
- Normaliza la query antes de buscar en cache
- Guarda embedding junto con la respuesta para futuras comparaciones
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from core import (
    EmbeddingServiceError,
    NoResultsFoundError,
    RAGPipelineError,
    InvalidQueryError,
)
from services.embeddings import get_embedding_client, get_embeddings_cache
from services.cache import get_redis_cache
from .faiss_manager import Chunk, FAISSManager, get_faiss_manager

logger = logging.getLogger(__name__)


# ── Dataclasses (sin cambios) ──────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    chunks: List[Chunk]
    total_found: int
    search_latency_ms: float


@dataclass
class QARequest:
    query: str
    top_k: int = 5
    min_score: float = 0.0
    include_metadata: bool = False
    use_cache: bool = True  # ← Nuevo: permite desactivar cache por request


@dataclass
class QAResponse:
    query: str
    answer: str
    sources: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    total_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    from_cache: bool = False


# ── Context Builder (sin cambios) ──────────────────────────────────────────────

class ContextBuilder:
    def build_context(self, chunks: List[Chunk], max_chars: int = 3000) -> str:
        if not chunks:
            return ""
        context_parts = []
        total_chars = 0
        for i, chunk in enumerate(chunks, start=1):
            header = f"[Fuente {i}: {chunk.title}]"
            part = f"{header}\n{chunk.content}"
            if total_chars + len(part) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    part = part[:remaining] + "..."
                    context_parts.append(part)
                break
            context_parts.append(part)
            total_chars += len(part)
        return "\n\n".join(context_parts)

    def build_sources_metadata(self, chunks: List[Chunk]) -> List[dict]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
                "content_preview": chunk.content[:150] + "..."
                if len(chunk.content) > 150 else chunk.content,
            }
            for chunk in chunks
        ]


# ── QA Service ─────────────────────────────────────────────────────────────────

class QAService:
    """
    Servicio Q&A con cache de 2 niveles:

    Nivel 1 (normalización textual):
        "¿Cuál es el horario?" == "cual es el horario" → HIT instantáneo

    Nivel 2 (similitud semántica):
        "¿a qué hora atienden?" ≈ "cual es el horario" → similitud 0.94 → HIT

    Solo llama al LLM si ambos niveles dan MISS.
    """

    def __init__(
        self,
        faiss_manager: Optional[FAISSManager] = None,
        llm_client=None,
        max_context_chars: int = 3000,
        min_chunks_required: int = 1,
    ) -> None:
        self._faiss_manager = faiss_manager or get_faiss_manager()
        self._llm_client = llm_client
        self._context_builder = ContextBuilder()
        self._max_context_chars = max_context_chars
        self._min_chunks_required = min_chunks_required
        logger.info("🤖 QAService inicializado")

    async def initialize(self, documents=None) -> dict:
        stats = await self._faiss_manager.load_or_build(documents)
        logger.info(f"✅ QAService inicializado: {stats}")
        return stats

    # ── Pipeline principal ─────────────────────────────────────────────────────

    async def answer(self, request: QARequest) -> QAResponse:
        """
        Pipeline RAG con cache semántico de 2 niveles.

        Flujo:
            1. Generar embedding (necesario para nivel 2 del cache)
            2. Buscar en cache (nivel 1 textual + nivel 2 semántico)
            3. Si HIT → retornar inmediatamente
            4. Si MISS → búsqueda híbrida → LLM → guardar en cache
        """
        query = request.query.strip()
        if not query:
            raise InvalidQueryError("La query no puede estar vacía")

        if not self._faiss_manager.is_ready:
            raise RAGPipelineError(
                "Los índices no están listos. Llama initialize() primero."
            )

        total_start = time.perf_counter()
        cache = get_redis_cache()

        # ── PASO 1: EMBEDDING (siempre, necesario para cache semántico) ──
        embedding_start = time.perf_counter()
        query_embedding = await self._get_query_embedding(query)
        embedding_latency = (time.perf_counter() - embedding_start) * 1000

        # ── PASO 2: BUSCAR EN CACHE (ambos niveles) ───────────────────
        if request.use_cache:
            cached = await cache.get_qa_response(
                query=query,
                query_embedding=query_embedding,  # ← activa nivel 2
            )
            if cached:
                total_latency = (time.perf_counter() - total_start) * 1000
                cached["from_cache"] = True
                cached["total_latency_ms"] = round(total_latency, 2)
                cached["embedding_latency_ms"] = round(embedding_latency, 2)
                return QAResponse(**{
                    k: cached.get(k, v)
                    for k, v in QAResponse.__dataclass_fields__.items()
                })

        # ── PASO 3: BÚSQUEDA HÍBRIDA ──────────────────────────────────
        search_start = time.perf_counter()
        candidates = max(request.top_k * 2, 20)
        search_response = await self._faiss_manager.hybrid_search.search(
            query_text=query,
            query_embedding=query_embedding,
            top_k=candidates,
        )
        search_latency = (time.perf_counter() - search_start) * 1000

        if not search_response.results:
            raise NoResultsFoundError(f"No se encontraron resultados para: '{query}'")

        filtered_results = [
            r for r in search_response.results
            if r.rrf_score >= request.min_score
        ][:request.top_k]

        if len(filtered_results) < self._min_chunks_required:
            raise NoResultsFoundError(
                f"Resultados insuficientes (encontrados: {len(filtered_results)})"
            )

        chunk_ids = [r.doc_id for r in filtered_results]
        chunks = self._faiss_manager.get_chunks_by_ids(chunk_ids)

        if not chunks:
            raise NoResultsFoundError("No se pudieron recuperar los chunks")

        top_score = filtered_results[0].rrf_score if filtered_results else 0
        confidence = min(top_score / 0.033, 1.0) if top_score > 0 else 0.0

        # ── PASO 4: CONTEXTO ──────────────────────────────────────────
        context = self._context_builder.build_context(
            chunks, max_chars=self._max_context_chars
        )

        # ── PASO 5: LLM ───────────────────────────────────────────────
        llm_start = time.perf_counter()
        answer = await self._generate_answer(query, context)
        llm_latency = (time.perf_counter() - llm_start) * 1000
        total_latency = (time.perf_counter() - total_start) * 1000

        sources = []
        if request.include_metadata:
            sources = self._context_builder.build_sources_metadata(chunks)

        response = QAResponse(
            query=query,
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3),
            total_latency_ms=round(total_latency, 2),
            search_latency_ms=round(search_latency, 2),
            embedding_latency_ms=round(embedding_latency, 2),
            llm_latency_ms=round(llm_latency, 2),
            from_cache=False,
        )

        # ── PASO 6: GUARDAR EN CACHE CON EMBEDDING ────────────────────
        if request.use_cache:
            await cache.set_qa_response(
                query=query,
                response={
                    "query": response.query,
                    "answer": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "total_latency_ms": response.total_latency_ms,
                    "search_latency_ms": response.search_latency_ms,
                    "embedding_latency_ms": response.embedding_latency_ms,
                    "llm_latency_ms": response.llm_latency_ms,
                    "from_cache": False,
                },
                query_embedding=query_embedding,  # ← guarda para futuras comparaciones
            )

        logger.info(
            f"✅ Q&A: '{query[:60]}' → confianza={confidence:.2%} | "
            f"total={total_latency:.0f}ms "
            f"(emb={embedding_latency:.0f}ms, search={search_latency:.0f}ms, "
            f"llm={llm_latency:.0f}ms)"
        )

        return response

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _get_query_embedding(self, query: str) -> List[float]:
        cache = get_embeddings_cache()
        cached = cache.get(query)
        if cached is not None:
            return cached
        client = get_embedding_client()
        try:
            embedding = await client.generate_embedding(query)
        except Exception as e:
            raise EmbeddingServiceError(f"Error generando embedding: {e}")
        cache.set(query, embedding)
        return embedding

    async def _generate_answer(self, query: str, context: str) -> str:
        if self._llm_client is not None:
            from services.llm import get_response_parser
            raw_response = await self._llm_client.generate(
                query=query,
                context=context,
            )
            parser = get_response_parser()
            return parser.parse_text(raw_response)

        if not context:
            return (
                "No encontré información relevante para responder tu pregunta. "
                "Por favor, intenta reformularla."
            )
        return (
            f"**Información encontrada para:** {query}\n\n"
            f"{context}"
        )

    async def search_context(self, query: str, top_k: int = 5) -> RetrievedContext:
        if not query.strip():
            raise InvalidQueryError("La query no puede estar vacía")
        query_embedding = await self._get_query_embedding(query)
        search_start = time.perf_counter()
        search_response = await self._faiss_manager.hybrid_search.search(
            query_text=query,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        search_latency = (time.perf_counter() - search_start) * 1000
        chunk_ids = [r.doc_id for r in search_response.results]
        chunks = self._faiss_manager.get_chunks_by_ids(chunk_ids)
        return RetrievedContext(
            chunks=chunks,
            total_found=search_response.total_found,
            search_latency_ms=round(search_latency, 2),
        )

    @property
    def is_ready(self) -> bool:
        return self._faiss_manager.is_ready

    def set_llm_client(self, llm_client) -> None:
        self._llm_client = llm_client
        logger.info("🦙 LLM client inyectado en QAService")

    def get_stats(self) -> dict:
        return {
            "is_ready": self.is_ready,
            "llm_available": self._llm_client is not None,
            "index_stats": self._faiss_manager.get_stats(),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_qa_service: Optional[QAService] = None


def get_qa_service(faiss_manager=None, llm_client=None) -> QAService:
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService(
            faiss_manager=faiss_manager,
            llm_client=llm_client,
        )
        logger.info("🤖 QAService singleton creado")
    return _qa_service