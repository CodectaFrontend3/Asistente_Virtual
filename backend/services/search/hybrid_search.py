"""
Orquestador de Búsqueda Híbrida: BM25 + FAISS + RRF.

Ejecuta BM25 y FAISS EN PARALELO con asyncio.gather() para
minimizar latencia total.

Flujo:
    Query
      │
      ├─── BM25 search (asyncio.gather) ───┐
      │                                    │
      └─── FAISS search ───────────────────┤
                                           │
                                    RRF Fusion
                                           │
                                    Top-K resultados

Latencia objetivo:
    BM25:   ~5ms
    FAISS: ~15ms
    PARALELO: max(5, 15) = ~15ms (no 5+15=20ms)
    RRF:    ~2ms
    TOTAL: ~17ms ← objetivo superado (plan: <40ms)
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .bm25_search import BM25SearchService, get_bm25_service
from .rrf_fusion import analyze_fusion_results, rrf_bm25_faiss
from .vector_search import VectorSearchService, get_vector_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Resultado individual de búsqueda híbrida."""
    doc_id: str
    rrf_score: float
    bm25_score: Optional[float] = None    # Score original BM25 (None si no apareció)
    faiss_score: Optional[float] = None   # Score original FAISS (None si no apareció)
    bm25_rank: Optional[int] = None       # Posición en resultados BM25
    faiss_rank: Optional[int] = None      # Posición en resultados FAISS


@dataclass
class HybridSearchResponse:
    """Respuesta completa de búsqueda híbrida."""
    query: str
    results: List[SearchResult]
    total_found: int

    # Métricas de latencia (ms)
    bm25_latency_ms: float = 0.0
    faiss_latency_ms: float = 0.0
    rrf_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Métricas de fusión
    bm25_candidates: int = 0
    faiss_candidates: int = 0
    overlap_count: int = 0
    overlap_pct: float = 0.0

    # Config usada
    top_k: int = 10
    rrf_k: int = 60


# ---------------------------------------------------------------------------
# HYBRID SEARCH ORCHESTRATOR
# ---------------------------------------------------------------------------

class HybridSearchService:
    """
    Servicio de búsqueda híbrida BM25 + FAISS + RRF.

    Combina búsqueda léxica (BM25) y semántica (FAISS) en paralelo,
    fusiona con RRF para obtener los mejores resultados.

    Configuración:
        - top_k_candidates: Candidatos de cada sistema antes de RRF
        - top_k_final: Resultados finales tras RRF
        - rrf_k: Constante de suavizado RRF (paper recomienda 60)
        - bm25_weight / faiss_weight: Pesos relativos (default=1.0 cada uno)

    Example:
        ```python
        service = HybridSearchService()

        # Indexar documentos (doc_id, text, embedding)
        await service.index_documents([
            ("doc_1", "Tiempo de envío 3-5 días", [0.1, 0.2, ...]),
            ("doc_2", "Política de devoluciones", [0.3, 0.4, ...]),
        ])

        # Buscar (query_text para BM25, query_embedding para FAISS)
        response = await service.search(
            query_text="cuánto tarda el envío",
            query_embedding=[0.15, 0.25, ...],
            top_k=5,
        )

        for result in response.results:
            print(f"{result.doc_id}: RRF={result.rrf_score:.4f}")
        ```
    """

    def __init__(
        self,
        rrf_k: int = 60,
        top_k_candidates: int = 50,    # Candidatos por sistema antes de RRF
        bm25_weight: float = 1.0,
        faiss_weight: float = 1.0,
        bm25_service: Optional[BM25SearchService] = None,
        vector_service: Optional[VectorSearchService] = None,
    ) -> None:
        self._rrf_k = rrf_k
        self._top_k_candidates = top_k_candidates
        self._bm25_weight = bm25_weight
        self._faiss_weight = faiss_weight

        # Usar singletons por defecto o inyectados (para tests)
        self._bm25 = bm25_service or get_bm25_service()
        self._faiss = vector_service or get_vector_service()

        logger.info(
            f"🔍 HybridSearchService inicializado "
            f"(rrf_k={rrf_k}, candidates={top_k_candidates})"
        )

    # ------------------------------------------------------------------
    # INDEXACIÓN
    # ------------------------------------------------------------------

    async def index_documents(
        self,
        documents: List[Tuple[str, str, List[float]]],
    ) -> dict:
        """
        Indexa documentos en BM25 y FAISS simultáneamente.

        Args:
            documents: Lista de (doc_id, text, embedding)
                - doc_id: Identificador único
                - text: Texto para indexar en BM25
                - embedding: Vector 384-dim para FAISS

        Returns:
            Dict con stats de indexación

        Example:
            ```python
            await service.index_documents([
                ("faq_1", "¿Cuánto tarda el envío?", embedding_1),
                ("faq_2", "Política de devoluciones", embedding_2),
            ])
            ```
        """
        if not documents:
            raise ValueError("documents no puede estar vacío")

        # Preparar datos para cada índice
        bm25_docs = [(doc_id, text) for doc_id, text, _ in documents]
        faiss_docs = [(doc_id, emb) for doc_id, _, emb in documents]

        # Indexar en paralelo
        bm25_count, faiss_count = await asyncio.gather(
            self._bm25.add_documents(bm25_docs),
            self._faiss.add_vectors(faiss_docs),
        )

        stats = {
            "bm25_indexed": bm25_count,
            "faiss_indexed": faiss_count,
            "total_documents": len(documents),
        }

        logger.info(f"✅ Indexación híbrida completa: {stats}")
        return stats

    # ------------------------------------------------------------------
    # BÚSQUEDA PRINCIPAL
    # ------------------------------------------------------------------

    async def search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
        top_k_candidates: Optional[int] = None,
    ) -> HybridSearchResponse:
        """
        Búsqueda híbrida BM25 + FAISS en paralelo con fusión RRF.

        Args:
            query_text: Texto de la query (para BM25)
            query_embedding: Embedding de la query (para FAISS, 384-dim)
            top_k: Número de resultados finales a retornar
            top_k_candidates: Override de candidatos por sistema (None = default)

        Returns:
            HybridSearchResponse con resultados y métricas

        Raises:
            RuntimeError: Si los índices no están construidos
        """
        if not self._bm25.is_ready or not self._faiss.is_ready:
            raise RuntimeError(
                "Los índices no están construidos. "
                "Llama index_documents() primero."
            )

        candidates = top_k_candidates or self._top_k_candidates
        total_start = time.perf_counter()

        # ── BM25 + FAISS EN PARALELO ──────────────────────────────────

        bm25_start = time.perf_counter()
        faiss_start = bm25_start  # Ambos inician juntos con gather

        bm25_results, faiss_results = await asyncio.gather(
            self._bm25.search(query_text, top_k=candidates),
            self._faiss.search(query_embedding, top_k=candidates),
            return_exceptions=True,
        )

        # Manejar errores individuales (no fallar todo si uno falla)
        if isinstance(bm25_results, Exception):
            logger.error(f"❌ BM25 falló: {bm25_results}")
            bm25_results = []

        if isinstance(faiss_results, Exception):
            logger.error(f"❌ FAISS falló: {faiss_results}")
            faiss_results = []

        bm25_latency = (time.perf_counter() - bm25_start) * 1000

        # ── RRF FUSION ────────────────────────────────────────────────

        rrf_start = time.perf_counter()

        fused = rrf_bm25_faiss(
            bm25_results=bm25_results,
            faiss_results=faiss_results,
            k=self._rrf_k,
            top_k=top_k,
            bm25_weight=self._bm25_weight,
            faiss_weight=self._faiss_weight,
        )

        rrf_latency = (time.perf_counter() - rrf_start) * 1000
        total_latency = (time.perf_counter() - total_start) * 1000

        # ── ENRIQUECER RESULTADOS CON SCORES ORIGINALES ───────────────

        # Mapas rápidos para lookup de scores y ranks
        bm25_score_map = {doc_id: score for doc_id, score in bm25_results}
        bm25_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results, 1)}
        faiss_score_map = {doc_id: score for doc_id, score in faiss_results}
        faiss_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(faiss_results, 1)}

        search_results = [
            SearchResult(
                doc_id=doc_id,
                rrf_score=round(rrf_score, 6),
                bm25_score=bm25_score_map.get(doc_id),
                faiss_score=faiss_score_map.get(doc_id),
                bm25_rank=bm25_rank_map.get(doc_id),
                faiss_rank=faiss_rank_map.get(doc_id),
            )
            for doc_id, rrf_score in fused
        ]

        # ── ANÁLISIS DE FUSIÓN ────────────────────────────────────────

        analysis = analyze_fusion_results(bm25_results, faiss_results, fused)

        # ── RESPUESTA ─────────────────────────────────────────────────

        response = HybridSearchResponse(
            query=query_text,
            results=search_results,
            total_found=len(search_results),
            bm25_latency_ms=round(bm25_latency, 2),
            faiss_latency_ms=round(bm25_latency, 2),  # Paralelo → misma medida
            rrf_latency_ms=round(rrf_latency, 2),
            total_latency_ms=round(total_latency, 2),
            bm25_candidates=analysis["bm25_count"],
            faiss_candidates=analysis["faiss_count"],
            overlap_count=analysis["overlap_count"],
            overlap_pct=analysis["overlap_pct"],
            top_k=top_k,
            rrf_k=self._rrf_k,
        )

        logger.info(
            f"✅ Hybrid search: '{query_text[:50]}' → "
            f"{len(search_results)} resultados | "
            f"total={total_latency:.1f}ms "
            f"(bm25+faiss paralelo={bm25_latency:.1f}ms, rrf={rrf_latency:.1f}ms)"
        )

        return response

    # ------------------------------------------------------------------
    # BÚSQUEDA SOLO BM25 (fallback sin embedding)
    # ------------------------------------------------------------------

    async def search_bm25_only(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Búsqueda solo BM25 (cuando no hay embedding disponible).

        Útil como fallback si el Embedding Service está caído.

        Args:
            query_text: Texto de búsqueda
            top_k: Máximo de resultados

        Returns:
            Lista de (doc_id, bm25_score)
        """
        if not self._bm25.is_ready:
            raise RuntimeError("BM25 index no está construido")

        return await self._bm25.search(query_text, top_k=top_k)

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------

    async def save_indexes(self, base_path: Path) -> None:
        """
        Guarda ambos índices en disco.

        Args:
            base_path: Directorio base (ej: Path("vector_store"))

        Crea:
            base_path/bm25/qa_bm25.pkl
            base_path/faiss/qa_faiss.bin
            base_path/faiss/qa_faiss_meta.pkl
        """
        bm25_path = base_path / "bm25" / "qa_bm25.pkl"
        faiss_index_path = base_path / "faiss" / "qa_faiss.bin"
        faiss_meta_path = base_path / "faiss" / "qa_faiss_meta.pkl"

        await asyncio.gather(
            self._bm25.save(bm25_path),
            self._faiss.save(faiss_index_path, faiss_meta_path),
        )

        logger.info(f"💾 Índices guardados en: {base_path}")

    async def load_indexes(self, base_path: Path) -> bool:
        """
        Carga ambos índices desde disco.

        Args:
            base_path: Directorio base donde están los índices

        Returns:
            True si ambos cargaron correctamente
        """
        bm25_path = base_path / "bm25" / "qa_bm25.pkl"
        faiss_index_path = base_path / "faiss" / "qa_faiss.bin"
        faiss_meta_path = base_path / "faiss" / "qa_faiss_meta.pkl"

        bm25_ok, faiss_ok = await asyncio.gather(
            self._bm25.load(bm25_path),
            self._faiss.load(faiss_index_path, faiss_meta_path),
        )

        if bm25_ok and faiss_ok:
            logger.info(f"📂 Índices cargados desde: {base_path}")
        else:
            logger.warning(
                f"⚠️ Carga parcial: bm25={bm25_ok}, faiss={faiss_ok}"
            )

        return bm25_ok and faiss_ok

    # ------------------------------------------------------------------
    # ESTADO Y STATS
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True si ambos índices están listos."""
        return self._bm25.is_ready and self._faiss.is_ready

    def get_stats(self) -> dict:
        """Retorna estadísticas de ambos índices."""
        return {
            "is_ready": self.is_ready,
            "bm25": self._bm25.get_stats(),
            "faiss": self._faiss.get_stats(),
            "config": {
                "rrf_k": self._rrf_k,
                "top_k_candidates": self._top_k_candidates,
                "bm25_weight": self._bm25_weight,
                "faiss_weight": self._faiss_weight,
            },
        }


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_hybrid_service: Optional[HybridSearchService] = None


def get_hybrid_search_service(
    rrf_k: int = 60,
    top_k_candidates: int = 50,
) -> HybridSearchService:
    """
    Obtiene instancia singleton del servicio de búsqueda híbrida.

    Returns:
        HybridSearchService singleton
    """
    global _hybrid_service
    if _hybrid_service is None:
        _hybrid_service = HybridSearchService(
            rrf_k=rrf_k,
            top_k_candidates=top_k_candidates,
        )
    return _hybrid_service