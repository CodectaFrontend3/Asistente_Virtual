"""
Reciprocal Rank Fusion (RRF) para combinar resultados BM25 + FAISS.

¿Por qué RRF?
- BM25 retorna scores en escala 0-20 (dependiendo del corpus)
- FAISS retorna cosine similarity en escala 0-1
- NO se pueden sumar directamente (escalas distintas)
- RRF combina por RANKING, no por score → agnóstico a la escala

Fórmula:
    RRF(doc) = Σ 1 / (k + rank_i)

    Donde:
    - k = constante de suavizado (default=60, recomendado en literatura)
    - rank_i = posición del doc en el resultado del sistema i (1-based)
    - Σ = suma sobre todos los sistemas (BM25, FAISS, etc.)

Propiedades:
- Robustez: Un doc en posición 1 contribuye 1/(60+1) ≈ 0.016
- Penalización suave: posición 10 → 0.014, posición 100 → 0.006
- Docs no rankeados: No contribuyen (score 0)
- Agnóstico: Funciona igual sin importar escala de scores originales

Referencia:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
    SIGIR 2009. k=60 es el valor usado en el paper original.
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RRF CORE
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Fusiona múltiples listas de resultados usando Reciprocal Rank Fusion.

    Args:
        result_lists: Lista de listas de (doc_id, score).
                      Cada lista debe estar ordenada por relevancia (DESC).
                      Los scores originales se IGNORAN, solo importa el orden.
        k: Constante de suavizado (default=60, paper original).
           Valores menores → más peso a los primeros resultados.
           Valores mayores → distribución más uniforme.
        top_k: Limitar resultados finales (None = todos)

    Returns:
        Lista de (doc_id, rrf_score) ordenada por RRF score DESC.
        RRF scores están en rango (0, n_lists/k].

    Example:
        ```python
        bm25_results = [
            ("doc_1", 8.5),   # rank 1 BM25
            ("doc_3", 4.2),   # rank 2 BM25
            ("doc_2", 1.1),   # rank 3 BM25
        ]
        faiss_results = [
            ("doc_2", 0.95),  # rank 1 FAISS
            ("doc_1", 0.87),  # rank 2 FAISS
            ("doc_4", 0.65),  # rank 3 FAISS
        ]

        fused = reciprocal_rank_fusion([bm25_results, faiss_results], k=60)
        # doc_1: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 ← winner
        # doc_2: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
        # doc_3: 1/(60+2) + 0         = 0.0161
        # doc_4: 0        + 1/(60+3)  = 0.0159
        ```
    """
    if not result_lists:
        return []

    # Acumular scores RRF por doc_id
    rrf_scores: Dict[str, float] = {}

    for ranked_list in result_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_contribution = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution

    # Ordenar por RRF score DESC
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Aplicar top_k si se especificó
    if top_k is not None:
        fused = fused[:top_k]

    logger.debug(
        f"🔀 RRF fusion: {sum(len(r) for r in result_lists)} candidatos "
        f"→ {len(fused)} resultados únicos (k={k})"
    )

    return fused


# ---------------------------------------------------------------------------
# VARIANTES ESPECIALIZADAS
# ---------------------------------------------------------------------------

def rrf_bm25_faiss(
    bm25_results: List[Tuple[str, float]],
    faiss_results: List[Tuple[str, float]],
    k: int = 60,
    top_k: int = 10,
    bm25_weight: float = 1.0,
    faiss_weight: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    RRF especializado para fusión BM25 + FAISS.

    Permite pesos diferenciados para cada sistema.
    Con pesos=1.0, es RRF estándar del paper original.

    Args:
        bm25_results: Resultados BM25 ordenados por score DESC
        faiss_results: Resultados FAISS ordenados por score DESC
        k: Constante RRF (default=60)
        top_k: Máximo de resultados finales
        bm25_weight: Multiplicador para contribución BM25 (default=1.0)
        faiss_weight: Multiplicador para contribución FAISS (default=1.0)

    Returns:
        Lista de (doc_id, rrf_score) fusionada y ordenada

    Example:
        ```python
        # Para Q&A: BM25 y FAISS igual de importantes
        results = rrf_bm25_faiss(bm25, faiss, k=60, top_k=10)

        # Para búsqueda semántica: dar más peso a FAISS
        results = rrf_bm25_faiss(
            bm25, faiss,
            k=60, top_k=10,
            bm25_weight=0.5,
            faiss_weight=1.5
        )
        ```
    """
    rrf_scores: Dict[str, float] = {}

    # Contribución BM25
    for rank, (doc_id, _score) in enumerate(bm25_results, start=1):
        contribution = bm25_weight / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

    # Contribución FAISS
    for rank, (doc_id, _score) in enumerate(faiss_results, start=1):
        contribution = faiss_weight / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

    # Ordenar y limitar
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Log de diagnóstico
    bm25_ids = {doc_id for doc_id, _ in bm25_results}
    faiss_ids = {doc_id for doc_id, _ in faiss_results}
    both = bm25_ids & faiss_ids
    only_bm25 = bm25_ids - faiss_ids
    only_faiss = faiss_ids - bm25_ids

    logger.debug(
        f"🔀 RRF: bm25={len(bm25_results)}, faiss={len(faiss_results)}, "
        f"ambos={len(both)}, solo_bm25={len(only_bm25)}, "
        f"solo_faiss={len(only_faiss)} → top_{top_k}={len(fused)}"
    )

    return fused


# ---------------------------------------------------------------------------
# ANÁLISIS DE RESULTADOS
# ---------------------------------------------------------------------------

def analyze_fusion_results(
    bm25_results: List[Tuple[str, float]],
    faiss_results: List[Tuple[str, float]],
    fused_results: List[Tuple[str, float]],
) -> dict:
    """
    Analiza la distribución de resultados tras la fusión.

    Útil para debugging y métricas de calidad.

    Args:
        bm25_results: Resultados BM25 originales
        faiss_results: Resultados FAISS originales
        fused_results: Resultados fusionados

    Returns:
        Dict con métricas de la fusión:
        - bm25_count: Docs únicos de BM25
        - faiss_count: Docs únicos de FAISS
        - overlap_count: Docs en ambos sistemas
        - fused_count: Docs en resultado final
        - overlap_pct: Porcentaje de solapamiento
        - top_doc_sources: Fuente de los top-3 docs
    """
    bm25_ids = {doc_id for doc_id, _ in bm25_results}
    faiss_ids = {doc_id for doc_id, _ in faiss_results}
    fused_ids = {doc_id for doc_id, _ in fused_results}

    overlap = bm25_ids & faiss_ids
    total_unique = len(bm25_ids | faiss_ids)

    # Analizar fuente de top-3 docs
    top_sources = []
    for doc_id, score in fused_results[:3]:
        in_bm25 = doc_id in bm25_ids
        in_faiss = doc_id in faiss_ids
        if in_bm25 and in_faiss:
            source = "both"
        elif in_bm25:
            source = "bm25_only"
        else:
            source = "faiss_only"
        top_sources.append({"doc_id": doc_id, "rrf_score": round(score, 6), "source": source})

    return {
        "bm25_count": len(bm25_ids),
        "faiss_count": len(faiss_ids),
        "overlap_count": len(overlap),
        "overlap_pct": round(len(overlap) / max(total_unique, 1) * 100, 1),
        "fused_count": len(fused_ids),
        "top_doc_sources": top_sources,
    }