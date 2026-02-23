"""
Módulo de búsqueda híbrida.

Combina BM25 (léxico) + FAISS (vectorial) con RRF para
obtener los mejores resultados de búsqueda.

Exports principales:
    - HybridSearchService: Orquestador principal
    - BM25SearchService: Búsqueda léxica standalone
    - VectorSearchService: Búsqueda vectorial standalone
    - SearchResult: Dataclass de resultado individual
    - HybridSearchResponse: Dataclass de respuesta completa
    - reciprocal_rank_fusion: Función RRF pura
    - rrf_bm25_faiss: RRF especializado BM25+FAISS

Usage:
    ```python
    from services.search import get_hybrid_search_service

    service = get_hybrid_search_service()

    # Indexar documentos
    await service.index_documents([
        ("doc_1", "texto del documento", embedding_1),
        ("doc_2", "otro documento", embedding_2),
    ])

    # Buscar
    response = await service.search(
        query_text="mi consulta",
        query_embedding=query_embedding,
        top_k=5,
    )

    for result in response.results:
        print(f"{result.doc_id}: {result.rrf_score:.4f}")
    ```
"""
from .bm25_search import (
    BM25Index,
    BM25SearchService,
    get_bm25_service,
    tokenize_text,
)
from .hybrid_search import (
    HybridSearchResponse,
    HybridSearchService,
    SearchResult,
    get_hybrid_search_service,
)
from .rrf_fusion import (
    analyze_fusion_results,
    reciprocal_rank_fusion,
    rrf_bm25_faiss,
)
from .vector_search import (
    EMBEDDING_DIM,
    FAISSIndex,
    VectorSearchService,
    get_vector_service,
)

__all__ = [
    # Hybrid (principal)
    "HybridSearchService",
    "HybridSearchResponse",
    "SearchResult",
    "get_hybrid_search_service",
    # BM25
    "BM25SearchService",
    "BM25Index",
    "get_bm25_service",
    "tokenize_text",
    # FAISS
    "VectorSearchService",
    "FAISSIndex",
    "get_vector_service",
    "EMBEDDING_DIM",
    # RRF
    "reciprocal_rank_fusion",
    "rrf_bm25_faiss",
    "analyze_fusion_results",
]