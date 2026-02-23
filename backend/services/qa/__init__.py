"""
Módulo Q&A: Pipeline RAG completo.

Orquesta embedding → búsqueda híbrida → contexto → LLM → respuesta.

Exports principales:
    - QAService: Orquestador principal del pipeline RAG
    - FAISSManager: Gestión de índices y chunking
    - Document: Documento a indexar
    - Chunk: Fragmento indexado
    - QARequest: Request de pregunta
    - QAResponse: Respuesta del pipeline
    - RetrievedContext: Contexto recuperado (sin LLM)

Usage:
    ```python
    from services.qa import get_qa_service, QARequest, Document

    # Preparar documentos
    docs = [
        Document(
            doc_id="faq_envio",
            title="Preguntas frecuentes: Envíos",
            content="Los envíos tardan 3-5 días hábiles...",
        ),
    ]

    # Inicializar servicio (carga o construye índices)
    qa = get_qa_service()
    await qa.initialize(documents=docs)

    # Hacer pregunta
    response = await qa.answer(QARequest(
        query="¿Cuánto tarda el envío?",
        top_k=5,
        include_metadata=True,
    ))

    print(response.answer)
    print(f"Latencia: {response.total_latency_ms:.0f}ms")
    ```
"""
from .faiss_manager import (
    Chunk,
    Document,
    DocumentChunker,
    FAISSManager,
    IndexStats,
    get_faiss_manager,
)
from .qa_service import (
    ContextBuilder,
    QARequest,
    QAResponse,
    QAService,
    RetrievedContext,
    get_qa_service,
)

__all__ = [
    # QA Pipeline
    "QAService",
    "QARequest",
    "QAResponse",
    "RetrievedContext",
    "get_qa_service",
    # FAISS Manager
    "FAISSManager",
    "IndexStats",
    "get_faiss_manager",
    # Document handling
    "Document",
    "Chunk",
    "DocumentChunker",
    # Context
    "ContextBuilder",
]