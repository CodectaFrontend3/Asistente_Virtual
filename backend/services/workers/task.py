"""
Tareas Celery para procesamiento asíncrono.

Tareas disponibles:
    build_qa_index_task  → Construye índices FAISS + BM25 desde documentos JSON
    warmup_cache_task    → Pre-calienta Redis con queries frecuentes

Cómo ejecutar el worker (Windows):
    cd backend
    entorno_backend\\Scripts\\activate
    celery -A workers.celery_app worker --loglevel=info -P solo

Cómo lanzar tareas desde código Python:
    from workers.tasks import build_qa_index_task, warmup_cache_task

    # Asíncrono (no espera):
    build_qa_index_task.delay(documents_data)

    # Con resultado (espera máx 10min):
    result = build_qa_index_task.delay(documents_data)
    stats = result.get(timeout=600)
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import List

from .celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Ejecuta una coroutine desde contexto síncrono (Celery task)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# TAREA 1: CONSTRUIR ÍNDICES Q&A
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="workers.tasks.build_qa_index",
    max_retries=2,
    default_retry_delay=30,
)
def build_qa_index_task(self, documents_data: List[dict]) -> dict:
    """
    Construye índices FAISS + BM25 desde documentos JSON.

    Se ejecuta como tarea Celery → no bloquea el servidor FastAPI.

    Args:
        documents_data: Lista de dicts con formato:
            [
                {
                    "doc_id": "faq_001",
                    "title": "Política de Envíos",
                    "content": "Los envíos tardan...",
                    "metadata": {}  (opcional)
                },
                ...
            ]

    Returns:
        Dict con estadísticas:
        {
            "status": "success",
            "doc_count": 10,
            "chunk_count": 45,
            "build_time_seconds": 12.3,
            "bm25_ready": True,
            "faiss_ready": True,
        }

    Raises:
        Retry automático si falla (máx 2 intentos)

    Ejemplo de uso:
        from workers.tasks import build_qa_index_task

        docs = [
            {"doc_id": "faq_1", "title": "Envíos", "content": "..."},
            {"doc_id": "faq_2", "title": "Devoluciones", "content": "..."},
        ]
        result = build_qa_index_task.delay(docs)
        print(result.get(timeout=300))
    """
    logger.info(
        f"🏗️ [Celery] build_qa_index_task iniciado: "
        f"{len(documents_data)} documentos"
    )

    try:
        async def _build():
            from services.qa import get_qa_service, Document, get_faiss_manager

            # Convertir dicts → Document objects
            documents = [
                Document(
                    doc_id=d["doc_id"],
                    title=d["title"],
                    content=d["content"],
                    metadata=d.get("metadata", {}),
                )
                for d in documents_data
            ]

            # Construir índices
            faiss_manager = get_faiss_manager()
            stats = await faiss_manager.build_from_documents(documents)

            # Guardar en disco
            await faiss_manager.save_indexes()

            return stats

        index_stats = _run_async(_build())

        result = {
            "status": "success",
            "doc_count": index_stats.doc_count,
            "chunk_count": index_stats.chunk_count,
            "build_time_seconds": index_stats.build_time_seconds,
            "bm25_ready": index_stats.bm25_ready,
            "faiss_ready": index_stats.faiss_ready,
        }

        logger.info(f"✅ [Celery] Índices construidos: {result}")
        return result

    except Exception as exc:
        logger.error(f"❌ [Celery] build_qa_index_task falló: {exc}")
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# TAREA 2: WARMUP CACHE
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="workers.tasks.warmup_cache",
    max_retries=1,
)
def warmup_cache_task(self, queries: List[str]) -> dict:
    """
    Pre-calienta el cache Redis con queries frecuentes.

    Ejecuta el pipeline RAG para cada query y guarda en Redis.
    Así los usuarios reciben respuestas instantáneas para las
    preguntas más comunes.

    Args:
        queries: Lista de preguntas frecuentes a pre-cachear
            ["¿Cuánto tarda el envío?", "¿Cómo devuelvo un producto?", ...]

    Returns:
        Dict con resultados:
        {
            "status": "success",
            "total": 10,
            "cached": 8,
            "failed": 2,
            "failed_queries": ["query que falló", ...]
        }

    Ejemplo de uso:
        from workers.tasks import warmup_cache_task

        frequent_queries = [
            "¿Cuánto tarda el envío?",
            "¿Cómo hago una devolución?",
            "¿Cuáles son los métodos de pago?",
        ]
        warmup_cache_task.delay(frequent_queries)
    """
    logger.info(
        f"🔥 [Celery] warmup_cache_task iniciado: {len(queries)} queries"
    )

    cached_count = 0
    failed_queries = []

    async def _warmup():
        nonlocal cached_count

        from services.qa import get_qa_service, QARequest
        from services.cache import get_redis_cache

        qa_service = get_qa_service()
        cache = get_redis_cache()

        if not qa_service.is_ready:
            raise RuntimeError(
                "QAService no está listo. Construye los índices primero."
            )

        # Conectar Redis si no está conectado
        if not cache.is_connected:
            await cache.connect()

        for query in queries:
            try:
                # Verificar si ya está en cache
                existing = await cache.get_qa_response(query)
                if existing:
                    logger.debug(f"  ⏭️ Ya cacheada: '{query[:40]}'")
                    cached_count += 1
                    continue

                # Ejecutar pipeline RAG
                response = await qa_service.answer(QARequest(query=query, top_k=5))

                # Guardar en cache
                await cache.set_qa_response(
                    query=query,
                    response={
                        "query": response.query,
                        "answer": response.answer,
                        "confidence": response.confidence,
                        "sources": [],
                    }
                )

                cached_count += 1
                logger.info(f"  ✅ Cacheada: '{query[:40]}'")

            except Exception as e:
                logger.error(f"  ❌ Falló: '{query[:40]}' → {e}")
                failed_queries.append(query)

    try:
        _run_async(_warmup())

        result = {
            "status": "success",
            "total": len(queries),
            "cached": cached_count,
            "failed": len(failed_queries),
            "failed_queries": failed_queries,
        }

        logger.info(f"✅ [Celery] Warmup completo: {result}")
        return result

    except Exception as exc:
        logger.error(f"❌ [Celery] warmup_cache_task falló: {exc}")
        raise self.retry(exc=exc)