"""
Configuración de Celery para tareas asíncronas del backend.

Tareas implementadas:
    - build_qa_index_task: Construye índices FAISS + BM25 desde documentos
    - warmup_cache_task: Pre-carga queries frecuentes en Redis

Uso:
    # Iniciar worker (en otra terminal):
    celery -A workers.celery_app worker --loglevel=info -P solo

    # Lanzar tarea desde código:
    from workers.tasks import build_qa_index_task
    result = build_qa_index_task.delay(documents_data)
"""
from celery import Celery
from config import settings

celery_app = Celery(
    "backend_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["workers.tasks"],
)

celery_app.conf.update(
    # Serialización
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timeouts
    task_soft_time_limit=settings.CELERY_TASK_TIMEOUT,
    task_time_limit=settings.CELERY_TASK_TIMEOUT + 60,

    # Resultados
    result_expires=3600,  # 1 hora

    # Windows requiere esto (no tiene fork)
    worker_pool="solo",

    # Timezone
    timezone="America/Lima",
    enable_utc=True,
)