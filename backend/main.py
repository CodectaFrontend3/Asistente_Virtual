"""
Backend Principal - Entry Point FastAPI.

Inicializa todos los servicios en el orden correcto:
    1. Logging
    2. Redis Cache
    3. Embedding Service (HTTP client)
    4. Índices Q&A (carga desde disco o construye)
    5. LLM Ollama (inyecta en QAService)

Luego levanta el servidor y registra los routers.
"""
import logging
import logging.config
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from api.qa.router import router as qa_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configura logging estructurado para el backend."""
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console"],
        },
        # Silenciar loggers muy verbosos
        "loggers": {
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "uvicorn.access": {"level": "WARNING"},
        },
    })


# ---------------------------------------------------------------------------
# LIFESPAN — Startup y Shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.

    STARTUP (antes del yield):
        1. Redis cache
        2. Embedding HTTP client
        3. Índices Q&A (carga o construye)
        4. LLM Ollama

    SHUTDOWN (después del yield):
        1. Cerrar Redis
        2. Cerrar HTTP client de embeddings
    """
    startup_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("🚀 BACKEND INICIANDO...")
    logger.info("=" * 60)

    # ── 1. REDIS CACHE ────────────────────────────────────────────────
    logger.info("🔴 [1/4] Inicializando Redis cache...")
    try:
        from services.cache import setup_cache
        cache_status = await setup_cache()
        if cache_status["connected"]:
            logger.info(f"✅ Redis: {cache_status['status']} (latency={cache_status.get('latency_ms', 0)}ms)")
        else:
            logger.warning("⚠️ Redis: modo degradado (sin cache)")
    except Exception as e:
        logger.warning(f"⚠️ Redis falló al iniciar: {e} → modo degradado")

    # ── 2. EMBEDDING SERVICE ──────────────────────────────────────────
    logger.info("📡 [2/4] Verificando Embedding Service...")
    try:
        from services.embeddings import get_embedding_client
        client = get_embedding_client()
        health = await client.health_check()
        if health.get("status") == "healthy":
            logger.info(f"✅ Embedding Service: healthy ({settings.embedding_service_http_url})")
        else:
            logger.warning(f"⚠️ Embedding Service: {health}")
    except Exception as e:
        logger.warning(f"⚠️ Embedding Service no disponible: {e}")

    # ── 3. ÍNDICES Q&A ────────────────────────────────────────────────
    logger.info("🗂️ [3/4] Cargando índices Q&A...")
    try:
        from services.qa import get_qa_service
        qa_service = get_qa_service()

        # Intentar cargar desde disco
        # Si no existen, QAService queda en is_ready=False
        # Los índices se construyen manualmente con el script build_qa_index.py
        from services.qa import get_faiss_manager
        faiss_manager = get_faiss_manager()
        loaded = await faiss_manager.load_indexes()

        if loaded:
            stats = faiss_manager.get_stats()
            logger.info(
                f"✅ Índices cargados: "
                f"chunks={stats['chunks_in_memory']}, "
                f"bm25={stats['bm25']['is_ready']}, "
                f"faiss={stats['faiss']['is_ready']}"
            )
        else:
            logger.warning(
                "⚠️ Índices Q&A no encontrados en disco. "
                "Ejecuta el script de indexación para construirlos."
            )
    except Exception as e:
        logger.error(f"❌ Error cargando índices: {e}")

    # ── 4. LLM OLLAMA ─────────────────────────────────────────────────
    logger.info("🦙 [4/4] Conectando con Ollama LLM...")
    try:
        from services.llm import setup_llm
        llm_status = await setup_llm()

        if llm_status["llm_injected"]:
            logger.info(
                f"✅ Ollama: {llm_status['model']} disponible → LLM activado"
            )
        else:
            logger.warning(
                f"⚠️ Ollama no disponible → QAService en modo stub. "
                f"Error: {llm_status.get('error', 'desconocido')}"
            )
    except Exception as e:
        logger.warning(f"⚠️ Ollama falló al iniciar: {e} → modo stub")

    # ── STARTUP COMPLETO ──────────────────────────────────────────────
    startup_time = (time.perf_counter() - startup_start) * 1000
    logger.info("=" * 60)
    logger.info(f"✅ BACKEND LISTO en {startup_time:.0f}ms")
    logger.info(f"   Docs: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"   API:  http://{settings.HOST}:{settings.PORT}/qa/ask")
    logger.info("=" * 60)

    # ── SERVIR REQUESTS ───────────────────────────────────────────────
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────
    logger.info("🛑 Backend apagándose...")

    try:
        from services.cache import get_redis_cache
        cache = get_redis_cache()
        await cache.disconnect()
        logger.info("✅ Redis desconectado")
    except Exception as e:
        logger.warning(f"⚠️ Error cerrando Redis: {e}")

    try:
        from services.embeddings.http_client import close_http_client
        await close_http_client()
        logger.info("✅ HTTP client cerrado")
    except Exception as e:
        logger.warning(f"⚠️ Error cerrando HTTP client: {e}")

    logger.info("👋 Backend apagado correctamente")


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Crea y configura la aplicación FastAPI."""

    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── ROUTERS ───────────────────────────────────────────────────────
    app.include_router(qa_router)

    # ── HEALTH CHECK GLOBAL ───────────────────────────────────────────
    @app.get("/health", tags=["Sistema"])
    async def global_health():
        """Health check global del backend."""
        return {
            "status": "ok",
            "service": settings.APP_TITLE,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/", tags=["Sistema"])
    async def root():
        """Raíz de la API."""
        return {
            "message": "Asistente Virtual - Backend API",
            "docs": "/docs",
            "health": "/health",
            "qa": "/qa/ask",
        }

    return app


# ---------------------------------------------------------------------------
# INSTANCIA GLOBAL
# ---------------------------------------------------------------------------

setup_logging()
app = create_app()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        # Silenciar access logs de uvicorn (usamos el nuestro)
        access_log=False,
    )