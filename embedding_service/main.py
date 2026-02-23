"""
Entry point del Embedding Service.

FastAPI application con:
- Configuración de logging
- Lifespan events (startup/shutdown)
- Middlewares (CORS, timing, logging)
- Router de embeddings
- Documentación automática
"""
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import settings
from api import router
from services import get_embedding_service, get_cpu_manager


# ===================================
# CONFIGURACIÓN DE LOGGING
# ===================================

def setup_logging() -> None:
    """
    Configura el sistema de logging.
    
    Formato simple para desarrollo con:
    - Timestamp
    - Nivel
    - Logger name
    - Mensaje
    """
    # Formato de logging
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configuración básica
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Silenciar logs muy verbosos de librerías
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    
    # Logger propio
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info(f"🚀 LOGGING CONFIGURADO - Nivel: {settings.LOG_LEVEL}")
    logger.info("=" * 70)


# Configurar logging al importar
setup_logging()
logger = logging.getLogger(__name__)


# ===================================
# LIFESPAN EVENTS
# ===================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Maneja eventos de inicio y cierre del servidor.
    
    Startup:
    - Pre-carga el modelo de embeddings
    - Inicializa CPU manager
    - Log de información del sistema
    
    Shutdown:
    - Limpieza de recursos
    """
    # STARTUP
    logger.info("=" * 70)
    logger.info("🚀 INICIANDO EMBEDDING SERVICE")
    logger.info("=" * 70)
    
    try:
        # Pre-cargar modelo
        logger.info("📥 Pre-cargando modelo de embeddings...")
        embedder = get_embedding_service()
        logger.info(f"✅ Modelo cargado: {embedder.model_name}")
        logger.info(f"   Dimensión: {embedder.embedding_dimension}")
        logger.info(f"   Dispositivo: {embedder.device}")
        
        # Inicializar CPU manager
        cpu_mgr = get_cpu_manager()
        cpu_info = cpu_mgr.get_cpu_info()
        logger.info(f"💻 CPU: {cpu_info['physical_cores']} cores físicos, "
                   f"{cpu_info['logical_cores']} lógicos")
        
        mem_info = cpu_mgr.get_memory_info()
        logger.info(f"🧠 RAM: {mem_info['total_gb']:.2f} GB total, "
                   f"{mem_info['available_gb']:.2f} GB disponible")
        
        logger.info("=" * 70)
        logger.info(f"✅ SERVIDOR LISTO EN http://{settings.HOST}:{settings.PORT}")
        logger.info(f"📖 Docs: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Error en startup: {e}")
        raise
    
    # Yield para que la app corra
    yield
    
    # SHUTDOWN
    logger.info("=" * 70)
    logger.info("🛑 CERRANDO EMBEDDING SERVICE")
    logger.info("=" * 70)
    
    try:
        # Limpieza de recursos
        embedder = get_embedding_service()
        embedder.clear_cache()
        logger.info("🧹 Recursos limpiados")
        
        # Stats finales
        cpu_mgr = get_cpu_manager()
        inference_count = cpu_mgr.get_inference_count()
        logger.info(f"📊 Total de inferencias: {inference_count}")
        
    except Exception as e:
        logger.error(f"⚠️ Error en shutdown: {e}")
    
    logger.info("👋 Servidor cerrado")


# ===================================
# CREAR APP
# ===================================

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ===================================
# MIDDLEWARES
# ===================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware de timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Agrega header con tiempo de procesamiento."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000  # ms
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response


# Middleware de logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log de todas las requests."""
    start_time = time.perf_counter()
    
    # Log de request
    logger.info(f"→ {request.method} {request.url.path}")
    
    # Procesar request
    try:
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        
        # Log de response
        logger.info(
            f"← {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error procesando request: {e}")
        raise


# ===================================
# EXCEPTION HANDLERS
# ===================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global para excepciones no capturadas."""
    logger.error(f"❌ Excepción no capturada: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor",
            "error": str(exc) if settings.is_development() else "Internal server error"
        }
    )


# ===================================
# ROUTERS
# ===================================

# Incluir router de embeddings
app.include_router(router)


# ===================================
# ROOT ENDPOINT
# ===================================

@app.get(
    "/",
    tags=["root"],
    summary="Root endpoint",
    description="Información básica del servicio"
)
async def root():
    """
    Endpoint raíz con información del servicio.
    
    Returns:
        Información básica
    """
    return {
        "service": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": f"http://{settings.HOST}:{settings.PORT}/docs",
        "endpoints": {
            "embeddings": "/embeddings",
            "health": "/embeddings/health",
            "stats": "/embeddings/stats",
        }
    }


# ===================================
# ENTRY POINT
# ===================================

if __name__ == "__main__":
    """
    Ejecuta el servidor con Uvicorn.
    
    Uso:
        python main.py
    """
    logger.info("🚀 Iniciando servidor Uvicorn...")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD and settings.is_development(),
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )