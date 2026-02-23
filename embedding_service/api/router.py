"""
API Router para el Embedding Service.

Endpoints HTTP REST:
- POST /embeddings - Generar embeddings
- GET /embeddings/health - Health check
- POST /embeddings/clear-cache - Limpiar cache
- GET /embeddings/stats - Estadísticas del servicio
"""
import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config import settings
from services import get_embedding_service, get_cpu_manager


logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ===================================
# SCHEMAS
# ===================================

class EmbeddingRequest(BaseModel):
    """Request para generar embeddings."""
    
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_TEXTS_PER_REQUEST,
        description="Lista de textos para generar embeddings"
    )
    
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=settings.MAX_BATCH_SIZE,
        description=f"Tamaño del batch (default: {settings.DEFAULT_BATCH_SIZE})"
    )
    
    normalize: bool = Field(
        default=True,
        description="Normalizar embeddings para cosine similarity"
    )
    
    @field_validator("texts")
    @classmethod
    def validate_texts_not_empty(cls, v: List[str]) -> List[str]:
        """Validar que los textos no estén vacíos."""
        if not v:
            raise ValueError("La lista de textos no puede estar vacía")
        
        # Validar que cada texto no esté vacío
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"El texto en posición {i} está vacío")
        
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": [
                        "¿Cuánto tarda el envío?",
                        "¿Puedo devolver un producto?"
                    ],
                    "batch_size": 16,
                    "normalize": True
                }
            ]
        }
    }


class EmbeddingResponse(BaseModel):
    """Response con embeddings generados."""
    
    embeddings: List[List[float]] = Field(
        ...,
        description="Lista de embeddings (vectores)"
    )
    
    dimension: int = Field(
        ...,
        description="Dimensión de cada embedding"
    )
    
    count: int = Field(
        ...,
        description="Número de embeddings generados"
    )
    
    device: str = Field(
        ...,
        description="Dispositivo usado (cpu/cuda)"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Tiempo de procesamiento en milisegundos"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "embeddings": [[0.123, 0.456, -0.789, "..."], ["..."]],
                    "dimension": 384,
                    "count": 2,
                    "device": "cpu",
                    "processing_time_ms": 125.34
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response de health check."""
    
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    model_name: str = Field(..., description="Nombre del modelo")
    embedding_dimension: int = Field(..., description="Dimensión de embeddings")
    device: str = Field(..., description="Dispositivo (cpu/cuda)")
    cpu_percent: Optional[float] = Field(None, description="Uso de CPU (%)")
    ram_used_gb: Optional[float] = Field(None, description="RAM usada (GB)")
    ram_total_gb: Optional[float] = Field(None, description="RAM total (GB)")
    ram_percent: Optional[float] = Field(None, description="RAM usada (%)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                    "embedding_dimension": 384,
                    "device": "cpu",
                    "cpu_percent": 15.2,
                    "ram_used_gb": 8.5,
                    "ram_total_gb": 16.0,
                    "ram_percent": 53.1
                }
            ]
        }
    }


class StatsResponse(BaseModel):
    """Response con estadísticas del servicio."""
    
    inference_count: int = Field(..., description="Número de inferencias")
    embedding_config: dict = Field(..., description="Configuración del modelo")
    cpu_info: dict = Field(..., description="Información de CPU")
    memory_info: dict = Field(..., description="Información de memoria")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "inference_count": 42,
                    "embedding_config": {
                        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                        "embedding_dimension": 384
                    },
                    "cpu_info": {
                        "physical_cores": 4,
                        "logical_cores": 8
                    },
                    "memory_info": {
                        "total_gb": 16.0,
                        "used_gb": 8.5
                    }
                }
            ]
        }
    }


class ClearCacheResponse(BaseModel):
    """Response de limpieza de cache."""
    
    status: str = Field(..., description="Estado de la operación")
    message: str = Field(..., description="Mensaje descriptivo")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "message": "Cache limpiado correctamente"
                }
            ]
        }
    }


# ===================================
# ENDPOINTS
# ===================================

@router.post(
    "",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generar embeddings",
    description="Genera embeddings para una lista de textos usando el modelo configurado"
)
async def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Genera embeddings para textos.
    
    Args:
        request: Request con textos y parámetros
    
    Returns:
        Response con embeddings generados
        
    Raises:
        HTTPException: Si hay error en la generación
    """
    start_time = time.perf_counter()
    
    try:
        # Obtener servicio
        embedder = get_embedding_service()
        
        # Generar embeddings
        embeddings_array = embedder.encode(
            texts=request.texts,
            batch_size=request.batch_size,
            normalize=request.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convertir a lista de listas (JSON serializable)
        embeddings_list = embeddings_array.tolist()
        
        # Tiempo de procesamiento
        processing_time = (time.perf_counter() - start_time) * 1000  # ms
        
        logger.info(
            f"✅ Embeddings generados: {len(request.texts)} textos, "
            f"{processing_time:.2f}ms"
        )
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimension=embeddings_array.shape[1],
            count=len(embeddings_list),
            device=embedder.device,
            processing_time_ms=round(processing_time, 2)
        )
        
    except ValueError as e:
        logger.error(f"❌ Error de validación: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"❌ Error al generar embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al generar embeddings: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Verifica el estado del servicio y recursos del sistema"
)
async def health_check() -> HealthResponse:
    """
    Health check del servicio.
    
    Returns:
        Estado del servicio y recursos
    """
    try:
        # Obtener servicio de embeddings
        embedder = get_embedding_service()
        
        # Obtener CPU manager
        cpu_mgr = get_cpu_manager()
        
        # Stats de CPU/RAM (sin bloquear)
        stats = cpu_mgr.get_stats(include_cpu_usage=False)
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=embedder.model_name,
            embedding_dimension=embedder.embedding_dimension,
            device=embedder.device,
            cpu_percent=stats.cpu_percent,
            ram_used_gb=stats.ram_used_gb,
            ram_total_gb=stats.ram_total_gb,
            ram_percent=stats.ram_percent
        )
        
    except Exception as e:
        logger.error(f"❌ Error en health check: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name="unknown",
            embedding_dimension=0,
            device="unknown"
        )


@router.post(
    "/clear-cache",
    response_model=ClearCacheResponse,
    status_code=status.HTTP_200_OK,
    summary="Limpiar cache",
    description="Limpia el cache de CPU (garbage collection)"
)
async def clear_cache() -> ClearCacheResponse:
    """
    Limpia cache de CPU.
    
    Returns:
        Confirmación de limpieza
    """
    try:
        embedder = get_embedding_service()
        embedder.clear_cache()
        
        logger.info("🧹 Cache limpiado manualmente")
        
        return ClearCacheResponse(
            status="success",
            message="Cache limpiado correctamente"
        )
        
    except Exception as e:
        logger.error(f"❌ Error al limpiar cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al limpiar cache: {str(e)}"
        )


@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Estadísticas",
    description="Obtiene estadísticas detalladas del servicio"
)
async def get_stats() -> StatsResponse:
    """
    Obtiene estadísticas del servicio.
    
    Returns:
        Estadísticas completas
    """
    try:
        embedder = get_embedding_service()
        cpu_mgr = get_cpu_manager()
        
        return StatsResponse(
            inference_count=cpu_mgr.get_inference_count(),
            embedding_config=embedder.get_config(),
            cpu_info=cpu_mgr.get_cpu_info(),
            memory_info=cpu_mgr.get_memory_info()
        )
        
    except Exception as e:
        logger.error(f"❌ Error al obtener stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al obtener estadísticas: {str(e)}"
        )