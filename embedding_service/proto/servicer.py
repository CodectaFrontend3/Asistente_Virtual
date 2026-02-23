"""
gRPC Servicer para el Embedding Service.

Implementación async de todos los métodos RPC.
Optimizado para baja latencia (10-20ms).
"""
import logging
import time
import grpc

from config import settings
from services import get_embedding_service, get_cpu_manager


logger = logging.getLogger(__name__)


class EmbeddingServicer:
    """
    Implementación async del servicio de embeddings.

    Todos los métodos son async para maximizar rendimiento.
    Los tipos pb2 se inyectan desde grpc_server.py para evitar
    conflictos de imports en tiempo de carga.
    """

    def __init__(self, pb2):
        """
        Inicializa el servicer.

        Args:
            pb2: Módulo embeddings_pb2 generado (inyectado desde grpc_server.py)
        """
        self.pb2 = pb2
        self.embedder = get_embedding_service()
        self.cpu_manager = get_cpu_manager()
        logger.info("✅ EmbeddingServicer inicializado")

    async def GenerateEmbedding(self, request, context):
        """
        Genera un solo embedding.

        Args:
            request: EmbeddingRequest con texto y normalize
            context: Contexto gRPC
        """
        start_time = time.perf_counter()

        try:
            # Validar texto
            if not request.text or not request.text.strip():
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "El texto no puede estar vacío"
                )
                return

            # Generar embedding
            embedding = self.embedder.encode_single(
                text=request.text,
                normalize=request.normalize
            )

            processing_time = (time.perf_counter() - start_time) * 1000

            logger.debug(f"✅ Embedding gRPC: {processing_time:.2f}ms")

            return self.pb2.EmbeddingResponse(
                embedding=embedding.tolist(),
                dimension=len(embedding),
                device=self.embedder.device,
                processing_time_ms=processing_time
            )

        except ValueError as e:
            logger.error(f"❌ Error de validación: {e}")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error interno: {str(e)}")

    async def GenerateBatch(self, request, context):
        """
        Genera batch de embeddings.

        Args:
            request: BatchRequest con lista de textos
            context: Contexto gRPC
        """
        start_time = time.perf_counter()

        try:
            if not request.texts:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "La lista de textos no puede estar vacía"
                )
                return

            if len(request.texts) > settings.MAX_TEXTS_PER_REQUEST:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Demasiados textos: {len(request.texts)}. "
                    f"Máximo: {settings.MAX_TEXTS_PER_REQUEST}"
                )
                return

            # Generar embeddings
            embeddings_array = self.embedder.encode(
                texts=list(request.texts),
                batch_size=request.batch_size if request.batch_size > 0 else None,
                normalize=request.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Convertir a formato protobuf
            embeddings_list = [
                self.pb2.FloatList(values=emb.tolist())
                for emb in embeddings_array
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"✅ Batch gRPC: {len(request.texts)} textos, "
                f"{processing_time:.2f}ms"
            )

            return self.pb2.BatchResponse(
                embeddings=embeddings_list,
                dimension=embeddings_array.shape[1],
                count=len(embeddings_list),
                device=self.embedder.device,
                processing_time_ms=processing_time
            )

        except ValueError as e:
            logger.error(f"❌ Error de validación: {e}")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        except Exception as e:
            logger.error(f"❌ Error generando batch: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error interno: {str(e)}")

    async def HealthCheck(self, request, context):
        """Health check del servicio."""
        try:
            return self.pb2.HealthResponse(
                status="healthy",
                model_loaded=True,
                model_name=self.embedder.model_name,
                embedding_dimension=self.embedder.embedding_dimension,
                device=self.embedder.device
            )
        except Exception as e:
            logger.error(f"❌ Error en health check: {e}")
            return self.pb2.HealthResponse(
                status="unhealthy",
                model_loaded=False,
                model_name="unknown",
                embedding_dimension=0,
                device="unknown"
            )

    async def GetStats(self, request, context):
        """Obtiene estadísticas del servicio."""
        try:
            stats = self.cpu_manager.get_stats(include_cpu_usage=False)

            return self.pb2.StatsResponse(
                inference_count=self.cpu_manager.get_inference_count(),
                model_name=self.embedder.model_name,
                embedding_dimension=self.embedder.embedding_dimension,
                device=self.embedder.device,
                cpu_count_physical=stats.cpu_count_physical,
                cpu_count_logical=stats.cpu_count_logical,
                ram_total_gb=stats.ram_total_gb,
                ram_used_gb=stats.ram_used_gb,
                ram_percent=stats.ram_percent
            )

        except Exception as e:
            logger.error(f"❌ Error obteniendo stats: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error interno: {str(e)}")