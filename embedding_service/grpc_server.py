"""
Servidor gRPC para el Embedding Service.

Usa grpc.aio para máximo rendimiento async.
Los módulos proto se cargan con importlib para evitar
conflictos con el paquete grpcio instalado.
"""
import asyncio
import importlib.util
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import grpc
from grpc import aio

from config import settings
from services import get_embedding_service, get_cpu_manager


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configura logging."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_module_from_file(module_name: str, file_path: Path):
    """
    Carga un módulo Python directamente desde un archivo.

    Usa importlib para cargar sin pasar por __init__.py del paquete,
    evitando conflictos de nombres con paquetes instalados.

    Args:
        module_name: Nombre del módulo
        file_path: Ruta al archivo .py

    Returns:
        Módulo cargado
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # Registrar en sys.modules para que imports internos funcionen
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_proto_modules():
    """
    Carga pb2 y pb2_grpc desde la carpeta proto/.

    Returns:
        Tupla (pb2, pb2_grpc)

    Raises:
        SystemExit: Si los archivos no existen
    """
    project_root = Path(__file__).parent
    proto_dir = project_root / "proto"
    pb2_file = proto_dir / "embeddings_pb2.py"
    pb2_grpc_file = proto_dir / "embeddings_pb2_grpc.py"

    # Verificar existencia
    if not pb2_file.exists() or not pb2_grpc_file.exists():
        logger.error("=" * 70)
        logger.error("❌ ARCHIVOS gRPC NO GENERADOS")
        logger.error("   Ejecuta primero:")
        logger.error("   python scripts/generate_grpc.py")
        logger.error("=" * 70)
        sys.exit(1)

    try:
        # Cargar pb2 (messages)
        pb2 = load_module_from_file("embeddings_pb2", pb2_file)

        # Cargar pb2_grpc (stubs/servicer base)
        pb2_grpc = load_module_from_file("embeddings_pb2_grpc", pb2_grpc_file)

        logger.info("✅ Módulos proto cargados correctamente")
        return pb2, pb2_grpc

    except Exception as e:
        logger.error(f"❌ Error cargando módulos proto: {e}")
        sys.exit(1)


class GrpcServer:
    """
    Servidor gRPC con gestión completa de lifecycle.

    Maneja:
    - Carga de módulos proto sin conflictos
    - Inyección de pb2 al servicer
    - Graceful shutdown
    """

    def __init__(self, port: Optional[int] = None):
        """
        Inicializa el servidor.

        Args:
            port: Puerto gRPC (None = usar settings.GRPC_PORT)
        """
        self.port = port or settings.GRPC_PORT
        self.server: Optional[aio.Server] = None

        # Cargar módulos proto
        self.pb2, self.pb2_grpc = load_proto_modules()

    async def start(self) -> None:
        """Inicia el servidor gRPC."""
        logger.info("=" * 70)
        logger.info("🚀 INICIANDO SERVIDOR gRPC")
        logger.info("=" * 70)

        try:
            # Pre-cargar modelo
            logger.info("📥 Pre-cargando modelo...")
            embedder = get_embedding_service()
            logger.info(f"✅ Modelo cargado: {embedder.model_name}")
            logger.info(f"   Dimensión: {embedder.embedding_dimension}")
            logger.info(f"   Dispositivo: {embedder.device}")

            # Info de CPU/RAM
            cpu_mgr = get_cpu_manager()
            cpu_info = cpu_mgr.get_cpu_info()
            mem_info = cpu_mgr.get_memory_info()
            logger.info(
                f"💻 CPU: {cpu_info['physical_cores']} cores físicos, "
                f"{cpu_info['logical_cores']} lógicos"
            )
            logger.info(
                f"🧠 RAM: {mem_info['total_gb']:.2f} GB total, "
                f"{mem_info['available_gb']:.2f} GB disponible"
            )

            # Crear servidor async
            self.server = aio.server(
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 10000),
                ]
            )

            # Importar servicer e inyectar pb2
            from proto.servicer import EmbeddingServicer
            servicer = EmbeddingServicer(pb2=self.pb2)

            # Registrar servicer en el servidor
            self.pb2_grpc.add_EmbeddingServiceServicer_to_server(
                servicer,
                self.server
            )

            # Abrir puerto sin TLS (desarrollo)
            listen_addr = f'[::]:{self.port}'
            self.server.add_insecure_port(listen_addr)

            # Iniciar
            await self.server.start()

            logger.info("=" * 70)
            logger.info(f"✅ SERVIDOR gRPC LISTO EN {listen_addr}")
            logger.info(f"   Puerto: {self.port}")
            logger.info(f"   Modelo: {embedder.model_name}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"❌ Error iniciando servidor gRPC: {e}")
            raise

    async def wait_for_termination(self) -> None:
        """Espera señal de terminación."""
        if self.server:
            await self.server.wait_for_termination()

    async def stop(self, grace: float = 5.0) -> None:
        """Detiene el servidor gracefully."""
        logger.info("=" * 70)
        logger.info("🛑 CERRANDO SERVIDOR gRPC")
        logger.info("=" * 70)

        if self.server:
            try:
                await self.server.stop(grace=grace)

                # Limpieza de recursos
                get_embedding_service().clear_cache()
                logger.info("🧹 Recursos limpiados")

                # Stats finales
                count = get_cpu_manager().get_inference_count()
                logger.info(f"📊 Total de inferencias: {count}")

            except Exception as e:
                logger.error(f"⚠️ Error en shutdown: {e}")

        logger.info("👋 Servidor gRPC cerrado")


async def main() -> None:
    """Función principal del servidor gRPC."""
    setup_logging()

    server = GrpcServer()
    loop = asyncio.get_event_loop()

    # Signal handlers (Ctrl+C)
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(server.stop())
            )
    except NotImplementedError:
        # Windows no soporta todos los signals
        pass

    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("⚠️ Interrupción por teclado (Ctrl+C)")
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        sys.exit(1)