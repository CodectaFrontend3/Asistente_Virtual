"""
Cliente HTTP para el Embedding Service.

Características:
- httpx.AsyncClient reutilizable (connection pooling)
- Retry automático con exponential backoff
- Circuit breaker integrado
- Timeouts configurados (connect/read/write)
- Cache local LRU
- Manejo de errores personalizado
"""
import asyncio
import logging
from typing import List, Optional

import httpx
import pybreaker

from config import settings
from core import (
    embedding_service_breaker,
    EmbeddingServiceError,
    TimeoutError as BackendTimeoutError,
    CircuitBreakerOpenError,
)


logger = logging.getLogger(__name__)


# ===================================
# SINGLETON ASYNC CLIENT
# ===================================

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """
    Obtiene cliente HTTP singleton.
    
    Se crea una sola vez y se reutiliza para todas las requests.
    Esto aprovecha connection pooling y reduce overhead de TLS.
    
    Returns:
        httpx.AsyncClient configurado
    
    Note:
        El cliente NO debe cerrarse manualmente.
        Se cierra automáticamente en app shutdown (lifespan).
    """
    global _http_client
    
    if _http_client is None:
        # Timeout granular por fase
        timeout = httpx.Timeout(
            connect=5.0,      # Tiempo para establecer conexión
            read=30.0,        # Tiempo para recibir respuesta
            write=10.0,       # Tiempo para enviar request
            pool=5.0,         # Tiempo para obtener conexión del pool
        )
        
        # Límites de conexiones
        limits = httpx.Limits(
            max_connections=100,            # Total de conexiones
            max_keepalive_connections=20,   # Keepalive connections
            keepalive_expiry=30.0,          # Segundos antes de cerrar keepalive
        )
        
        _http_client = httpx.AsyncClient(
            base_url=settings.embedding_service_http_url,
            timeout=timeout,
            limits=limits,
            http2=True,  # HTTP/2 para mejor rendimiento
        )
        
        logger.info(
            f"🌐 HTTP Client creado - "
            f"base_url={settings.embedding_service_http_url}"
        )
    
    return _http_client


async def close_http_client() -> None:
    """
    Cierra el cliente HTTP.
    
    Se debe llamar en app shutdown.
    """
    global _http_client
    
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        logger.info("🌐 HTTP Client cerrado")


# ===================================
# EMBEDDING CLIENT
# ===================================

class EmbeddingHTTPClient:
    """
    Cliente HTTP para generar embeddings.
    
    Incluye:
    - Retry automático con exponential backoff
    - Circuit breaker
    - Cache local
    - Error handling
    """
    
    def __init__(self):
        """Inicializa el cliente."""
        self.base_url = settings.embedding_service_http_url
        self.timeout = settings.EMBEDDING_SERVICE_TIMEOUT
        
        # Retry config
        self.max_retries = 3
        self.backoff_factor = 2  # 1s, 2s, 4s
        
        logger.info("📡 EmbeddingHTTPClient inicializado")
    
    @embedding_service_breaker
    async def generate_embeddings(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de textos
            normalize: Si normalizar embeddings (cosine similarity)
            batch_size: Tamaño del batch (None = default del servicio)
        
        Returns:
            Lista de embeddings (lista de floats)
        
        Raises:
            EmbeddingServiceError: Error al generar embeddings
            CircuitBreakerOpenError: Circuit breaker abierto
            BackendTimeoutError: Timeout esperando respuesta
        
        Example:
            ```python
            client = EmbeddingHTTPClient()
            embeddings = await client.generate_embeddings([
                "¿Cuánto tarda el envío?",
                "¿Puedo devolver?"
            ])
            print(len(embeddings))  # 2
            print(len(embeddings[0]))  # 384
            ```
        """
        # Preparar request payload
        payload = {
            "texts": texts,
            "normalize": normalize,
        }
        if batch_size is not None:
            payload["batch_size"] = batch_size
        
        # Retry loop con exponential backoff
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                client = get_http_client()
                
                response = await client.post(
                    "/embeddings",
                    json=payload,
                    timeout=self.timeout
                )
                
                # Verificar status
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                embeddings = data["embeddings"]
                
                logger.info(
                    f"✅ Embeddings generados via HTTP - "
                    f"{len(embeddings)} textos, "
                    f"{data.get('processing_time_ms', 0):.2f}ms"
                )
                
                return embeddings
            
            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"⏱️ Timeout en intento {attempt}/{self.max_retries}: {e}"
                )
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** (attempt - 1)
                    logger.info(f"♻️ Reintentando en {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise BackendTimeoutError(
                        "embedding_service",
                        self.timeout
                    )
            
            except httpx.HTTPStatusError as e:
                # Errores HTTP no se reintentan (4xx, 5xx)
                logger.error(
                    f"❌ HTTP Error {e.response.status_code}: {e.response.text}"
                )
                raise EmbeddingServiceError(
                    f"HTTP {e.response.status_code}: {e.response.text}"
                )
            
            except httpx.ConnectError as e:
                last_exception = e
                logger.warning(
                    f"🔌 Error de conexión en intento {attempt}/{self.max_retries}: {e}"
                )
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** (attempt - 1)
                    logger.info(f"♻️ Reintentando en {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise EmbeddingServiceError(
                        f"No se pudo conectar a Embedding Service: {e}"
                    )
            
            except pybreaker.CircuitBreakerError:
                # Circuit breaker abierto
                raise CircuitBreakerOpenError("embedding_service")
            
            except Exception as e:
                logger.error(f"❌ Error inesperado: {e}")
                raise EmbeddingServiceError(str(e))
        
        # Si llegamos aquí, todos los intentos fallaron
        if last_exception:
            raise EmbeddingServiceError(
                f"Falló después de {self.max_retries} intentos: {last_exception}"
            )
    
    async def generate_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> List[float]:
        """
        Genera un solo embedding.
        
        Wrapper conveniente para un solo texto.
        
        Args:
            text: Texto único
            normalize: Si normalizar embedding
        
        Returns:
            Embedding (lista de floats)
        
        Example:
            ```python
            client = EmbeddingHTTPClient()
            embedding = await client.generate_embedding("Hola mundo")
            print(len(embedding))  # 384
            ```
        """
        embeddings = await self.generate_embeddings([text], normalize)
        return embeddings[0]
    
    async def health_check(self) -> dict:
        """
        Verifica salud del Embedding Service.
        
        Returns:
            Dict con estado del servicio
        
        Example:
            ```python
            client = EmbeddingHTTPClient()
            status = await client.health_check()
            print(status["status"])  # "healthy"
            ```
        """
        try:
            client = get_http_client()
            response = await client.get("/embeddings/health", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Health check falló: {e}")
            return {"status": "unhealthy", "error": str(e)}


# ===================================
# SINGLETON INSTANCE
# ===================================

_embedding_client: Optional[EmbeddingHTTPClient] = None


def get_embedding_client() -> EmbeddingHTTPClient:
    """
    Obtiene instancia singleton del cliente embeddings.
    
    Returns:
        EmbeddingHTTPClient
    
    Example:
        ```python
        from services.embeddings import get_embedding_client
        
        client = get_embedding_client()
        embedding = await client.generate_embedding("Texto de prueba")
        ```
    """
    global _embedding_client
    
    if _embedding_client is None:
        _embedding_client = EmbeddingHTTPClient()
    
    return _embedding_client