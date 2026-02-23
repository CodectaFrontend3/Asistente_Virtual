"""
Servicio de generación de embeddings usando Sentence Transformers.

Implementa singleton pattern para cargar el modelo una sola vez.
Optimizado para procesamiento con CPU.

Basado en sentence-transformers 5.2.2 con mejores prácticas:
- Batch processing optimizado
- Normalización automática para cosine similarity
- Manejo de OOM con retry logic
- Progress bars para operaciones largas
- Integración con CPU Manager
"""
import logging
import numpy as np
import torch
from typing import Union, List, Optional
from sentence_transformers import SentenceTransformer
from numpy import ndarray

from config import settings
from .cpu_manager import get_cpu_manager


logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Servicio de generación de embeddings con Sentence Transformers.
    
    Singleton que maneja:
    - Carga del modelo desde HuggingFace
    - Generación de embeddings (single + batch)
    - Normalización automática
    - Gestión de recursos con CPU Manager
    - Manejo de errores y OOM
    
    Example:
        ```python
        from services.embedder import get_embedding_service
        
        embedder = get_embedding_service()
        
        # Single embedding
        emb = embedder.encode_single("Hola mundo")
        print(emb.shape)  # (384,)
        
        # Batch embeddings
        embs = embedder.encode([
            "Primera frase",
            "Segunda frase",
            "Tercera frase"
        ])
        print(embs.shape)  # (3, 384)
        ```
    """
    
    _instance: Optional['EmbeddingService'] = None
    
    def __new__(cls, *args, **kwargs):
        """Implementa singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        Inicializa el servicio de embeddings.
        
        Args:
            model_name: Nombre del modelo de HuggingFace
            max_seq_length: Longitud máxima de secuencia
            device: Dispositivo (cpu/cuda)
        """
        # Evitar re-inicialización
        if hasattr(self, '_initialized'):
            return
        
        # Configuración
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.max_seq_length = max_seq_length or settings.EMBEDDING_MAX_SEQ_LENGTH
        self.device = device
        
        # Inicializar CPU Manager
        self.cpu_manager = get_cpu_manager(
            num_threads=settings.CPU_NUM_THREADS,
            log_enabled=settings.CPU_LOG_USAGE
        )
        
        # Cargar modelo
        self.model = self._load_model()
        
        # Dimensión de embeddings
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        # Validar configuración
        if self.embedding_dimension != settings.EMBEDDING_DIMENSION:
            logger.warning(
                f"⚠️ Dimensión del modelo ({self.embedding_dimension}) "
                f"difiere de configuración ({settings.EMBEDDING_DIMENSION})"
            )
        
        self._initialized = True
    
    def _load_model(self) -> SentenceTransformer:
        """
        Carga el modelo de Sentence Transformers.
        
        Returns:
            Modelo cargado
            
        Raises:
            RuntimeError: Si falla la carga del modelo
        """
        try:
            logger.info("=" * 70)
            logger.info(f"📥 CARGANDO MODELO: {self.model_name}")
            logger.info(f"   Dispositivo: {self.device}")
            logger.info(f"   Max Sequence Length: {self.max_seq_length}")
            
            # Cargar modelo
            model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Configurar max_seq_length
            model.max_seq_length = self.max_seq_length
            
            # Log de información
            dimension = model.get_sentence_embedding_dimension()
            logger.info(f"   Embedding Dimension: {dimension}")
            logger.info(f"✅ Modelo cargado correctamente")
            logger.info("=" * 70)
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {e}")
            raise RuntimeError(f"No se pudo cargar el modelo {self.model_name}: {e}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True
    ) -> Union[ndarray, torch.Tensor]:
        """
        Genera embeddings para uno o más textos.
        
        Args:
            texts: Texto único o lista de textos
            batch_size: Tamaño del batch (None = usar default)
            normalize: Normalizar vectores para cosine similarity
            show_progress_bar: Mostrar barra de progreso
            convert_to_numpy: Convertir a numpy array
        
        Returns:
            Array de embeddings (n_texts, embedding_dim)
            
        Example:
            ```python
            # Single text
            emb = embedder.encode("Hola mundo")
            
            # Multiple texts
            embs = embedder.encode([
                "Primera frase",
                "Segunda frase"
            ], batch_size=16)
            ```
        """
        # Convertir texto único a lista
        if isinstance(texts, str):
            texts = [texts]
        
        # Validar que no esté vacío
        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")
        
        # Validar límite de textos
        if len(texts) > settings.MAX_TEXTS_PER_REQUEST:
            raise ValueError(
                f"Demasiados textos: {len(texts)}. "
                f"Máximo: {settings.MAX_TEXTS_PER_REQUEST}"
            )
        
        # Batch size por defecto
        if batch_size is None:
            batch_size = settings.DEFAULT_BATCH_SIZE
        
        # Validar batch size
        batch_size = min(batch_size, settings.MAX_BATCH_SIZE)
        
        try:
            # Generar embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                device=self.device
            )
            
            # Trackear inferencia
            self.cpu_manager.track_inference()
            
            return embeddings
            
        except RuntimeError as e:
            # Manejo de OOM (Out of Memory)
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"⚠️ OOM detectado con batch_size={batch_size}. "
                    f"Reduciendo batch_size..."
                )
                
                # Retry con batch_size reducido
                new_batch_size = max(1, batch_size // 2)
                return self.encode(
                    texts=texts,
                    batch_size=new_batch_size,
                    normalize=normalize,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=convert_to_numpy
                )
            else:
                # Otro error de runtime
                logger.error(f"Error en encode: {e}")
                raise
        
        except Exception as e:
            logger.error(f"Error inesperado en encode: {e}")
            raise
    
    def encode_single(
        self,
        text: str,
        normalize: bool = True
    ) -> ndarray:
        """
        Genera embedding para un solo texto.
        
        Wrapper conveniente para encode() con un solo texto.
        
        Args:
            text: Texto a codificar
            normalize: Normalizar vector
        
        Returns:
            Embedding como numpy array (embedding_dim,)
            
        Example:
            ```python
            emb = embedder.encode_single("Hola mundo")
            print(emb.shape)  # (384,)
            ```
        """
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
        
        # Encode y retornar primer elemento
        embeddings = self.encode(
            texts=[text],
            batch_size=1,
            normalize=normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings[0]
    
    def batch_encode_with_progress(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        description: str = "Generating embeddings"
    ) -> ndarray:
        """
        Genera embeddings en batch con barra de progreso.
        
        Útil para procesar grandes cantidades de textos
        (ej: construir índices FAISS con miles de documentos).
        
        Args:
            texts: Lista de textos
            batch_size: Tamaño del batch
            normalize: Normalizar vectores
            description: Descripción para progress bar
        
        Returns:
            Array de embeddings (n_texts, embedding_dim)
            
        Example:
            ```python
            texts = ["texto1", "texto2", ..., "texto1000"]
            embs = embedder.batch_encode_with_progress(
                texts,
                batch_size=16,
                description="Building FAISS index"
            )
            ```
        """
        logger.info(f"📊 {description}: {len(texts)} textos")
        
        embeddings = self.encode(
            texts=texts,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=True,  # Mostrar progress bar
            convert_to_numpy=True
        )
        
        logger.info(f"✅ Embeddings generados: {embeddings.shape}")
        
        return embeddings
    
    def get_config(self) -> dict:
        """
        Obtiene configuración del servicio.
        
        Returns:
            Dict con configuración
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
            "default_batch_size": settings.DEFAULT_BATCH_SIZE,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "max_texts_per_request": settings.MAX_TEXTS_PER_REQUEST,
        }
    
    def get_dimension(self) -> int:
        """Retorna la dimensión de los embeddings."""
        return self.embedding_dimension
    
    def clear_cache(self) -> None:
        """Limpia cache de CPU."""
        self.cpu_manager.clear_cache()
        logger.debug("🧹 Cache limpiado")
    
    def get_model_info(self) -> dict:
        """
        Obtiene información detallada del modelo.
        
        Returns:
            Dict con información del modelo
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_seq_length": self.max_seq_length,
            "device": str(self.model.device),
            "num_parameters": sum(
                p.numel() for p in self.model.parameters()
            ),
        }


# ===================================
# SINGLETON GLOBAL
# ===================================

_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service(
    model_name: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    device: str = "cpu"
) -> EmbeddingService:
    """
    Obtiene instancia única del servicio de embeddings.
    
    Args:
        model_name: Nombre del modelo (None = usar config)
        max_seq_length: Longitud máxima (None = usar config)
        device: Dispositivo (cpu/cuda)
    
    Returns:
        Instancia singleton de EmbeddingService
        
    Example:
        ```python
        # Primera llamada: carga modelo
        embedder = get_embedding_service()
        
        # Siguientes llamadas: retorna misma instancia
        embedder2 = get_embedding_service()
        assert embedder is embedder2  # True
        ```
    """
    global _embedding_service_instance
    
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService(
            model_name=model_name,
            max_seq_length=max_seq_length,
            device=device
        )
    
    return _embedding_service_instance