"""
Servicio de búsqueda vectorial con FAISS.

Usa faiss-cpu 1.13.2 con IndexFlatIP (inner product).

Características:
- IndexFlatIP para similitud coseno (con vectores normalizados 384-dim)
- Thread-safe con RLock
- Async mediante run_in_executor (FAISS CPU es síncrono)
- Persistencia en disco (faiss.write_index / read_index + pickle para metadata)
- Doc IDs mapeados por posición (FAISS maneja índices enteros 0-based)

Notas importantes:
- FAISS IndexFlatIP asume vectores float32 normalizados para cosine similarity
- index.search() retorna (D, I): D=scores, I=índices de posición en el índice
- Los doc_ids reales se mapean desde I usando self._id_map[i]
- Dimensión fija: 384 (paraphrase-multilingual-MiniLM-L12-v2)
"""
import asyncio
import logging
import pickle
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Dimensión del modelo paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# FAISS INDEX
# ---------------------------------------------------------------------------

class FAISSIndex:
    """
    Índice FAISS para búsqueda vectorial por similitud coseno.

    Usa IndexFlatIP (inner product) con vectores normalizados.
    Con normalización L2, inner product = cosine similarity.

    Thread-safe mediante RLock.

    Attributes:
        _index: faiss.IndexFlatIP
        _id_map: Lista de doc_ids en el mismo orden que vectores en el índice
        _lock: RLock para thread-safety

    Example:
        ```python
        index = FAISSIndex()

        # Indexar
        docs = [
            ("doc_1", [0.1, 0.2, ...]),  # embedding normalizado 384-dim
            ("doc_2", [0.3, 0.4, ...]),
        ]
        index.add_vectors(docs)

        # Buscar
        query_embedding = [0.15, 0.25, ...]
        results = index.search(query_embedding, top_k=5)
        # → [("doc_1", 0.98), ("doc_2", 0.75)]
        ```
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self._dimension = dimension
        self._index: Optional[faiss.IndexFlatIP] = None
        self._id_map: List[str] = []  # posición → doc_id real
        self._lock = threading.RLock()

        # Inicializar índice vacío
        self._init_index()

    def _init_index(self) -> None:
        """Inicializa un índice FAISS vacío."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._id_map = []
        logger.debug(f"🔢 FAISS IndexFlatIP({self._dimension}) inicializado")

    # ------------------------------------------------------------------
    # INDEXACIÓN
    # ------------------------------------------------------------------

    def add_vectors(self, documents: List[Tuple[str, List[float]]]) -> int:
        """
        Agrega vectores al índice FAISS.

        REEMPLAZA el índice anterior (reconstruye desde cero).

        Args:
            documents: Lista de (doc_id, embedding)
                       Embedding debe ser lista de floats con dim=384

        Returns:
            Número de vectores indexados

        Raises:
            ValueError: Si documents está vacío o dimensión incorrecta
        """
        if not documents:
            raise ValueError("documents no puede estar vacío")

        doc_ids = [doc_id for doc_id, _ in documents]
        embeddings = [emb for _, emb in documents]

        # Convertir a numpy float32 (requerido por FAISS)
        vectors = np.array(embeddings, dtype=np.float32)

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Dimensión incorrecta: esperado {self._dimension}, "
                f"recibido {vectors.shape[1]}"
            )

        # Normalizar L2 para que inner product = cosine similarity
        faiss.normalize_L2(vectors)

        with self._lock:
            # Resetear índice
            self._index = faiss.IndexFlatIP(self._dimension)
            self._id_map = doc_ids

            # Agregar vectores
            self._index.add(vectors)

        logger.info(
            f"📐 FAISS indexado: {len(doc_ids)} vectores "
            f"(dim={self._dimension})"
        )
        return len(doc_ids)

    # ------------------------------------------------------------------
    # BÚSQUEDA
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Busca vectores más cercanos por similitud coseno.

        Args:
            query_embedding: Embedding de la query (lista de floats, dim=384)
            top_k: Número máximo de resultados

        Returns:
            Lista de (doc_id, cosine_score) ordenada por score DESC

        Raises:
            RuntimeError: Si el índice está vacío
            ValueError: Si la dimensión del query es incorrecta
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                raise RuntimeError(
                    "FAISS index está vacío. Llama add_vectors() primero."
                )
            index = self._index
            id_map = self._id_map.copy()

        # Convertir query a numpy float32 y normalizar
        query_vec = np.array([query_embedding], dtype=np.float32)

        if query_vec.shape[1] != self._dimension:
            raise ValueError(
                f"Dimensión del query incorrecta: esperado {self._dimension}, "
                f"recibido {query_vec.shape[1]}"
            )

        faiss.normalize_L2(query_vec)

        # Ajustar top_k al total disponible
        actual_k = min(top_k, index.ntotal)

        # search() → (D, I): arrays shape (n_queries, k)
        # D = scores (inner product), I = índices en el índice
        distances, indices = index.search(query_vec, actual_k)

        # Mapear índices FAISS → doc_ids reales
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS retorna -1 si no hay suficientes resultados
                continue
            if idx < len(id_map):
                doc_id = id_map[idx]
                results.append((doc_id, float(score)))

        logger.debug(
            f"🔍 FAISS search: {len(results)} resultados (top_k={top_k})"
        )

        return results

    # ------------------------------------------------------------------
    # PROPIEDADES
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True si hay vectores indexados."""
        with self._lock:
            return self._index is not None and self._index.ntotal > 0

    @property
    def vector_count(self) -> int:
        """Número de vectores en el índice."""
        with self._lock:
            return self._index.ntotal if self._index else 0

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------

    def save(self, index_path: Path, meta_path: Path) -> None:
        """
        Guarda índice FAISS + metadata en disco.

        Args:
            index_path: Ruta al archivo .bin del índice FAISS
            meta_path: Ruta al archivo .pkl con doc_ids

        Raises:
            RuntimeError: Si el índice está vacío
        """
        with self._lock:
            if not self.is_ready:
                raise RuntimeError("No hay índice que guardar")
            index = self._index
            id_map = self._id_map.copy()

        # Crear directorios
        index_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar índice FAISS
        faiss.write_index(index, str(index_path))

        # Guardar metadata (doc_ids)
        with open(meta_path, "wb") as f:
            pickle.dump({"id_map": id_map, "dimension": self._dimension}, f)

        logger.info(
            f"💾 FAISS guardado: {index_path} ({self.vector_count} vectores)"
        )

    def load(self, index_path: Path, meta_path: Path) -> bool:
        """
        Carga índice FAISS + metadata desde disco.

        Args:
            index_path: Ruta al archivo .bin
            meta_path: Ruta al archivo .pkl

        Returns:
            True si cargó correctamente
        """
        if not index_path.exists() or not meta_path.exists():
            logger.warning(
                f"⚠️ FAISS files no encontrados: {index_path} / {meta_path}"
            )
            return False

        # Cargar índice FAISS
        loaded_index = faiss.read_index(str(index_path))

        # Cargar metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        with self._lock:
            self._index = loaded_index
            self._id_map = meta["id_map"]
            self._dimension = meta.get("dimension", EMBEDDING_DIM)

        logger.info(
            f"📂 FAISS cargado: {index_path} ({self.vector_count} vectores)"
        )
        return True

    def get_stats(self) -> dict:
        """Retorna estadísticas del índice."""
        with self._lock:
            return {
                "is_ready": self.is_ready,
                "vector_count": self.vector_count,
                "dimension": self._dimension,
                "index_type": "IndexFlatIP",
            }


# ---------------------------------------------------------------------------
# ASYNC WRAPPER
# ---------------------------------------------------------------------------

class VectorSearchService:
    """
    Servicio async de búsqueda vectorial con FAISS.

    Envuelve FAISSIndex (síncrono) en executor para no bloquear asyncio.

    Example:
        ```python
        service = VectorSearchService()

        # Indexar
        await service.add_vectors([
            ("doc_1", embedding_1),
            ("doc_2", embedding_2),
        ])

        # Buscar
        results = await service.search(query_embedding, top_k=10)
        # → [("doc_1", 0.95), ("doc_2", 0.87)]
        ```
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self._index = FAISSIndex(dimension=dimension)

    async def add_vectors(
        self,
        documents: List[Tuple[str, List[float]]],
    ) -> int:
        """
        Indexa vectores de forma async.

        Args:
            documents: Lista de (doc_id, embedding)

        Returns:
            Número de vectores indexados
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._index.add_vectors,
            documents,
        )

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Busca vectores similares de forma async.

        Args:
            query_embedding: Embedding de la query (384-dim)
            top_k: Máximo de resultados

        Returns:
            Lista de (doc_id, cosine_score)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._index.search,
            query_embedding,
            top_k,
        )

    async def save(self, index_path: Path, meta_path: Path) -> None:
        """Guarda índice en disco (async)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._index.save, index_path, meta_path
        )

    async def load(self, index_path: Path, meta_path: Path) -> bool:
        """Carga índice desde disco (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._index.load, index_path, meta_path
        )

    @property
    def is_ready(self) -> bool:
        return self._index.is_ready

    @property
    def vector_count(self) -> int:
        return self._index.vector_count

    def get_stats(self) -> dict:
        return self._index.get_stats()


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_vector_service: Optional[VectorSearchService] = None


def get_vector_service(dimension: int = EMBEDDING_DIM) -> VectorSearchService:
    """
    Obtiene instancia singleton del servicio vectorial FAISS.

    Returns:
        VectorSearchService singleton
    """
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorSearchService(dimension=dimension)
        logger.info(f"📐 VectorSearchService inicializado (dim={dimension})")
    return _vector_service