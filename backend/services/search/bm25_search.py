"""
Servicio de búsqueda BM25 léxico.

Usa rank-bm25 0.2.2 con BM25Okapi.

Características:
- BM25Okapi con tokenización NLTK (lowercase + stopwords español)
- Thread-safe con RLock
- Async mediante run_in_executor (BM25 es síncrono internamente)
- Persistencia en disco (pickle)
- Serializa doc_ids junto al índice

Notas:
- BM25Okapi requiere corpus pre-tokenizado (lista de listas de strings)
- get_scores() retorna array numpy con score por doc (mismo orden que corpus)
- NO hace preprocessing propio → lo hacemos nosotros con NLTK
"""
import asyncio
import logging
import pickle
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK: Descargar recursos si no están presentes
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Descarga recursos NLTK necesarios si no están disponibles."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"📥 Descargando NLTK resource: {name}")
            nltk.download(name, quiet=True)


_ensure_nltk_resources()

# Stopwords español + inglés combinadas
try:
    from nltk.corpus import stopwords as _sw
    _STOP_WORDS = set(_sw.words("spanish")) | set(_sw.words("english"))
except Exception:
    _STOP_WORDS = set()

# ---------------------------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------------------------

def tokenize_text(text: str) -> List[str]:
    """
    Tokeniza texto para BM25.

    Pasos:
    1. Lowercase
    2. Eliminar caracteres especiales (mantiene letras, números, espacios)
    3. Split por espacios
    4. Eliminar stopwords y tokens vacíos

    Args:
        text: Texto a tokenizar

    Returns:
        Lista de tokens limpios

    Example:
        >>> tokenize_text("¿Cuánto tarda el envío a Lima?")
        ['cuánto', 'tarda', 'envío', 'lima']
    """
    # Lowercase
    text = text.lower()

    # Eliminar caracteres especiales (conservar letras unicode y números)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Split
    tokens = text.split()

    # Filtrar stopwords y tokens muy cortos
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    return tokens


# ---------------------------------------------------------------------------
# BM25 INDEX
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Índice BM25 con BM25Okapi.

    Thread-safe mediante RLock.

    Attributes:
        _bm25: Instancia BM25Okapi (None si no hay documentos)
        _doc_ids: Lista de IDs originales (str) — mismo orden que corpus
        _lock: RLock para thread-safety

    Example:
        ```python
        index = BM25Index()

        # Indexar documentos
        docs = [
            ("doc_1", "¿Cuánto tarda el envío?"),
            ("doc_2", "Política de devoluciones en 30 días"),
        ]
        index.add_documents(docs)

        # Buscar
        results = index.search("tiempo de envío", top_k=5)
        # → [("doc_1", 2.34), ("doc_2", 0.12)]
        ```
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # INDEXACIÓN
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Tuple[str, str]]) -> int:
        """
        Construye el índice BM25 con los documentos dados.

        Reemplaza el índice anterior si existía.

        Args:
            documents: Lista de (doc_id, text)

        Returns:
            Número de documentos indexados

        Raises:
            ValueError: Si documents está vacío
        """
        if not documents:
            raise ValueError("documents no puede estar vacío")

        doc_ids = [doc_id for doc_id, _ in documents]
        texts = [text for _, text in documents]

        # Tokenizar corpus
        tokenized_corpus = [tokenize_text(text) for text in texts]

        with self._lock:
            self._doc_ids = doc_ids
            self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"📚 BM25 indexado: {len(doc_ids)} documentos")
        return len(doc_ids)

    # ------------------------------------------------------------------
    # BÚSQUEDA
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Busca documentos relevantes por BM25.

        Args:
            query: Texto de la consulta
            top_k: Número máximo de resultados

        Returns:
            Lista de (doc_id, bm25_score) ordenada por score DESC

        Raises:
            RuntimeError: Si el índice no está construido
        """
        with self._lock:
            if self._bm25 is None:
                raise RuntimeError(
                    "BM25 index no está construido. Llama add_documents() primero."
                )
            bm25 = self._bm25
            doc_ids = self._doc_ids.copy()

        # Tokenizar query con el mismo preprocesamiento
        tokenized_query = tokenize_text(query)

        if not tokenized_query:
            logger.warning("⚠️ Query vacía después de tokenizar")
            return []

        # get_scores() retorna array numpy: score por cada doc del corpus
        scores = bm25.get_scores(tokenized_query)

        # Crear lista (doc_id, score) y ordenar
        scored_docs = list(zip(doc_ids, scores.tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Filtrar docs con score 0 y limitar top_k
        results = [(doc_id, score) for doc_id, score in scored_docs[:top_k] if score > 0]

        logger.debug(
            f"🔍 BM25 search: '{query[:50]}' → {len(results)} resultados (top_k={top_k})"
        )

        return results

    # ------------------------------------------------------------------
    # PROPIEDADES
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True si el índice está construido."""
        with self._lock:
            return self._bm25 is not None

    @property
    def doc_count(self) -> int:
        """Número de documentos indexados."""
        with self._lock:
            return len(self._doc_ids)

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Guarda el índice en disco (pickle).

        Args:
            path: Ruta al archivo .pkl

        Raises:
            RuntimeError: Si el índice no está construido
        """
        with self._lock:
            if self._bm25 is None:
                raise RuntimeError("No hay índice que guardar")

            data = {
                "bm25": self._bm25,
                "doc_ids": self._doc_ids,
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"💾 BM25 guardado: {path} ({self.doc_count} docs)")

    def load(self, path: Path) -> bool:
        """
        Carga el índice desde disco.

        Args:
            path: Ruta al archivo .pkl

        Returns:
            True si cargó correctamente, False si el archivo no existe
        """
        if not path.exists():
            logger.warning(f"⚠️ BM25 index no encontrado: {path}")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        with self._lock:
            self._bm25 = data["bm25"]
            self._doc_ids = data["doc_ids"]

        logger.info(f"📂 BM25 cargado: {path} ({self.doc_count} docs)")
        return True

    def get_stats(self) -> dict:
        """Retorna estadísticas del índice."""
        with self._lock:
            return {
                "is_ready": self._bm25 is not None,
                "doc_count": len(self._doc_ids),
                "avgdl": round(self._bm25.avgdl, 2) if self._bm25 else 0,
            }


# ---------------------------------------------------------------------------
# ASYNC WRAPPER
# ---------------------------------------------------------------------------

class BM25SearchService:
    """
    Servicio async de búsqueda BM25.

    Envuelve BM25Index (síncrono) en un executor para no bloquear
    el event loop de asyncio.

    Example:
        ```python
        service = BM25SearchService()

        # Indexar (async)
        await service.add_documents([
            ("doc_1", "Tiempo de envío 3-5 días hábiles"),
            ("doc_2", "Política de devoluciones"),
        ])

        # Buscar (async)
        results = await service.search("cuánto tarda envío", top_k=5)
        ```
    """

    def __init__(self) -> None:
        self._index = BM25Index()
        self._loop = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Obtiene el event loop actual."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def add_documents(self, documents: List[Tuple[str, str]]) -> int:
        """
        Indexa documentos de forma async.

        Args:
            documents: Lista de (doc_id, text)

        Returns:
            Número de documentos indexados
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Default ThreadPoolExecutor
            self._index.add_documents,
            documents,
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Busca de forma async (no bloquea event loop).

        Args:
            query: Texto de búsqueda
            top_k: Máximo de resultados

        Returns:
            Lista de (doc_id, bm25_score)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._index.search,
            query,
            top_k,
        )

    async def save(self, path: Path) -> None:
        """Guarda índice en disco (async)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._index.save, path)

    async def load(self, path: Path) -> bool:
        """Carga índice desde disco (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._index.load, path)

    @property
    def is_ready(self) -> bool:
        return self._index.is_ready

    @property
    def doc_count(self) -> int:
        return self._index.doc_count

    def get_stats(self) -> dict:
        return self._index.get_stats()


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_bm25_service: Optional[BM25SearchService] = None


def get_bm25_service() -> BM25SearchService:
    """
    Obtiene instancia singleton del servicio BM25.

    Returns:
        BM25SearchService singleton
    """
    global _bm25_service
    if _bm25_service is None:
        _bm25_service = BM25SearchService()
        logger.info("🔤 BM25SearchService inicializado")
    return _bm25_service