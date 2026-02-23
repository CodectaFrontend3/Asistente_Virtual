"""
FAISS Manager: Gestión del ciclo de vida de índices para Q&A.

Responsabilidades:
- Cargar/guardar índices FAISS + BM25 desde disco
- Indexar documentos de la knowledge base
- Chunking de documentos largos
- Exponer HybridSearchService listo para usar en QAService

Este módulo NO hace búsquedas directamente.
Las búsquedas las hace HybridSearchService.

Flujo de indexación:
    1. load_or_build()  → carga desde disco si existe
    2. Si no existe → build_from_documents() → genera embeddings → indexa
    3. save() → persiste en disco

Flujo de búsqueda (desde QAService):
    hybrid_search.search(query_text, query_embedding, top_k=10)
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from services.embeddings import get_embedding_client, get_embeddings_cache
from services.search import HybridSearchService, get_hybrid_search_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    Documento de la knowledge base antes de chunking.

    Attributes:
        doc_id: Identificador único del documento fuente
        title: Título del documento
        content: Contenido completo
        metadata: Metadatos adicionales (categoría, fuente, etc.)
    """
    doc_id: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """
    Fragmento de documento listo para indexar.

    Attributes:
        chunk_id: ID único del chunk (doc_id + índice)
        doc_id: ID del documento fuente
        title: Título del documento fuente
        content: Texto del chunk
        chunk_index: Posición del chunk dentro del documento
        metadata: Metadatos heredados + chunk-específicos
    """
    chunk_id: str
    doc_id: str
    title: str
    content: str
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexStats:
    """Estadísticas del índice actual."""
    is_ready: bool
    doc_count: int
    chunk_count: int
    bm25_ready: bool
    faiss_ready: bool
    index_path: Optional[str] = None
    build_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# CHUNKER
# ---------------------------------------------------------------------------

class DocumentChunker:
    """
    Divide documentos en chunks para indexación.

    Estrategia: Split por párrafos con overlap para mantener contexto.

    Args:
        max_chunk_size: Caracteres máximos por chunk (default=800)
        min_chunk_size: Mínimo para evitar chunks triviales (default=50)
        overlap_sentences: Líneas de overlap entre chunks (default=1)
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_chunk_size: int = 50,
        overlap_sentences: int = 1,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Divide un documento en chunks.

        Estrategia:
        1. Split por doble salto de línea (párrafos)
        2. Si párrafo > max_chunk_size → split por oración
        3. Agrega overlap de párrafos anteriores

        Args:
            document: Documento a dividir

        Returns:
            Lista de Chunk con IDs únicos
        """
        # Incluir título en el contenido para enriquecer búsqueda
        full_content = f"{document.title}\n\n{document.content}".strip()

        # Split por párrafos (doble salto de línea)
        paragraphs = [p.strip() for p in full_content.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_idx = 0

        for paragraph in paragraphs:
            # Si el párrafo solo cabe en el chunk actual
            if len(current_chunk) + len(paragraph) + 2 <= self.max_chunk_size:
                current_chunk = (current_chunk + "\n\n" + paragraph).strip()
            else:
                # Guardar chunk actual si tiene contenido suficiente
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._make_chunk(
                        document, current_chunk, chunk_idx
                    ))
                    chunk_idx += 1

                # Iniciar nuevo chunk con este párrafo
                # Si el párrafo es muy largo → truncar
                if len(paragraph) > self.max_chunk_size:
                    # Split por oraciones (punto + espacio)
                    sentences = paragraph.replace(". ", ".|").split("|")
                    sentence_chunk = ""
                    for sentence in sentences:
                        if len(sentence_chunk) + len(sentence) <= self.max_chunk_size:
                            sentence_chunk = (sentence_chunk + " " + sentence).strip()
                        else:
                            if len(sentence_chunk) >= self.min_chunk_size:
                                chunks.append(self._make_chunk(
                                    document, sentence_chunk, chunk_idx
                                ))
                                chunk_idx += 1
                            sentence_chunk = sentence
                    current_chunk = sentence_chunk
                else:
                    current_chunk = paragraph

        # Guardar último chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(self._make_chunk(document, current_chunk, chunk_idx))

        logger.debug(
            f"📄 Chunked '{document.title}': "
            f"{len(document.content)} chars → {len(chunks)} chunks"
        )

        return chunks

    def _make_chunk(self, document: Document, content: str, idx: int) -> Chunk:
        """Crea un Chunk desde documento y contenido."""
        chunk_id = f"{document.doc_id}_chunk_{idx}"
        metadata = {**document.metadata, "chunk_index": idx}
        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            title=document.title,
            content=content.strip(),
            chunk_index=idx,
            metadata=metadata,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Divide múltiples documentos en chunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(
            f"📚 Chunked {len(documents)} documentos → {len(all_chunks)} chunks"
        )
        return all_chunks


# ---------------------------------------------------------------------------
# FAISS MANAGER
# ---------------------------------------------------------------------------

class FAISSManager:
    """
    Gestor de índices híbridos (FAISS + BM25) para Q&A.

    Ciclo de vida:
    1. Inicializar FAISSManager
    2. load_indexes() → intentar cargar desde disco
    3. Si no hay índices → build_from_documents()
    4. QAService usa self.hybrid_search para buscar

    Attributes:
        hybrid_search: HybridSearchService listo para búsquedas
        _chunks_meta: Metadatos de chunks indexados (doc_id → Chunk)
        _index_path: Directorio donde se persisten los índices
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        chunk_size: int = 800,
        top_k_candidates: int = 50,
        rrf_k: int = 60,
        hybrid_service: Optional[HybridSearchService] = None,
    ) -> None:
        # Directorio para persistir índices
        self._index_path = index_path or Path("vector_store")

        # Chunker
        self._chunker = DocumentChunker(max_chunk_size=chunk_size)

        # Metadata de chunks para recuperar contexto
        self._chunks_meta: dict[str, Chunk] = {}

        # HybridSearchService (singleton o inyectado)
        self.hybrid_search = hybrid_service or get_hybrid_search_service(
            rrf_k=rrf_k,
            top_k_candidates=top_k_candidates,
        )

        logger.info(
            f"🗂️ FAISSManager inicializado "
            f"(index_path={self._index_path}, chunk_size={chunk_size})"
        )

    # ------------------------------------------------------------------
    # INDEXACIÓN
    # ------------------------------------------------------------------

    async def build_from_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
    ) -> IndexStats:
        """
        Construye índices desde documentos raw.

        Pasos:
        1. Chunking de documentos
        2. Generación de embeddings en batches
        3. Indexación en BM25 + FAISS simultáneo
        4. Guardar metadatos de chunks

        Args:
            documents: Lista de documentos a indexar
            batch_size: Chunks por batch al llamar al embedding service

        Returns:
            IndexStats con resultado de la indexación

        Raises:
            ValueError: Si no hay documentos
            EmbeddingServiceError: Si el embedding service falla
        """
        if not documents:
            raise ValueError("No hay documentos para indexar")

        start_time = time.perf_counter()
        logger.info(f"🏗️ Construyendo índices desde {len(documents)} documentos...")

        # ── CHUNKING ─────────────────────────────────────────────────
        chunks = self._chunker.chunk_documents(documents)

        if not chunks:
            raise ValueError("Chunking no generó ningún chunk")

        logger.info(f"📄 {len(chunks)} chunks generados")

        # ── GENERAR EMBEDDINGS EN BATCHES ─────────────────────────────
        embedding_client = get_embedding_client()
        embeddings_cache = get_embeddings_cache()

        all_embeddings = []
        texts = [chunk.content for chunk in chunks]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(
                f"⚡ Generando embeddings batch {batch_num}/{total_batches} "
                f"({len(batch_texts)} chunks)"
            )

            # Intentar cache primero
            cached_results = embeddings_cache.get_many(batch_texts)
            batch_embeddings = []

            texts_to_embed = []
            texts_to_embed_indices = []

            for idx, text in enumerate(batch_texts):
                cached = cached_results.get(text)
                if cached is not None:
                    batch_embeddings.append((idx, cached))
                else:
                    texts_to_embed.append(text)
                    texts_to_embed_indices.append(idx)

            # Generar los que no están en cache
            if texts_to_embed:
                new_embeddings = await embedding_client.generate_embeddings(
                    texts_to_embed
                )
                # Guardar en cache
                embeddings_cache.set_many(list(zip(texts_to_embed, new_embeddings)))
                # Agregar a batch
                for idx, emb in zip(texts_to_embed_indices, new_embeddings):
                    batch_embeddings.append((idx, emb))

            # Ordenar por índice original y agregar
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb for _, emb in batch_embeddings])

        logger.info(f"✅ {len(all_embeddings)} embeddings generados")

        # ── INDEXAR EN BM25 + FAISS SIMULTÁNEO ────────────────────────
        # Preparar datos para indexación híbrida
        index_docs = [
            (chunk.chunk_id, chunk.content, embedding)
            for chunk, embedding in zip(chunks, all_embeddings)
        ]

        stats = await self.hybrid_search.index_documents(index_docs)

        # ── GUARDAR METADATA DE CHUNKS ─────────────────────────────────
        self._chunks_meta = {chunk.chunk_id: chunk for chunk in chunks}

        elapsed = time.perf_counter() - start_time

        logger.info(
            f"🎉 Indexación completa: {len(chunks)} chunks en {elapsed:.1f}s "
            f"| BM25: {stats['bm25_indexed']} | FAISS: {stats['faiss_indexed']}"
        )

        return IndexStats(
            is_ready=self.hybrid_search.is_ready,
            doc_count=len(documents),
            chunk_count=len(chunks),
            bm25_ready=self.hybrid_search._bm25.is_ready,
            faiss_ready=self.hybrid_search._faiss.is_ready,
            index_path=str(self._index_path),
            build_time_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------

    async def save_indexes(self) -> None:
        """
        Guarda índices FAISS + BM25 + metadata de chunks en disco.

        Crea:
            vector_store/bm25/qa_bm25.pkl
            vector_store/faiss/qa_faiss.bin
            vector_store/faiss/qa_faiss_meta.pkl
            vector_store/chunks_meta.json
        """
        await self.hybrid_search.save_indexes(self._index_path)

        # Guardar metadata de chunks (para recuperar contexto en QAService)
        meta_path = self._index_path / "chunks_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        chunks_data = {
            chunk_id: {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk_id, chunk in self._chunks_meta.items()
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"💾 Metadata guardada: {len(self._chunks_meta)} chunks → {meta_path}"
        )

    async def load_indexes(self) -> bool:
        """
        Carga índices desde disco.

        Returns:
            True si todos los índices cargaron correctamente
        """
        # Cargar índices FAISS + BM25
        loaded = await self.hybrid_search.load_indexes(self._index_path)

        if not loaded:
            return False

        # Cargar metadata de chunks
        meta_path = self._index_path / "chunks_meta.json"
        if not meta_path.exists():
            logger.warning(f"⚠️ chunks_meta.json no encontrado: {meta_path}")
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self._chunks_meta = {
            chunk_id: Chunk(
                chunk_id=data["chunk_id"],
                doc_id=data["doc_id"],
                title=data["title"],
                content=data["content"],
                chunk_index=data["chunk_index"],
                metadata=data["metadata"],
            )
            for chunk_id, data in chunks_data.items()
        }

        logger.info(
            f"📂 Índices cargados: {len(self._chunks_meta)} chunks"
        )
        return True

    async def load_or_build(
        self,
        documents: Optional[List[Document]] = None,
    ) -> IndexStats:
        """
        Carga índices desde disco o los construye si no existen.

        Args:
            documents: Documentos para construir si no hay índices en disco.
                       Requerido si load falla y no hay índices.

        Returns:
            IndexStats del resultado

        Raises:
            ValueError: Si no hay índices en disco y no se proporcionaron docs
        """
        # Intentar cargar primero
        loaded = await self.load_indexes()

        if loaded:
            stats = self.get_stats()
            logger.info(f"✅ Índices cargados desde disco: {stats}")
            return IndexStats(
                is_ready=stats["is_ready"],
                doc_count=0,  # No sabemos cuántos docs originales había
                chunk_count=len(self._chunks_meta),
                bm25_ready=stats["bm25"]["is_ready"],
                faiss_ready=stats["faiss"]["is_ready"],
                index_path=str(self._index_path),
            )

        # Si no pudo cargar, construir desde documentos
        if not documents:
            raise ValueError(
                "No hay índices en disco y no se proporcionaron documentos. "
                "Proporciona documents= para construir índices."
            )

        logger.info("📭 No hay índices en disco → construyendo desde cero...")
        stats = await self.build_from_documents(documents)

        # Guardar para próxima vez
        await self.save_indexes()

        return stats

    # ------------------------------------------------------------------
    # RECUPERACIÓN DE CONTEXTO
    # ------------------------------------------------------------------

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Recupera chunks por sus IDs para construir contexto RAG.

        Args:
            chunk_ids: Lista de chunk_ids de los resultados de búsqueda

        Returns:
            Lista de Chunk en el mismo orden que chunk_ids
            (omite IDs no encontrados)
        """
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self._chunks_meta.get(chunk_id)
            if chunk:
                chunks.append(chunk)
            else:
                logger.warning(f"⚠️ Chunk no encontrado: {chunk_id}")
        return chunks

    # ------------------------------------------------------------------
    # ESTADO
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True si los índices están listos para búsqueda."""
        return self.hybrid_search.is_ready and bool(self._chunks_meta)

    def get_stats(self) -> dict:
        """Retorna estadísticas completas del manager."""
        return {
            "is_ready": self.is_ready,
            "chunks_in_memory": len(self._chunks_meta),
            **self.hybrid_search.get_stats(),
        }


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_faiss_manager: Optional[FAISSManager] = None


def get_faiss_manager(
    index_path: Optional[Path] = None,
) -> FAISSManager:
    """
    Obtiene instancia singleton del FAISSManager.

    Returns:
        FAISSManager singleton
    """
    global _faiss_manager
    if _faiss_manager is None:
        _faiss_manager = FAISSManager(index_path=index_path)
        logger.info("🗂️ FAISSManager singleton creado")
    return _faiss_manager