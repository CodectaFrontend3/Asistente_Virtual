"""
Script para construir los índices Q&A desde la knowledge base real.

Lee automáticamente todos los archivos .md y .json de knowledge_base/
y los indexa en FAISS + BM25.

Ejecutar UNA VEZ (o cuando cambies el contenido):
    python build_index.py

Requiere:
    - Embedding Service corriendo en :8001
    - knowledge_base/ con tus archivos md/json
"""
import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# READERS
# ---------------------------------------------------------------------------

def read_markdown(path: Path) -> str:
    """Lee un archivo .md y retorna su contenido."""
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> str:
    """
    Lee un archivo .json y lo convierte a texto legible.

    Soporta:
    - Lista de objetos: [{"pregunta": "...", "respuesta": "..."}]
    - Dict con clave "items", "faqs", "data", etc.
    - Cualquier estructura → se serializa bonito
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # Si es una lista de dicts con pregunta/respuesta (formato FAQ)
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                # Intentar extraer pregunta y respuesta
                pregunta = (
                    item.get("pregunta") or item.get("question") or
                    item.get("titulo") or item.get("title") or ""
                )
                respuesta = (
                    item.get("respuesta") or item.get("answer") or
                    item.get("contenido") or item.get("content") or
                    item.get("descripcion") or item.get("description") or ""
                )
                if pregunta and respuesta:
                    parts.append(f"Pregunta: {pregunta}\nRespuesta: {respuesta}")
                elif respuesta:
                    parts.append(respuesta)
                else:
                    # Serializar el dict completo
                    parts.append(json.dumps(item, ensure_ascii=False, indent=2))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)

    # Si es un dict
    if isinstance(data, dict):
        # Buscar la lista principal
        for key in ["items", "faqs", "data", "preguntas", "entries", "content"]:
            if key in data and isinstance(data[key], list):
                return read_json_list(data[key])
        # Serializar el dict completo
        return json.dumps(data, ensure_ascii=False, indent=2)

    return str(data)


def read_json_list(items: list) -> str:
    """Convierte lista de items a texto."""
    parts = []
    for item in items:
        if isinstance(item, dict):
            pregunta = (
                item.get("pregunta") or item.get("question") or
                item.get("titulo") or item.get("title") or ""
            )
            respuesta = (
                item.get("respuesta") or item.get("answer") or
                item.get("contenido") or item.get("content") or ""
            )
            if pregunta and respuesta:
                parts.append(f"Pregunta: {pregunta}\nRespuesta: {respuesta}")
            elif respuesta:
                parts.append(respuesta)
        else:
            parts.append(str(item))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CARGADOR DE KNOWLEDGE BASE
# ---------------------------------------------------------------------------

def load_knowledge_base(kb_path: Path) -> list:
    """
    Carga todos los archivos .md y .json de knowledge_base/.

    Estructura esperada (la tuya):
        knowledge_base/
        ├── company/        → contexto.md, horarios.md
        ├── contacts/       → contactos.json
        ├── faq/            → devoluciones.json, envios.json, ...
        ├── policies/       → devoluciones.md, garantias.md, ...
        └── support/        → configuracion.md, diagnostico.md, ...

    Returns:
        Lista de dicts con doc_id, title, content, metadata
    """
    from services.qa import Document

    documents = []

    if not kb_path.exists():
        logger.error(f"❌ knowledge_base no encontrada: {kb_path}")
        return documents

    # Recorrer todos los archivos recursivamente
    all_files = list(kb_path.rglob("*.md")) + list(kb_path.rglob("*.json"))
    all_files.sort()  # Orden consistente

    logger.info(f"📁 Encontrados {len(all_files)} archivos en {kb_path}")

    for file_path in all_files:
        try:
            # Generar doc_id desde ruta relativa
            relative = file_path.relative_to(kb_path)
            doc_id = str(relative).replace("\\", "_").replace("/", "_").replace(".", "_")

            # Título desde nombre del archivo (sin extensión, capitalizado)
            stem = file_path.stem.replace("_", " ").replace("-", " ").title()
            category = file_path.parent.name.title()
            title = f"{category} - {stem}"

            # Leer contenido según tipo
            if file_path.suffix == ".md":
                content = read_markdown(file_path)
            elif file_path.suffix == ".json":
                content = read_json(file_path)
            else:
                continue

            # Saltar archivos vacíos
            if not content.strip():
                logger.warning(f"⚠️ Archivo vacío, saltando: {relative}")
                continue

            doc = Document(
                doc_id=doc_id,
                title=title,
                content=content,
                metadata={
                    "source_file": str(relative),
                    "category": file_path.parent.name,
                    "file_type": file_path.suffix,
                },
            )
            documents.append(doc)

            logger.info(
                f"  ✅ {relative} → "
                f"{len(content)} chars"
            )

        except Exception as e:
            logger.error(f"  ❌ Error leyendo {file_path}: {e}")

    return documents


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

async def main():
    logger.info("=" * 55)
    logger.info("🏗️  CONSTRUYENDO ÍNDICES Q&A")
    logger.info("=" * 55)

    # knowledge_base está UN NIVEL ARRIBA de backend/
    kb_path = Path(__file__).parent.parent / "knowledge_base"

    # 1. Cargar documentos
    logger.info(f"\n📚 Cargando knowledge base desde: {kb_path.absolute()}")
    documents = load_knowledge_base(kb_path)

    if not documents:
        logger.error(
            "❌ No se encontraron documentos. "
            "Verifica que knowledge_base/ exista y tenga archivos .md/.json"
        )
        return

    logger.info(f"\n✅ {len(documents)} documentos cargados")

    # 2. Construir índices
    logger.info("\n⚡ Construyendo índices (esto puede tardar ~30s)...")
    logger.info("   Requiere Embedding Service corriendo en :8001")

    from services.qa import get_qa_service, get_faiss_manager

    # Limpiar singletons para forzar reconstrucción
    import services.qa.faiss_manager as fm_module
    import services.qa.qa_service as qs_module
    import services.search.bm25_search as bm25_module
    import services.search.vector_search as vs_module
    import services.search.hybrid_search as hs_module

    fm_module._faiss_manager = None
    qs_module._qa_service = None
    bm25_module._bm25_service = None
    vs_module._vector_service = None
    hs_module._hybrid_search_service = None

    faiss_manager = get_faiss_manager()
    stats = await faiss_manager.build_from_documents(documents)

    # 3. Guardar en disco
    logger.info("\n💾 Guardando índices en disco...")
    await faiss_manager.save_indexes()

    # 4. Resumen
    logger.info("\n" + "=" * 55)
    logger.info("✅ ÍNDICES CONSTRUIDOS EXITOSAMENTE")
    logger.info("=" * 55)
    logger.info(f"   Documentos:    {len(documents)}")
    logger.info(f"   Chunks:        {stats.chunk_count}")
    logger.info(f"   BM25 listo:    {stats.bm25_ready}")
    logger.info(f"   FAISS listo:   {stats.faiss_ready}")
    logger.info(f"   Tiempo:        {stats.build_time_seconds:.1f}s")
    logger.info(f"   Guardado en:   {faiss_manager._index_path.absolute()}")
    logger.info("\n🚀 Reinicia el backend para usar los nuevos índices:")
    logger.info("   python main.py")


if __name__ == "__main__":
    asyncio.run(main())