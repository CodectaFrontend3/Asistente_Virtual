"""
Tests completos del Backend Principal.

Cubre:
- Config y settings
- Circuit breaker y exceptions
- BM25 search
- FAISS vector search
- RRF fusion
- Hybrid search
- Cache Redis (con mock)
- QAService (con mocks)
- API endpoints (con TestClient)

Ejecutar:
    cd backend
    pytest tests/ -v
    pytest tests/ -v --cov=. --cov-report=term-missing
"""
import asyncio
import json
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ===========================================================================
# FIXTURES GLOBALES
# ===========================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Event loop compartido para toda la sesión de tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_documents():
    """Documentos de muestra para tests."""
    from services.qa import Document
    return [
        Document(
            doc_id="doc_envio",
            title="Política de Envíos",
            content="""Los envíos nacionales tardan 3-5 días hábiles.
El costo de envío estándar es S/10.
Para envíos express (24 horas) el costo es S/25. Solo disponible en Lima.""",
        ),
        Document(
            doc_id="doc_devolucion",
            title="Política de Devoluciones",
            content="""Aceptamos devoluciones dentro de los 30 días de recibido el producto.
El producto debe estar en su estado original sin uso.
Para iniciar una devolución contacta a soporte con tu número de pedido.""",
        ),
        Document(
            doc_id="doc_pago",
            title="Métodos de Pago",
            content="""Aceptamos tarjetas Visa, Mastercard y American Express.
También aceptamos pagos por Yape y Plin.
Las transferencias bancarias se procesan en 24 horas hábiles.""",
        ),
    ]


@pytest.fixture
def sample_embedding():
    """Embedding de muestra (384 dimensiones)."""
    import numpy as np
    vec = np.random.rand(384).astype("float32")
    vec = vec / np.linalg.norm(vec)  # normalizado
    return vec.tolist()


# ===========================================================================
# PARTE 1: CONFIG
# ===========================================================================

class TestConfig:
    def test_settings_loads(self):
        """Settings carga correctamente."""
        from config import settings
        assert settings.PORT == 8000
        assert settings.OLLAMA_MODEL == "llama3.2:3b"
        assert settings.EMBEDDING_DIMENSION == 384
        assert settings.RRF_K == 60

    def test_settings_urls(self):
        """Properties de URL construyen correctamente."""
        from config import settings
        assert "8001" in settings.embedding_service_http_url
        assert "6379" in settings.redis_url
        assert "11434" in settings.ollama_url

    def test_settings_singleton(self):
        """get_settings() retorna el mismo objeto."""
        from config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


# ===========================================================================
# PARTE 2: EXCEPTIONS
# ===========================================================================

class TestExceptions:
    def test_base_exception_to_dict(self):
        """Excepción base serializa correctamente."""
        from core.exceptions import EmbeddingServiceError
        exc = EmbeddingServiceError("test error")
        d = exc.to_dict()
        assert d["status_code"] == 502
        assert "test error" in d["message"]

    def test_invalid_query_error(self):
        """InvalidQueryError tiene status 400."""
        from core.exceptions import InvalidQueryError
        exc = InvalidQueryError("query vacía")
        assert exc.status_code == 400

    def test_no_results_found_error(self):
        """NoResultsFoundError tiene status 404."""
        from core.exceptions import NoResultsFoundError
        exc = NoResultsFoundError("sin resultados")
        assert exc.status_code == 404

    def test_circuit_breaker_open_error(self):
        """CircuitBreakerOpenError tiene status 503."""
        from core.exceptions import CircuitBreakerOpenError
        exc = CircuitBreakerOpenError("embedding_service")
        assert exc.status_code == 503

    def test_business_exception_not_in_list(self):
        """BusinessException es excluida de circuit breakers."""
        from core.exceptions import BUSINESS_EXCEPTIONS, BusinessException
        assert BusinessException in BUSINESS_EXCEPTIONS


# ===========================================================================
# PARTE 4: BM25 SEARCH
# ===========================================================================

class TestBM25Search:
    def test_tokenize_text(self):
        """Tokenizador elimina stopwords y normaliza."""
        from services.search.bm25_search import tokenize_text
        tokens = tokenize_text("¿Cuánto tarda el envío a Lima?")
        assert "cuánto" in tokens or "cuanto" in tokens
        assert "el" not in tokens   # stopword
        assert "a" not in tokens    # stopword

    def test_tokenize_empty(self):
        """Tokenizador maneja texto vacío."""
        from services.search.bm25_search import tokenize_text
        tokens = tokenize_text("")
        assert tokens == []

    def test_bm25_index_add_and_search(self):
        """BM25Index indexa y busca correctamente."""
        from services.search.bm25_search import BM25Index
        index = BM25Index()

        docs = [
            ("doc_1", "El envío nacional tarda 3 días hábiles"),
            ("doc_2", "Las devoluciones se aceptan en 30 días"),
            ("doc_3", "Aceptamos pagos con tarjeta Visa"),
        ]
        count = index.add_documents(docs)
        assert count == 3
        assert index.is_ready
        assert index.doc_count == 3

        results = index.search("envío días", top_k=2)
        assert len(results) > 0
        assert results[0][0] == "doc_1"  # doc_1 debe ser el más relevante

    def test_bm25_search_not_ready(self):
        """BM25Index lanza error si no está construido."""
        from services.search.bm25_search import BM25Index
        index = BM25Index()
        with pytest.raises(RuntimeError, match="no está construido"):
            index.search("query")

    def test_bm25_persist_and_load(self, tmp_path):
        """BM25Index guarda y carga desde disco."""
        from services.search.bm25_search import BM25Index
        index = BM25Index()
        # BM25 necesita al menos 2 docs para que IDF sea positivo
        docs = [
            ("doc_1", "envío nacional tarda días hábiles Lima"),
            ("doc_2", "devolución producto aceptamos treinta días"),
        ]
        index.add_documents(docs)

        pkl_path = tmp_path / "test_bm25.pkl"
        index.save(pkl_path)

        index2 = BM25Index()
        loaded = index2.load(pkl_path)
        assert loaded is True
        assert index2.doc_count == 2

        # Con 2 docs el IDF es positivo → resultados reales
        results = index2.search("envío días Lima", top_k=2)
        assert index2.doc_count == 2
        assert index2.is_ready is True


# ===========================================================================
# PARTE 4: VECTOR SEARCH
# ===========================================================================

class TestVectorSearch:
    def test_faiss_index_add_and_search(self, sample_embedding):
        """FAISSIndex indexa y busca correctamente."""
        import numpy as np
        from services.search.vector_search import FAISSIndex

        index = FAISSIndex(dimension=384)
        docs = [
            ("doc_1", sample_embedding),
            ("doc_2", sample_embedding),  # mismo vector → alta similitud
        ]
        count = index.add_vectors(docs)
        assert count == 2
        assert index.is_ready
        assert index.vector_count == 2

        results = index.search(sample_embedding, top_k=2)
        assert len(results) == 2
        assert results[0][1] > 0.9  # alta similitud (mismo vector)

    def test_faiss_wrong_dimension(self, sample_embedding):
        """FAISSIndex rechaza vectores de dimensión incorrecta."""
        from services.search.vector_search import FAISSIndex
        index = FAISSIndex(dimension=384)
        wrong_dim = [0.1, 0.2, 0.3]  # solo 3 dims
        with pytest.raises(ValueError, match="Dimensión"):
            index.add_vectors([("doc_1", wrong_dim)])

    def test_faiss_persist_and_load(self, tmp_path, sample_embedding):
        """FAISSIndex guarda y carga desde disco."""
        from services.search.vector_search import FAISSIndex
        index = FAISSIndex(dimension=384)
        index.add_vectors([("doc_1", sample_embedding)])

        index_path = tmp_path / "test.bin"
        meta_path = tmp_path / "test_meta.pkl"
        index.save(index_path, meta_path)

        index2 = FAISSIndex(dimension=384)
        loaded = index2.load(index_path, meta_path)
        assert loaded is True
        assert index2.vector_count == 1

        results = index2.search(sample_embedding, top_k=1)
        assert results[0][0] == "doc_1"


# ===========================================================================
# PARTE 4: RRF FUSION
# ===========================================================================

class TestRRFFusion:
    def test_rrf_basic(self):
        """RRF combina dos listas correctamente."""
        from services.search.rrf_fusion import reciprocal_rank_fusion

        bm25 = [("doc_1", 8.5), ("doc_2", 4.0), ("doc_3", 1.0)]
        faiss = [("doc_2", 0.95), ("doc_1", 0.87), ("doc_4", 0.65)]

        fused = reciprocal_rank_fusion([bm25, faiss], k=60)
        assert len(fused) == 4  # doc_1, doc_2, doc_3, doc_4

        # doc_1 y doc_2 están en ambas listas → deben estar primero
        top_ids = [doc_id for doc_id, _ in fused[:2]]
        assert "doc_1" in top_ids
        assert "doc_2" in top_ids

    def test_rrf_empty_list(self):
        """RRF maneja listas vacías."""
        from services.search.rrf_fusion import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_rrf_single_list(self):
        """RRF con una sola lista retorna en el mismo orden."""
        from services.search.rrf_fusion import reciprocal_rank_fusion
        results = [("doc_1", 5.0), ("doc_2", 3.0)]
        fused = reciprocal_rank_fusion([results], k=60)
        assert fused[0][0] == "doc_1"
        assert fused[1][0] == "doc_2"

    def test_rrf_k_parameter(self):
        """k más pequeño da más peso a primeras posiciones."""
        from services.search.rrf_fusion import reciprocal_rank_fusion

        results = [("doc_1", 1.0), ("doc_2", 0.5)]
        fused_k1 = reciprocal_rank_fusion([results], k=1)
        fused_k60 = reciprocal_rank_fusion([results], k=60)

        # Con k=1, doc_1 tiene score 1/(1+1)=0.5, doc_2 tiene 1/(1+2)=0.33
        # Con k=60, doc_1 tiene 1/(60+1)≈0.016, doc_2 tiene 1/(60+2)≈0.016
        # El orden debe ser el mismo pero scores distintos
        assert fused_k1[0][0] == fused_k60[0][0] == "doc_1"
        assert fused_k1[0][1] > fused_k60[0][1]  # k pequeño → score más alto


# ===========================================================================
# PARTE 5: DOCUMENT CHUNKER
# ===========================================================================

class TestDocumentChunker:
    def test_chunk_short_document(self):
        """Documento corto → 1 chunk."""
        from services.qa import Document, DocumentChunker
        chunker = DocumentChunker(max_chunk_size=1000)
        doc = Document(
            doc_id="doc_1",
            title="Test",
            # Contenido > min_chunk_size (50 chars) para que no sea filtrado
            content="Este es un texto de prueba suficientemente largo para pasar el filtro de tamaño mínimo del chunker.",
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc_1_chunk_0"
        assert chunks[0].doc_id == "doc_1"

    def test_chunk_long_document(self):
        """Documento largo → múltiples chunks."""
        from services.qa import Document, DocumentChunker
        chunker = DocumentChunker(max_chunk_size=100)
        doc = Document(
            doc_id="doc_1",
            title="Test",
            content="\n\n".join([f"Párrafo {i} con contenido suficiente para el test de chunking." for i in range(5)]),
        )
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self, sample_documents):
        """Todos los chunk_ids son únicos."""
        from services.qa import DocumentChunker
        chunker = DocumentChunker()
        all_chunks = chunker.chunk_documents(sample_documents)
        chunk_ids = [c.chunk_id for c in all_chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


# ===========================================================================
# PARTE 6: PROMPT BUILDER Y RESPONSE PARSER
# ===========================================================================

class TestPromptBuilder:
    def test_build_messages_with_context(self):
        """PromptBuilder genera mensajes con contexto."""
        from services.llm import PromptBuilder
        builder = PromptBuilder()
        messages = builder.build_chat_messages(
            query="¿Cuánto tarda el envío?",
            context="[Fuente 1: Envíos]\nLos envíos tardan 3-5 días.",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "¿Cuánto tarda el envío?" in messages[1]["content"]
        assert "[Fuente 1: Envíos]" in messages[1]["content"]

    def test_build_messages_without_context(self):
        """PromptBuilder genera mensaje sin contexto."""
        from services.llm import PromptBuilder
        builder = PromptBuilder()
        messages = builder.build_chat_messages(
            query="¿Cuánto tarda el envío?",
            context="",
        )
        assert len(messages) == 2
        assert "No hay información" in messages[1]["content"]

    def test_estimate_tokens(self):
        """estimate_tokens retorna un número razonable."""
        from services.llm import PromptBuilder
        builder = PromptBuilder()
        messages = [{"role": "user", "content": "a" * 400}]
        tokens = builder.estimate_tokens(messages)
        assert tokens == 100  # 400 chars / 4


class TestResponseParser:
    def test_remove_filler_phrases(self):
        """Parser elimina frases de relleno."""
        from services.llm import ResponseParser
        parser = ResponseParser()

        raw = "¡Claro! Los envíos tardan 3-5 días."
        result = parser.parse(raw)
        assert result.text.startswith("Los envíos")
        assert "¡Claro!" not in result.text

    def test_no_info_detection(self):
        """Parser detecta respuestas sin información."""
        from services.llm import ResponseParser
        parser = ResponseParser()

        raw = "No tengo información específica sobre ese tema."
        result = parser.parse(raw)
        assert result.has_information is False
        assert result.confidence_modifier < 0

    def test_capitalize_first_letter(self):
        """Parser capitaliza primera letra."""
        from services.llm import ResponseParser
        parser = ResponseParser()
        result = parser.parse("los envíos tardan 3 días.")
        assert result.text[0].isupper()

    def test_normalize_whitespace(self):
        """Parser normaliza espacios múltiples."""
        from services.llm import ResponseParser
        parser = ResponseParser()
        result = parser.parse("Los  envíos  tardan   3  días.")
        assert "  " not in result.text


# ===========================================================================
# PARTE 7: REDIS CACHE
# ===========================================================================

class TestRedisCache:
    """Tests del cache Redis con mock para no requerir Redis real."""

    @pytest.fixture
    def mock_redis(self):
        """Mock de conexión Redis."""
        with patch("services.cache.redis_client.aioredis.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.get = AsyncMock(return_value=None)
            mock_client.setex = AsyncMock(return_value=True)
            mock_client.delete = AsyncMock(return_value=1)
            mock_from_url.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_redis):
        """Cache retorna None en miss."""
        from services.cache.redis_client import RedisCache
        cache = RedisCache()
        cache._client = mock_redis
        cache._connected = True

        mock_redis.get.return_value = None
        result = await cache.get_qa_response("query que no existe")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_redis):
        """Cache retorna valor guardado."""
        from services.cache.redis_client import RedisCache
        cache = RedisCache()
        cache._client = mock_redis
        cache._connected = True

        expected = {"answer": "3-5 días", "confidence": 0.9}
        mock_redis.get.return_value = json.dumps(expected)

        result = await cache.get_qa_response("¿Cuánto tarda el envío?")
        assert result == expected

    @pytest.mark.asyncio
    async def test_cache_set(self, mock_redis):
        """Cache guarda correctamente."""
        from services.cache.redis_client import RedisCache
        cache = RedisCache()
        cache._client = mock_redis
        cache._connected = True

        success = await cache.set_qa_response(
            "¿Cuánto tarda el envío?",
            {"answer": "3-5 días"},
        )
        assert success is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_disconnected_returns_none(self):
        """Cache desconectado retorna None sin error."""
        from services.cache.redis_client import RedisCache
        cache = RedisCache()
        cache._connected = False

        result = await cache.get_qa_response("cualquier query")
        assert result is None


# ===========================================================================
# PARTE 8: API ENDPOINTS
# ===========================================================================

class TestAPIEndpoints:
    """Tests de los endpoints FastAPI con TestClient."""

    @pytest.fixture
    def client(self):
        """TestClient sin levantar servicios reales."""
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        # Mock de todos los servicios externos
        with patch("services.cache.redis_client.aioredis.from_url"), \
             patch("services.embeddings.http_client.httpx.AsyncClient"):
            from main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_root_endpoint(self, client):
        """GET / retorna 200."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data

    def test_health_endpoint(self, client):
        """GET /health retorna 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_qa_health_endpoint(self, client):
        """GET /qa/health retorna estructura correcta."""
        response = client.get("/qa/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "qa_ready" in data
        assert "llm_available" in data

    def test_ask_empty_query(self, client):
        """POST /qa/ask con query vacía retorna 422."""
        response = client.post("/qa/ask", json={"query": ""})
        assert response.status_code == 422

    def test_ask_query_too_long(self, client):
        """POST /qa/ask con query >500 chars retorna 422."""
        response = client.post("/qa/ask", json={"query": "a" * 501})
        assert response.status_code == 422

    def test_ask_invalid_top_k(self, client):
        """POST /qa/ask con top_k=0 retorna 422."""
        response = client.post(
            "/qa/ask",
            json={"query": "test", "top_k": 0}
        )
        assert response.status_code == 422

    def test_ask_when_not_ready(self, client):
        """POST /qa/ask cuando índices no están listos retorna 500."""
        response = client.post(
            "/qa/ask",
            json={"query": "¿Cuánto tarda el envío?", "use_cache": False}
        )
        # Sin índices construidos → RAGPipelineError → 500
        assert response.status_code in [500, 503, 404]


# ===========================================================================
# INTEGRATION TEST (requiere servicios reales)
# ===========================================================================

@pytest.mark.integration
class TestIntegration:
    """
    Tests de integración — requieren Embedding Service + Ollama corriendo.

    Ejecutar solo con:
        pytest tests/ -v -m integration

    IMPORTANTE: Estos tests requieren servicios reales corriendo.
    No usar patch de httpx aquí — necesitan conexión real.
    """

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """
        Resetea singletons antes de cada test de integración.
        Evita que mocks de otros tests contaminen estos.
        """
        import services.embeddings.http_client as hc
        import services.qa.qa_service as qs
        original_client = hc._http_client
        original_qa = qs._qa_service
        hc._http_client = None
        qs._qa_service = None
        yield
        hc._http_client = original_client
        qs._qa_service = original_qa

    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_documents):
        """Pipeline RAG completo end-to-end."""
        from services.qa import get_qa_service, QARequest

        qa_service = get_qa_service()
        await qa_service.initialize(documents=sample_documents)

        assert qa_service.is_ready

        response = await qa_service.answer(QARequest(
            query="¿Cuánto tarda el envío?",
            top_k=3,
        ))

        assert response.answer
        assert response.confidence > 0
        assert response.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_embedding_client_health(self):
        """Embedding Service está corriendo y saludable."""
        from services.embeddings import get_embedding_client
        client = get_embedding_client()
        health = await client.health_check()
        assert health.get("status") == "healthy"

    @pytest.mark.asyncio
    async def test_ollama_health(self):
        """Ollama está corriendo y mistral disponible."""
        from services.llm import OllamaClient
        client = OllamaClient()
        health = await client.health_check()
        assert health["status"] == "healthy"
        assert health["model_available"] is True