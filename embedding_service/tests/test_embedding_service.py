"""
Tests del Embedding Service.

Cubre:
- EmbeddingsCache (unit tests)
- CPUManager (unit tests)
- EmbeddingService (unit tests)
- API HTTP endpoints (integration tests con TestClient)

Uso:
    cd embedding_service
    pytest tests/test_embedding_service.py -v
    pytest tests/test_embedding_service.py -v --cov=. --cov-report=term-missing
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ===================================
# FIXTURES
# ===================================

@pytest.fixture(scope="module")
def client():
    """
    Cliente HTTP para tests de API.

    scope="module" = se crea una vez por módulo de tests.
    El modelo se carga solo una vez para todos los tests.
    """
    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def embedder():
    """Instancia del servicio de embeddings."""
    from services import get_embedding_service
    return get_embedding_service()


@pytest.fixture(scope="module")
def cpu_mgr():
    """Instancia del CPU manager."""
    from services import get_cpu_manager
    return get_cpu_manager()


@pytest.fixture
def cache():
    """
    Cache fresco para cada test.

    scope por defecto = "function" → cache nuevo en cada test.
    """
    from services.embeddings_cache import EmbeddingsCache
    return EmbeddingsCache(max_size=100)


# ===================================
# TESTS: EMBEDDINGS CACHE
# ===================================

class TestEmbeddingsCache:
    """Tests del cache LRU de embeddings."""

    def test_cache_inicializa_vacio(self, cache):
        """Cache debe iniciar sin embeddings."""
        assert len(cache) == 0

    def test_set_y_get_embedding(self, cache):
        """Debe guardar y recuperar un embedding correctamente."""
        texto = "¿Cuánto tarda el envío?"
        embedding = np.random.rand(384).astype(np.float32)

        cache.set(texto, embedding)
        recuperado = cache.get(texto)

        assert recuperado is not None
        np.testing.assert_array_equal(recuperado, embedding)

    def test_get_texto_inexistente_retorna_none(self, cache):
        """Texto no cacheado debe retornar None."""
        resultado = cache.get("texto que no existe en cache")
        assert resultado is None

    def test_hit_rate_se_actualiza(self, cache):
        """Hit rate debe incrementarse con cada hit."""
        texto = "texto de prueba"
        embedding = np.random.rand(384).astype(np.float32)

        # Primer acceso → miss
        cache.get(texto)
        stats_antes = cache.get_stats()
        assert stats_antes["misses"] == 1
        assert stats_antes["hits"] == 0

        # Guardar
        cache.set(texto, embedding)

        # Segundo acceso → hit
        cache.get(texto)
        stats_despues = cache.get_stats()
        assert stats_despues["hits"] == 1
        assert stats_despues["hit_rate_pct"] == 50.0

    def test_clear_vacia_cache(self, cache):
        """Clear debe eliminar todos los embeddings."""
        texto = "texto a borrar"
        embedding = np.random.rand(384).astype(np.float32)

        cache.set(texto, embedding)
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0
        assert cache.get(texto) is None

    def test_lru_elimina_menos_usado(self):
        """Cache lleno debe eliminar el menos usado recientemente."""
        from services.embeddings_cache import EmbeddingsCache
        cache_pequeño = EmbeddingsCache(max_size=3)

        # Llenar el cache
        for i in range(3):
            emb = np.random.rand(384).astype(np.float32)
            cache_pequeño.set(f"texto_{i}", emb)

        assert len(cache_pequeño) == 3

        # Agregar uno más → debe eliminarse el más antiguo (texto_0)
        cache_pequeño.set("texto_nuevo", np.random.rand(384).astype(np.float32))

        assert len(cache_pequeño) == 3
        # texto_0 debería haber sido eliminado (LRU)
        assert cache_pequeño.get("texto_0") is None

    def test_normalize_distingue_cache_keys(self, cache):
        """normalize=True y normalize=False deben tener claves distintas."""
        texto = "mismo texto"
        emb_normalizado = np.random.rand(384).astype(np.float32)
        emb_sin_normalizar = np.random.rand(384).astype(np.float32)

        cache.set(texto, emb_normalizado, normalize=True)
        cache.set(texto, emb_sin_normalizar, normalize=False)

        assert len(cache) == 2
        np.testing.assert_array_equal(
            cache.get(texto, normalize=True),
            emb_normalizado
        )
        np.testing.assert_array_equal(
            cache.get(texto, normalize=False),
            emb_sin_normalizar
        )

    def test_get_stats_retorna_dict_correcto(self, cache):
        """get_stats debe retornar diccionario con las claves esperadas."""
        stats = cache.get_stats()

        claves_esperadas = {
            "current_size", "max_size", "hits", "misses",
            "total_requests", "hit_rate_pct", "ram_used_mb"
        }
        assert claves_esperadas == set(stats.keys())
        assert stats["max_size"] == 100
        assert stats["current_size"] == 0


# ===================================
# TESTS: CPU MANAGER
# ===================================

class TestCPUManager:
    """Tests del gestor de recursos CPU."""

    def test_cpu_manager_es_singleton(self, cpu_mgr):
        """Dos llamadas a get_cpu_manager deben retornar la misma instancia."""
        from services import get_cpu_manager
        otra_instancia = get_cpu_manager()
        assert cpu_mgr is otra_instancia

    def test_device_es_cpu(self, cpu_mgr):
        """El dispositivo debe ser siempre 'cpu'."""
        assert cpu_mgr.device == "cpu"

    def test_get_cpu_info_retorna_datos_validos(self, cpu_mgr):
        """get_cpu_info debe retornar dict con datos válidos."""
        info = cpu_mgr.get_cpu_info()

        assert isinstance(info, dict)
        assert info["physical_cores"] >= 1
        assert info["logical_cores"] >= info["physical_cores"]
        assert info["pytorch_threads"] >= 1

    def test_get_memory_info_retorna_datos_validos(self, cpu_mgr):
        """get_memory_info debe retornar datos de RAM coherentes."""
        mem = cpu_mgr.get_memory_info()

        assert isinstance(mem, dict)
        assert mem["total_gb"] > 0
        assert mem["used_gb"] > 0
        assert mem["available_gb"] >= 0
        assert 0.0 <= mem["percent"] <= 100.0
        # total = used + available (aproximadamente)
        assert mem["used_gb"] + mem["available_gb"] <= mem["total_gb"] + 1.0

    def test_track_inference_incrementa_contador(self, cpu_mgr):
        """track_inference debe incrementar el contador."""
        count_antes = cpu_mgr.get_inference_count()
        cpu_mgr.track_inference()
        assert cpu_mgr.get_inference_count() == count_antes + 1

    def test_check_memory_available_con_ram_suficiente(self, cpu_mgr):
        """check_memory_available debe retornar True con RAM disponible."""
        # Pedir solo 0.001 GB → siempre disponible
        assert cpu_mgr.check_memory_available(required_gb=0.001) is True

    def test_check_memory_available_con_ram_insuficiente(self, cpu_mgr):
        """check_memory_available debe retornar False con RAM imposible."""
        # Pedir 999999 GB → nunca disponible
        assert cpu_mgr.check_memory_available(required_gb=999999.0) is False

    def test_clear_cache_no_falla(self, cpu_mgr):
        """clear_cache debe ejecutarse sin errores."""
        cpu_mgr.clear_cache()  # Solo verificar que no lanza excepción


# ===================================
# TESTS: EMBEDDING SERVICE
# ===================================

class TestEmbeddingService:
    """Tests del servicio de generación de embeddings."""

    def test_embedding_service_es_singleton(self, embedder):
        """Dos llamadas deben retornar la misma instancia."""
        from services import get_embedding_service
        otra_instancia = get_embedding_service()
        assert embedder is otra_instancia

    def test_modelo_cargado_correctamente(self, embedder):
        """El modelo debe estar cargado con el nombre correcto."""
        assert embedder.model is not None
        assert "MiniLM" in embedder.model_name

    def test_dimension_es_384(self, embedder):
        """La dimensión de embeddings debe ser 384."""
        assert embedder.embedding_dimension == 384
        assert embedder.get_dimension() == 384

    def test_device_es_cpu(self, embedder):
        """El dispositivo debe ser 'cpu'."""
        assert embedder.device == "cpu"

    def test_encode_single_retorna_array_correcto(self, embedder):
        """encode_single debe retornar array de dimensión 384."""
        texto = "¿Cuánto tarda el envío a domicilio?"
        embedding = embedder.encode_single(texto)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype in [np.float32, np.float64]

    def test_encode_single_normalizado(self, embedder):
        """Embedding normalizado debe tener norma ≈ 1.0."""
        texto = "Quiero devolver un producto"
        embedding = embedder.encode_single(texto, normalize=True)

        norma = np.linalg.norm(embedding)
        assert abs(norma - 1.0) < 1e-5, f"Norma esperada ≈1.0, got {norma}"

    def test_encode_batch_retorna_shape_correcto(self, embedder):
        """encode con lista debe retornar (n_textos, 384)."""
        textos = [
            "Primera pregunta sobre envíos",
            "Segunda pregunta sobre devoluciones",
            "Tercera pregunta sobre garantías",
        ]
        embeddings = embedder.encode(textos)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_encode_textos_similares_alta_similitud(self, embedder):
        """Textos similares deben tener alta similitud coseno."""
        texto1 = "¿Cuánto tarda el envío?"
        texto2 = "¿Cuál es el tiempo de entrega?"

        emb1 = embedder.encode_single(texto1, normalize=True)
        emb2 = embedder.encode_single(texto2, normalize=True)

        similitud = float(np.dot(emb1, emb2))
        # Textos semánticamente similares deben tener similitud > 0.7
        assert similitud > 0.7, f"Similitud esperada > 0.7, got {similitud:.4f}"

    def test_encode_textos_distintos_baja_similitud(self, embedder):
        """Textos muy distintos deben tener baja similitud coseno."""
        texto1 = "¿Cuánto tarda el envío?"
        texto2 = "El precio del producto es muy alto"

        emb1 = embedder.encode_single(texto1, normalize=True)
        emb2 = embedder.encode_single(texto2, normalize=True)

        similitud = float(np.dot(emb1, emb2))
        # Textos distintos deben tener similitud < 0.95
        assert similitud < 0.95, f"Similitud muy alta para textos distintos: {similitud:.4f}"

    def test_encode_single_texto_vacio_lanza_error(self, embedder):
        """Texto vacío debe lanzar ValueError."""
        with pytest.raises(ValueError):
            embedder.encode_single("")

        with pytest.raises(ValueError):
            embedder.encode_single("   ")

    def test_encode_lista_vacia_lanza_error(self, embedder):
        """Lista vacía debe lanzar ValueError."""
        with pytest.raises(ValueError):
            embedder.encode([])

    def test_get_config_retorna_dict_correcto(self, embedder):
        """get_config debe retornar dict con claves esperadas."""
        config = embedder.get_config()

        claves_esperadas = {
            "model_name", "embedding_dimension", "max_seq_length",
            "device", "default_batch_size", "max_batch_size",
            "max_texts_per_request"
        }
        assert claves_esperadas == set(config.keys())

    def test_get_model_info_retorna_parametros(self, embedder):
        """get_model_info debe incluir número de parámetros."""
        info = embedder.get_model_info()

        assert "num_parameters" in info
        # paraphrase-multilingual-MiniLM-L12-v2 tiene 117M parámetros
        assert info["num_parameters"] > 100_000_000


# ===================================
# TESTS: API HTTP ENDPOINTS
# ===================================

class TestAPIEndpoints:
    """Tests de los endpoints HTTP REST."""

    def test_root_endpoint(self, client):
        """GET / debe retornar info del servicio."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_health_check_retorna_healthy(self, client):
        """GET /embeddings/health debe retornar status healthy."""
        response = client.get("/embeddings/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["embedding_dimension"] == 384
        assert data["device"] == "cpu"
        assert data["ram_total_gb"] > 0

    def test_generate_embeddings_un_texto(self, client):
        """POST /embeddings con un texto debe retornar embedding correcto."""
        response = client.post(
            "/embeddings",
            json={"texts": ["¿Cuánto tarda el envío?"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["dimension"] == 384
        assert data["device"] == "cpu"
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 384
        assert data["processing_time_ms"] > 0

    def test_generate_embeddings_multiples_textos(self, client):
        """POST /embeddings con múltiples textos debe retornar batch correcto."""
        textos = [
            "¿Cuánto tarda el envío?",
            "¿Puedo devolver un producto?",
            "¿Cuál es la garantía?",
        ]
        response = client.post(
            "/embeddings",
            json={"texts": textos}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 3
        assert len(data["embeddings"]) == 3
        for emb in data["embeddings"]:
            assert len(emb) == 384

    def test_generate_embeddings_con_batch_size(self, client):
        """POST /embeddings debe respetar el batch_size."""
        response = client.post(
            "/embeddings",
            json={
                "texts": ["texto uno", "texto dos"],
                "batch_size": 1,
                "normalize": True
            }
        )

        assert response.status_code == 200
        assert response.json()["count"] == 2

    def test_generate_embeddings_lista_vacia_da_422(self, client):
        """POST /embeddings con lista vacía debe retornar 422."""
        response = client.post(
            "/embeddings",
            json={"texts": []}
        )
        assert response.status_code == 422

    def test_generate_embeddings_texto_vacio_da_422(self, client):
        """POST /embeddings con texto vacío en la lista debe retornar 422."""
        response = client.post(
            "/embeddings",
            json={"texts": [""]}
        )
        # 422 de Pydantic o 400 del validador
        assert response.status_code in [400, 422]

    def test_get_stats_retorna_estructura_correcta(self, client):
        """GET /embeddings/stats debe retornar estructura esperada."""
        response = client.get("/embeddings/stats")

        assert response.status_code == 200
        data = response.json()
        assert "inference_count" in data
        assert "embedding_config" in data
        assert "cpu_info" in data
        assert "memory_info" in data
        assert data["inference_count"] >= 0

    def test_clear_cache_retorna_success(self, client):
        """POST /embeddings/clear-cache debe retornar status success."""
        response = client.post("/embeddings/clear-cache")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_process_time_header_presente(self, client):
        """Response debe incluir header X-Process-Time-Ms."""
        response = client.get("/embeddings/health")

        assert "x-process-time-ms" in response.headers

    def test_embeddings_normalizados_norma_uno(self, client):
        """Embeddings con normalize=True deben tener norma ≈ 1.0."""
        response = client.post(
            "/embeddings",
            json={
                "texts": ["texto de prueba para normalización"],
                "normalize": True
            }
        )

        assert response.status_code == 200
        embedding = np.array(response.json()["embeddings"][0])
        norma = np.linalg.norm(embedding)
        assert abs(norma - 1.0) < 1e-5, f"Norma esperada ≈1.0, got {norma}"