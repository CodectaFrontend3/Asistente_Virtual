"""
Configuración central del Backend Principal.

Usa pydantic-settings 2.7.1 para:
- Validación automática de tipos
- Carga desde .env
- Singleton con @lru_cache
"""
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración del Backend Principal.

    Todas las variables se pueden sobreescribir con .env
    o variables de entorno (prefijo BACKEND_).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="BACKEND_",
        case_sensitive=False,
        extra="ignore",
    )

    # ===================================
    # APLICACIÓN
    # ===================================
    APP_TITLE: str = "Asistente Virtual - Backend Principal"
    APP_DESCRIPTION: str = "Backend de búsqueda híbrida con RAG"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development")

    # ===================================
    # SERVIDOR
    # ===================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    LOG_LEVEL: str = "INFO"

    # ===================================
    # EMBEDDING SERVICE
    # ===================================
    EMBEDDING_SERVICE_HOST: str = "localhost"
    EMBEDDING_SERVICE_HTTP_PORT: int = 8001
    EMBEDDING_SERVICE_GRPC_PORT: int = 50051
    EMBEDDING_SERVICE_TIMEOUT: int = 30        # segundos
    EMBEDDING_DIMENSION: int = 384

    # ===================================
    # REDIS
    # ===================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_TIMEOUT: int = 5
    REDIS_CACHE_TTL: int = 3600               # 1 hora en segundos

    # ===================================
    # CELERY
    # ===================================
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_TIMEOUT: int = 300            # 5 minutos

    # ===================================
    # GOOGLE GEMINI 1.5 FLASH / LLM
    # ===================================
    GEMINI_API_KEY: str = Field(
        default="#####",
        description="API Key de Google Gemini"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Modelo a usar en Google Gemini"
    )
    GEMINI_TIMEOUT: int = Field(
        default=60,
        description="Timeout en segundos para llamadas a la API"
    )
    GEMINI_MAX_TOKENS: int = Field(
        default=512,
        description="Máximo de tokens a generar"
    )
    GEMINI_TEMPERATURE: float = Field(
        default=0.3,
        description="Temperatura de generación (0.0 - 2.0)"
    )

    # ===================================
    # ZHIPU AI / LLM (Legacy - mantenido para compatibilidad)
    # ===================================
    ZHIPU_API_KEY: str = Field(
        default="95c160bb6fcf44b7b581b213c2e0390a.U4nuorngGxnp7P6i",
        description="API Key de Zhipu AI (Legacy)"
    )
    ZHIPU_BASE_URL: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4/",
        description="Base URL de la API de Zhipu (compatible con OpenAI)"
    )
    ZHIPU_MODEL: str = Field(
        default="glm-4.6",
        description="Modelo a usar en Zhipu AI"
    )
    ZHIPU_TIMEOUT: int = Field(
        default=60,
        description="Timeout en segundos para llamadas a la API"
    )
    ZHIPU_MAX_TOKENS: int = Field(
        default=512,
        description="Máximo de tokens a generar"
    )
    ZHIPU_TEMPERATURE: float = Field(
        default=0.1,
        description="Temperatura de generación (0.0 - 2.0)"
    )

    # ===================================
    # BÚSQUEDA HÍBRIDA
    # ===================================
    BM25_TOP_K: int = 10
    BM25_MIN_SCORE: float = 0.0

    VECTOR_TOP_K: int = 10
    VECTOR_MIN_SCORE: float = 0.0

    RRF_K: int = 60
    HYBRID_TOP_K: int = 5

    # ===================================
    # CHROMADB
    # ===================================
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8002
    CHROMA_COLLECTION_NAME: str = "qa_collection"
    CHROMA_PERSIST_DIR: str = "./chroma_data"

    # ===================================
    # CIRCUIT BREAKER
    # ===================================
    CB_FAIL_MAX: int = 5
    CB_RESET_TIMEOUT: int = 60
    CB_EXPECTED_EXCEPTION: str = "Exception"

    # ===================================
    # CACHE LOCAL (cachetools)
    # ===================================
    CACHE_MAX_SIZE: int = 500
    CACHE_TTL: int = 1800

    # ===================================
    # CORS
    # ===================================
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8001",
    ]

    # ===================================
    # VALIDATORS
    # ===================================

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"ENVIRONMENT debe ser uno de: {allowed}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL debe ser uno de: {allowed}")
        return v_upper

    @field_validator("ZHIPU_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("ZHIPU_TEMPERATURE debe estar entre 0.0 y 2.0")
        return v

    @field_validator("GEMINI_TEMPERATURE")
    @classmethod
    def validate_gemini_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("GEMINI_TEMPERATURE debe estar entre 0.0 y 2.0")
        return v

    @field_validator("RRF_K")
    @classmethod
    def validate_rrf_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("RRF_K debe ser >= 1 (recomendado: 60)")
        return v

    # ===================================
    # HELPERS
    # ===================================

    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "testing"

    @property
    def embedding_service_http_url(self) -> str:
        """URL completa del Embedding Service HTTP."""
        return (
            f"http://{self.EMBEDDING_SERVICE_HOST}"
            f":{self.EMBEDDING_SERVICE_HTTP_PORT}"
        )

    @property
    def embedding_service_grpc_address(self) -> str:
        """Dirección gRPC del Embedding Service."""
        return (
            f"{self.EMBEDDING_SERVICE_HOST}"
            f":{self.EMBEDDING_SERVICE_GRPC_PORT}"
        )

    @property
    def redis_url(self) -> str:
        """URL completa de Redis."""
        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD}"
                f"@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


@lru_cache()
def get_settings() -> Settings:
    """
    Retorna instancia única de Settings (singleton).

    El decorador @lru_cache garantiza que Settings()
    se instancia una sola vez en toda la aplicación.
    """
    return Settings()


# Instancia global conveniente
settings = get_settings()