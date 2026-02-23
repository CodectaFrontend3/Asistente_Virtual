"""
Configuración del Embedding Service usando Pydantic Settings.

Carga configuración desde:
1. Variables de entorno
2. Archivo .env
3. Valores por defecto

Sigue las mejores prácticas de Twelve-Factor App:
- Separación de configuración y código
- Validación automática de tipos
- Fail-fast en caso de configuración inválida
"""
from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración del Embedding Service.
    
    Todas las variables pueden ser sobrescritas por variables de entorno.
    Ejemplo: EMBEDDING_MODEL_NAME="otro-modelo" python main.py
    """
    
    # ===================================
    # GENERAL
    # ===================================
    
    APP_TITLE: str = Field(
        default="Embedding Service (CPU)",
        description="Título de la aplicación"
    )
    
    APP_DESCRIPTION: str = Field(
        default="Microservicio de generación de embeddings con CPU",
        description="Descripción de la aplicación"
    )
    
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Versión de la aplicación"
    )
    
    ENVIRONMENT: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Entorno de ejecución"
    )
    
    # ===================================
    # SERVER
    # ===================================
    
    HOST: str = Field(
        default="0.0.0.0",
        description="Host del servidor"
    )
    
    PORT: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        description="Puerto del servidor HTTP"
    )
    
    GRPC_PORT: int = Field(
        default=50051,
        ge=1024,
        le=65535,
        description="Puerto del servidor gRPC"
    )
    
    RELOAD: bool = Field(
        default=True,
        description="Auto-reload en desarrollo (solo development)"
    )
    
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nivel de logging"
    )
    
    # ===================================
    # CORS
    # ===================================
    
    CORS_ORIGINS: list[str] = Field(
        default=[
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
        ],
        description="Orígenes permitidos para CORS"
    )
    
    # ===================================
    # EMBEDDING MODEL
    # ===================================
    
    EMBEDDING_MODEL_NAME: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        description="Nombre del modelo de embeddings de HuggingFace"
    )
    
    EMBEDDING_MAX_SEQ_LENGTH: int = Field(
        default=384,
        ge=128,
        le=512,
        description="Longitud máxima de secuencia del modelo"
    )
    
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        ge=128,
        le=1024,
        description="Dimensión de los embeddings (debe coincidir con el modelo)"
    )
    
    # ===================================
    # CPU CONFIGURATION
    # ===================================
    
    CPU_NUM_THREADS: Optional[int] = Field(
        default=None,
        ge=1,
        description="Número de hilos de CPU a usar (None = auto-detect)"
    )
    
    CPU_LOG_USAGE: bool = Field(
        default=True,
        description="Registrar uso de CPU/RAM periódicamente"
    )
    
    # ===================================
    # API LIMITS
    # ===================================
    
    MAX_TEXTS_PER_REQUEST: int = Field(
        default=500,
        ge=1,
        le=1000,
        description="Máximo de textos por request"
    )
    
    MAX_BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Máximo batch size para procesamiento"
    )
    
    DEFAULT_BATCH_SIZE: int = Field(
        default=16,
        ge=1,
        le=32,
        description="Batch size por defecto (optimizado para CPU)"
    )
    
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=180,
        ge=30,
        le=600,
        description="Timeout por request en segundos (CPU es más lento)"
    )
    
    # ===================================
    # CACHE
    # ===================================
    
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Habilitar cache LRU de embeddings"
    )
    
    CACHE_MAX_SIZE: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Tamaño máximo del cache LRU"
    )
    
    # ===================================
    # PATHS
    # ===================================
    
    CONFIG_PATH: str = Field(
        default="config/cpu_config.yaml",
        description="Ruta al archivo de configuración YAML"
    )
    
    # ===================================
    # VALIDATORS
    # ===================================
    
    @field_validator("RELOAD")
    @classmethod
    def validate_reload(cls, v: bool, info) -> bool:
        """Solo permitir reload en development."""
        environment = info.data.get("ENVIRONMENT", "development")
        if v and environment == "production":
            return False  # Force disable en producción
        return v
    
    @field_validator("DEFAULT_BATCH_SIZE")
    @classmethod
    def validate_batch_size(cls, v: int, info) -> int:
        """Validar que default batch size no exceda el máximo."""
        max_batch = info.data.get("MAX_BATCH_SIZE", 32)
        if v > max_batch:
            return max_batch
        return v
    
    # ===================================
    # HELPERS
    # ===================================
    
    def is_production(self) -> bool:
        """Verifica si está en modo producción."""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Verifica si está en modo desarrollo."""
        return self.ENVIRONMENT == "development"
    
    def is_testing(self) -> bool:
        """Verifica si está en modo testing."""
        return self.ENVIRONMENT == "testing"
    
    # ===================================
    # MODEL CONFIG
    # ===================================
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignorar variables extra del .env
        validate_default=True,  # Validar valores por defecto
    )


# ===================================
# SINGLETON GLOBAL
# ===================================

@lru_cache()
def get_settings() -> Settings:
    """
    Obtiene instancia única de configuración.
    
    Usa lru_cache para cargar una sola vez y reutilizar.
    Validación automática de todas las variables.
    
    Returns:
        Instancia validada de Settings
        
    Raises:
        ValidationError: Si configuración es inválida
        
    Example:
        ```python
        from config.settings import get_settings
        
        settings = get_settings()
        print(settings.PORT)  # 8001
        print(settings.is_production())  # False
        ```
    """
    return Settings()


# ===================================
# CONVENIENCIA
# ===================================

# Instancia global para imports directos
settings = get_settings()