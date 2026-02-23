"""
Módulo de configuración del Embedding Service.

Exporta:
- get_settings: Función para obtener configuración singleton
- settings: Instancia global de configuración
- Settings: Clase de configuración (para type hints)
"""
from .settings import get_settings, settings, Settings

__all__ = [
    "get_settings",
    "settings",
    "Settings",
]