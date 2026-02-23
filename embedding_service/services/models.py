"""
Utilidades y helpers para modelos de embeddings.

Funciones auxiliares para:
- Validación de modelos disponibles
- Información de modelos soportados
- Descarga y verificación de modelos
"""
import logging
from typing import Dict, List
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


# ===================================
# MODELOS RECOMENDADOS
# ===================================

RECOMMENDED_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "max_seq_length": 128,
        "size_mb": 470,
        "languages": ["multilingual"],
        "speed": "fast",
        "quality": "good",
        "description": "Modelo multilingüe ligero. Balance velocidad/calidad (RECOMENDADO para CPU)",
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "dimension": 768,
        "max_seq_length": 128,
        "size_mb": 1100,
        "languages": ["multilingual"],
        "speed": "medium",
        "quality": "excellent",
        "description": "Modelo multilingüe de alta calidad. Más lento pero más preciso",
    },
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_seq_length": 256,
        "size_mb": 80,
        "languages": ["english"],
        "speed": "very_fast",
        "quality": "good",
        "description": "Modelo inglés muy rápido. Ideal para CPU si solo usas inglés",
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_seq_length": 384,
        "size_mb": 420,
        "languages": ["english"],
        "speed": "medium",
        "quality": "excellent",
        "description": "Modelo inglés de alta calidad. Buen balance",
    },
}


def get_model_info(model_name: str) -> Dict[str, any]:
    """
    Obtiene información de un modelo recomendado.
    
    Args:
        model_name: Nombre del modelo
    
    Returns:
        Dict con información del modelo
        
    Raises:
        ValueError: Si el modelo no está en la lista recomendada
        
    Example:
        ```python
        info = get_model_info("paraphrase-multilingual-MiniLM-L12-v2")
        print(info["dimension"])  # 384
        print(info["speed"])  # "fast"
        ```
    """
    if model_name not in RECOMMENDED_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no está en la lista recomendada. "
            f"Modelos disponibles: {list(RECOMMENDED_MODELS.keys())}"
        )
    
    return RECOMMENDED_MODELS[model_name]


def list_recommended_models() -> List[Dict[str, any]]:
    """
    Lista todos los modelos recomendados con su información.
    
    Returns:
        Lista de dicts con información de modelos
        
    Example:
        ```python
        models = list_recommended_models()
        for model in models:
            print(f"{model['name']}: {model['description']}")
        ```
    """
    return [
        {"name": name, **info}
        for name, info in RECOMMENDED_MODELS.items()
    ]


def validate_model_name(model_name: str) -> bool:
    """
    Valida si un nombre de modelo es correcto.
    
    Args:
        model_name: Nombre del modelo
    
    Returns:
        True si el modelo está en la lista recomendada
        
    Example:
        ```python
        is_valid = validate_model_name("paraphrase-multilingual-MiniLM-L12-v2")
        # True
        ```
    """
    return model_name in RECOMMENDED_MODELS


def download_model(model_name: str) -> SentenceTransformer:
    """
    Descarga y verifica un modelo de embeddings.
    
    Args:
        model_name: Nombre del modelo de HuggingFace
    
    Returns:
        Modelo cargado
        
    Raises:
        RuntimeError: Si falla la descarga
        
    Example:
        ```python
        model = download_model("paraphrase-multilingual-MiniLM-L12-v2")
        print(model.get_sentence_embedding_dimension())  # 384
        ```
    """
    try:
        logger.info(f"📥 Descargando modelo: {model_name}")
        
        # Cargar modelo (descarga automática si no existe)
        model = SentenceTransformer(model_name)
        
        # Verificar
        dimension = model.get_sentence_embedding_dimension()
        max_seq = model.max_seq_length
        
        logger.info(f"✅ Modelo descargado correctamente")
        logger.info(f"   Dimensión: {dimension}")
        logger.info(f"   Max Sequence Length: {max_seq}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error al descargar modelo: {e}")
        raise RuntimeError(f"No se pudo descargar el modelo {model_name}: {e}")


def verify_model_dimension(model_name: str, expected_dimension: int) -> bool:
    """
    Verifica que un modelo tenga la dimensión esperada.
    
    Args:
        model_name: Nombre del modelo
        expected_dimension: Dimensión esperada
    
    Returns:
        True si coincide la dimensión
        
    Example:
        ```python
        is_correct = verify_model_dimension(
            "paraphrase-multilingual-MiniLM-L12-v2",
            384
        )
        # True
        ```
    """
    try:
        if model_name in RECOMMENDED_MODELS:
            actual_dimension = RECOMMENDED_MODELS[model_name]["dimension"]
            return actual_dimension == expected_dimension
        else:
            # Si no está en lista, cargar para verificar
            model = SentenceTransformer(model_name)
            actual_dimension = model.get_sentence_embedding_dimension()
            return actual_dimension == expected_dimension
            
    except Exception as e:
        logger.error(f"Error al verificar dimensión: {e}")
        return False


def get_best_model_for_cpu() -> str:
    """
    Retorna el mejor modelo para procesamiento con CPU.
    
    Returns:
        Nombre del modelo recomendado para CPU
        
    Example:
        ```python
        best_model = get_best_model_for_cpu()
        # "paraphrase-multilingual-MiniLM-L12-v2"
        ```
    """
    # Para CPU, priorizar velocidad y tamaño pequeño
    return "paraphrase-multilingual-MiniLM-L12-v2"


def get_best_model_for_quality() -> str:
    """
    Retorna el mejor modelo para máxima calidad.
    
    Returns:
        Nombre del modelo con mejor calidad
        
    Example:
        ```python
        best_model = get_best_model_for_quality()
        # "paraphrase-multilingual-mpnet-base-v2"
        ```
    """
    # Para calidad máxima
    return "paraphrase-multilingual-mpnet-base-v2"


def compare_models() -> Dict[str, Dict[str, any]]:
    """
    Compara todos los modelos recomendados.
    
    Returns:
        Dict con comparación de modelos
        
    Example:
        ```python
        comparison = compare_models()
        for name, info in comparison.items():
            print(f"{name}: {info['speed']} speed, {info['quality']} quality")
        ```
    """
    return RECOMMENDED_MODELS.copy()