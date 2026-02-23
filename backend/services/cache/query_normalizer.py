"""
Normalizador de queries para mejorar el hit rate del cache.

PROBLEMA:
    "¿Cuál es el horario?"     → MD5: abc123
    "cual es el horario"       → MD5: def456  ← MISS innecesario
    "¿CUAL ES EL HORARIO?"     → MD5: ghi789  ← MISS innecesario

SOLUCIÓN - 2 niveles:
    Nivel 1 — Normalización textual (este archivo):
        Elimina diferencias superficiales → mismo MD5
        "¿Cuál es el horario?" == "cual es el horario" == "CUAL ES EL HORARIO?"

    Nivel 2 — Cache semántico (redis_client.py):
        Compara embeddings para queries con mismo significado pero distintas palabras
        "¿a qué hora atienden?" ≈ "¿cuál es el horario?" → similarity 0.94 → HIT

Transformaciones aplicadas:
    1. Lowercase
    2. Eliminar acentos (á→a, é→e, í→i, ó→o, ú→u, ñ→n)
    3. Eliminar puntuación (¿?¡!.,;:-_)
    4. Colapsar espacios múltiples
    5. Strip (eliminar espacios al inicio/fin)
"""
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Puntuación a eliminar (incluye signos de apertura españoles)
_PUNCTUATION_RE = re.compile(r"[¿?¡!.,;:\-_\"\'()\[\]{}/\\]")

# Espacios múltiples
_WHITESPACE_RE = re.compile(r"\s+")


def _remove_accents(text: str) -> str:
    """
    Elimina acentos usando unicodedata (robusto, sin maketrans).
    NFD descompone caracteres acentuados en base + diacrítico.
    Filtramos diacríticos (categoría Mn). ñ → n se hace manualmente.
    """
    text = text.replace("ñ", "n").replace("Ñ", "N")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def normalize_query(text: str) -> str:
    """
    Normaliza una query para mejorar el hit rate del cache.

    Args:
        text: Query original del usuario

    Returns:
        Query normalizada para usar como clave de cache

    Examples:
        >>> normalize_query("¿Cuál es el horario?")
        'cual es el horario'

        >>> normalize_query("  HACEN ENVÍOS A TODO EL PAÍS??  ")
        'hacen envios a todo el pais'

        >>> normalize_query("¿Cómo hago una devolución?")
        'como hago una devolucion'
    """
    if not text:
        return ""

    # 1. Lowercase
    normalized = text.lower()

    # 2. Eliminar acentos
    normalized = _remove_accents(normalized)

    # 3. Eliminar puntuación
    normalized = _PUNCTUATION_RE.sub(" ", normalized)

    # 4. Colapsar espacios
    normalized = _WHITESPACE_RE.sub(" ", normalized)

    # 5. Strip
    normalized = normalized.strip()

    logger.debug(f"🔤 Normalizado: '{text}' → '{normalized}'")
    return normalized


def queries_are_similar_text(q1: str, q2: str) -> bool:
    """
    Verifica si dos queries son textualmente similares tras normalizar.

    Args:
        q1, q2: Queries a comparar

    Returns:
        True si son idénticas después de normalizar
    """
    return normalize_query(q1) == normalize_query(q2)