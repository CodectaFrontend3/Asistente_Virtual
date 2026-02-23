"""
Parser de respuestas del LLM.

Limpia, valida y normaliza las respuestas de Mistral antes
de enviarlas al usuario.

Mistral a veces incluye:
- Espacios/saltos de línea extra al inicio o final
- Frases de relleno innecesarias ("¡Claro!", "Por supuesto!")
- Repetición de la pregunta del usuario
- Marcadores markdown que no aplican en el contexto

Este módulo normaliza todo eso para dar respuestas limpias.
"""
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FRASES A FILTRAR
# ---------------------------------------------------------------------------

# Frases de relleno que Mistral suele agregar al inicio
_FILLER_STARTS = [
    "¡claro!",
    "¡por supuesto!",
    "¡claro que sí!",
    "claro,",
    "por supuesto,",
    "entendido,",
    "con gusto,",
    "con mucho gusto,",
    "¡hola!",
    "¡buenas!",
    "¡buenos días!",
    "¡buenas tardes!",
    "¡buenas noches!",
    "basándome en la información proporcionada,",
    "basado en la información disponible,",
    "según la información disponible,",
    "según el contexto proporcionado,",
    "de acuerdo con la información disponible,",
    "de acuerdo con el contexto,",
]

# Respuestas que indican que el modelo no tiene información
_NO_INFO_PATTERNS = [
    "no tengo información",
    "no dispongo de información",
    "no cuento con información",
    "no hay información",
    "no se menciona",
    "no se especifica",
    "no puedo responder",
    "no tengo datos",
]


# ---------------------------------------------------------------------------
# DATACLASSES
# ---------------------------------------------------------------------------

@dataclass
class ParsedResponse:
    """
    Respuesta parseada y validada del LLM.

    Attributes:
        text: Texto limpio de la respuesta
        has_information: True si el modelo encontró información relevante
        confidence_modifier: Modificador de confianza (-0.5 si no hay info)
        original_length: Longitud del texto original (antes de limpiar)
        cleaned_length: Longitud del texto limpio
    """
    text: str
    has_information: bool
    confidence_modifier: float
    original_length: int
    cleaned_length: int


# ---------------------------------------------------------------------------
# RESPONSE PARSER
# ---------------------------------------------------------------------------

class ResponseParser:
    """
    Limpia y valida respuestas del LLM Mistral.

    Aplica una serie de transformaciones para normalizar el texto:
    1. Strip de espacios/saltos de línea
    2. Eliminar frases de relleno iniciales
    3. Normalizar espacios múltiples
    4. Detectar respuestas vacías o sin información
    5. Capitalizar primera letra

    Example:
        ```python
        parser = ResponseParser()

        raw = "  ¡Claro! Basándome en la información proporcionada, los envíos tardan 3-5 días.  "
        result = parser.parse(raw)

        print(result.text)
        # "Los envíos tardan 3-5 días."

        print(result.has_information)
        # True
        ```
    """

    def parse(self, raw_response: str) -> ParsedResponse:
        """
        Parsea y limpia la respuesta del LLM.

        Args:
            raw_response: Texto crudo de la respuesta de Ollama

        Returns:
            ParsedResponse con texto limpio y metadata
        """
        original_length = len(raw_response)

        # Aplicar transformaciones en orden
        text = raw_response

        # 1. Strip inicial/final
        text = text.strip()

        # 2. Eliminar frases de relleno al inicio
        text = self._remove_filler_phrases(text)

        # 3. Normalizar saltos de línea múltiples (máx 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Normalizar espacios múltiples
        text = re.sub(r" {2,}", " ", text)

        # 5. Strip final
        text = text.strip()

        # 6. Capitalizar primera letra si no empieza con mayúscula
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # 7. Detectar si el modelo indica que no tiene información
        has_information = not self._is_no_info_response(text)
        confidence_modifier = 0.0 if has_information else -0.3

        if not has_information:
            logger.debug("⚠️ LLM indica que no tiene información suficiente")

        cleaned_length = len(text)

        logger.debug(
            f"🔧 Respuesta parseada: "
            f"{original_length} → {cleaned_length} chars, "
            f"has_info={has_information}"
        )

        return ParsedResponse(
            text=text,
            has_information=has_information,
            confidence_modifier=confidence_modifier,
            original_length=original_length,
            cleaned_length=cleaned_length,
        )

    def parse_text(self, raw_response: str) -> str:
        """
        Versión simplificada que retorna solo el texto limpio.

        Útil cuando solo necesitas el string sin metadata.

        Args:
            raw_response: Texto crudo del LLM

        Returns:
            Texto limpio listo para el usuario
        """
        return self.parse(raw_response).text

    def _remove_filler_phrases(self, text: str) -> str:
        """
        Elimina frases de relleno al inicio del texto.

        Compara el inicio del texto (lowercase) con la lista de frases.
        Si coincide, elimina la frase y limpia el resultado.

        Args:
            text: Texto a limpiar

        Returns:
            Texto sin frases de relleno iniciales
        """
        text_lower = text.lower()

        for phrase in _FILLER_STARTS:
            if text_lower.startswith(phrase):
                # Eliminar la frase (preservando case del resto)
                text = text[len(phrase):].strip()

                # Limpiar puntuación/coma que quede al inicio
                text = text.lstrip(",. ")

                # Capitalizar
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]

                # Solo eliminar una frase (no hacer múltiples pasadas)
                break

        return text

    def _is_no_info_response(self, text: str) -> bool:
        """
        Detecta si la respuesta indica falta de información.

        Args:
            text: Texto de la respuesta

        Returns:
            True si el modelo indica que no tiene información
        """
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in _NO_INFO_PATTERNS)

    def validate_response(self, text: str) -> Optional[str]:
        """
        Valida que la respuesta sea usable.

        Args:
            text: Texto de la respuesta ya limpio

        Returns:
            None si es válida, string con el error si no lo es
        """
        if not text or not text.strip():
            return "Respuesta vacía"

        if len(text) < 10:
            return f"Respuesta demasiado corta ({len(text)} chars)"

        if len(text) > 5000:
            return f"Respuesta demasiado larga ({len(text)} chars)"

        return None  # Válida


# ---------------------------------------------------------------------------
# SINGLETON
# ---------------------------------------------------------------------------

_parser: Optional[ResponseParser] = None


def get_response_parser() -> ResponseParser:
    """
    Obtiene instancia singleton del ResponseParser.

    Returns:
        ResponseParser singleton
    """
    global _parser
    if _parser is None:
        _parser = ResponseParser()
    return _parser