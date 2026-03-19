"""
Cliente Google Gemini 2.5 Flash para generación de respuestas LLM.
Reemplaza ZhipuClient con soporte para RAG (Generación Aumentada por Recuperación).

El modelo Gemini 2.5 Flash está optimizado para:
- Ventana de contexto muy grande (1M tokens)
- Procesamiento rápido de datos técnicos
- Síntesis de información en lenguaje natural
"""
import logging
import time
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import (
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
    Unauthenticated,
    ServiceUnavailable,
)

from config import settings

logger = logging.getLogger(__name__)

# Códigos de error conocidos de Google AI
_GEMINI_ERROR_CODES = {
    "INVALID_ARGUMENT": "Argumento inválido — verifica el formato de la request",
    "PERMISSION_DENIED": "Permiso denegado — API Key inválida o sin permisos",
    "RESOURCE_EXHAUSTED": "Límite de cuota excedido — reintenta más tarde",
    "UNAUTHENTICATED": "No autenticado — verifica tu API Key",
    "SERVICE_UNAVAILABLE": "Servicio no disponible — intenta más tarde",
}


class GeminiClient:
    """
    Cliente async para Google Gemini 2.5 Flash optimizado para RAG.
    """

    MAX_CONTEXT_CHARS = 8000  # Gemini soporta mucho más, pero limitamos
    MAX_QUERY_CHARS = 500
    MAX_HISTORY_TURNS = 3
    MAX_TOKENS = 512

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self.temperature = (
            temperature if temperature is not None 
            else settings.GEMINI_TEMPERATURE
        )

        # Configurar Google Generative AI
        genai.configure(api_key=self.api_key)

        # Crear el modelo con instrucción del sistema
        self._model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=(
                "Eres un especialista en atención al cliente con formación en comunicación profesional.\n"
                "Tu rol es recibir información de una base de datos y transformarla en respuestas claras, "
                "organizadas y de diseño profesional.\n\n"
                "ESTRUCTURA DE RESPUESTA (OBLIGATORIA):\n"
                "1. SALUDO/INTRO (1 línea): 'Con gusto te proporciono la información solicitada:'\n"
                "2. CONTENIDO - Usa viñetas con guiones (-), separación clara, párrafos bien definidos\n"
                "3. CIERRE (1-2 líneas): '¿Hay algo más que necesites saber?'\n\n"
                "REGLAS DE FORMATO:\n"
                "• Texto PLANO LIMPIO - sin Markdown, sin asteriscos, sin símbolos especiales\n"
                "• Viñetas con guiones: '- Item' (NUNCA asteriscos)\n"
                "• Horarios: 'Lunes a viernes: 9:00 AM - 6:00 PM'\n"
                "• Máximo 2 saltos de línea consecutivos\n"
                "• Párrafos máximo 3 oraciones\n\n"
                "TONO:\n"
                "• Profesional, amigable, accesible\n"
                "• Directo sin rodeos\n"
                "• Solo información de base de datos\n"
                "• NUNCA digas 'según la información' o 'basándome en'\n"
                "• Si falta info: 'Te recomendamos contactar a soporte'\n\n"
                "EJEMPLO:\n"
                "Te proporciono los horarios:\n"
                "ATENCIÓN AL CLIENTE\n"
                "- Lunes a viernes: 9:00 AM - 6:00 PM\n"
                "- Sábados: 9:00 AM - 1:00 PM\n"
                "¿Necesitas algo más?"
            ),
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.MAX_TOKENS,
                top_p=0.9,
            ),
        )

        logger.info(
            f"🤖 GeminiClient inicializado: model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.MAX_TOKENS}"
        )

    # ── Método principal ───────────────────────────────────────────────────────

    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None,
    ) -> str:
        """
        Genera una respuesta utilizando Gemini 1.5 Flash.
        
        Args:
            query: Pregunta del usuario
            context: Contexto RAG recuperado
            system_prompt: Instrucción del sistema (se ignora, usa la configurada)
            conversation_history: Historial de la conversación
            
        Returns:
            Texto de la respuesta generada
        """
        from .prompt_builder import PromptBuilder

        context_truncated = context[:self.MAX_CONTEXT_CHARS]

        # Construir el mensaje del usuario con contexto
        messages = []

        if conversation_history:
            history = conversation_history[-(self.MAX_HISTORY_TURNS * 2) :]
            for msg in history:
                if msg["role"] == "user":
                    messages.append(msg)
                else:
                    messages.append(msg)

        user_content = PromptBuilder.build_user_message(
            query=query,
            context=context_truncated,
        )
        messages.append({"role": "user", "content": user_content})

        return await self._call_gemini(messages)

    # ── Implementación interna ─────────────────────────────────────────────────

    async def _call_gemini(self, messages: list) -> str:
        """Realiza la llamada a la API de Gemini."""
        start_time = time.perf_counter()
        try:
            # Convertir mensajes al formato de Gemini
            # Gemini usa start_chat() para mantener conversación
            chat = self._model.start_chat(history=[])

            # Generar respuesta
            response = chat.send_message(
                messages[-1]["content"] if messages else "",
                stream=False,
            )

            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"🤖 Gemini respondió en {latency:.0f}ms")

            content = response.text
            if not content or not content.strip():
                raise ValueError("Gemini retornó una respuesta vacía")

            return content.strip()

        except Unauthenticated as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"❌ Gemini autenticación fallida en {latency:.0f}ms: {e}"
            )
            raise RuntimeError(
                "API Key de Gemini inválida o expirada — "
                "verifica GEMINI_API_KEY en .env"
            ) from e

        except PermissionDenied as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"❌ Gemini permiso denegado en {latency:.0f}ms: {e}"
            )
            raise RuntimeError(
                "API Key de Gemini sin permisos para este modelo"
            ) from e

        except ResourceExhausted as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"❌ Gemini cuota excedida en {latency:.0f}ms: {e}"
            )
            raise RuntimeError(
                "Límite de cuota de Gemini excedido — reintenta más tarde"
            ) from e

        except InvalidArgument as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"❌ Gemini argumento inválido en {latency:.0f}ms: {e}"
            )
            raise RuntimeError(f"Request inválido para Gemini: {e}") from e

        except ServiceUnavailable as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"⏰ Gemini servicio no disponible en {latency:.0f}ms"
            )
            raise RuntimeError(
                "Servicio de Gemini no disponible — intenta más tarde"
            ) from e

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"❌ Error inesperado Gemini en {latency:.0f}ms: "
                f"{type(e).__name__}: {e}"
            )
            raise

    # ── Health check ───────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """
        Verifica que la API de Gemini responde.
        """
        try:
            test_response = self._model.generate_content(
                "hola",
                stream=False,
            )

            if test_response.text:
                logger.info(
                    f"✅ Gemini health check OK — model={self.model}"
                )
                return {
                    "status": "healthy",
                    "model": self.model,
                    "model_available": True,
                    "provider": "gemini",
                    "error_code": None,
                    "error": None,
                }
            else:
                raise ValueError("Gemini retornó respuesta vacía")

        except Unauthenticated as e:
            logger.warning(
                f"⚠️ Gemini health check falló | "
                f"Auth error: {_GEMINI_ERROR_CODES.get('UNAUTHENTICATED', str(e))}"
            )
            return {
                "status": "unhealthy",
                "model": self.model,
                "model_available": False,
                "provider": "gemini",
                "error_code": "UNAUTHENTICATED",
                "error": _GEMINI_ERROR_CODES.get(
                    "UNAUTHENTICATED", str(e)
                ),
                "error_original": str(e),
            }

        except PermissionDenied as e:
            logger.warning(
                f"⚠️ Gemini health check falló | "
                f"Permission error: {_GEMINI_ERROR_CODES.get('PERMISSION_DENIED', str(e))}"
            )
            return {
                "status": "unhealthy",
                "model": self.model,
                "model_available": False,
                "provider": "gemini",
                "error_code": "PERMISSION_DENIED",
                "error": _GEMINI_ERROR_CODES.get(
                    "PERMISSION_DENIED", str(e)
                ),
                "error_original": str(e),
            }

        except Exception as e:
            logger.warning(
                f"⚠️ Gemini health check falló: "
                f"{type(e).__name__}: {e}"
            )
            return {
                "status": "unhealthy",
                "model": self.model,
                "model_available": False,
                "provider": "gemini",
                "error_code": None,
                "error": str(e),
                "error_original": str(e),
            }


# ── Singleton del cliente ──────────────────────────────────────────────────────

_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Obtiene o crea el singleton del cliente Gemini."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
        logger.info("🤖 GeminiClient singleton creado")
    return _gemini_client
