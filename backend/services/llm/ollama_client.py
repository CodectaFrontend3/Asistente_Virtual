"""
Cliente Ollama async para generación de respuestas LLM.
"""
import logging
import time
from typing import Optional

import ollama as ollama_lib

from config import settings
from core import (
    OllamaServiceError,
    CircuitBreakerOpenError,
    TimeoutError as BackendTimeoutError,
    ollama_breaker,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Cliente async para Ollama LLM con límites optimizados para CPU.

    Límites de tokens con num_ctx=2048:
        - System prompt:  ~200 tokens
        - Contexto RAG:   ~375 tokens  (3 fuentes × ~500 chars)
        - Query usuario:  ~50 tokens
        - Respuesta:      250 tokens   (num_predict=250)
        - Total usado:    ~875 / 2048  ✅

    Historial de conversación:
        - Máx 3 turnos anteriores (user+assistant) = ~300 tokens extra
        - Total con historial: ~1175 / 2048  ✅
    """

    # Límites explícitos para referencia
    MAX_CONTEXT_CHARS = 3000   # chars para el contexto RAG
    MAX_QUERY_CHARS   = 500    # chars para la query (ya validado en API)
    MAX_HISTORY_TURNS = 3      # turnos de historial (user+assistant cada uno)
    NUM_PREDICT       = 250    # tokens máx a generar en la respuesta

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> None:
        self.base_url    = base_url    or settings.ollama_url
        self.model       = model       or settings.OLLAMA_MODEL
        self.timeout     = timeout     or settings.OLLAMA_TIMEOUT
        self.temperature = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        self.num_ctx     = num_ctx     or 2048

        self._client = ollama_lib.AsyncClient(
            host=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            f"🦙 OllamaClient inicializado: model={self.model}, "
            f"timeout={self.timeout}s, num_ctx={self.num_ctx}, "
            f"num_predict={self.NUM_PREDICT}"
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
        Genera respuesta RAG dado query + contexto + historial opcional.

        Args:
            query: Pregunta del usuario (máx 500 chars)
            context: Contexto de búsqueda híbrida (se trunca a MAX_CONTEXT_CHARS)
            system_prompt: System prompt personalizado (None = default)
            conversation_history: Lista de turnos anteriores:
                [
                    {"role": "user",      "content": "¿Hacen envíos?"},
                    {"role": "assistant", "content": "Sí, enviamos a..."},
                    ...
                ]
                Máx 3 turnos. Se ignoran los más antiguos automáticamente.

        Returns:
            Texto de respuesta generado (máx ~250 tokens)
        """
        from .prompt_builder import PromptBuilder
        builder = PromptBuilder()

        # Truncar contexto si es muy largo
        context_truncated = context[:self.MAX_CONTEXT_CHARS] if len(context) > self.MAX_CONTEXT_CHARS else context
        if len(context) > self.MAX_CONTEXT_CHARS:
            logger.debug(f"✂️ Contexto truncado: {len(context)} → {self.MAX_CONTEXT_CHARS} chars")

        messages = builder.build_chat_messages(
            query=query,
            context=context_truncated,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            max_history_turns=self.MAX_HISTORY_TURNS,
        )

        return await self._call_ollama(messages)

    async def chat(self, messages: list) -> str:
        """Llamada directa con mensajes ya construidos."""
        return await self._call_ollama(messages)

    # ── Implementación interna ─────────────────────────────────────────────────

    async def _call_ollama(self, messages: list) -> str:
        start_time = time.perf_counter()
        try:
            response = await self._call_with_breaker(messages)
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"🦙 Ollama respondió en {latency:.0f}ms")
            return response

        except ollama_lib.ResponseError as e:
            logger.error(f"❌ Ollama ResponseError: {e.status_code} - {e.error}")
            raise OllamaServiceError(f"Ollama error {e.status_code}: {e.error}")

        except Exception as e:
            error_type = type(e).__name__
            error_msg  = str(e).strip() or error_type

            if "timeout" in error_type.lower() or "timeout" in error_msg.lower():
                latency = (time.perf_counter() - start_time) * 1000
                logger.error(f"⏰ Ollama timeout después de {latency:.0f}ms")
                raise BackendTimeoutError("ollama", self.timeout)

            if isinstance(e, CircuitBreakerOpenError):
                raise

            logger.error(f"❌ Error inesperado en Ollama: {error_type}: {error_msg}")
            raise OllamaServiceError(f"Error LLM ({error_type}): {error_msg}")

    @ollama_breaker
    async def _call_with_breaker(self, messages: list) -> str:
        response = await self._client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={
                "temperature": self.temperature,
                "num_ctx":     self.num_ctx,
                "num_predict": self.NUM_PREDICT,  # ← CLAVE: limita tokens generados
                "num_thread":  4,                 # hilos CPU (i5 gen11 = 4 cores)
                "top_k":       40,
                "top_p":       0.9,
                "repeat_penalty": 1.1,            # evita repeticiones
            },
        )

        content = response.message.content
        if not content or not content.strip():
            raise OllamaServiceError("Ollama retornó una respuesta vacía")

        return content.strip()

    # ── Health check ───────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        try:
            models_response = await self._client.list()
            model_names     = [m.model for m in models_response.models]
            model_available = any(self.model in name for name in model_names)

            if not model_available:
                logger.warning(f"⚠️ Modelo '{self.model}' no encontrado. Disponibles: {model_names}")

            return {
                "status": "healthy",
                "model": self.model,
                "model_available": model_available,
                "models": model_names,
                "num_predict": self.NUM_PREDICT,
                "num_ctx": self.num_ctx,
                "error": None,
            }
        except Exception as e:
            logger.error(f"❌ Ollama health check falló: {e}")
            return {"status": "unhealthy", "model": self.model,
                    "model_available": False, "models": [], "error": str(e)}

    async def list_models(self) -> list:
        try:
            response = await self._client.list()
            return [m.model for m in response.models]
        except Exception as e:
            logger.error(f"❌ Error listando modelos: {e}")
            return []


# ── Singleton ──────────────────────────────────────────────────────────────────

_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
        logger.info("🦙 OllamaClient singleton creado")
    return _ollama_client