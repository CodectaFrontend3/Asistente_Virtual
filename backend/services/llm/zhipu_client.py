"""
Cliente Zhipu AI (GLM-4.6) async para generación de respuestas LLM.
Compatible con OpenAI SDK - reemplaza OllamaClient.
"""
import logging
import time
from typing import Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


class ZhipuClient:
    """
    Cliente async para Zhipu AI (GLM-4.6) con API compatible con OpenAI.

    Límites de tokens:
        - System prompt:  ~200 tokens
        - Contexto RAG:   ~375 tokens  (3 fuentes × ~500 chars)
        - Query usuario:  ~50 tokens
        - Respuesta:      250 tokens   (max_tokens=250)
        - Total usado:    ~875 tokens  ✅

    Historial de conversación:
        - Máx 3 turnos anteriores (user+assistant) = ~300 tokens extra
    """

    MAX_CONTEXT_CHARS = 3000
    MAX_QUERY_CHARS   = 500
    MAX_HISTORY_TURNS = 3
    MAX_TOKENS        = 250

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.api_key     = api_key     or settings.ZHIPU_API_KEY
        self.base_url    = base_url    or settings.ZHIPU_BASE_URL
        self.model       = model       or settings.ZHIPU_MODEL
        self.timeout     = timeout     or settings.ZHIPU_TIMEOUT
        self.temperature = temperature if temperature is not None else settings.ZHIPU_TEMPERATURE

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.info(
            f"🤖 ZhipuClient inicializado: model={self.model}, "
            f"timeout={self.timeout}s, max_tokens={self.MAX_TOKENS}"
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
                    {"role": "assistant", "content": "Sí, enviamos a todo el país."},
                ]

        Returns:
            Respuesta generada por el modelo

        Raises:
            Exception: Si falla la llamada a la API
        """
        from .prompt_builder import PromptBuilder, SYSTEM_PROMPT_QA

        # Truncar contexto si excede límite
        context_truncated = context[:self.MAX_CONTEXT_CHARS]

        # System prompt
        sys_prompt = system_prompt or SYSTEM_PROMPT_QA

        # Construir mensajes
        messages = [{"role": "system", "content": sys_prompt}]

        # Historial (máx MAX_HISTORY_TURNS turnos)
        if conversation_history:
            history = conversation_history[-(self.MAX_HISTORY_TURNS * 2):]
            messages.extend(history)

        # Prompt del usuario con contexto RAG
        user_content = PromptBuilder.build_user_message(
            query=query,
            context=context_truncated,
        )
        messages.append({"role": "user", "content": user_content})

        return await self._call_zhipu(messages)

    # ── Implementación interna ─────────────────────────────────────────────────

    async def _call_zhipu(self, messages: list) -> str:
        start_time = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.MAX_TOKENS,
                temperature=self.temperature,
                top_p=0.9,
            )
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"🤖 Zhipu GLM respondió en {latency:.0f}ms")

            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Zhipu retornó una respuesta vacía")

            return content.strip()

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            error_type = type(e).__name__
            error_msg  = str(e).strip() or error_type
            logger.error(
                f"❌ Error Zhipu ({error_type}) después de {latency:.0f}ms: {error_msg}"
            )
            raise

    # ── Health check ───────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """
        Verifica que la API de Zhipu responde correctamente.

        Returns:
            Dict con status, model e info adicional.
        """
        try:
            # Llamada mínima para verificar conectividad
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hola"}],
                max_tokens=5,
            )
            return {
                "status": "healthy",
                "model": self.model,
                "model_available": True,
                "provider": "zhipu",
                "base_url": self.base_url,
            }
        except Exception as e:
            logger.warning(f"⚠️ Zhipu health check falló: {e}")
            return {
                "status": "unhealthy",
                "model": self.model,
                "model_available": False,
                "provider": "zhipu",
                "error": str(e),
            }


# ── Singleton ──────────────────────────────────────────────────────────────────

_zhipu_client: Optional[ZhipuClient] = None


def get_zhipu_client() -> ZhipuClient:
    """Retorna instancia singleton del cliente Zhipu."""
    global _zhipu_client
    if _zhipu_client is None:
        _zhipu_client = ZhipuClient()
        logger.info("🤖 ZhipuClient singleton creado")
    return _zhipu_client