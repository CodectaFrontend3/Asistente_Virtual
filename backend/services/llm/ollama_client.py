"""
Cliente Zhipu AI (GLM-4.6) para generación de respuestas LLM.
Reemplaza OllamaClient — compatible con OpenAI API.
"""
import logging
import time
from typing import Optional

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Cliente async para Zhipu AI GLM-4.6.
    Mantiene el nombre OllamaClient para no romper imports existentes.
    """

    MAX_CONTEXT_CHARS = 3000
    MAX_QUERY_CHARS   = 500
    MAX_HISTORY_TURNS = 3
    NUM_PREDICT       = 512

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.model       = model       or settings.ZHIPU_MODEL
        self.timeout     = timeout     or settings.ZHIPU_TIMEOUT
        self.temperature = temperature if temperature is not None else settings.ZHIPU_TEMPERATURE

        self._client = AsyncOpenAI(
            api_key=settings.ZHIPU_API_KEY,
            base_url=settings.ZHIPU_BASE_URL,
            timeout=self.timeout,
        )

        logger.info(
            f"🤖 ZhipuClient inicializado: model={self.model}, "
            f"timeout={self.timeout}s"
        )

    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None,
    ) -> str:
        from .prompt_builder import PromptBuilder
        builder = PromptBuilder()

        context_truncated = context[:self.MAX_CONTEXT_CHARS]

        messages = builder.build_chat_messages(
            query=query,
            context=context_truncated,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            max_history_turns=self.MAX_HISTORY_TURNS,
        )

        return await self._call_zhipu(messages)

    async def chat(self, messages: list) -> str:
        return await self._call_zhipu(messages)

    async def _call_zhipu(self, messages: list) -> str:
        start_time = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.NUM_PREDICT,
                temperature=self.temperature,
            )
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"🤖 Zhipu GLM respondió en {latency:.0f}ms")

            content = response.choices[0].message.content
            if not content or not content.strip():
                raise Exception("Zhipu retornó una respuesta vacía")
            return content.strip()

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"❌ Error en Zhipu después de {latency:.0f}ms: {e}")
            raise

    async def health_check(self) -> dict:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hola"}],
                max_tokens=5,
            )
            return {
                "status": "healthy",
                "model": self.model,
                "model_available": True,
                "error": None,
            }
        except Exception as e:
            logger.error(f"❌ Zhipu health check falló: {e}")
            return {
                "status": "unhealthy",
                "model": self.model,
                "model_available": False,
                "error": str(e),
            }

    async def list_models(self) -> list:
        return [self.model]


# ── Singleton ───────────────────────────────────────────────────────────────

_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
        logger.info("🤖 ZhipuClient singleton creado")
    return _ollama_client