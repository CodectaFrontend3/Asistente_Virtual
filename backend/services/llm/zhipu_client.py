"""
Cliente Zhipu AI (GLM-4.6) async para generación de respuestas LLM.
Compatible con OpenAI SDK - reemplaza OllamaClient.
"""
import logging
import time
from typing import Optional

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, APITimeoutError

from config import settings

logger = logging.getLogger(__name__)

# Códigos de error conocidos de Zhipu AI
_ZHIPU_ERROR_CODES = {
    "1113": "Saldo insuficiente — recarga en open.bigmodel.cn",
    "1101": "API Key inválida o expirada",
    "1102": "API Key sin permisos para este modelo",
    "1110": "Límite de peticiones por minuto alcanzado (rate limit)",
    "1111": "Límite diario de tokens alcanzado",
}


class ZhipuClient:
    """
    Cliente async para Zhipu AI (GLM-4.6) con API compatible con OpenAI.
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
        from .prompt_builder import PromptBuilder, SYSTEM_PROMPT_QA

        context_truncated = context[:self.MAX_CONTEXT_CHARS]
        sys_prompt = system_prompt or SYSTEM_PROMPT_QA

        messages = [{"role": "system", "content": sys_prompt}]

        if conversation_history:
            history = conversation_history[-(self.MAX_HISTORY_TURNS * 2):]
            messages.extend(history)

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

        except APIStatusError as e:
            latency = (time.perf_counter() - start_time) * 1000
            error_body = {}
            try:
                error_body = e.response.json()
            except Exception:
                pass
            code = str(error_body.get("error", {}).get("code", e.status_code))
            msg_zh = error_body.get("error", {}).get("message", "")
            friendly = _ZHIPU_ERROR_CODES.get(code, f"Error HTTP {e.status_code}")
            logger.error(
                f"❌ Zhipu API error [{code}] en {latency:.0f}ms | "
                f"{friendly} | msg_original: {msg_zh}"
            )
            raise RuntimeError(f"[Zhipu {code}] {friendly}") from e

        except APIConnectionError as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"❌ Zhipu sin conexión después de {latency:.0f}ms: {e}")
            raise RuntimeError("No se pudo conectar con Zhipu AI — verifica tu internet") from e

        except APITimeoutError as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"⏰ Zhipu timeout después de {latency:.0f}ms")
            raise RuntimeError(f"Zhipu AI no respondió en {self.timeout}s") from e

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"❌ Error inesperado Zhipu en {latency:.0f}ms: "
                f"{type(e).__name__}: {e}"
            )
            raise

    # ── Health check ───────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """
        Verifica que la API de Zhipu responde.
        Clasifica el error con código y mensaje amigable si falla.
        """
        try:
            await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hola"}],
                max_tokens=5,
            )
            logger.info(f"✅ Zhipu health check OK — model={self.model}")
            return {
                "status": "healthy",
                "model": self.model,
                "model_available": True,
                "provider": "zhipu",
                "base_url": self.base_url,
                "error_code": None,
                "error": None,
            }

        except APIStatusError as e:
            error_body = {}
            try:
                error_body = e.response.json()
            except Exception:
                pass
            code = str(error_body.get("error", {}).get("code", e.status_code))
            msg_zh = error_body.get("error", {}).get("message", str(e))
            friendly = _ZHIPU_ERROR_CODES.get(code, f"Error HTTP {e.status_code}")
            logger.warning(
                f"⚠️ Zhipu health check falló | "
                f"código={code} | {friendly} | original: {msg_zh}"
            )
            return {
                "status": "unhealthy",
                "model": self.model,
                "model_available": False,
                "provider": "zhipu",
                "error_code": code,
                "error": friendly,
                "error_original": msg_zh,
            }

        except Exception as e:
            logger.warning(f"⚠️ Zhipu health check falló: {type(e).__name__}: {e}")
            return {
                "status": "unhealthy",
                "model": self.model,
                "model_available": False,
                "provider": "zhipu",
                "error_code": None,
                "error": str(e),
                "error_original": str(e),
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