"""
Módulo LLM: Cliente Google Gemini 2.5 Flash + Prompt Builder + Response Parser.

Utiliza la IA de Google Gemini 2.5 Flash para generación aumentada por 
recuperación (RAG) con soporte de contexto muy grande y síntesis de datos.

Exports principales:
    - GeminiClient: Cliente async para Google Gemini 2.5 Flash
    - PromptBuilder: Constructor de prompts RAG
    - ResponseParser: Limpieza de respuestas LLM
    - get_gemini_client: Singleton del cliente
    - setup_llm: Función de integración con QAService

Usage:
    ```python
    # Integración completa en main.py:
    from services.llm import setup_llm

    # Conecta Gemini con QAService automáticamente
    await setup_llm()

    # Ahora QAService usa LLM real en vez del stub
    ```
"""
import logging

from .gemini_client import GeminiClient, get_gemini_client
from .prompt_builder import PromptBuilder, SYSTEM_PROMPT_QA
from .response_parser import ResponseParser, ParsedResponse, get_response_parser

logger = logging.getLogger(__name__)


async def setup_llm() -> dict:
    """
    Inicializa el LLM (Gemini 1.5 Flash) e inyecta el cliente en QAService.

    Esta función se llama en el startup de FastAPI.

    Pasos:
    1. Crear GeminiClient
    2. Verificar que la API responde
    3. Inyectar en QAService (activa LLM real, desactiva stub)

    Returns:
        Dict con status de la inicialización:
        {
            "llm_status": "healthy" | "unhealthy",
            "model": "gemini-1.5-flash",
            "model_available": True | False,
            "llm_injected": True | False,
            "error": None | "mensaje"
        }
    """
    from services.qa import get_qa_service

    client = get_gemini_client()

    # Verificar health
    health = await client.health_check()

    llm_injected = False

    if health["status"] == "healthy" and health["model_available"]:
        # Inyectar en QAService: activa LLM real
        qa_service = get_qa_service()
        qa_service.set_llm_client(client)
        llm_injected = True
        logger.info(
            f"✅ LLM activado: {health['model']} (Gemini) → QAService conectado"
        )
    else:
        logger.warning(
            f"⚠️ Gemini no disponible → QAService usa stub. "
            f"Health: {health}"
        )

    return {
        "llm_status": health["status"],
        "model": health["model"],
        "model_available": health["model_available"],
        "provider": "gemini",
        "llm_injected": llm_injected,
        "error": health.get("error"),
    }


__all__ = [
    # Cliente
    "GeminiClient",
    "get_gemini_client",
    # Prompt
    "PromptBuilder",
    "SYSTEM_PROMPT_QA",
    # Parser
    "ResponseParser",
    "ParsedResponse",
    "get_response_parser",
    # Setup
    "setup_llm",
]