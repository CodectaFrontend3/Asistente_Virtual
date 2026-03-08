"""
Módulo LLM: Cliente Zhipu GLM-4.6 + Prompt Builder + Response Parser.

Reemplaza Ollama por la API de Zhipu AI (compatible con OpenAI).

Exports principales:
    - ZhipuClient: Cliente async para Zhipu GLM
    - PromptBuilder: Constructor de prompts RAG
    - ResponseParser: Limpieza de respuestas LLM
    - get_zhipu_client: Singleton del cliente
    - setup_llm: Función de integración con QAService

Usage:
    ```python
    # Integración completa en main.py:
    from services.llm import setup_llm

    # Conecta Zhipu con QAService automáticamente
    await setup_llm()

    # Ahora QAService usa LLM real en vez del stub
    ```
"""
import logging

from .zhipu_client import ZhipuClient, get_zhipu_client
from .prompt_builder import PromptBuilder, SYSTEM_PROMPT_QA
from .response_parser import ResponseParser, ParsedResponse, get_response_parser

logger = logging.getLogger(__name__)


async def setup_llm() -> dict:
    """
    Inicializa el LLM e inyecta el cliente en QAService.

    Esta función se llama en el startup de FastAPI.

    Pasos:
    1. Crear ZhipuClient
    2. Verificar que la API responde
    3. Inyectar en QAService (activa LLM real, desactiva stub)

    Returns:
        Dict con status de la inicialización:
        {
            "llm_status": "healthy" | "unhealthy",
            "model": "glm-4.6",
            "model_available": True | False,
            "llm_injected": True | False,
            "error": None | "mensaje"
        }
    """
    from services.qa import get_qa_service

    client = get_zhipu_client()

    # Verificar health
    health = await client.health_check()

    llm_injected = False

    if health["status"] == "healthy" and health["model_available"]:
        # Inyectar en QAService: activa LLM real
        qa_service = get_qa_service()
        qa_service.set_llm_client(client)
        llm_injected = True
        logger.info(
            f"✅ LLM activado: {health['model']} (Zhipu) → QAService conectado"
        )
    else:
        logger.warning(
            f"⚠️ Zhipu no disponible → QAService usa stub. "
            f"Health: {health}"
        )

    return {
        "llm_status": health["status"],
        "model": health["model"],
        "model_available": health["model_available"],
        "provider": "zhipu",
        "llm_injected": llm_injected,
        "error": health.get("error"),
    }


__all__ = [
    # Cliente
    "ZhipuClient",
    "get_zhipu_client",
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