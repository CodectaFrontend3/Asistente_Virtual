"""
Módulo LLM: Cliente Ollama + Prompt Builder + Response Parser.

Integra el LLM al pipeline RAG conectándolo con el QAService de Parte 5.

Exports principales:
    - OllamaClient: Cliente async para Ollama
    - PromptBuilder: Constructor de prompts RAG
    - ResponseParser: Limpieza de respuestas LLM
    - get_ollama_client: Singleton del cliente
    - setup_llm: Función de integración con QAService

Usage:
    ```python
    # Integración completa en main.py (Parte 8):
    from services.llm import setup_llm

    # Conecta Ollama con QAService automáticamente
    await setup_llm()

    # Ahora QAService usa LLM real en vez del stub
    ```
"""
import logging

from .ollama_client import OllamaClient, get_ollama_client
from .prompt_builder import PromptBuilder, SYSTEM_PROMPT_QA
from .response_parser import ResponseParser, ParsedResponse, get_response_parser

logger = logging.getLogger(__name__)


async def setup_llm() -> dict:
    """
    Inicializa el LLM e inyecta el cliente en QAService.

    Esta función se llama en el startup de FastAPI (Parte 8).

    Pasos:
    1. Crear OllamaClient
    2. Verificar que Ollama está corriendo
    3. Verificar que el modelo está disponible
    4. Inyectar en QAService (activa LLM real, desactiva stub)

    Returns:
        Dict con status de la inicialización:
        {
            "ollama_status": "healthy" | "unhealthy",
            "model": "mistral",
            "model_available": True | False,
            "llm_injected": True | False,
            "error": None | "mensaje"
        }

    Example:
        ```python
        # En lifespan de FastAPI:
        from services.llm import setup_llm

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            llm_status = await setup_llm()
            if not llm_status["model_available"]:
                logger.warning("⚠️ Modelo no disponible, usando stub")
            yield
        ```
    """
    from services.qa import get_qa_service

    client = get_ollama_client()

    # Verificar health
    health = await client.health_check()

    llm_injected = False

    if health["status"] == "healthy" and health["model_available"]:
        # Inyectar en QAService: activa LLM real
        qa_service = get_qa_service()
        qa_service.set_llm_client(client)
        llm_injected = True
        logger.info(
            f"✅ LLM activado: {health['model']} → QAService conectado"
        )
    else:
        logger.warning(
            f"⚠️ Ollama no disponible o modelo ausente → QAService usa stub. "
            f"Health: {health}"
        )

    return {
        "ollama_status": health["status"],
        "model": health["model"],
        "model_available": health["model_available"],
        "available_models": health.get("models", []),
        "llm_injected": llm_injected,
        "error": health.get("error"),
    }


__all__ = [
    # Cliente
    "OllamaClient",
    "get_ollama_client",
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