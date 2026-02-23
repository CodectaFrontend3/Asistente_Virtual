"""
Constructor de prompts RAG para Mistral - optimizado para CPU.

Límites diseñados para num_ctx=2048 tokens:
    System prompt:           ~200 tokens  (~800 chars)
    Historial (3 turnos):    ~300 tokens  (~1200 chars)
    Contexto RAG:            ~375 tokens  (~1500 chars activos)
    Query usuario:           ~50 tokens   (~200 chars)
    ─────────────────────────────────────────────────
    Total input:             ~925 tokens  de 2048
    Reservado para respuesta: 250 tokens  (num_predict=250)
    Margen de seguridad:     ~873 tokens  ✅
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_QA = """Eres un asistente virtual de atención al cliente para una tienda online J&P.

Responde preguntas basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS:
1. Responde SOLO en español
2. Usa ÚNICAMENTE la información del contexto. No inventes datos
3. Si el contexto no tiene la información, dilo claramente y sugiere contactar soporte
4. Sé conciso: máximo 3 oraciones
5. Tono amable y profesional
6. Incluye datos específicos (precios, plazos, políticas) cuando estén disponibles
7. No repitas la pregunta del usuario"""

SYSTEM_PROMPT_NO_CONTEXT = """Eres un asistente virtual de J&P. No tienes información específica para esta consulta. Informa amablemente y sugiere contactar al soporte."""


# ── Prompt Builder ─────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructor de prompts RAG con soporte de historial de conversación.

    El historial permite al modelo entender el contexto de la sesión:
        Usuario: ¿Cuánto tarda el envío?
        Asistente: Los envíos tardan 3-5 días hábiles.
        Usuario: ¿Y a provincias?   ← el modelo entiende que habla de envíos
    """

    def build_chat_messages(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None,
        max_history_turns: int = 3,
    ) -> list:
        """
        Construye mensajes para ollama.chat() con historial opcional.

        Args:
            query: Pregunta actual del usuario
            context: Contexto RAG (ya truncado por OllamaClient)
            system_prompt: Personalizado o None para usar default
            conversation_history: Turnos anteriores de la sesión:
                [
                    {"role": "user",      "content": "pregunta anterior"},
                    {"role": "assistant", "content": "respuesta anterior"},
                    ...
                ]
                Máximo max_history_turns turnos (los más recientes).
            max_history_turns: Cuántos turnos incluir (default 3)

        Returns:
            Lista de mensajes para Ollama
        """
        # 1. System prompt
        sys_prompt = system_prompt or (
            SYSTEM_PROMPT_QA if context.strip() else SYSTEM_PROMPT_NO_CONTEXT
        )

        messages = [{"role": "system", "content": sys_prompt}]

        # 2. Historial de conversación (turnos recientes)
        if conversation_history:
            # Tomar solo los últimos N turnos completos (user+assistant = 1 turno)
            # Cada turno son 2 mensajes, así que tomamos los últimos max*2
            recent = conversation_history[-(max_history_turns * 2):]
            messages.extend(recent)
            logger.debug(f"💬 Historial incluido: {len(recent)} mensajes ({len(recent)//2} turnos)")

        # 3. Mensaje actual con contexto
        messages.append({
            "role": "user",
            "content": self._build_user_message(query, context),
        })

        # Log diagnóstico
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est_tokens  = total_chars // 4
        logger.debug(
            f"📝 Prompt: {len(messages)} mensajes, "
            f"~{est_tokens} tokens estimados "
            f"(chars: sys={len(sys_prompt)}, ctx={len(context)}, query={len(query)})"
        )

        return messages

    def _build_user_message(self, query: str, context: str) -> str:
        """Construye el mensaje usuario con contexto embebido."""
        if context.strip():
            return (
                f"INFORMACIÓN DISPONIBLE:\n"
                f"{context}\n\n"
                f"PREGUNTA DEL CLIENTE:\n"
                f"{query}"
            )
        return (
            f"PREGUNTA DEL CLIENTE:\n"
            f"{query}\n\n"
            f"(Sin contexto disponible para esta consulta)"
        )

    def estimate_tokens(self, messages: list) -> int:
        """Estimación rápida: 1 token ≈ 4 chars en español."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4

    def build_fallback_message(self) -> str:
        return (
            "Lo siento, en este momento no puedo procesar tu consulta. "
            "Por favor intenta nuevamente o contacta a nuestro soporte."
        )