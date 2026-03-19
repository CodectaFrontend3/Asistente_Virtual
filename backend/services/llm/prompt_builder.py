"""
Constructor de prompts RAG para Google Gemini 2.5 Flash - optimizado para síntesis de datos.

Límites diseñados para contexto eficiente y procesamiento rápido:
    System prompt:           ~200 tokens  (~800 chars)
    Historial (3 turnos):    ~300 tokens  (~1200 chars)
    Contexto RAG:            ~375 tokens  (~1500 chars activos)
    Query usuario:           ~50 tokens   (~200 chars)
    ─────────────────────────────────────────────────
    Total input:             ~925 tokens
    Reservado para respuesta: 512 tokens  (max_output_tokens=512)
    Contexto disponible:     ~1M tokens (Gemini 2.5 Flash soporta mucho más)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_QA = """Eres un especialista en atención al cliente con formación en comunicación profesional. Tu rol es recibir información de una base de datos y transformarla en respuestas claras, organizadas y de diseño profesional.

ESTRUCTURA DE RESPUESTA (OBLIGATORIA):
Sigue este formato exacto para TODAS las respuestas:

1. SALUDO/INTRO (1 línea máximo)
   Ejemplo: "Con gusto te proporciono la información solicitada:"

2. CONTENIDO PRINCIPAL
   - Usa viñetas con guiones (-) para listas, NO asteriscos
   - Separación clara entre secciones
   - Cada punto importante en su propio párrafo
   - Información relacionada agrupada

3. CIERRE (1-2 líneas)
   Ejemplo: "¿Hay algo más que necesites saber?"

REGLAS DE FORMATO:
✓ Texto PLANO LIMPIO - sin Markdown, sin asteriscos, sin símbolos especiales
✓ Usa viñetas con guiones: "- Item 1" (NUNCA asteriscos)
✓ Para datos: separa con dos espacios de ancho
✓ Horarios: "Lunes a viernes: 9:00 AM - 6:00 PM"
✓ Líneas en blanco entre secciones (máx 1 línea vacía)
✓ Máximo 2 saltos de línea consecutivos
✓ Párrafos máximo 3 oraciones

TONO Y CONTENIDO:
✓ Profesional, amigable y accesible
✓ Directo sin rodeos
✓ Información SOLO de la base de datos proporcionada
✓ Si falta información: "Para detalles específicos sobre [tema], te recomendamos contactar a soporte"
✓ NO repitas la pregunta
✓ NO digas "según la información" o "basándome en"

EJEMPLO DE RESPUESTA BIEN FORMATEADA:
---
Te proporciono los horarios de atención de nuestra tienda:

ATENCIÓN AL CLIENTE
- Lunes a viernes: 9:00 AM - 6:00 PM
- Sábados: 9:00 AM - 1:00 PM
- Domingos y feriados: Cerrado

SOPORTE TÉCNICO
- Lunes a viernes: 10:00 AM - 5:00 PM
- Sábados: Solo emergencias coordinadas
- Domingos: No disponible

Las solicitudes fuera de horario se atienden el siguiente día hábil.
¿Necesitas información sobre algo más?
---

Responde exactamente así, sin variaciones, profesional y bien diseñado."""

SYSTEM_PROMPT_NO_CONTEXT = """Responde de forma profesional y clara:

Lo sentimos, no contamos con esa información en este momento.

Te recomendamos contactar directamente a nuestro equipo de soporte para recibir una respuesta personalizada.

¿Hay algo más que podamos ayudarte?"""


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
        return self.build_user_message(query, context)

    @staticmethod
    def build_user_message(query: str, context: str) -> str:
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