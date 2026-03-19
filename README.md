# Asistente Virtual - Estructura Final

Este repositorio contiene **SOLO el backend** del asistente virtual.

## 📁 Estructura

```
Vs1/
├── backend/              ← Servidor FastAPI (Python)
│   ├── main.py           ← Aplicación principal
│   ├── requirements.txt  ← Dependencias
│   ├── api/qa/           ← Endpoint /api/chat
│   ├── services/         ← Búsqueda, LLM, embeddings
│   ├── knowledge_base/   ← Base de datos de información
│   ├── config/           ← Configuración
│   └── tests/            ← Tests unitarios
│
├── embedding_service/    ← Servicio de embeddings (gRPC)
├── docker/               ← Configuración Docker
├── scripts/              ← Scripts de utilidad
├── knowledge_base/       ← Base de datos compartida (O en backend/)
├── vector_store/         ← FAISS/BM25 indexes
│
└── .git/
```

## 🚀 Usar el Backend

```bash
# Instalar dependencias
cd backend
pip install -r requirements.txt

# Ejecutar
python main.py
# Corre en http://localhost:3000

# Tests
pytest
```

## 📱 Frontend (Separado)

El **frontend en Next.js es un proyecto aparte** con:
- `types/api/chat.ts` - Tipos TypeScript
- `services/chatService.ts` - Fetch al backend
- `hooks/useChat.ts` - Hook React
- `components/` - Componentes UI

**Frontend debe usar:** `NEXT_PUBLIC_API_URL=http://localhost:3000` (en desarrollo)

## 🔗 Comunicación

```
Frontend (Next.js)
    |
    | POST /api/chat
    |
    ↓
Backend (FastAPI)
    service.py
    search/
    llm/
    knowledge_base/
    |
    | JSON Response
    ↓
Frontend (renderiza)
```

## 📝 Endpoints

- `POST /api/chat` - Procesar pregunta del usuario
- `GET /api/health` - Health check (si existe)

Ver `backend/README.md` para más detalles.

## 🧪 Testing

```bash
cd backend
pytest              # Ejecutar todos los tests
pytest -v          # Con verbose
pytest --cov       # Con coverage
```

## ⚙️ Configuración

Variables de entorno están en `backend/.env`

Ver `backend/.env.example` para referencia.

## 📚 Documentación

- Backend: [backend/README.md](backend/README.md)
- API: Revisar `backend/api/qa/router.py`
- Endpoints: Swagger en `http://localhost:3000/docs` (si FastAPI está configurado)

---

**Nota:** Solo Backend permanece aquí. Frontend está en tu proyecto Next.js separado.
