# Módulo 7: Agentes de IA y Orquestación con n8n

## Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- **Diseñar** agentes con herramientas y memoria (Bloom 6)
- **Implementar** loops de razonamiento (ReAct, MRKL) (Bloom 6)
- **Crear** workflows complejos en n8n (Bloom 6)
- **Integrar** agentes con APIs y servicios externos (Bloom 3)
- **Aplicar** buenas prácticas de seguridad en n8n (Bloom 3)

## Duración Estimada
**18-20 horas** (2 semanas)

## Contenidos

### Parte A: Agentes de IA (10 horas)

#### 1. Fundamentos de Agentes
- Definición: percepción → razonamiento → acción
- Tipos: reactivos, deliberativos, híbridos
- Componentes: tools, memory, planning

#### 2. Herramientas (Tools)
- Definición y ejecución
- Parsing de outputs
- Error handling
- Ejemplos: calculadora, búsqueda web, API calls, SQL queries

#### 3. Memoria
- Short-term: conversación actual
- Long-term: base de conocimiento
- Vector memory: embeddings de contexto
- Implementación con diccionarios, DBs, vector stores

#### 4. Planificación y Razonamiento
- **ReAct:** Reasoning + Acting
- **MRKL:** Modular Reasoning, Knowledge and Language
- **Plan-and-Execute:** planificación top-down
- Chain-of-Thought prompting

#### 5. Frameworks (opcional)
- LangChain agents
- LlamaIndex agents
- Comparación con implementación desde cero

### Parte B: n8n (8 horas)

#### 1. Instalación y Setup
- Docker vs Desktop app
- Configuración inicial
- Credenciales y autenticación

#### 2. Conceptos Core
- Nodes, connections, workflows
- Triggers: Manual, Webhook, Schedule, Cron
- Data flow: JSON, parámetros, expresiones
- Variables de entorno

#### 3. Nodes Esenciales para IA
- **HTTP Request:** Llamadas a APIs
- **Webhook:** Recibir requests
- **Code (JS/Python):** Lógica personalizada
- **Google Sheets:** Logs y datos
- **Slack/Discord:** Notificaciones
- **Schedule/Cron:** Automatización temporal

#### 4. Workflows para IA
- ETL: ingesta → limpieza → almacenamiento
- Orquestación de entrenamiento: cron → train → log → notify
- Pipeline de inferencia: webhook → preprocesar → predict → respond
- RAG pipeline: chunking → embeddings → indexado

#### 5. Buenas Prácticas
- Error handling: try/catch, retry logic
- Logging: audit trail
- Secretos: credentials, env vars
- Performance: async, caching
- Debugging: execution logs

## Prácticas Guiadas

### Agentes
1. **Agente Simple:** 2-3 herramientas básicas (2h)
2. **Agente con Memoria:** Conversacional (2h)
3. **Agente RAG:** Búsqueda vectorial + razonamiento (3h)

### n8n
4. **Workflow ETL:** Google Sheets → limpieza → storage (2h)
5. **Orquestación ML:** Cron → train → log → notify (2h)
6. **Pipeline RAG:** Webhook → chunking → embeddings → query (3h)

## Ejercicios

1. Implementar agente con 4+ herramientas
2. Agente que decide qué herramienta usar dinámicamente
3. Workflow n8n: ingesta → preprocessing → API call → notify
4. RAG con n8n: chunking → indexado → endpoint de consulta
5. Sistema de alertas con error handling

## Mini-Quiz (8 preguntas)

Conceptuales sobre agentes, ReAct, y n8n workflows.

## Proyecto Integrador (Preparación para Capstone 3)

**Mini RAG + Agente + n8n:**
- Backend FastAPI con endpoint de consulta
- Agente con 2 herramientas (vector search + web search)
- Workflow n8n: webhook → agente → respuesta → log

## Anti-Patrones

❌ **Hardcodear API keys en código o n8n**
❌ **No manejar errores en workflows**
❌ **Loops infinitos en agentes sin límite**
❌ **No loguear decisiones del agente**

✅ **Variables de entorno para secretos**
✅ **Error handling y retries**
✅ **Max iterations en agentes**
✅ **Logging estructurado de razonamiento**

## Recursos

### Agentes
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Agents Docs](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

### n8n
- [n8n Documentation](https://docs.n8n.io/)
- [n8n Community](https://community.n8n.io/)
- [n8n YouTube Channel](https://www.youtube.com/@n8n-io)

## Herramientas Necesarias

```bash
# n8n (Docker)
docker run -it --rm --name n8n -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n

# O con npm
npm install n8n -g
n8n start
```

## Checklist de Evaluación

- [ ] Implementar agente funcional con herramientas
- [ ] Agente usa razonamiento para decidir herramientas
- [ ] Memoria conversacional implementada
- [ ] 3+ workflows n8n funcionales
- [ ] Error handling en todos los workflows
- [ ] Secretos manejados correctamente
- [ ] Documentación de workflows

## Siguientes Pasos

➡️ **Capstone 3:** Sistema RAG + Agente + n8n End-to-End (proyecto final)

---

**Este es el módulo más avanzado. ¡El proyecto final integrará todo lo aprendido!**
