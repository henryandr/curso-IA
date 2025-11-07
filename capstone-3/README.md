# Capstone 3: Sistema RAG + Agente + n8n End-to-End

## Información General

**Peso en el Curso:** 40% de la nota final  
**Duración Estimada:** 20-25 horas (2 semanas)  
**Módulos Prerequisitos:** 6, 7  
**Nivel de Dificultad:** ⭐⭐⭐⭐⭐ (Avanzado)

## Objetivos de Aprendizaje

Al completar este capstone, demostrarás que puedes:
- Implementar un sistema RAG completo desde cero
- Diseñar y programar un agente con herramientas y memoria
- Orquestar flujos complejos con n8n
- Desplegar servicios de inferencia con FastAPI
- Aplicar buenas prácticas de seguridad y manejo de errores
- Evaluar y documentar sistemas de IA end-to-end

## Descripción del Proyecto

Construirás un sistema completo que integra:
1. **Backend RAG:** Ingesta, chunking, embeddings, búsqueda vectorial
2. **Agente de IA:** Razonamiento, herramientas, memoria conversacional
3. **API con FastAPI:** Endpoints de ingesta y consulta
4. **Orquestación n8n:** Workflows automatizados
5. **Evaluación:** Casos de prueba y métricas de calidad

El sistema debe ser funcional, seguro, documentado y desplegable.

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                         n8n Workflows                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Workflow 1: Ingesta                                         │
│  [Webhook] → [Validate] → [Process Docs] → [Chunk]          │
│       ↓                                                       │
│  [Generate Embeddings] → [Store in Vector DB] → [Notify]    │
│                                                               │
│  Workflow 2: Consulta                                        │
│  [Webhook] → [FastAPI /query] → [Agent] → [Response]        │
│       ↓                                                       │
│  [Log] → [Notify if needed]                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  POST /ingest     - Ingesta de documentos                   │
│  POST /query      - Consultas al agente                      │
│  GET /health      - Health check                             │
│  GET /docs        - Swagger documentation                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      RAG + Agent Layer                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  RAG Module:                                                 │
│  - Document processing (PDF, txt, markdown)                 │
│  - Chunking (fixed/semantic)                                │
│  - Embeddings generation                                     │
│  - Vector storage (FAISS/Chroma)                            │
│  - Retrieval & re-ranking                                    │
│                                                               │
│  Agent Module:                                               │
│  - Tools: [Vector Search, HTTP Request, Calculator, ...]   │
│  - Memory: Conversation history                             │
│  - Planning: ReAct loop                                      │
│  - LLM: OpenAI/local model                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data & Model Layer                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  - Vector Database (FAISS/ChromaDB)                         │
│  - Document store (local files / S3)                        │
│  - Configuration (environment variables)                     │
│  - Logging (structured logs)                                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Requisitos Técnicos

### 1. Backend RAG con FastAPI

#### 1.1. Ingesta de Documentos
**Endpoint:** `POST /ingest`

**Funcionalidades:**
- [ ] Aceptar múltiples formatos: PDF, TXT, Markdown, HTML
- [ ] Extraer texto de PDFs (PyPDF2, pdfplumber)
- [ ] Chunking configurable:
  - Fixed-size con overlap
  - Semantic chunking (por párrafo/sección)
- [ ] Generación de embeddings (Sentence-BERT o OpenAI)
- [ ] Almacenamiento en vector DB con metadata
- [ ] Validación de entrada (tamaño máximo, formato)
- [ ] Respuesta con ID de documento y estadísticas

**Ejemplo de request:**
```json
{
  "content": "Texto del documento...",
  "metadata": {
    "source": "manual_usuario.pdf",
    "category": "documentacion",
    "date": "2024-01-15"
  },
  "chunking_strategy": "fixed",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Código ejemplo:**
```python
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(title="RAG System API")

class IngestRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = {}
    chunking_strategy: str = "fixed"
    chunk_size: int = 1000
    chunk_overlap: int = 200

@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    # Validar contenido
    if len(request.content) > 1_000_000:  # 1MB max
        raise HTTPException(400, "Document too large")
    
    # Chunking
    chunks = create_chunks(
        request.content,
        strategy=request.chunking_strategy,
        size=request.chunk_size,
        overlap=request.chunk_overlap
    )
    
    # Embeddings
    embeddings = generate_embeddings(chunks)
    
    # Almacenar
    doc_id = store_in_vector_db(chunks, embeddings, request.metadata)
    
    return {
        "document_id": doc_id,
        "chunks_created": len(chunks),
        "status": "success"
    }
```

#### 1.2. Endpoint de Consulta
**Endpoint:** `POST /query`

**Funcionalidades:**
- [ ] Recibir query del usuario
- [ ] Búsqueda vectorial (top-k documentos relevantes)
- [ ] Re-ranking opcional
- [ ] Invocación del agente con contexto
- [ ] Streaming de respuesta (opcional)
- [ ] Logging de query y respuesta

**Ejemplo de request:**
```json
{
  "query": "¿Cómo resetear la contraseña?",
  "top_k": 5,
  "use_agent": true,
  "conversation_id": "uuid-1234"
}
```

**Respuesta:**
```json
{
  "answer": "Para resetear la contraseña...",
  "sources": [
    {
      "chunk": "Texto del chunk relevante",
      "source": "manual_usuario.pdf",
      "score": 0.89
    }
  ],
  "agent_thoughts": [
    "Buscaré en la base de conocimiento...",
    "Encontré información relevante..."
  ]
}
```

### 2. Sistema RAG

#### 2.1. Chunking Module
**Archivo:** `src/rag/chunking.py`

Implementar:
```python
from typing import List, Dict

def chunk_by_fixed_size(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """Fixed-size chunking con overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        chunks.append({
            'text': chunk_text,
            'start': start,
            'end': end
        })
        
        start += (chunk_size - overlap)
    
    return chunks

def chunk_by_semantic(
    text: str,
    separator: str = "\n\n"
) -> List[Dict[str, Any]]:
    """Semantic chunking por párrafos o secciones"""
    paragraphs = text.split(separator)
    chunks = []
    
    current_chunk = ""
    current_start = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < 1500:
            current_chunk += para + separator
        else:
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'start': current_start,
                    'end': current_start + len(current_chunk)
                })
            current_chunk = para + separator
            current_start += len(current_chunk)
    
    if current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'start': current_start,
            'end': current_start + len(current_chunk)
        })
    
    return chunks
```

#### 2.2. Vector Store Module
**Archivo:** `src/rag/vector_store.py`

**Con FAISS:**
```python
import faiss
import numpy as np
import pickle
from typing import List, Tuple

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: List[dict]
    ):
        """Agregar documentos al índice"""
        self.index.add(embeddings)
        self.documents.extend(texts)
        self.metadata.extend(metadata)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """Buscar documentos similares"""
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((
                self.documents[idx],
                float(1 / (1 + distance)),  # Convert to similarity
                self.metadata[idx]
            ))
        
        return results
    
    def save(self, path: str):
        """Guardar índice y metadata"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    def load(self, path: str):
        """Cargar índice y metadata"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
```

### 3. Agente de IA

#### 3.1. Herramientas (Tools)
**Archivo:** `src/agent/tools.py`

Implementar al menos 2 herramientas:

```python
from typing import Dict, Any, List

class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, *args, **kwargs) -> str:
        raise NotImplementedError

class VectorSearchTool(Tool):
    def __init__(self, vector_store):
        super().__init__(
            name="vector_search",
            description="Busca información relevante en la base de conocimiento. Entrada: query (string)"
        )
        self.vector_store = vector_store
        self.embedder = load_embedder()
    
    def execute(self, query: str, top_k: int = 3) -> str:
        """Buscar en la base vectorial"""
        query_embedding = self.embedder.encode([query])
        results = self.vector_store.search(query_embedding, top_k)
        
        # Formatear resultados
        context = "\n\n---\n\n".join([
            f"[Fuente: {meta.get('source', 'unknown')}]\n{text}"
            for text, score, meta in results
        ])
        
        return context

class HTTPRequestTool(Tool):
    def __init__(self):
        super().__init__(
            name="http_request",
            description="Realiza request HTTP GET. Entrada: url (string)"
        )
    
    def execute(self, url: str) -> str:
        """GET request a URL"""
        import requests
        try:
            response = requests.get(url, timeout=10)
            return response.text[:1000]  # Truncar
        except Exception as e:
            return f"Error: {str(e)}"

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evalúa expresiones matemáticas. Entrada: expression (string)"
        )
    
    def execute(self, expression: str) -> str:
        """Evaluar expresión matemática de forma segura"""
        try:
            # Usar ast.literal_eval o eval controlado
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
```

#### 3.2. Agente ReAct
**Archivo:** `src/agent/react_agent.py`

```python
from typing import List, Dict
import re

class ReActAgent:
    def __init__(self, llm, tools: List[Tool], max_iterations: int = 5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.memory = []
    
    def run(self, query: str) -> Dict[str, Any]:
        """Ejecutar loop ReAct"""
        self.memory.append({"role": "user", "content": query})
        
        thoughts = []
        
        for i in range(self.max_iterations):
            # Thought: Razonamiento
            thought_prompt = self._build_thought_prompt(query, thoughts)
            thought = self.llm.generate(thought_prompt)
            thoughts.append(f"Thought {i+1}: {thought}")
            
            # Decidir si usar herramienta
            if "Action:" in thought:
                # Action: Extraer herramienta y input
                action_match = re.search(r"Action: (\w+)", thought)
                input_match = re.search(r"Action Input: (.+)", thought)
                
                if action_match and input_match:
                    tool_name = action_match.group(1)
                    tool_input = input_match.group(1).strip()
                    
                    # Observation: Ejecutar herramienta
                    if tool_name in self.tools:
                        observation = self.tools[tool_name].execute(tool_input)
                        thoughts.append(f"Observation: {observation}")
                    else:
                        thoughts.append(f"Observation: Tool '{tool_name}' not found")
            
            # Si hay "Final Answer", terminar
            if "Final Answer:" in thought:
                answer_match = re.search(r"Final Answer: (.+)", thought, re.DOTALL)
                if answer_match:
                    final_answer = answer_match.group(1).strip()
                    
                    return {
                        "answer": final_answer,
                        "thoughts": thoughts,
                        "iterations": i + 1
                    }
        
        # Si no llegó a respuesta final
        return {
            "answer": "No pude encontrar una respuesta en el número de iteraciones permitido.",
            "thoughts": thoughts,
            "iterations": self.max_iterations
        }
    
    def _build_thought_prompt(self, query: str, thoughts: List[str]) -> str:
        """Construir prompt para el LLM"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        history = "\n".join(thoughts) if thoughts else "No history yet."
        
        prompt = f"""You are a helpful assistant with access to tools.

Available Tools:
{tools_desc}

Question: {query}

Previous thoughts and observations:
{history}

Think step by step. You can use tools by writing:
Action: tool_name
Action Input: input_for_tool

When you have the final answer, write:
Final Answer: your answer here

Think:"""
        
        return prompt
```

### 4. n8n Workflows

#### 4.1. Workflow de Ingesta
**Archivo:** `n8n/workflow-ingestion.json`

**Nodos:**
1. **Webhook Trigger** (POST)
   - URL: `/webhook/ingest`
   - Acepta: file upload o text
2. **Code: Extract Text**
   - Lee archivo o texto
   - Valida tamaño
3. **HTTP Request: FastAPI /ingest**
   - POST a tu API
   - Body: documento procesado
4. **IF: Success**
   - Condición: status code 200
5. **Google Sheets: Log Success**
   - Registra documento ingestado
6. **Slack: Notify Success**
   - Mensaje de confirmación
7. **Error Handler**
   - En caso de fallo
   - Slack: Notify Error

#### 4.2. Workflow de Consulta
**Archivo:** `n8n/workflow-query.json`

**Nodos:**
1. **Webhook Trigger** (POST)
   - URL: `/webhook/query`
   - Body: `{"query": "..."}`
2. **HTTP Request: FastAPI /query**
   - POST a tu API
3. **Code: Format Response**
   - Formatear para usuario
4. **IF: Response Quality**
   - Validar si respuesta es útil
5. **Slack/Discord: Send Response**
   - Enviar respuesta al usuario
6. **Google Sheets: Log Query**
   - Timestamp, query, respuesta
7. **Error Handler**

### 5. Evaluación del Sistema

#### 5.1. Casos de Prueba
**Archivo:** `evaluation/test_cases.json`

Definir 10+ casos de prueba:
```json
[
  {
    "id": 1,
    "query": "¿Cómo resetear mi contraseña?",
    "expected_answer_contains": ["resetear", "contraseña", "email"],
    "relevant_docs": ["manual_usuario.pdf"],
    "category": "authentication"
  },
  {
    "id": 2,
    "query": "¿Cuál es el horario de soporte?",
    "expected_answer_contains": ["lunes", "viernes", "9am", "6pm"],
    "relevant_docs": ["FAQ.md"],
    "category": "support"
  }
]
```

#### 5.2. Métricas
**Archivo:** `evaluation/evaluate.py`

Medir:
- **Retrieval:** Recall@k, MRR (Mean Reciprocal Rank)
- **Generation:** Faithfulness (respuesta basada en contexto)
- **End-to-end:** Exactitud (% respuestas correctas)
- **Latencia:** Tiempo de respuesta promedio

```python
def evaluate_rag_system(test_cases, rag_system):
    results = {
        'total': len(test_cases),
        'correct': 0,
        'retrieval_recall': [],
        'avg_latency': []
    }
    
    for case in test_cases:
        start_time = time.time()
        
        # Query
        response = rag_system.query(case['query'])
        latency = time.time() - start_time
        
        # Evaluar retrieval
        retrieved_sources = [s['source'] for s in response['sources']]
        recall = len(set(retrieved_sources) & set(case['relevant_docs'])) / len(case['relevant_docs'])
        results['retrieval_recall'].append(recall)
        
        # Evaluar respuesta
        answer_correct = all(
            keyword.lower() in response['answer'].lower()
            for keyword in case['expected_answer_contains']
        )
        if answer_correct:
            results['correct'] += 1
        
        results['avg_latency'].append(latency)
    
    # Promedios
    results['accuracy'] = results['correct'] / results['total']
    results['mean_recall'] = np.mean(results['retrieval_recall'])
    results['mean_latency'] = np.mean(results['avg_latency'])
    
    return results
```

### 6. Infraestructura y Despliegue

#### 6.1. Docker Compose
**Archivo:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_DB_PATH=/data/vector_db
    volumes:
      - ./data:/data
      - ./src:/app/src
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

  n8n:
    image: n8nio/n8n
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
    volumes:
      - n8n_data:/home/node/.n8n

volumes:
  n8n_data:
```

#### 6.2. Environment Variables
**Archivo:** `.env.example`

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# Vector Database
VECTOR_DB_PATH=./data/vector_db
VECTOR_DB_TYPE=faiss  # or chromadb

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# n8n Configuration
N8N_PASSWORD=secure_password_here

# Logging
LOG_LEVEL=INFO

# Slack (opcional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

### 7. Documentación

#### 7.1. README.md
Debe incluir:
- Descripción del proyecto
- Arquitectura (diagrama)
- Instalación paso a paso
- Configuración de environment variables
- Cómo ejecutar con Docker
- Cómo usar los endpoints
- Cómo importar workflows de n8n
- Estructura de archivos
- Resultados de evaluación

#### 7.2. Architecture Diagram
Crear diagrama visual (draw.io, mermaid, etc.) mostrando flujo completo.

#### 7.3. Demo Video
Grabar demo de 3-5 minutos mostrando:
1. Ingesta de documentos via n8n
2. Consulta via API o n8n
3. Respuesta del agente con fuentes
4. Workflows funcionando

## Rúbrica de Evaluación

| Componente | Peso | Criterios |
|------------|------|-----------|
| **RAG Backend** | 25% | Ingesta, chunking, embeddings, búsqueda |
| **Agente** | 25% | Herramientas, memoria, razonamiento, logging |
| **n8n Workflows** | 20% | Ingesta, consulta, errores, notificaciones |
| **Infraestructura** | 10% | FastAPI, Docker, secretos, reproducibilidad |
| **Evaluación** | 10% | Métricas, casos de prueba, análisis |
| **Documentación** | 10% | README, arquitectura, decisiones, demo |

## Criterios de Aprobación

- [ ] ≥70% de la calificación
- [ ] Sistema ejecutable end-to-end
- [ ] Sin secretos en código
- [ ] ≥8/10 queries respondidas correctamente
- [ ] Workflows n8n funcionales
- [ ] Docker Compose funcional
- [ ] Documentación completa

## Extras (Puntos Bonus)

- [ ] **+5:** UI con Streamlit/Gradio
- [ ] **+3:** Tests automatizados (pytest)
- [ ] **+2:** CI/CD pipeline
- [ ] **+2:** Monitoring con Prometheus/Grafana

## Timeline Sugerido

| Días | Actividad |
|------|-----------|
| 1-3 | RAG backend (ingesta, chunking, vectores) |
| 4-5 | Agente (herramientas, ReAct loop) |
| 6-8 | FastAPI endpoints |
| 9-10 | n8n workflows |
| 11-12 | Docker + infraestructura |
| 13-14 | Evaluación y testing |
| 15-16 | Documentación y demo |

## Entrega

**Repositorio Git** con:
- Código completo
- Docker Compose funcional
- README con instrucciones
- Workflows n8n exportados (JSON)
- Video demo (link)

---

**Este es el proyecto culminante del curso. ¡Dale todo!** 🚀
