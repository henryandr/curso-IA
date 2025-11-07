# Módulo 6: LLMs, Prompting y RAG

## 📚 Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- **Diseñar** prompts efectivos para LLMs (Bloom 6)
- **Implementar** sistema RAG básico (Bloom 6)
- **Evaluar** respuestas de LLMs (Bloom 5)
- **Usar** APIs de LLMs de forma segura (Bloom 3)
- **Optimizar** retrieval y generación (Bloom 5)

## ⏱️ Duración Estimada
**12-15 horas** (1.5 semanas)

## 🗺️ Mapa Conceptual

```
                        LLMs, PROMPTING Y RAG
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   PROMPTING                    RAG                    EVALUACIÓN
        │                         │                         │
  ┌─────┴─────┐           ┌───────┴───────┐         ┌──────┴──────┐
  │           │           │               │         │             │
Zero-shot  Few-shot   Chunking       Retrieval   Metrics      Testing
  │           │           │               │         │             │
  • Direct    • Examples  • Fixed         • Dense   • Exact       • Test
  • CoT       • In-context • Semantic     • Sparse    Match         Cases
  • ReAct       Prompts   • Overlap       • Hybrid  • BLEU/ROUGE  • A/B
                                                    • Faithfulness
```

## 📖 Contenidos Detallados

### 1. Fundamentos de LLMs (2 horas)

#### Conceptos Clave

**¿Qué es un LLM?**
- Large Language Model - modelo entrenado en billones de palabras
- Predice siguiente token dado contexto
- Emergent abilities: razonamiento, código, traducción

**Arquitectura:**
```
Input Text
    ↓
[Tokenización] → ["Hello", ",", "how", "are", "you"]
    ↓
[Token IDs] → [15496, 11, 703, 389, 345]
    ↓
[Embeddings] → Vector denso por token
    ↓
[Transformer Layers] × 12-96 layers
    ↓
[Output Logits] → Probabilidades de próximo token
    ↓
[Sampling/Greedy] → Token seleccionado
```

**Context Window:**
- GPT-3.5: 4,096 tokens (~3,000 palabras)
- GPT-4: 8,192 - 32,768 tokens
- Claude: 100,000 tokens
- Llama 2: 4,096 tokens

**Tokenización:**
```python
import tiktoken

# Encoder para GPT-3.5/4
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = "Hello, how are you doing today?"
tokens = encoding.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Num tokens: {len(tokens)}")

# Decodificar
decoded = encoding.decode(tokens)
print(f"Decoded: {decoded}")
```

### 2. Prompting Efectivo (3 horas)

#### Zero-Shot Prompting

**Estructura básica:**
```
System: Define el rol y comportamiento
User: La instrucción/pregunta
Assistant: Respuesta del modelo
```

**Ejemplo con OpenAI API:**
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant expert in machine learning."},
        {"role": "user", "content": "What is the difference between supervised and unsupervised learning?"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

#### Few-Shot Prompting

**Proporcionar ejemplos:**
```python
few_shot_prompt = """
Classify the sentiment of these movie reviews:

Review: "This movie was absolutely fantastic! I loved every minute."
Sentiment: Positive

Review: "Worst movie I've ever seen. Complete waste of time."
Sentiment: Negative

Review: "It was okay, nothing special but not terrible either."
Sentiment: Neutral

Review: "An absolute masterpiece of cinema. Brilliant acting and direction."
Sentiment: """

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": few_shot_prompt}],
    temperature=0
)

print(response.choices[0].message.content)  # "Positive"
```

#### Chain-of-Thought (CoT) Prompting

**Hacer que el modelo "piense paso a paso":**
```python
cot_prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans
3. Each can has 3 balls, so 2 cans = 2 × 3 = 6 balls
4. Total = 5 + 6 = 11 balls

Therefore, Roger has 11 tennis balls.

Q: A juggler can juggle 16 balls. Half of the balls are golf balls,
and half of the golf balls are blue. How many blue golf balls are there?

A: Let's think step by step:"""

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": cot_prompt}],
    temperature=0,
    max_tokens=150
)

print(response.choices[0].message.content)
```

#### Prompting Best Practices

**✅ Buenos Prompts:**
- Claros y específicos
- Con contexto suficiente
- Ejemplos cuando es posible
- Formato de salida definido

**❌ Malos Prompts:**
- Ambiguos o vagos
- Sin contexto
- Múltiples preguntas mezcladas

**Ejemplo de mejora:**
```python
# ❌ Malo
"Tell me about Python"

# ✅ Mejor
"Explain the main advantages of Python for data science in 3 bullet points."

# ✅ Aún mejor
"""You are a data science educator. Explain to a beginner programmer 
the main advantages of Python for data science.

Requirements:
- Use 3 bullet points
- Focus on practical benefits
- Mention relevant libraries
- Keep it concise (max 150 words)"""
```

### 3. RAG (Retrieval-Augmented Generation) (4 horas)

#### Arquitectura de RAG

```
User Query: "What is the return policy?"
    ↓
[Query Embedding] → Vector 384D
    ↓
[Vector Database Search] → Top-K documentos relevantes
    ↓
[Retrieved Context]:
- "Our return policy allows 30 days..."
- "Items must be in original packaging..."
- "Refunds processed within 7 business days..."
    ↓
[Prompt Construction]:
System: "Answer based only on the provided context"
Context: [Retrieved docs]
Query: "What is the return policy?"
    ↓
[LLM Generation] → Respuesta basada en contexto
    ↓
"Based on our policy, you have 30 days to return items..."
```

#### Implementación Básica de RAG

**Paso 1: Chunking de Documentos**
```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """Divide texto en chunks con overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap para contexto
    
    return chunks

# Ejemplo
document = """
[Documento largo sobre política de devoluciones...]
"""

chunks = chunk_text(document, chunk_size=500, overlap=100)
print(f"Documento dividido en {len(chunks)} chunks")
```

**Paso 2: Generar Embeddings**
```python
from sentence_transformers import SentenceTransformer

# Modelo de embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generar embeddings de chunks
chunk_embeddings = embedder.encode(chunks)
print(f"Embeddings shape: {chunk_embeddings.shape}")
```

**Paso 3: Indexar en Vector Database**
```python
import faiss
import numpy as np

# Crear índice FAISS
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings.astype('float32'))

# Metadata de chunks
chunk_metadata = [
    {'text': chunk, 'source': 'return_policy.pdf', 'chunk_id': i}
    for i, chunk in enumerate(chunks)
]
```

**Paso 4: Retrieval**
```python
def retrieve_relevant_chunks(query, embedder, index, chunks, top_k=3):
    """Recupera chunks más relevantes para la query"""
    # Embedding de query
    query_embedding = embedder.encode([query]).astype('float32')
    
    # Búsqueda en índice
    distances, indices = index.search(query_embedding, top_k)
    
    # Recuperar chunks
    retrieved = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        retrieved.append({
            'text': chunks[idx],
            'score': 1 / (1 + dist),  # Convertir distancia a similarity
            'rank': i + 1
        })
    
    return retrieved

# Ejemplo
query = "What is the return policy?"
relevant_chunks = retrieve_relevant_chunks(query, embedder, index, chunks)

for chunk in relevant_chunks:
    print(f"Rank {chunk['rank']} (Score: {chunk['score']:.3f})")
    print(chunk['text'][:200])
    print()
```

**Paso 5: Generación con Contexto**
```python
def rag_query(query, embedder, index, chunks, llm_client):
    """Sistema RAG completo"""
    # 1. Retrieve
    relevant = retrieve_relevant_chunks(query, embedder, index, chunks, top_k=3)
    
    # 2. Construir contexto
    context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant])
    
    # 3. Prompt para LLM
    prompt = f"""Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
    
    # 4. Generate
    response = llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    
    return {
        'query': query,
        'answer': answer,
        'sources': relevant
    }

# Usar RAG
result = rag_query(query, embedder, index, chunks, client)
print("Query:", result['query'])
print("\nAnswer:", result['answer'])
print("\nSources used:", len(result['sources']))
```

#### RAG Avanzado: Chunking Semántico

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Chunking más inteligente
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Prioriza párrafos
)

chunks = splitter.split_text(document)
```

#### Re-ranking de Resultados

```python
from sentence_transformers import CrossEncoder

# Modelo de re-ranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, retrieved_chunks, top_k=3):
    """Re-rankea resultados con cross-encoder"""
    # Pares query-documento
    pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
    
    # Scores de re-ranking
    scores = reranker.predict(pairs)
    
    # Reordenar
    for chunk, score in zip(retrieved_chunks, scores):
        chunk['rerank_score'] = score
    
    reranked = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)
    return reranked[:top_k]
```

### 4. Evaluación de RAG (2 horas)

#### Métricas de Retrieval

**Recall@K:**
```python
def recall_at_k(retrieved_ids, relevant_ids, k=3):
    """
    Recall@K: ¿Cuántos docs relevantes recuperamos en top-K?
    """
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    
    if len(relevant) == 0:
        return 0.0
    
    recall = len(retrieved_k & relevant) / len(relevant)
    return recall

# Ejemplo
retrieved = [1, 5, 3, 8, 2]  # IDs recuperados
relevant = [1, 3, 7]         # IDs realmente relevantes

print(f"Recall@3: {recall_at_k(retrieved, relevant, k=3)}")  # 2/3 = 0.67
print(f"Recall@5: {recall_at_k(retrieved, relevant, k=5)}")  # 2/3 = 0.67
```

**Mean Reciprocal Rank (MRR):**
```python
def mrr(retrieved_ids, relevant_ids):
    """
    MRR: Posición del primer documento relevante
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

print(f"MRR: {mrr(retrieved, relevant)}")  # 1/1 = 1.0 (primer resultado es relevante)
```

#### Métricas de Generación

**Faithfulness (Fidelidad al Contexto):**
```python
def check_faithfulness(answer, context):
    """
    Verifica si respuesta está basada en contexto
    (Simplificado - en realidad usarías otro LLM)
    """
    prompt = f"""Given the following context and answer, 
determine if the answer is faithful to the context (only uses information from context).

Context: {context}

Answer: {answer}

Is the answer faithful to the context? Respond with only "Yes" or "No"."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip().lower() == "yes"
```

**Evaluación con Test Cases:**
```python
test_cases = [
    {
        'query': 'What is the return policy?',
        'expected_keywords': ['30 days', 'original packaging', 'refund'],
        'relevant_doc_ids': [0, 1, 5]
    },
    {
        'query': 'How long does shipping take?',
        'expected_keywords': ['5-7 business days', 'standard shipping'],
        'relevant_doc_ids': [3, 4]
    }
]

def evaluate_rag(rag_system, test_cases):
    results = []
    
    for case in test_cases:
        response = rag_system.query(case['query'])
        
        # Recall
        retrieved_ids = [doc['id'] for doc in response['sources']]
        recall = recall_at_k(retrieved_ids, case['relevant_doc_ids'], k=3)
        
        # Keyword presence
        answer_lower = response['answer'].lower()
        keywords_found = sum(
            1 for kw in case['expected_keywords'] 
            if kw.lower() in answer_lower
        )
        keyword_score = keywords_found / len(case['expected_keywords'])
        
        results.append({
            'query': case['query'],
            'recall@3': recall,
            'keyword_score': keyword_score,
            'answer': response['answer']
        })
    
    return results

# Evaluar
eval_results = evaluate_rag(rag_system, test_cases)
for result in eval_results:
    print(f"Query: {result['query']}")
    print(f"Recall@3: {result['recall@3']:.2f}")
    print(f"Keyword Score: {result['keyword_score']:.2f}")
    print()
```

### 5. Manejo de Costos y Rate Limits (1 hora)

```python
import time
from functools import wraps

def retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1,
    backoff_factor=2
):
    """Decorator para retry con backoff exponencial"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
            
        return wrapper
    return decorator

@retry_with_exponential_backoff(max_retries=3)
def call_llm_with_retry(client, messages):
    """Llamada a LLM con retry"""
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

# Estimar costos
def estimate_cost(text, model="gpt-3.5-turbo"):
    """Estima costo de procesar texto"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = len(encoding.encode(text))
    
    # Precios aproximados (verificar precios actuales)
    prices = {
        "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000}
    }
    
    # Asumiendo ratio output/input = 0.5
    total_cost = tokens * prices[model]["input"] + (tokens * 0.5) * prices[model]["output"]
    
    return {
        'tokens': tokens,
        'estimated_cost_usd': total_cost
    }

# Ejemplo
text = "Long document..."
cost = estimate_cost(text)
print(f"Tokens: {cost['tokens']}, Cost: ${cost['estimated_cost_usd']:.4f}")
```

## 🎯 Prácticas Guiadas

1. **Prompting Basics** (2h) - Zero-shot, few-shot, CoT
2. **RAG Básico** (3h) - Implementación completa
3. **Chunking Strategies** (2h) - Comparar métodos
4. **Evaluación de RAG** (2h) - Test cases, métricas
5. **Optimización** (2h) - Re-ranking, costos

## ✏️ Ejercicios

1. Sistema RAG para documentación técnica
2. Comparar estrategias de chunking (fixed vs semantic)
3. Implementar re-ranking
4. A/B testing de prompts

## 📚 Recursos Externos

### 📹 Videos
- [Andrew Ng: Building Systems with LLMs](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)
- [RAG Explained](https://www.youtube.com/watch?v=T-D1OfcDW1M)

### 📖 Lecturas
- [RAG Paper](https://arxiv.org/abs/2005.11401) - Lewis et al.
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)

### 🛠️ Herramientas
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Chroma](https://www.trychroma.com/)

## ✅ Checklist

- [ ] Dominar técnicas de prompting
- [ ] Implementar RAG funcional
- [ ] Chunking apropiado
- [ ] Sistema de evaluación
- [ ] Manejo de costos y errores

## ➡️ Siguientes Pasos

**Módulo 7:** Agentes de IA y Orquestación con n8n
