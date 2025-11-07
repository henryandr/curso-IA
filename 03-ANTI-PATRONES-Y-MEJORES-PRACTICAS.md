# Anti-Patrones, Riesgos y Mejores Prácticas en ML/IA

## Tabla de Contenidos
1. [Data Leakage](#data-leakage)
2. [Evaluación Incorrecta](#evaluación-incorrecta)
3. [Overfitting y Underfitting](#overfitting-y-underfitting)
4. [Reproducibilidad](#reproducibilidad)
5. [Seguridad y Secretos](#seguridad-y-secretos)
6. [n8n y Orquestación](#n8n-y-orquestación)
7. [LLMs y RAG](#llms-y-rag)
8. [Infraestructura y Despliegue](#infraestructura-y-despliegue)

---

## 1. Data Leakage

### ❌ Anti-Patrón 1.1: Normalizar/Escalar Antes del Split

**Problema:**
```python
# MAL: Escala usando estadísticas de TODO el dataset
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # ❌ Usa info del test set

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

**¿Por qué es malo?**
- El scaler aprende estadísticas (media, std) del test set
- En producción, no tendrás acceso al test set
- Genera métricas optimistas e irreales

**✅ Solución:**
```python
# BIEN: Split primero, luego escala solo con train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✅ Aprende de train
X_test_scaled = scaler.transform(X_test)        # ✅ Solo transforma test
```

**Mejor aún: Usar pipelines**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# El pipeline maneja el fit/transform correctamente
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

### ❌ Anti-Patrón 1.2: Feature Engineering con Info del Test

**Problema:**
```python
# MAL: Calcula target encoding con todo el dataset
def target_encode(df, col, target):
    means = df.groupby(col)[target].mean()  # ❌ Incluye test
    return df[col].map(means)

df['category_encoded'] = target_encode(df, 'category', 'target')
train, test = train_test_split(df)
```

**✅ Solución:**
```python
# BIEN: Calcula solo con train, aplica a test
train, test = train_test_split(df)

means = train.groupby('category')['target'].mean()  # ✅ Solo train
train['category_encoded'] = train['category'].map(means)
test['category_encoded'] = test['category'].map(means)
```

---

### ❌ Anti-Patrón 1.3: Imputación Incorrecta

**Problema:**
```python
# MAL: Imputa con estadísticas de todo el dataset
df['age'].fillna(df['age'].median(), inplace=True)  # ❌
train, test = train_test_split(df)
```

**✅ Solución:**
```python
# BIEN: Split primero, imputa con estadísticas de train
train, test = train_test_split(df)
median_age = train['age'].median()  # ✅ Solo train
train['age'].fillna(median_age, inplace=True)
test['age'].fillna(median_age, inplace=True)
```

---

## 2. Evaluación Incorrecta

### ❌ Anti-Patrón 2.1: Evaluar Repetidamente en Test Set

**Problema:**
```python
# MAL: Optimiza hiperparámetros mirando el test set
for C in [0.1, 1, 10]:
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌ Repites test
    print(f'C={C}, score={score}')
# Eliges el mejor C basado en test
```

**¿Por qué es malo?**
- El test set se "contamina" con tus decisiones
- Overfitting indirecto al test set
- Métricas finales optimistas

**✅ Solución: Usar validación cruzada**
```python
from sklearn.model_selection import cross_val_score

for C in [0.1, 1, 10]:
    model = LogisticRegression(C=C)
    # ✅ Evalúa en train con CV
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f'C={C}, CV score={scores.mean():.3f} ± {scores.std():.3f}')

# Eliges mejor C, LUEGO evalúas UNA VEZ en test
best_model = LogisticRegression(C=1)
best_model.fit(X_train, y_train)
final_score = best_model.score(X_test, y_test)  # ✅ Solo una vez
```

---

### ❌ Anti-Patrón 2.2: Métricas Inapropiadas

**Problema:**
```python
# MAL: Usar accuracy en dataset muy desbalanceado
# Dataset: 95% clase 0, 5% clase 1
accuracy = model.score(X_test, y_test)  # ❌ Puede ser 95% solo prediciendo 0
```

**✅ Solución:**
```python
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# Para desbalance, usa: Precision, Recall, F1, ROC-AUC
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f'ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}')
print(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
```

---

### ❌ Anti-Patrón 2.3: Olvidar Estratificación

**Problema:**
```python
# MAL: Split aleatorio en dataset desbalanceado
train, test = train_test_split(X, y)  # ❌ Puede tener distribuciones diferentes
```

**✅ Solución:**
```python
# BIEN: Split estratificado mantiene proporciones
train, test = train_test_split(X, y, stratify=y, random_state=42)  # ✅
```

---

## 3. Overfitting y Underfitting

### ❌ Anti-Patrón 3.1: No Validar Complejidad del Modelo

**Problema:**
```python
# MAL: Modelo muy complejo sin validación
model = RandomForestClassifier(n_estimators=1000, max_depth=None)  # ❌
model.fit(X_train, y_train)
# Sin evaluar en validación, solo en train
```

**✅ Solución:**
```python
# BIEN: Learning curves para diagnosticar
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.legend()
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.show()
```

---

### ❌ Anti-Patrón 3.2: No Usar Regularización

**Problema:**
```python
# MAL: Modelo lineal sin regularización con muchas features
model = LinearRegression()  # ❌ No penaliza complejidad
```

**✅ Solución:**
```python
# BIEN: Usa Ridge/Lasso con CV para encontrar alpha
from sklearn.linear_model import RidgeCV

model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)  # ✅
model.fit(X_train, y_train)
print(f'Best alpha: {model.alpha_}')
```

---

## 4. Reproducibilidad

### ❌ Anti-Patrón 4.1: No Fijar Seeds

**Problema:**
```python
# MAL: Resultados no reproducibles
train, test = train_test_split(X, y)  # ❌ Diferentes cada vez
model = RandomForestClassifier()
model.fit(train_X, train_y)  # ❌ Árboles diferentes cada ejecución
```

**✅ Solución:**
```python
# BIEN: Fija seeds en todas partes
import numpy as np
import random

# Fija seeds globales
np.random.seed(42)
random.seed(42)

train, test = train_test_split(X, y, random_state=42)  # ✅
model = RandomForestClassifier(random_state=42)  # ✅
model.fit(train_X, train_y)
```

---

### ❌ Anti-Patrón 4.2: No Documentar Versiones

**Problema:**
```bash
# MAL: requirements.txt sin versiones
# pandas
# scikit-learn
# numpy
```

**✅ Solución:**
```bash
# BIEN: requirements.txt con versiones específicas
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3

# O genera automáticamente:
pip freeze > requirements.txt
```

---

### ❌ Anti-Patrón 4.3: No Versionar Datos y Modelos

**Problema:**
```python
# MAL: Sobreescribe modelos sin tracking
import joblib
joblib.dump(model, 'model.pkl')  # ❌ Pierdes versión anterior
```

**✅ Solución:**
```python
# BIEN: Versiona con timestamp o MLflow
from datetime import datetime
import mlflow

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(model, f'models/model_{timestamp}.pkl')  # ✅

# Mejor aún: usa MLflow
mlflow.sklearn.log_model(model, 'model')
mlflow.log_params(model.get_params())
mlflow.log_metrics({'accuracy': accuracy})
```

---

## 5. Seguridad y Secretos

### ❌ Anti-Patrón 5.1: Hardcodear API Keys

**Problema:**
```python
# MAL: API key en el código
import openai
openai.api_key = "sk-proj-abc123def456..."  # ❌ NUNCA HAGAS ESTO
```

**✅ Solución:**
```python
# BIEN: Usa variables de entorno
import os
from dotenv import load_dotenv

load_dotenv()  # Carga .env
openai.api_key = os.getenv('OPENAI_API_KEY')  # ✅

# .env (en .gitignore)
# OPENAI_API_KEY=sk-proj-abc123def456...

# .gitignore
# .env
# *.key
# secrets/
```

---

### ❌ Anti-Patrón 5.2: Commitear Secretos

**Problema:**
```python
# MAL: config.py en el repo
DATABASE_URL = "postgresql://user:password@host/db"  # ❌
AWS_SECRET_KEY = "wJalrXUtn..."  # ❌
```

**✅ Solución:**
```python
# config.py - solo estructura
import os

DATABASE_URL = os.getenv('DATABASE_URL')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')

# .env.example (en repo)
# DATABASE_URL=postgresql://user:password@localhost/db
# AWS_SECRET_KEY=your_secret_here

# .env (en .gitignore, con valores reales)
```

---

### ❌ Anti-Patrón 5.3: Logs con Información Sensible

**Problema:**
```python
# MAL: Loguea datos sensibles
logger.info(f"User query: {user_email} - {sensitive_data}")  # ❌
```

**✅ Solución:**
```python
# BIEN: Sanitiza o enmascara
logger.info(f"User query: {user_email[:3]}*** - [REDACTED]")  # ✅
```

---

## 6. n8n y Orquestación

### ❌ Anti-Patrón 6.1: No Manejar Errores en Workflows

**Problema:**
```
[HTTP Request] → [Process Data] → [Save to DB]
```
Si algún nodo falla, todo el workflow se detiene sin recuperación.

**✅ Solución:**
```
[HTTP Request] 
    ↓ (on success)
[Process Data]
    ↓ (on success)        ↓ (on error)
[Save to DB]          [Error Handler]
                           ↓
                    [Log Error + Notify Slack]
                           ↓
                    [Retry Logic (optional)]
```

**Configuración en n8n:**
- Usar "On Error" connections
- Configurar "Retry On Fail" en nodes críticos
- Implementar nodo de error handling

---

### ❌ Anti-Patrón 6.2: Secretos Hardcodeados en n8n

**Problema:**
- Poner API keys directamente en HTTP Request nodes

**✅ Solución:**
- Usar n8n Credentials para almacenar secretos
- Referencia credentials en nodes, no valores directos
- Variables de entorno en n8n Docker

```yaml
# docker-compose.yml
services:
  n8n:
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
```

---

### ❌ Anti-Patrón 6.3: Workflows Sin Logging

**Problema:**
```
[Webhook] → [Process] → [API Call]
```
No hay visibilidad de qué pasó cuando algo falla.

**✅ Solución:**
```
[Webhook] → [Log Input] → [Process] → [Log Process] → [API Call] → [Log Output]
                                  ↓ (on error)
                            [Log Error Details]
```

Usar:
- Function nodes para logging
- Google Sheets/Database para audit trail
- Timestamps en todos los logs

---

## 7. LLMs y RAG

### ❌ Anti-Patrón 7.1: No Validar Entrada del Usuario

**Problema:**
```python
# MAL: Usa input del usuario directamente
user_query = request.json['query']
response = llm.invoke(user_query)  # ❌ Prompt injection
```

**✅ Solución:**
```python
# BIEN: Valida y sanitiza
from pydantic import BaseModel, validator

class Query(BaseModel):
    query: str
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        if any(word in v.lower() for word in ['ignore', 'forget']):
            raise ValueError('Potentially malicious query')
        return v

try:
    validated_query = Query(query=request.json['query'])
    response = llm.invoke(validated_query.query)
except ValueError as e:
    return {'error': str(e)}
```

---

### ❌ Anti-Patrón 7.2: Chunks Demasiado Grandes o Pequeños

**Problema:**
```python
# MAL: Chunks fijos sin considerar contenido
chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # ❌
```

**✅ Solución:**
```python
# BIEN: Chunking semántico con overlap
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # ✅ Overlap para contexto
    separators=["\n\n", "\n", ". ", " ", ""]  # ✅ Respeta estructura
)
chunks = splitter.split_text(text)
```

---

### ❌ Anti-Patrón 7.3: No Evaluar Calidad de RAG

**Problema:**
```python
# MAL: Implementa RAG sin medir calidad
def rag_query(query):
    docs = retriever.get_relevant_documents(query)
    return llm(f"Context: {docs}\nQuestion: {query}")
# No hay evaluación
```

**✅ Solución:**
```python
# BIEN: Define métricas y evalúa
test_cases = [
    {'query': 'What is X?', 'expected': 'X is...', 'relevant_docs': [...]},
    # ...
]

for case in test_cases:
    docs = retriever.get_relevant_documents(case['query'])
    response = llm(f"Context: {docs}\nQuestion: {case['query']}")
    
    # Métricas
    recall = calculate_recall(docs, case['relevant_docs'])
    faithfulness = check_faithfulness(response, docs)
    
    print(f"Query: {case['query']}")
    print(f"Recall@k: {recall:.2f}")
    print(f"Faithfulness: {faithfulness}")
```

---

## 8. Infraestructura y Despliegue

### ❌ Anti-Patrón 8.1: No Manejar Timeouts

**Problema:**
```python
# MAL: Sin timeout, puede colgar indefinidamente
@app.post('/predict')
def predict(data: dict):
    result = model.predict(data)  # ❌ Puede tardar mucho
    return result
```

**✅ Solución:**
```python
# BIEN: Con timeout y async
from fastapi import HTTPException
import asyncio

@app.post('/predict')
async def predict(data: dict):
    try:
        result = await asyncio.wait_for(
            run_prediction(data),
            timeout=30.0  # ✅ 30 segundos máximo
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail='Prediction timeout')
```

---

### ❌ Anti-Patrón 8.2: No Limitar Rate Limits

**Problema:**
```python
# MAL: API sin rate limiting
@app.post('/expensive-operation')
def expensive_op():
    # Operación costosa sin límite
    pass
```

**✅ Solución:**
```python
# BIEN: Con rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post('/expensive-operation')
@limiter.limit("5/minute")  # ✅ Max 5 requests por minuto
def expensive_op():
    pass
```

---

### ❌ Anti-Patrón 8.3: No Validar Entrada de API

**Problema:**
```python
# MAL: Acepta cualquier entrada
@app.post('/predict')
def predict(data: dict):  # ❌ dict genérico
    return model.predict(data['features'])
```

**✅ Solución:**
```python
# BIEN: Valida con Pydantic
from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    features: list[float] = Field(..., min_items=10, max_items=10)
    model_version: str = Field(default='v1')

@app.post('/predict')
def predict(data: PredictionInput):  # ✅ Validación automática
    return model.predict(data.features)
```

---

## Checklist General de Mejores Prácticas

### Antes de Entrenar
- [ ] Datos divididos ANTES de cualquier transformación
- [ ] Seeds fijados para reproducibilidad
- [ ] Estratificación en splits si hay desbalance
- [ ] Sin información del test en transformaciones

### Durante el Desarrollo
- [ ] Validación cruzada para evaluar
- [ ] Métricas apropiadas al problema
- [ ] Learning curves para diagnosticar
- [ ] Tracking de experimentos (MLflow/W&B)

### Antes de Desplegar
- [ ] Evaluación final UNA VEZ en test
- [ ] Código versionado en Git
- [ ] Secretos en variables de entorno
- [ ] Requirements.txt con versiones
- [ ] Documentación actualizada
- [ ] Tests automatizados

### En Producción
- [ ] Logging estructurado
- [ ] Manejo de errores
- [ ] Timeouts configurados
- [ ] Rate limiting
- [ ] Monitoreo de métricas
- [ ] Versionado de modelos

---

**Recuerda:** Estos anti-patrones son comunes incluso entre profesionales. El objetivo es reconocerlos y evitarlos de forma sistemática.
