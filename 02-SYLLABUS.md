# Syllabus: Curso de Inteligencia Artificial y Análisis de Datos con Python

## Información General del Curso

**Título:** Inteligencia Artificial y Análisis de Datos con Python: De ML Clásico a Agentes y Orquestación con n8n

**Nivel:** Intermedio-Avanzado

**Prerrequisitos:**
- Python 3.10+ (control de flujo, funciones, OOP básica, paquetes, virtualenv/poetry)
- Nociones básicas de estadística (media, varianza, correlación)
- Álgebra lineal básica (vectores, matrices)
- Probabilidad básica

**Duración Total:** 8 semanas (10-15 horas/semana) = 80-120 horas

**Modalidad:** Autoguiado con prácticas, ejercicios, quizzes y proyectos capstone

---

## Objetivos Generales del Curso

Al completar este curso, serás capaz de:

1. **Análizar y preparar datos** usando pandas, numpy, matplotlib y seaborn
2. **Construir modelos de ML clásico** con scikit-learn y evaluar su rendimiento
3. **Implementar pipelines reproducibles** con validación, tuning y tracking
4. **Aplicar técnicas de NLP moderno** con embeddings y transformers (Hugging Face)
5. **Desarrollar modelos de Deep Learning** con PyTorch
6. **Trabajar con LLMs**: prompting, RAG (Retrieval-Augmented Generation) y evaluación
7. **Diseñar y programar agentes** con herramientas, memoria y planificación
8. **Orquestar flujos de IA** con n8n (webhooks, APIs, automatización)
9. **Aplicar MLOps ligero**: tracking, reproducibilidad, versionado
10. **Implementar IA responsable**: privacidad, seguridad, reducción de sesgos

---

## Estructura del Curso

### Módulo 0 (Opcional): Fundamentos Matemáticos para ML
**Duración:** 1 semana (6-8 horas)  
**Objetivo:** Repasar conceptos matemáticos esenciales

**Temas:**
- Estadística descriptiva: media, mediana, varianza, desviación estándar
- Distribuciones de probabilidad: normal, binomial
- Álgebra lineal: vectores, matrices, producto punto, normas
- Cálculo básico: derivadas, gradientes, optimización
- Correlación y covarianza

**Prácticas:**
1. Análisis estadístico exploratorio con numpy
2. Operaciones con matrices y vectores
3. Normalización y estandarización de datos
4. Correlaciones y scatter plots

**Ejercicios:**
1. Calcular estadísticas descriptivas de un dataset
2. Implementar normalización min-max y z-score desde cero
3. Visualizar distribuciones y correlaciones

**Quiz:** 8 preguntas sobre estadística y álgebra lineal aplicada

**Entregables:**
- Notebook de repaso completado
- Ejercicios resueltos

---

### Módulo 1: EDA y Preparación de Datos con Pandas
**Duración:** 1 semana (10-12 horas)  
**Objetivos Medibles:**
- Cargar, explorar y limpiar datasets con pandas (Aplicar, Bloom 3)
- Identificar y tratar valores perdidos y outliers (Analizar, Bloom 4)
- Realizar encoding de variables categóricas (Aplicar, Bloom 3)
- Crear visualizaciones informativas con seaborn/matplotlib (Crear, Bloom 6)

**Contenidos:**
- Introducción a pandas: DataFrames, Series, indexación
- Limpieza de datos: valores nulos, duplicados, inconsistencias
- Detección y tratamiento de outliers: IQR, z-score
- Feature engineering básico: binning, one-hot encoding, label encoding
- Escalado: StandardScaler, MinMaxScaler, RobustScaler
- Visualización: histogramas, boxplots, scatter plots, correlation heatmaps

**Prácticas Guiadas:**
1. EDA completo de dataset de churn de clientes (Telco)
2. Limpieza y transformación de dataset de ventas
3. Detección de outliers en datos de precios inmobiliarios
4. Feature engineering para dataset de marketing
5. Dashboard de visualización con seaborn

**Ejercicios de Consolidación:**
1. EDA de dataset de Titanic: limpieza, visualización, insights
2. Imputación de valores faltantes: comparar 3 estrategias y su impacto
3. Crear pipeline de preprocesamiento reproducible
4. Informe ejecutivo con visualizaciones clave

**Mini-Quiz:** 6 preguntas (conceptuales y de código)

**Tiempo Estimado por Actividad:**
- Teoría y lectura: 2 horas
- Prácticas guiadas: 5 horas
- Ejercicios: 3-4 horas
- Quiz y revisión: 1 hora

**Recursos:**
- Documentación oficial de pandas
- "Python for Data Analysis" (Wes McKinney) - capítulos seleccionados
- Kaggle: Titanic, House Prices datasets

**Criterios de Evaluación:**
- Código limpio y documentado
- Identificación correcta de problemas en datos
- Visualizaciones claras y relevantes
- Justificación de decisiones de limpieza

---

### Módulo 2: Fundamentos de Machine Learning con Scikit-Learn
**Duración:** 1.5 semanas (12-15 horas)  
**Objetivos Medibles:**
- Entrenar modelos de regresión y clasificación (Aplicar, Bloom 3)
- Evaluar modelos con métricas apropiadas (Analizar, Bloom 4)
- Implementar validación cruzada correctamente (Aplicar, Bloom 3)
- Construir pipelines con scikit-learn (Crear, Bloom 6)

**Contenidos:**
- Conceptos: aprendizaje supervisado, función de costo, overfitting/underfitting
- Flujo de trabajo ML: train/test split, validación cruzada
- Regresión: Linear, Ridge, Lasso
- Clasificación: Logistic Regression, KNN, Naive Bayes
- Métricas:
  - Regresión: MAE, MSE, RMSE, R², MAPE
  - Clasificación: Accuracy, Precision, Recall, F1, ROC-AUC
- Pipelines y ColumnTransformer
- Estrategias de baseline

**Prácticas Guiadas:**
1. Predicción de precios de casas (regresión)
2. Clasificación de churn de clientes
3. Construcción de pipeline completo con preprocesamiento
4. Comparación de modelos baseline
5. Análisis de matriz de confusión y curvas ROC

**Ejercicios de Consolidación:**
1. Pipeline de regresión: datos → preprocesamiento → modelo → evaluación
2. Clasificación binaria: baseline vs modelo optimizado
3. Selección de métricas apropiadas para problema de negocio
4. Diagnóstico de overfitting con learning curves

**Mini-Quiz:** 8 preguntas

**Tiempo Estimado:**
- Teoría: 3 horas
- Prácticas: 6 horas
- Ejercicios: 4-5 horas
- Quiz: 1 hora

**Recursos:**
- Documentación scikit-learn
- "Hands-On Machine Learning" (Aurélien Géron) - caps. 1-4
- Datasets: California Housing, Adult Income

**Criterios de Evaluación:**
- Splits de datos correctos (sin leakage)
- Métricas apropiadas al problema
- Pipeline reproducible
- Interpretación correcta de resultados

---

### Módulo 3: Selección y Optimización de Modelos
**Duración:** 1.5 semanas (12-15 horas)  
**Objetivos Medibles:**
- Comparar múltiples algoritmos sistemáticamente (Evaluar, Bloom 5)
- Aplicar técnicas de feature selection (Aplicar, Bloom 3)
- Optimizar hiperparámetros con Grid/RandomSearch (Aplicar, Bloom 3)
- Usar experiment tracking (MLflow/W&B) (Aplicar, Bloom 3)

**Contenidos:**
- Modelos basados en árboles: Decision Trees, Random Forest
- Gradient Boosting: XGBoost, LightGBM, CatBoost
- Feature importance y selection
- Hyperparameter tuning: Grid Search, Random Search, Bayesian Optimization (opcional)
- Ensembles: bagging, boosting, stacking
- Experiment tracking con MLflow o Weights & Biases
- Estrategias de imbalanced data: SMOTE, class weights

**Prácticas Guiadas:**
1. Comparación RF vs XGBoost en clasificación
2. Feature importance y selection
3. Hyperparameter tuning con GridSearchCV
4. Tracking de experimentos con MLflow
5. Manejo de clases desbalanceadas

**Ejercicios de Consolidación:**
1. Optimizar modelo para dataset de fraude (imbalanced)
2. Comparar 5 modelos con validación cruzada y tracking
3. Feature selection: backward elimination o RFE
4. Ensemble de modelos heterogéneos

**Mini-Quiz:** 7 preguntas

**Tiempo Estimado:**
- Teoría: 3 horas
- Prácticas: 6-7 horas
- Ejercicios: 4-5 horas
- Quiz: 1 hora

**Recursos:**
- XGBoost documentation
- MLflow tutorials
- Papers: "XGBoost: A Scalable Tree Boosting System"
- Datasets: Credit Card Fraud, Lending Club

**Evaluación:**
- Justificación de elección de modelos
- Experimentos documentados y reproducibles
- Interpretación de feature importance
- Prevención de overfitting

**Entregable Clave:** Reporte de comparación de modelos con MLflow

---

### CAPSTONE 1: Pipeline ML Clásico End-to-End
**Duración:** 1 semana (10-12 horas)  
**Peso:** 30% de la nota final

**Descripción:**
Construir un pipeline completo de ML para un problema tabular real (clasificación o regresión), desde EDA hasta modelo final con evaluación honesta.

**Requisitos:**
1. EDA completo documentado
2. Limpieza y feature engineering justificados
3. Pipeline de scikit-learn con preprocesamiento
4. Comparación de al menos 3 modelos
5. Validación cruzada apropiada
6. Evaluación en test set (sin leakage)
7. Tracking con MLflow/W&B
8. Reporte técnico (2-3 páginas)
9. Código reproducible con requirements.txt

**Datasets Sugeridos:**
- Bank Marketing
- Employee Attrition
- Loan Default Prediction
- Customer Segmentation

**Rúbrica:**
- EDA y visualizaciones (20%)
- Feature engineering y justificación (20%)
- Pipeline y reproducibilidad (20%)
- Evaluación y métricas (20%)
- Interpretación y reporte (20%)

**Criterios de Aprobación:**
- Sin data leakage
- Métricas apropiadas y bien interpretadas
- Código ejecutable y documentado
- Mejora sobre baseline demostrada

---

### Módulo 4: NLP Moderno y Embeddings
**Duración:** 1.5 semanas (12-15 horas)  
**Objetivos Medibles:**
- Preprocesar texto para NLP (Aplicar, Bloom 3)
- Generar y usar embeddings de palabras y documentos (Aplicar, Bloom 3)
- Implementar clasificación de texto con transformers (Crear, Bloom 6)
- Realizar búsqueda semántica con vectores (Aplicar, Bloom 3)

**Contenidos:**
- Preprocesamiento NLP: tokenización, stemming, lemmatization, stop words
- Representaciones clásicas: Bag of Words, TF-IDF
- Word embeddings: Word2Vec, GloVe (conceptos)
- Sentence embeddings: Sentence-BERT
- Transformers: arquitectura, atención, tokenización
- Hugging Face Transformers: pipelines, fine-tuning
- Bases de datos vectoriales: FAISS, ChromaDB
- Búsqueda semántica y similitud de coseno

**Prácticas Guiadas:**
1. Clasificación de sentiment con TF-IDF + LogisticRegression
2. Generación de embeddings con Sentence-BERT
3. Búsqueda semántica con FAISS
4. Fine-tuning de BERT para clasificación de texto
5. Comparación: TF-IDF vs embeddings vs transformers

**Ejercicios de Consolidación:**
1. Clasificador de spam con embeddings
2. Sistema de búsqueda semántica de documentos
3. Análisis de similarity entre textos
4. Evaluación de calidad de embeddings (recall@k)

**Mini-Quiz:** 8 preguntas

**Tiempo Estimado:**
- Teoría: 4 horas
- Prácticas: 6-7 horas
- Ejercicios: 3-4 horas
- Quiz: 1 hora

**Recursos:**
- Hugging Face course (modules 1-4)
- "Natural Language Processing with Transformers"
- Papers: "Attention Is All You Need", "BERT"
- Datasets: IMDB reviews, AG News, Twitter Sentiment

**Evaluación:**
- Preprocesamiento correcto
- Uso apropiado de embeddings
- Evaluación con métricas de NLP
- Interpretación de resultados

---

### Módulo 5: Deep Learning con PyTorch (Introducción Pragmática)
**Duración:** 1.5 semanas (12-15 horas)  
**Objetivos Medibles:**
- Construir redes neuronales con PyTorch (Crear, Bloom 6)
- Entrenar modelos con backpropagation (Aplicar, Bloom 3)
- Implementar early stopping y regularización (Aplicar, Bloom 3)
- Aplicar transfer learning (Aplicar, Bloom 3)

**Contenidos:**
- Fundamentos de Deep Learning: perceptrón, activaciones, backpropagation
- PyTorch: tensors, autograd, nn.Module
- Construcción de arquitecturas: feedforward, CNN (conceptual)
- Loss functions: MSE, CrossEntropy
- Optimizers: SGD, Adam, learning rate scheduling
- Regularización: Dropout, L1/L2, batch normalization
- Early stopping y checkpointing
- Transfer learning con modelos preentrenados

**Prácticas Guiadas:**
1. Red neuronal simple para datos tabulares
2. Clasificación de texto con red feedforward
3. Early stopping y regularización
4. Fine-tuning de modelo preentrenado
5. Visualización de learning curves

**Ejercicios de Consolidación:**
1. MLP para clasificación multi-clase
2. Comparación: sklearn vs PyTorch para mismo problema
3. Experimentar con arquitecturas y hiperparámetros
4. Transfer learning para NLP

**Mini-Quiz:** 7 preguntas

**Tiempo Estimado:**
- Teoría: 4 horas
- Prácticas: 6 horas
- Ejercicios: 4-5 horas
- Quiz: 1 hora

**Recursos:**
- PyTorch tutorials oficiales
- "Deep Learning with PyTorch" (Manning)
- Fast.ai course (selected lessons)
- Datasets: MNIST (opcional), Adult, News Classification

**Evaluación:**
- Arquitectura apropiada al problema
- Entrenamiento estable (loss curves)
- Prevención de overfitting
- Código reproducible con seeds

---

### Módulo 6: LLMs, Prompting y RAG
**Duración:** 1.5 semanas (12-15 horas)  
**Objetivos Medibles:**
- Diseñar prompts efectivos (Crear, Bloom 6)
- Implementar sistema RAG básico (Crear, Bloom 6)
- Evaluar respuestas de LLMs (Evaluar, Bloom 5)
- Usar APIs de LLMs de forma segura (Aplicar, Bloom 3)

**Contenidos:**
- LLMs: arquitectura (conceptual), context window, tokenización
- Prompting: zero-shot, few-shot, chain-of-thought
- APIs: OpenAI, Anthropic, modelos locales (Llama, Mistral)
- Embeddings para RAG: text-embedding-ada, sentence-transformers
- RAG (Retrieval-Augmented Generation):
  - Chunking strategies
  - Indexado y búsqueda vectorial
  - Re-ranking
  - Generación con contexto
- Evaluación:
  - Exact match, BLEU, ROUGE (conceptual)
  - Faithfulness, relevance
  - Human-in-the-loop
- Costos, rate limits y manejo de errores

**Prácticas Guiadas:**
1. Prompting básico y avanzado con OpenAI API
2. Generación de embeddings e indexado con FAISS
3. Chunking de documentos (fixed-size, semantic)
4. Mini-RAG: pregunta → retrieval → generación
5. Evaluación A/B de prompts

**Ejercicios de Consolidación:**
1. Sistema RAG para documentación técnica
2. Comparar estrategias de chunking
3. Implementar re-ranking simple
4. Evaluar respuestas con métricas automáticas y manuales

**Mini-Quiz:** 8 preguntas

**Tiempo Estimado:**
- Teoría: 3-4 horas
- Prácticas: 6-7 horas
- Ejercicios: 4-5 horas
- Quiz: 1 hora

**Recursos:**
- OpenAI documentation
- Hugging Face "Embeddings" guide
- Papers: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- LangChain documentation (referencia)
- Datasets: Wikipedia articles, technical docs

**Evaluación:**
- Calidad de prompts
- Diseño de pipeline RAG
- Métricas de evaluación apropiadas
- Manejo seguro de API keys

---

### CAPSTONE 2: Proyecto NLP con Embeddings/Transformers
**Duración:** 1 semana (10-12 horas)  
**Peso:** 30% de la nota final

**Descripción:**
Construir un sistema de NLP usando embeddings o transformers para clasificación, QA o búsqueda semántica, con experimentos documentados y evaluación rigurosa.

**Requisitos:**
1. Preprocesamiento de corpus de texto
2. Implementación con embeddings o transformers
3. Experimentos con al menos 2 enfoques diferentes
4. Evaluación con métricas de NLP apropiadas
5. Análisis de errores
6. Reporte técnico con ablation study
7. Código reproducible

**Opciones de Proyecto:**
- Clasificador de documentos multi-clase
- Sistema de QA con búsqueda + generación
- Análisis de sentiment de reviews
- Clustering temático de artículos

**Datasets Sugeridos:**
- Reuters News
- Amazon Reviews
- SQuAD (QA)
- ArXiv papers

**Rúbrica:**
- Preparación y análisis de datos (15%)
- Diseño de experimentos (25%)
- Implementación técnica (25%)
- Evaluación y métricas (20%)
- Análisis de resultados (15%)

**Criterios de Aprobación:**
- Al menos 2 enfoques comparados
- Métricas de NLP correctas
- Análisis de errores documentado
- Reproducibilidad completa

---

### Módulo 7: Agentes de IA y Orquestación con n8n
**Duración:** 2 semanas (18-20 horas)  
**Objetivos Medibles:**
- Diseñar agentes con herramientas y memoria (Crear, Bloom 6)
- Implementar loops de planificación-ejecución (Crear, Bloom 6)
- Crear workflows en n8n para IA (Crear, Bloom 6)
- Integrar agentes con APIs y servicios externos (Aplicar, Bloom 3)

**Contenidos:**
- **Agentes de IA:**
  - Definición: percepción → razonamiento → acción
  - Herramientas (tools): definición, ejecución, parsing
  - Memoria: corto plazo, largo plazo, vectorial
  - Planificación: ReAct, MRKL, Plan-and-Execute
  - Frameworks: LangChain agents (opcional), implementación desde cero
  - Evaluación de agentes

- **n8n:**
  - Instalación: Docker, desktop app
  - Conceptos: nodes, connections, credentials, variables
  - Nodes clave:
    - HTTP Request, Webhook
    - Code (Python/JavaScript)
    - Cron/Schedule
    - Google Sheets, Slack, Discord, GitHub
    - Error handling, retries
  - Workflows para IA:
    - ETL de datos
    - Orquestación de inferencia
    - Pipelines RAG
  - Seguridad: manejo de secretos, environment variables
  - Debugging y logging

**Prácticas Guiadas:**
1. Agente simple con 2-3 herramientas (calculadora, búsqueda)
2. Agente con memoria conversacional
3. n8n Workflow 1: ETL (CSV/Google Sheets → limpieza → PostgreSQL/Sheets)
4. n8n Workflow 2: Orquestación (cron → llamar API FastAPI → log a Sheets → notificar Slack)
5. n8n Workflow 3: Pipeline RAG (webhook → chunking → embeddings → upsert FAISS → responder)

**Ejercicios de Consolidación:**
1. Agente que decide qué herramienta usar según consulta
2. Workflow n8n: ingesta de datos → preprocesamiento → entrenar modelo → notificar
3. RAG con n8n: chunking → indexado → endpoint de consulta
4. Manejo de errores y reintentos en workflows

**Mini-Quiz:** 8 preguntas

**Tiempo Estimado:**
- Teoría: 5 horas
- Prácticas: 10 horas
- Ejercicios: 4-5 horas
- Quiz: 1 hora

**Recursos:**
- n8n documentation
- LangChain agents guide (referencia)
- Papers: "ReAct: Synergizing Reasoning and Acting in Language Models"
- Tutoriales de n8n YouTube channel

**Evaluación:**
- Diseño de agente funcional
- Workflows n8n ejecutables
- Manejo correcto de errores
- Seguridad (secretos)
- Documentación clara

---

### CAPSTONE 3: Sistema RAG + Agente + n8n End-to-End
**Duración:** 2 semanas (20-25 horas)  
**Peso:** 40% de la nota final

**Descripción:**
Construir un sistema completo que integre RAG, un agente en Python con herramientas, y orquestación con n8n, desplegado como servicio FastAPI.

**Requisitos Técnicos:**

1. **RAG Backend (Python + FastAPI):**
   - Ingesta de documentos (PDF, txt, markdown)
   - Chunking configurable
   - Generación de embeddings
   - Almacenamiento en BD vectorial (FAISS/Chroma)
   - Endpoint `/ingest` para agregar documentos
   - Endpoint `/query` para consultas

2. **Agente:**
   - Al menos 2 herramientas (búsqueda vectorial + otra, ej: API externa)
   - Memoria conversacional
   - Loop de razonamiento (ReAct o similar)
   - Logging de decisiones

3. **Orquestación n8n:**
   - Workflow de ingesta: webhook → validar → procesar → indexar → notificar
   - Workflow de consulta: HTTP request → agente → respuesta → log
   - Manejo de errores y reintentos
   - Notificaciones (Slack/Discord/Email)

4. **Infraestructura:**
   - FastAPI con documentación (Swagger)
   - Docker Compose (opcional pero recomendado)
   - Variables de entorno para secretos
   - Logging estructurado

5. **Evaluación:**
   - 10 preguntas de prueba con respuestas esperadas
   - Métricas: exactitud, faithfulness, latencia
   - Análisis de calidad de respuestas

6. **Documentación:**
   - README con instrucciones de instalación
   - Arquitectura del sistema (diagrama)
   - Decisiones técnicas justificadas
   - Demo en video (3-5 min)

**Datasets/Dominio Sugerido:**
- Documentación técnica de un proyecto
- Base de conocimiento de soporte al cliente
- Artículos científicos de un área
- Regulaciones y compliance

**Rúbrica Detallada:**

| Componente | Peso | Criterios |
|------------|------|-----------|
| RAG Backend | 25% | Chunking, embeddings, búsqueda, API funcional |
| Agente | 25% | Herramientas, memoria, razonamiento, logging |
| n8n Workflows | 20% | Ingesta, consulta, errores, notificaciones |
| Infraestructura | 10% | FastAPI, Docker, secretos, reproducibilidad |
| Evaluación | 10% | Métricas, casos de prueba, análisis |
| Documentación | 10% | README, arquitectura, decisiones, demo |

**Criterios de Aprobación:**
- Sistema ejecutable end-to-end
- Sin secretos en código
- Al menos 8/10 queries respondidas correctamente
- Workflows n8n funcionales
- Documentación completa

**Extras (Puntos Bonus):**
- Docker Compose completo
- Tests automatizados
- Monitoring/logging avanzado
- Interface de usuario (Streamlit/Gradio)

---

## Evaluación General del Curso

### Distribución de Calificaciones

| Componente | Peso |
|------------|------|
| Capstone 1: Pipeline ML Clásico | 30% |
| Capstone 2: Proyecto NLP | 30% |
| Capstone 3: RAG + Agente + n8n | 40% |
| **Total** | **100%** |

**Nota:** Los quizzes y ejercicios son evaluaciones formativas y no cuentan para la nota final, pero son indicadores clave de progreso.

### Criterios de Aprobación del Curso

- **Aprobado:** ≥70% de la nota final
- **Bueno:** ≥80%
- **Excelente:** ≥90%

**Requisitos adicionales:**
- Completar los 3 capstones
- Código reproducible en todos los proyectos
- Sin plagios (se permite consultar recursos, pero el código debe ser original)

---

## Estructura de Repositorio Recomendada

```
curso-ia/
├── modulo-0-fundamentos/
│   ├── notebooks/
│   ├── ejercicios/
│   └── README.md
├── modulo-1-eda/
│   ├── notebooks/
│   ├── ejercicios/
│   ├── datos/
│   └── README.md
├── modulo-2-ml-fundamentos/
├── modulo-3-seleccion-modelos/
├── modulo-4-nlp/
├── modulo-5-deep-learning/
├── modulo-6-llms-rag/
├── modulo-7-agentes-n8n/
├── capstone-1/
│   ├── src/
│   ├── notebooks/
│   ├── data/
│   ├── models/
│   ├── reports/
│   ├── requirements.txt
│   └── README.md
├── capstone-2/
├── capstone-3/
│   ├── src/
│   ├── api/
│   ├── n8n/
│   ├── tests/
│   ├── docker-compose.yml
│   └── README.md
├── recursos/
│   ├── datasets/
│   ├── papers/
│   └── enlaces.md
└── README.md
```

---

## Especificación del Entorno

### Python Packages (requirements.txt)

```
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.3.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# NLP
transformers>=4.30.0
datasets>=2.14.0
sentence-transformers>=2.2.0
tiktoken>=0.4.0

# Vector Databases
faiss-cpu>=1.7.4
# or chromadb>=0.4.0

# APIs
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
httpx>=0.24.0

# MLOps
mlflow>=2.5.0
# or wandb>=0.15.0

# Utilities
jupyter>=1.0.0
notebook>=7.0.0
python-dotenv>=1.0.0
pyarrow>=12.0.0
tqdm>=4.65.0

# Optional
# langchain>=0.0.200
# openai>=0.27.0
```

### Alternativas para diferentes ecosistemas
- **Poetry:** `pyproject.toml` con grupos de dependencias
- **Conda:** `environment.yml`

---

## Herramientas Requeridas

1. **Python:** 3.10 o superior
2. **Git:** Control de versiones
3. **Docker:** Para n8n (opcional pero recomendado)
4. **n8n:** Desktop app o Docker
5. **Editor:** VS Code, PyCharm, o similar
6. **Jupyter:** Para notebooks interactivos

---

## Recursos de Aprendizaje

### Libros
- "Hands-On Machine Learning" (Aurélien Géron)
- "Natural Language Processing with Transformers" (Tunstall et al.)
- "Deep Learning with PyTorch" (Stevens et al.)
- "Python for Data Analysis" (Wes McKinney)

### Cursos Online (Complementarios)
- Fast.ai Practical Deep Learning
- Hugging Face NLP Course
- MLflow tutorials

### Papers Clave
- "Attention Is All You Need" (Vaswani et al.)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al.)
- "Retrieval-Augmented Generation" (Lewis et al.)
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al.)

### Documentación
- scikit-learn.org
- pytorch.org
- huggingface.co
- n8n.io/docs
- mlflow.org

---

## Principios Pedagógicos

1. **Aprender Haciendo:** 80% práctica, 20% teoría
2. **Progresión Gradual:** De simple a complejo, de tabular a texto a agentes
3. **Proyectos Reales:** Datasets públicos, problemas aplicables
4. **Reproducibilidad:** Seeds, requirements, documentación
5. **Evaluación Continua:** Quizzes, ejercicios, checkpoints
6. **Ética y Responsabilidad:** Integrada en cada módulo

---

## Anti-Patrones y Riesgos a Evitar

### Data Leakage
- ❌ Normalizar antes de split
- ❌ Feature engineering con información del test set
- ✅ Usar pipelines que transforman train y test independientemente

### Evaluación
- ❌ Evaluar repetidamente en test set
- ❌ Optimizar hiperparámetros en test
- ✅ Usar validación cruzada, reservar test para evaluación final

### Reproducibilidad
- ❌ No fijar seeds aleatorias
- ❌ No documentar versiones de librerías
- ✅ Requirements.txt, seeds, documentación clara

### Seguridad
- ❌ Hardcodear API keys
- ❌ Commitear secretos en Git
- ✅ Variables de entorno, .env en .gitignore

### n8n
- ❌ No manejar errores en workflows
- ❌ Workflows sin logging
- ✅ Error handling, retries, logging estructurado

---

## Soporte y Comunidad

- **Issues:** Para preguntas técnicas sobre el material
- **Discussions:** Para compartir proyectos y experiencias
- **Office Hours:** (Si aplicable) Sesiones de Q&A

---

## Próximos Pasos

1. ✅ Revisar este syllabus
2. ⏳ Responder preguntas de calibración
3. ⏳ Confirmar recursos disponibles (hardware, tiempo, presupuesto)
4. ⏳ Aprobar plan y comenzar con materiales detallados

---

**Última actualización:** 2025-11-07  
**Versión:** 1.0
