# Capstone 2: Proyecto NLP con Embeddings/Transformers

## Información General

**Peso en el Curso:** 30% de la nota final  
**Duración Estimada:** 10-12 horas (1 semana)  
**Módulos Prerequisitos:** 4, 5  
**Nivel de Dificultad:** ⭐⭐⭐⭐ (Intermedio-Avanzado)

## Objetivos de Aprendizaje

Al completar este capstone, demostrarás que puedes:
- Procesar y preparar corpus de texto para NLP
- Implementar soluciones con embeddings y/o transformers
- Diseñar y ejecutar experimentos comparativos
- Evaluar modelos de NLP con métricas apropiadas
- Realizar análisis de errores y ablation studies
- Documentar decisiones técnicas de forma profesional

## Descripción del Proyecto

Construirás un sistema de NLP completo usando embeddings pre-entrenados o transformers fine-tuneados para resolver uno de estos problemas:
- Clasificación de documentos multi-clase
- Análisis de sentimiento
- Sistema de Question-Answering
- Búsqueda semántica / Sistema de recomendación de contenido
- Clustering y análisis temático

El proyecto debe incluir **al menos 2 enfoques diferentes** comparados de forma rigurosa.

## Opciones de Proyecto

### Opción 1: Clasificador de Artículos de Noticias
**Dataset:** AG News, Reuters-21578, o 20 Newsgroups  
**Tarea:** Clasificación multi-clase de categorías  
**Enfoques sugeridos:**
1. TF-IDF + Logistic Regression (baseline)
2. Sentence-BERT embeddings + clasificador simple
3. Fine-tuning de BERT/DistilBERT

**Desafíos:** Múltiples clases, texto largo, vocabulario amplio

### Opción 2: Análisis de Sentimiento de Reviews
**Dataset:** IMDB Reviews, Amazon Reviews, Yelp Reviews  
**Tarea:** Clasificación de sentimiento (positivo/negativo/neutral)  
**Enfoques sugeridos:**
1. Embeddings pre-entrenados (GloVe) + LSTM
2. Sentence-BERT + clasificador
3. Fine-tuning de RoBERTa/DeBERTa

**Desafíos:** Sarcasmo, lenguaje coloquial, desbalance

### Opción 3: Sistema de Question-Answering
**Dataset:** SQuAD 2.0, Natural Questions, MS MARCO  
**Tarea:** Extraer respuestas de contexto o ranking de pasajes  
**Enfoques sugeridos:**
1. TF-IDF retrieval + heurísticas
2. Dense retrieval con Sentence-BERT
3. Fine-tuning de BERT para QA

**Desafíos:** Comprensión de contexto, respuestas imposibles

### Opción 4: Clustering Temático de Documentos
**Dataset:** ArXiv papers, Wikipedia, BBC News  
**Tarea:** Descubrir y agrupar temas automáticamente  
**Enfoques sugeridos:**
1. TF-IDF + K-Means
2. Sentence-BERT embeddings + HDBSCAN
3. Topic modeling con BERTopic

**Desafíos:** Determinar número de clusters, interpretabilidad

## Requisitos Técnicos

### 1. Preparación de Datos
**Entregable:** Notebook `01_data_preparation.ipynb`

Debe incluir:
- [ ] Exploración del corpus (distribución de clases, longitud de textos)
- [ ] Limpieza de texto (HTML, caracteres especiales, etc.)
- [ ] Análisis de vocabulario (palabras más frecuentes, rare words)
- [ ] Splits: train (70%), validation (15%), test (15%)
- [ ] Estratificación si es clasificación
- [ ] Guardado de splits reproducibles

### 2. Baseline con TF-IDF o BOW
**Entregable:** Notebook `02_baseline.ipynb`

Implementar modelo baseline:
- [ ] TF-IDF o Bag-of-Words vectorization
- [ ] Modelo simple (Logistic Regression, Naive Bayes)
- [ ] Evaluación en validation set
- [ ] Análisis de errores del baseline

**Objetivo:** Establecer línea base para comparar enfoques avanzados

### 3. Enfoque con Embeddings
**Entregable:** Notebook `03_embeddings.ipynb`

Implementar solución con embeddings:
- [ ] Uso de embeddings pre-entrenados (Sentence-BERT, Universal Sentence Encoder)
- [ ] Generación de embeddings para corpus
- [ ] Clasificador sobre embeddings (RF, XGBoost, o NN simple)
- [ ] Evaluación y comparación con baseline

### 4. Enfoque con Transformers
**Entregable:** Notebook `04_transformers.ipynb`

Implementar con Hugging Face Transformers:
- [ ] Selección de modelo pre-entrenado (BERT, RoBERTa, DistilBERT)
- [ ] Tokenización apropiada
- [ ] Fine-tuning con early stopping
- [ ] Evaluación en validation y comparación

### 5. Experimentos y Ablation Study
**Entregable:** Notebook `05_experiments.ipynb`

Realizar experimentos sistemáticos:
- [ ] Comparar al menos 3 configuraciones diferentes
- [ ] Ablation: impacto de preprocesamiento, longitud de secuencia, etc.
- [ ] Tracking de experimentos (MLflow/W&B)
- [ ] Tabla comparativa de resultados

**Ejemplos de experimentos:**
- Diferentes modelos pre-entrenados
- Con/sin data augmentation
- Diferentes longitudes máximas de secuencia
- Diferentes learning rates
- Frozen vs fine-tuned layers

### 6. Evaluación Final
**Entregable:** Notebook `06_final_evaluation.ipynb`

Evaluación rigurosa del mejor modelo:
- [ ] Métricas completas en test set (una sola vez)
- [ ] Métricas apropiadas:
  - Clasificación: accuracy, precision, recall, F1, confusion matrix
  - QA: Exact Match, F1 score
  - Clustering: silhouette score, coherence
- [ ] Análisis de errores (falsos positivos/negativos)
- [ ] Ejemplos de predicciones correctas e incorrectas
- [ ] Interpretabilidad: attention weights, SHAP (opcional)

### 7. Reporte Técnico
**Entregable:** `REPORT.md` (3-4 páginas)

Estructura:
1. **Introducción** (1/2 página)
   - Problema y motivación
   - Contribución del proyecto
2. **Dataset y Preprocesamiento** (1/2 página)
   - Descripción del corpus
   - Estrategias de limpieza
   - Estadísticas clave
3. **Metodología** (1.5 páginas)
   - Enfoques implementados
   - Arquitecturas y configuraciones
   - Experimentos realizados
4. **Resultados** (1 página)
   - Tabla comparativa de todos los enfoques
   - Ablation study results
   - Análisis de errores
5. **Conclusiones** (1/2 página)
   - Hallazgos principales
   - Limitaciones
   - Trabajo futuro

### 8. Código y Reproducibilidad
**Entregables:** `src/`, `requirements.txt`, `README.md`

Estructura del proyecto:
```
capstone-2/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── embeddings/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_transformers.ipynb
│   ├── 05_experiments.ipynb
│   └── 06_final_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
├── models/
│   ├── baseline/
│   ├── embeddings/
│   └── transformers/
├── reports/
│   ├── REPORT.md
│   └── figures/
└── mlruns/ (si se usa MLflow)
```

## Rúbrica de Evaluación

| Componente | Peso | Criterios de Excelencia |
|------------|------|------------------------|
| **Preparación de Datos** | 15% | Limpieza apropiada, análisis profundo, splits correctos |
| **Diseño de Experimentos** | 25% | Al menos 2 enfoques, comparación justa, ablation study |
| **Implementación Técnica** | 25% | Código limpio, uso correcto de transformers, reproducible |
| **Evaluación y Métricas** | 20% | Métricas apropiadas, análisis de errores, interpretación correcta |
| **Reporte y Documentación** | 15% | Narrativa clara, decisiones justificadas, visualizaciones |

### Desglose Detallado

#### Preparación de Datos (15%)
- **Excelente (13-15):** Limpieza exhaustiva, análisis de vocabulario, splits estratificados, sin leakage
- **Bueno (11-12):** Limpieza correcta, splits apropiados
- **Aceptable (9-10):** Limpieza básica, splits sin leakage
- **Insuficiente (<9):** Problemas de calidad de datos o leakage

#### Diseño de Experimentos (25%)
- **Excelente (22-25):** 3+ enfoques, ablation study completo, tracking riguroso
- **Bueno (19-21):** 2 enfoques bien comparados, algunos experimentos adicionales
- **Aceptable (16-18):** 2 enfoques comparados básicamente
- **Insuficiente (<16):** Solo 1 enfoque o comparación injusta

#### Implementación Técnica (25%)
- **Excelente (22-25):** Código profesional, uso avanzado de Hugging Face, optimizaciones
- **Bueno (19-21):** Código limpio, uso correcto de transformers
- **Aceptable (16-18):** Implementación funcional, algunos issues menores
- **Insuficiente (<16):** Código con errores, mal uso de librerías

#### Evaluación y Métricas (20%)
- **Excelente (18-20):** Métricas correctas y profundas, análisis de errores detallado, interpretabilidad
- **Bueno (15-17):** Métricas apropiadas, análisis de errores básico
- **Aceptable (12-14):** Métricas correctas, evaluación básica
- **Insuficiente (<12):** Métricas inapropiadas o evaluación incorrecta

#### Reporte y Documentación (15%)
- **Excelente (13-15):** Reporte profesional, decisiones bien justificadas, comunicación clara
- **Bueno (11-12):** Reporte completo y claro
- **Aceptable (9-10):** Reporte básico con información necesaria
- **Insuficiente (<9):** Reporte incompleto o confuso

## Criterios de Aprobación

Para aprobar el Capstone 2:
- [ ] Obtener ≥70% (≥70 puntos de 100)
- [ ] Implementar al menos 2 enfoques diferentes
- [ ] Usar métricas de NLP apropiadas
- [ ] Modelo final supera baseline
- [ ] Código ejecutable y reproducible
- [ ] Reporte técnico completo

## Extras (Puntos Bonus)

Hasta +10 puntos por:
- [ ] **+3:** Interpretabilidad (attention visualization, SHAP, LIME)
- [ ] **+3:** Data augmentation efectivo (back-translation, EDA, etc.)
- [ ] **+2:** Demo interactivo (Gradio, Streamlit)
- [ ] **+2:** Análisis de fairness/bias en predicciones

## Timeline Sugerido

| Día | Actividad | Tiempo |
|-----|-----------|--------|
| 1-2 | Preparación y análisis de datos | 3h |
| 3 | Baseline con TF-IDF | 2h |
| 4 | Enfoque con embeddings | 3h |
| 5-6 | Fine-tuning de transformers | 4h |
| 7 | Experimentos y ablation | 2h |
| 8 | Evaluación final y análisis | 2h |
| 9 | Reporte técnico | 2h |

## Datasets Recomendados

### Clasificación de Texto
- **AG News:** 120k artículos, 4 clases
- **20 Newsgroups:** 20k posts, 20 categorías
- **Reuters-21578:** Noticias financieras
- **DBPedia:** Categorización de Wikipedia

### Sentiment Analysis
- **IMDB Reviews:** 50k reviews de películas
- **Amazon Reviews:** Multi-dominio
- **Yelp Reviews:** Reviews de restaurantes
- **Twitter Sentiment:** Tweets etiquetados

### Question Answering
- **SQuAD 2.0:** 150k preguntas sobre Wikipedia
- **Natural Questions:** Preguntas de Google Search
- **MS MARCO:** Ranking de pasajes

### Otros
- **ArXiv Papers:** Para clustering temático
- **BBC News:** Artículos categorizados

## Herramientas Recomendadas

### Librerías
```python
transformers>=4.30.0
sentence-transformers>=2.2.0
datasets>=2.14.0
torch>=2.0.0
scikit-learn>=1.3.0
mlflow>=2.5.0
```

### Modelos Pre-entrenados Sugeridos
- **Clasificación general:** bert-base-uncased, distilbert-base-uncased
- **Sentiment:** cardiffnlp/twitter-roberta-base-sentiment
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **QA:** deepset/roberta-base-squad2

## Checklist de Entrega

Antes de entregar:
- [ ] Todos los notebooks ejecutan sin errores
- [ ] Al menos 2 enfoques implementados
- [ ] Experimentos tracked (MLflow/W&B)
- [ ] Métricas apropiadas calculadas
- [ ] Análisis de errores completo
- [ ] Reporte técnico finalizado
- [ ] README con instrucciones claras
- [ ] requirements.txt actualizado
- [ ] Código limpio y comentado

## Preguntas Frecuentes

**P: ¿Debo entrenar desde cero un transformer?**  
R: No. Fine-tuning de modelos pre-entrenados es suficiente y recomendado.

**P: ¿Necesito GPU?**  
R: No es estrictamente necesario. Puedes usar modelos pequeños (DistilBERT) en CPU o Google Colab gratis.

**P: ¿Cuántos experimentos debo hacer?**  
R: Mínimo 3-5 configuraciones diferentes. Enfócate en calidad, no cantidad.

**P: ¿Puedo usar ChatGPT API para clasificación?**  
R: Puedes incluirlo como punto de comparación, pero no como enfoque principal.

## Recursos de Ayuda

- [Hugging Face Course - NLP](https://huggingface.co/learn/nlp-course/)
- [Sentence-BERT Documentation](https://www.sbert.net/)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- Módulos 4 y 5 del curso

## Entrega

**Formato:** Repositorio Git con todo el código  
**Deadline:** Consulta calendario del curso  
**Método:** Link al repo + README con setup

---

**¡Éxito! Este proyecto te dará experiencia real en NLP moderno.**
