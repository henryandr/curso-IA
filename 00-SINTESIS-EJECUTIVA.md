# Síntesis Ejecutiva: Curso de IA y Análisis de Datos con Python

## Visión General

Este curso completo y práctico está diseñado para llevar a desarrolladores Python con experiencia desde los fundamentos del análisis de datos y machine learning hasta la construcción de agentes de IA y la automatización con n8n. El enfoque es **80% práctico** con teoría mínima pero suficiente para tomar decisiones técnicas informadas.

## Público Objetivo

Desarrolladores y desarrolladoras con:
- Experiencia sólida en Python (control de flujo, funciones, paquetes, virtualenv/poetry)
- Conocimientos limitados en IA, ML, LLMs y agentes
- Nociones básicas de estadística, álgebra lineal y probabilidad

## Duración y Estructura

- **8 módulos** (8 semanas de contenido)
- **Tiempo estimado:** 10-15 horas/semana
- **3 proyectos capstone** integrados
- **Módulo 0 opcional** de repaso matemático

## Resultados de Aprendizaje

Al finalizar, los estudiantes serán capaces de:

1. **Análisis de Datos:** Preparar, analizar y visualizar datos con pandas, seaborn y matplotlib
2. **Machine Learning Clásico:** Construir, entrenar y evaluar modelos con scikit-learn
3. **Pipelines Reproducibles:** Implementar flujos completos de preprocesamiento, validación, métricas y tuning
4. **Deep Learning:** Aplicar PyTorch para problemas tabulares y de texto
5. **NLP Moderno:** Trabajar con embeddings, transformers (Hugging Face) y modelos de lenguaje
6. **LLMs Avanzados:** Implementar prompting efectivo, RAG (Retrieval-Augmented Generation) y evaluación
7. **Agentes de IA:** Diseñar y programar agentes con herramientas, memoria y planificación
8. **Orquestación con n8n:** Automatizar flujos de datos e inferencia con webhooks, APIs y notificaciones
9. **MLOps Ligero:** Aplicar tracking, reproducibilidad, versionado de datos y principios de IA responsable

## Metodología

- **Aprender Haciendo:** 3-5 prácticas guiadas por módulo con datasets reales
- **Consolidación:** 2-4 ejercicios autónomos con criterios de corrección
- **Evaluación Continua:** Mini-quizzes (5-8 preguntas), katas de código, checklists de revisión
- **Proyectos Integradores:** 3 capstones con rúbricas detalladas

## Stack Tecnológico

### Python Libraries
- **Análisis:** numpy, pandas, matplotlib, seaborn, scipy
- **ML Clásico:** scikit-learn
- **Deep Learning:** PyTorch, transformers (Hugging Face)
- **Vectores y RAG:** faiss-cpu/chromadb
- **APIs:** FastAPI, uvicorn, pydantic
- **Tracking:** MLflow o Weights & Biases
- **Utilidades:** jupyter, python-dotenv, tiktoken

### Herramientas de Orquestación
- **n8n:** Instalación local (Docker/desktop)
- **Integraciones:** HTTP Request, Webhook, Code nodes, Google Sheets, Slack/Discord, GitHub

### Infraestructura
- **CPU por defecto** (con notas para GPU)
- **Python 3.10+**
- **Control de versiones:** Git/GitHub

## Proyectos Capstone

### Capstone 1 (30%): Pipeline ML Clásico
Modelo de clasificación o regresión con scikit-learn para datos tabulares, incluyendo EDA completo, feature engineering, validación cruzada y evaluación honesta.

### Capstone 2 (30%): NLP con Transformers
Sistema de clasificación de textos o QA usando embeddings y/o transformers, con experimentos documentados y evaluación rigurosa.

### Capstone 3 (40%): Sistema RAG + Agente + n8n
Sistema end-to-end que integra:
- RAG (chunking, embeddings, búsqueda vectorial)
- Agente en Python con herramientas y memoria
- Orquestación n8n (webhook → ingesta → indexado → servicio FastAPI → inferencia → notificación)

## Diferenciadores

1. **Progresión Natural:** De datos tabulares simples a sistemas de agentes complejos
2. **Enfoque Práctico:** Datasets reales, problemas aplicables, código ejecutable
3. **Orquestación Moderna:** Integración profunda de n8n para automatización
4. **MLOps Desde el Inicio:** Reproducibilidad, tracking y versionado incorporados
5. **Ética y Seguridad:** Manejo de secretos, privacidad, reducción de sesgos
6. **Evaluabilidad:** Rúbricas claras, criterios objetivos, anti-patrones documentados

## Entregables del Curso

- Syllabus detallado con objetivos SMART por módulo
- 8 planes de módulo con prácticas, ejercicios y soluciones
- 3 proyectos capstone con rúbricas completas
- 3 guías de workflows n8n
- Especificación de entorno y estructura de repositorio
- Quizzes y evaluaciones por módulo
- Lista de riesgos y anti-patrones

## Criterios de Éxito

El curso será exitoso si los estudiantes pueden:
- Implementar un sistema RAG funcional desde cero
- Automatizar flujos de ML/IA con n8n
- Tomar decisiones técnicas informadas sobre modelos y arquitecturas
- Aplicar mejores prácticas de reproducibilidad y evaluación
- Desplegar servicios de inferencia básicos con FastAPI

## Próximos Pasos

1. Responder preguntas de calibración (duración, recursos, preferencias)
2. Revisar y aprobar el syllabus propuesto
3. Generar materiales completos (prácticas, ejercicios, quizzes, rúbricas)
