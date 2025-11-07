# Curso de Inteligencia Artificial y Análisis de Datos con Python

**De Machine Learning Clásico a Agentes de IA y Orquestación con n8n**

---

## 🎯 Visión General

Este es un curso completo y práctico (80% práctica, 20% teoría) diseñado para llevar a desarrolladores Python con experiencia desde los fundamentos del análisis de datos y machine learning hasta la construcción de agentes de IA y la automatización con n8n.

**Duración:** 8 semanas (10-15 horas/semana)  
**Nivel:** Intermedio-Avanzado  
**Modalidad:** Autoguiado con prácticas, ejercicios y proyectos capstone

## 🎓 ¿Qué Aprenderás?

Al completar este curso, serás capaz de:

✅ **Análisis de Datos:** Preparar, analizar y visualizar datos con pandas, seaborn y matplotlib  
✅ **Machine Learning Clásico:** Construir, entrenar y evaluar modelos con scikit-learn  
✅ **Pipelines Reproducibles:** Implementar flujos completos de preprocesamiento, validación y tuning  
✅ **Deep Learning:** Aplicar PyTorch para problemas tabulares y de texto  
✅ **NLP Moderno:** Trabajar con embeddings, transformers (Hugging Face) y modelos de lenguaje  
✅ **LLMs y RAG:** Implementar sistemas de Retrieval-Augmented Generation y evaluación  
✅ **Agentes de IA:** Diseñar y programar agentes con herramientas, memoria y planificación  
✅ **Orquestación con n8n:** Automatizar flujos de datos e inferencia con webhooks, APIs y notificaciones  
✅ **MLOps Ligero:** Aplicar tracking, reproducibilidad, versionado y IA responsable

## 📚 Estructura del Curso

### Módulos de Aprendizaje

- **[Módulo 0](modulo-0-fundamentos/README.md) (Opcional):** Fundamentos Matemáticos para ML
- **[Módulo 1](modulo-1-eda/README.md):** EDA y Preparación de Datos con Pandas
- **Módulo 2:** Fundamentos de Machine Learning con Scikit-Learn
- **Módulo 3:** Selección y Optimización de Modelos
- **Módulo 4:** NLP Moderno y Embeddings
- **Módulo 5:** Deep Learning con PyTorch
- **Módulo 6:** LLMs, Prompting y RAG
- **Módulo 7:** Agentes de IA y Orquestación con n8n

### Proyectos Capstone

1. **[Capstone 1](capstone-1/README.md) (30%):** Pipeline de ML Clásico End-to-End
2. **Capstone 2 (30%):** Proyecto NLP con Embeddings/Transformers
3. **Capstone 3 (40%):** Sistema RAG + Agente + n8n

## 🚀 Inicio Rápido

### 1. Prerrequisitos

Antes de comenzar, asegúrate de tener:
- **Python 3.10+** instalado
- Experiencia sólida en Python (funciones, clases, paquetes)
- Nociones básicas de estadística y álgebra lineal
- Git instalado

### 2. Instalación

```bash
# Clona el repositorio
git clone https://github.com/henryandr/curso-IA.git
cd curso-IA

# Crea un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instala las dependencias
pip install -r requirements.txt

# Verifica la instalación
python -c "import pandas, sklearn, torch; print('✅ Todo instalado correctamente')"
```

### 3. Estructura de Directorios

```
curso-IA/
├── 00-SINTESIS-EJECUTIVA.md          # Resumen del curso
├── 01-PREGUNTAS-CALIBRACION.md      # Personaliza tu experiencia
├── 02-SYLLABUS.md                    # Temario completo
├── 03-ANTI-PATRONES-Y-MEJORES-PRACTICAS.md
├── modulo-0-fundamentos/             # Repaso matemático (opcional)
├── modulo-1-eda/                     # EDA con pandas
├── modulo-2-ml-fundamentos/          # ML clásico
├── modulo-3-seleccion-modelos/       # Optimización
├── modulo-4-nlp/                     # NLP moderno
├── modulo-5-deep-learning/           # PyTorch
├── modulo-6-llms-rag/                # LLMs y RAG
├── modulo-7-agentes-n8n/             # Agentes y n8n
├── capstone-1/                       # Proyecto ML clásico
├── capstone-2/                       # Proyecto NLP
├── capstone-3/                       # Proyecto RAG + Agente
├── recursos/
│   ├── datasets/                     # Datasets del curso
│   ├── guias-n8n/                    # Guías de n8n
│   └── papers/                       # Papers de referencia
└── requirements.txt                  # Dependencias
```

## 📖 Cómo Usar Este Curso

### Paso 1: Lee la Síntesis Ejecutiva
Comienza con [00-SINTESIS-EJECUTIVA.md](00-SINTESIS-EJECUTIVA.md) para entender el alcance completo.

### Paso 2: Responde las Preguntas de Calibración
Completa [01-PREGUNTAS-CALIBRACION.md](01-PREGUNTAS-CALIBRACION.md) para personalizar tu experiencia.

### Paso 3: Revisa el Syllabus
Lee [02-SYLLABUS.md](02-SYLLABUS.md) para ver el plan completo de 8 semanas.

### Paso 4: Comienza con los Módulos
- Si necesitas repaso: empieza con **Módulo 0**
- Si tienes buena base: empieza con **Módulo 1**
- Completa las prácticas guiadas antes de los ejercicios
- Haz los mini-quizzes para validar tu aprendizaje

### Paso 5: Proyectos Capstone
- Completa los capstones al terminar los módulos prerequisitos
- Sigue las rúbricas para autoevaluarte
- Documenta todo tu proceso

## 🛠️ Stack Tecnológico

### Librerías Python
- **Análisis:** numpy, pandas, matplotlib, seaborn, scipy
- **ML:** scikit-learn
- **DL:** PyTorch, transformers (Hugging Face)
- **Vectores:** faiss-cpu / chromadb
- **APIs:** FastAPI, uvicorn, pydantic
- **Tracking:** MLflow / Weights & Biases

### Herramientas
- **n8n:** Orquestación de workflows
- **Jupyter:** Notebooks interactivos
- **Docker:** Contenedores (opcional)
- **Git:** Control de versiones

## 📊 Evaluación

| Componente | Peso |
|------------|------|
| Capstone 1: Pipeline ML Clásico | 30% |
| Capstone 2: Proyecto NLP | 30% |
| Capstone 3: RAG + Agente + n8n | 40% |

**Aprobación:** ≥70%  
**Los quizzes y ejercicios son formativos (no califican)**

## 🎯 Público Objetivo

Este curso es ideal para ti si:
- Eres desarrollador/a Python con experiencia
- Quieres aprender IA/ML de forma práctica
- Tienes poco conocimiento de modelos de IA y agentes
- Quieres construir sistemas completos end-to-end
- Te interesa la automatización con n8n

**No es para ti si:**
- No sabes programar en Python
- Buscas solo teoría matemática profunda
- Quieres un curso de "hola mundo" de IA

## 📝 Documentos Importantes

- **[Síntesis Ejecutiva](00-SINTESIS-EJECUTIVA.md):** Visión general del curso (1 página)
- **[Preguntas de Calibración](01-PREGUNTAS-CALIBRACION.md):** Personaliza tu experiencia
- **[Syllabus Completo](02-SYLLABUS.md):** Temario detallado de 8 semanas
- **[Anti-Patrones](03-ANTI-PATRONES-Y-MEJORES-PRACTICAS.md):** Errores comunes y cómo evitarlos

## 🔧 Solución de Problemas

### Error al instalar PyTorch
```bash
# Para CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Para GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Error con FAISS
```bash
# Asegúrate de instalar la versión CPU
pip install faiss-cpu
```

### n8n no inicia
```bash
# Con Docker
docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n

# Con npm
npm install n8n -g
n8n start
```

## 🤝 Contribuciones

Este es un proyecto educativo en evolución. Si encuentras errores, tienes sugerencias o quieres contribuir:
1. Abre un Issue describiendo el problema o mejora
2. Si quieres contribuir código, abre un Pull Request
3. Comparte tus proyectos capstone en Discussions

## 📚 Recursos Adicionales

### Libros Recomendados
- "Hands-On Machine Learning" (Aurélien Géron)
- "Natural Language Processing with Transformers" (Tunstall et al.)
- "Deep Learning with PyTorch" (Stevens et al.)

### Cursos Complementarios
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)
- [MLflow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/)

### Comunidades
- [n8n Community](https://community.n8n.io/)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)

## 🔐 Ética y Seguridad

Este curso enfatiza:
- **Privacidad:** Manejo responsable de datos personales
- **Seguridad:** Gestión correcta de secretos y API keys
- **Reducción de Sesgos:** Análisis y mitigación de sesgos en modelos
- **Transparencia:** Documentación clara de decisiones técnicas
- **IA Responsable:** Consideraciones éticas en cada módulo

## 📄 Licencia

Este material educativo está disponible bajo [licencia a definir]. El código de ejemplo puede usarse libremente con atribución.

## 🙏 Agradecimientos

Este curso utiliza y se inspira en:
- Datasets públicos de Kaggle, UCI ML Repository
- Documentación oficial de scikit-learn, PyTorch, Hugging Face
- Comunidad open-source de Python y ML

## 📞 Contacto y Soporte

- **Issues:** Para problemas técnicos con el material
- **Discussions:** Para preguntas y compartir proyectos
- **Pull Requests:** Para contribuciones

---

## 🚀 ¡Comienza Ahora!

1. ✅ Instala las dependencias: `pip install -r requirements.txt`
2. 📖 Lee la [Síntesis Ejecutiva](00-SINTESIS-EJECUTIVA.md)
3. 📝 Responde las [Preguntas de Calibración](01-PREGUNTAS-CALIBRACION.md)
4. 🎯 Comienza con el [Módulo 1](modulo-1-eda/README.md) (o [Módulo 0](modulo-0-fundamentos/README.md) si necesitas repaso)

---

**¿Listo para convertirte en un/a profesional de IA y ML?** 🚀

**Última actualización:** 2025-11-07  
**Versión:** 1.0
