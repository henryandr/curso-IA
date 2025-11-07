# Capstone 1: Pipeline de Machine Learning Clásico End-to-End

## Información General

**Peso en el Curso:** 30% de la nota final  
**Duración Estimada:** 10-12 horas (1 semana)  
**Módulos Prerequisitos:** 1, 2, 3  
**Nivel de Dificultad:** ⭐⭐⭐ (Intermedio)

## Objetivos de Aprendizaje

Al completar este capstone, demostrarás que puedes:
- Realizar un análisis exploratorio completo y documentado
- Diseñar e implementar un pipeline de ML reproducible
- Comparar múltiples algoritmos de forma sistemática
- Evaluar modelos honestamente sin data leakage
- Comunicar resultados técnicos de forma clara

## Descripción del Proyecto

Construirás un pipeline completo de Machine Learning para resolver un problema de clasificación o regresión con datos tabulares. El proyecto debe incluir todas las etapas: desde la exploración inicial hasta la evaluación final, con énfasis en reproducibilidad y buenas prácticas.

## Datasets Sugeridos

Elige **UNO** de los siguientes datasets (o propón uno similar):

### Opción 1: Bank Marketing (Clasificación)
- **Descripción:** Predecir si un cliente suscribirá un depósito a plazo
- **Registros:** ~45,000
- **Features:** 16 (mix de numéricos y categóricos)
- **Target:** Binario (yes/no)
- **Descarga:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Desafíos:** Desbalance de clases, variables categóricas, outliers

### Opción 2: Employee Attrition (Clasificación)
- **Descripción:** Predecir rotación de empleados
- **Registros:** ~1,500
- **Features:** 30+ (demográficos, satisfacción, salario)
- **Target:** Binario (left/stayed)
- **Descarga:** [Kaggle - IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Desafíos:** Dataset pequeño, multicolinealidad, desbalance

### Opción 3: House Prices (Regresión)
- **Descripción:** Predecir precio de venta de viviendas
- **Registros:** ~1,460
- **Features:** 79 (características de la casa)
- **Target:** Continuo (precio)
- **Descarga:** [Kaggle - House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Desafíos:** Muchos features, valores faltantes, outliers

### Opción 4: Loan Default Prediction (Clasificación)
- **Descripción:** Predecir si un prestatario incumplirá un préstamo
- **Registros:** Variable según dataset
- **Features:** Historial crediticio, ingresos, monto del préstamo
- **Target:** Binario (default/no default)
- **Descarga:** [Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) o similar
- **Desafíos:** Desbalance severo, features financieros

## Requisitos Técnicos

### 1. Análisis Exploratorio de Datos (EDA)
**Entregable:** Notebook `01_eda.ipynb`

Debe incluir:
- [ ] Información del dataset (shape, tipos, estadísticas)
- [ ] Análisis de valores faltantes (porcentaje, visualización)
- [ ] Distribución de la variable target
- [ ] Análisis univariado de top 10 features
- [ ] Análisis bivariado (features vs target)
- [ ] Matriz de correlación con heatmap
- [ ] Detección y visualización de outliers
- [ ] **Al menos 5 insights clave documentados**

**Herramientas:** pandas, numpy, matplotlib, seaborn

### 2. Limpieza y Preprocesamiento
**Entregable:** Script `src/preprocessing.py` y/o notebook `02_preprocessing.ipynb`

Debe incluir:
- [ ] Tratamiento de valores faltantes (con justificación de estrategia)
- [ ] Manejo de outliers (con análisis previo)
- [ ] Encoding de variables categóricas
- [ ] Feature engineering (al menos 2 features nuevos)
- [ ] Documentación de todas las transformaciones aplicadas

**Criterios:**
- ✅ Sin data leakage (transformaciones se aplican post-split)
- ✅ Estrategias justificadas (no arbitrarias)
- ✅ Código modular y reutilizable

### 3. Pipeline de Scikit-Learn
**Entregable:** Script `src/pipeline.py`

Debe implementar:
- [ ] ColumnTransformer para diferentes tipos de features
- [ ] Pipeline completo (preprocesamiento + modelo)
- [ ] Splits: train (60%), validation (20%), test (20%)
- [ ] Estratificación si es clasificación
- [ ] Seeds fijados para reproducibilidad

**Ejemplo de estructura:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['age', 'salary', ...]
categorical_features = ['job', 'education', ...]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])
```

### 4. Comparación de Modelos
**Entregable:** Notebook `03_model_comparison.ipynb`

Debes comparar **al menos 3 modelos**, por ejemplo:
- Baseline simple (ej: DummyClassifier, media para regresión)
- Modelo lineal (LogisticRegression, Ridge)
- Modelo basado en árboles (RandomForest, XGBoost)

Para cada modelo:
- [ ] Entrenamiento con validación cruzada (5-fold)
- [ ] Registro de hiperparámetros
- [ ] Métricas en train y validation
- [ ] Tiempo de entrenamiento
- [ ] Feature importance (si aplica)

**Tracking:** Usar MLflow o W&B para registrar experimentos

### 5. Optimización de Hiperparámetros
**Entregable:** Notebook `04_hyperparameter_tuning.ipynb`

Para el **mejor modelo** de la comparación:
- [ ] Definir espacio de búsqueda (al menos 3 hiperparámetros)
- [ ] GridSearchCV o RandomizedSearchCV
- [ ] Validación cruzada en cada configuración
- [ ] Selección del mejor modelo
- [ ] Análisis de impacto de cada hiperparámetro

### 6. Evaluación Final
**Entregable:** Notebook `05_final_evaluation.ipynb`

Con el modelo final optimizado:
- [ ] **Una única evaluación en test set**
- [ ] Métricas completas:
  - Clasificación: accuracy, precision, recall, F1, ROC-AUC, matriz de confusión
  - Regresión: MAE, MSE, RMSE, R², MAPE
- [ ] Análisis de errores (top errores, patrones)
- [ ] Learning curves
- [ ] Comparación baseline vs modelo final

### 7. Reporte Técnico
**Entregable:** `REPORT.md` (2-3 páginas)

Estructura:
1. **Resumen Ejecutivo** (1 párrafo)
   - Problema, enfoque, resultado clave
2. **Dataset y EDA** (1/2 página)
   - Descripción, estadísticas clave, insights
3. **Metodología** (1 página)
   - Preprocesamiento aplicado
   - Modelos evaluados
   - Estrategia de validación
4. **Resultados** (1/2 página)
   - Comparación de modelos (tabla)
   - Modelo final y métricas en test
5. **Conclusiones y Siguientes Pasos** (1/2 página)
   - Lecciones aprendidas
   - Limitaciones
   - Mejoras futuras

### 8. Reproducibilidad
**Entregables:** `requirements.txt`, `README.md`, código organizado

Estructura del proyecto:
```
capstone-1/
├── README.md                      # Instrucciones de uso
├── requirements.txt               # Dependencias con versiones
├── data/
│   ├── raw/                      # Datos originales (no modificar)
│   │   └── dataset.csv
│   └── processed/                # Datos procesados
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── 05_final_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Funciones de preprocesamiento
│   ├── pipeline.py               # Definición de pipelines
│   ├── models.py                 # Configuraciones de modelos
│   └── evaluation.py             # Métricas y evaluación
├── models/
│   ├── baseline.pkl
│   ├── final_model.pkl
│   └── model_metadata.json
├── reports/
│   ├── REPORT.md
│   └── figures/
│       ├── correlation_heatmap.png
│       ├── roc_curve.png
│       └── learning_curves.png
└── mlruns/                       # MLflow artifacts (si se usa)
```

**README.md debe incluir:**
- Descripción del proyecto
- Instalación y setup (`pip install -r requirements.txt`)
- Cómo ejecutar el pipeline completo
- Estructura de archivos
- Resultados clave

## Rúbrica de Evaluación

| Componente | Peso | Criterios de Excelencia |
|------------|------|------------------------|
| **EDA y Visualizaciones** | 20% | Análisis profundo, visualizaciones claras, insights accionables |
| **Feature Engineering** | 20% | Transformaciones justificadas, features creativos pero sensatos, sin leakage |
| **Pipeline y Reproducibilidad** | 20% | Pipeline completo, código modular, seeds fijados, fácil de ejecutar |
| **Evaluación de Modelos** | 20% | Comparación justa, validación cruzada correcta, métricas apropiadas |
| **Reporte y Comunicación** | 20% | Narrativa clara, decisiones justificadas, resultados bien presentados |

### Desglose Detallado

#### EDA y Visualizaciones (20%)
- **Excelente (18-20 pts):** Análisis completo con insights profundos, visualizaciones profesionales, narrativa clara
- **Bueno (15-17 pts):** Análisis sólido, buenas visualizaciones, algunos insights interesantes
- **Aceptable (12-14 pts):** Análisis básico completo, visualizaciones estándar
- **Insuficiente (<12 pts):** Análisis superficial, visualizaciones pobres o faltantes

#### Feature Engineering (20%)
- **Excelente (18-20 pts):** Transformaciones creativas y bien justificadas, sin leakage, features mejoran modelo significativamente
- **Bueno (15-17 pts):** Transformaciones estándar bien aplicadas, sin leakage
- **Aceptable (12-14 pts):** Transformaciones básicas correctas
- **Insuficiente (<12 pts):** Data leakage, transformaciones no justificadas o incorrectas

#### Pipeline y Reproducibilidad (20%)
- **Excelente (18-20 pts):** Pipeline completo y elegante, código ejecutable sin errores, fácil de reproducir
- **Bueno (15-17 pts):** Pipeline funcional, reproducible con ajustes menores
- **Aceptable (12-14 pts):** Pipeline básico funcional, algún problema de reproducibilidad
- **Insuficiente (<12 pts):** Pipeline incompleto o no funcional, no reproducible

#### Evaluación de Modelos (20%)
- **Excelente (18-20 pts):** Comparación exhaustiva, validación rigurosa, mejora clara sobre baseline
- **Bueno (15-17 pts):** Comparación sólida de 3+ modelos, validación correcta
- **Aceptable (12-14 pts):** Comparación básica de modelos, validación aceptable
- **Insuficiente (<12 pts):** Evaluación incorrecta (leakage, métricas inapropiadas)

#### Reporte y Comunicación (20%)
- **Excelente (18-20 pts):** Reporte profesional, decisiones bien justificadas, fácil de entender
- **Bueno (15-17 pts):** Reporte claro y completo
- **Aceptable (12-14 pts):** Reporte básico con información necesaria
- **Insuficiente (<12 pts):** Reporte incompleto o confuso

## Criterios de Aprobación

Para aprobar el Capstone 1, debes:
- [ ] Obtener ≥70% (≥70 puntos de 100)
- [ ] No tener data leakage verificable
- [ ] Pipeline ejecutable sin errores
- [ ] Modelo final supera baseline en al menos 10%
- [ ] Reporte técnico completo

## Extras (Puntos Bonus)

Puedes ganar hasta 10 puntos adicionales con:
- [ ] **+3 pts:** Feature selection sistemático (RFE, feature importance)
- [ ] **+3 pts:** Análisis de interpretabilidad (SHAP, LIME)
- [ ] **+2 pts:** Tests automatizados (pytest)
- [ ] **+2 pts:** CI/CD básico (GitHub Actions para tests)

## Timeline Sugerido

| Día | Actividad | Tiempo |
|-----|-----------|--------|
| 1 | EDA completo | 3 horas |
| 2 | Limpieza y feature engineering | 3 horas |
| 3 | Implementación de pipeline + baseline | 2 horas |
| 4 | Comparación de modelos | 2 horas |
| 5 | Hyperparameter tuning | 2 horas |
| 6 | Evaluación final + análisis de errores | 2 horas |
| 7 | Reporte técnico + revisión | 2 horas |

**Total:** ~16 horas (con margen)

## Checklist de Entrega

Antes de entregar, verifica:
- [ ] Todos los notebooks ejecutan sin errores
- [ ] `requirements.txt` está actualizado
- [ ] README.md tiene instrucciones claras
- [ ] No hay data leakage
- [ ] Seeds están fijados
- [ ] Reporte técnico completo
- [ ] Código limpio y comentado
- [ ] Visualizaciones exportadas en alta calidad
- [ ] MLflow run registrado (si se usa)

## Preguntas Frecuentes

**P: ¿Puedo usar mi propio dataset?**  
R: Sí, pero debe ser aprobado. Debe tener >1000 registros, mix de features, y un problema de negocio claro.

**P: ¿Cuántos modelos debo comparar?**  
R: Mínimo 3 (incluyendo baseline). Más es mejor, pero enfócate en calidad sobre cantidad.

**P: ¿Debo usar deep learning?**  
R: No. Este capstone es sobre ML clásico. Deep learning es tema de capstones posteriores.

**P: ¿Puedo usar AutoML (TPOT, Auto-sklearn)?**  
R: No como solución principal, pero puedes usarlo como punto de comparación adicional.

**P: ¿Qué hago si mi modelo final es peor que el baseline?**  
R: Documéntalo honestamente y analiza por qué. A veces los datos no son predictivos, y eso también es un resultado válido.

## Recursos de Ayuda

- [Scikit-learn Pipeline Tutorial](https://scikit-learn.org/stable/modules/compose.html)
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)
- [Kaggle: Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- Módulos 1-3 del curso

## Entrega

**Formato:** Repositorio Git (GitHub, GitLab) con todo el código  
**Deadline:** Consulta el calendario del curso  
**Método:** Enviar link al repositorio + README con instrucciones

---

**¡Buena suerte! Este es tu primer proyecto completo de ML. Tómate tu tiempo para hacerlo bien.**
