# Módulo 1: EDA y Preparación de Datos con Pandas

## Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- **Aplicar** pandas para cargar, explorar y limpiar datasets (Bloom 3)
- **Analizar** distribuciones de datos e identificar problemas (Bloom 4)
- **Evaluar** diferentes estrategias de imputación y tratamiento de outliers (Bloom 5)
- **Crear** visualizaciones informativas para comunicar insights (Bloom 6)
- **Implementar** pipelines reproducibles de preprocesamiento (Bloom 6)

## Duración Estimada
**10-12 horas** (1 semana)

## Prerrequisitos
- Python 3.10+
- Módulo 0 (o conocimientos equivalentes de estadística)
- Jupyter Notebook instalado

## Contenidos

### 1. Introducción a Pandas (2 horas)
- DataFrames y Series
- Indexación: loc, iloc, boolean indexing
- Operaciones básicas: filtrado, ordenamiento, agrupación
- Lectura/escritura: CSV, Excel, Parquet

### 2. Análisis Exploratorio de Datos - EDA (2.5 horas)
- Información del dataset: shape, dtypes, info()
- Estadísticas descriptivas: describe()
- Valores únicos y frecuencias: value_counts()
- Valores faltantes: isnull(), isna()
- Distribuciones y asimetría

### 3. Limpieza de Datos (2.5 horas)
- Detección de valores perdidos
- Estrategias de imputación:
  - Media/mediana/moda
  - Forward fill / Backward fill
  - KNN imputation
  - Eliminación estratégica
- Detección de duplicados
- Inconsistencias en datos (tipos, formatos)

### 4. Detección y Tratamiento de Outliers (2 horas)
- Métodos de detección:
  - IQR (Interquartile Range)
  - Z-score
  - Visualización con boxplots
- Estrategias de tratamiento:
  - Eliminación
  - Capping/Flooring
  - Transformación (log, sqrt)

### 5. Feature Engineering Básico (1.5 horas)
- Encoding categórico:
  - Label Encoding
  - One-Hot Encoding
  - Target Encoding (conceptual)
- Binning y discretización
- Creación de features derivados
- Feature scaling: StandardScaler, MinMaxScaler, RobustScaler

### 6. Visualización con Matplotlib y Seaborn (2 horas)
- Histogramas y distribuciones
- Box plots y violin plots
- Scatter plots y pair plots
- Heatmaps de correlación
- Time series plots
- Personalización de gráficos

## Prácticas Guiadas

### Práctica 1: EDA Completo - Dataset de Churn
**Objetivo:** Realizar un análisis exploratorio completo de un dataset de churn de telecomunicaciones
**Dataset:** Telco Customer Churn (Kaggle)
**Tiempo:** 2 horas
**Notebook:** `notebooks/practica-1-eda-churn.ipynb`

**Actividades:**
1. Cargar datos y exploración inicial
2. Análisis de tipos de datos y conversiones necesarias
3. Estadísticas descriptivas por variable
4. Análisis de valores faltantes
5. Distribución de la variable target (churn)
6. Análisis univariado de variables numéricas y categóricas
7. Análisis bivariado: relación features vs target
8. Correlaciones entre variables numéricas
9. Insights y conclusiones

**Entregable:** Notebook completado con narrativa clara

### Práctica 2: Limpieza y Transformación - Dataset de Ventas
**Objetivo:** Limpiar y preparar un dataset "sucio" de ventas
**Dataset:** Sales Dataset con problemas de calidad
**Tiempo:** 1.5 horas
**Notebook:** `notebooks/practica-2-limpieza-ventas.ipynb`

**Actividades:**
1. Identificar problemas de calidad
2. Corregir tipos de datos
3. Manejar valores faltantes con justificación
4. Eliminar duplicados
5. Estandarizar formatos (fechas, strings)
6. Validar rangos de valores
7. Crear dataset limpio

**Entregable:** Dataset limpio + reporte de transformaciones

### Práctica 3: Detección de Outliers - Precios Inmobiliarios
**Objetivo:** Detectar y tratar outliers en datos de bienes raíces
**Dataset:** House Prices Dataset
**Tiempo:** 1.5 horas
**Notebook:** `notebooks/practica-3-outliers-casas.ipynb`

**Actividades:**
1. Visualización de distribuciones (histogramas, boxplots)
2. Detección con IQR
3. Detección con Z-score
4. Análisis de outliers: ¿errores o casos legítimos?
5. Aplicar diferentes estrategias de tratamiento
6. Comparar impacto en estadísticas descriptivas
7. Documentar decisiones

**Entregable:** Análisis comparativo de estrategias

### Práctica 4: Feature Engineering - Dataset de Marketing
**Objetivo:** Crear y transformar features para modelo de clasificación
**Dataset:** Bank Marketing Dataset
**Tiempo:** 2 horas
**Notebook:** `notebooks/practica-4-feature-engineering.ipynb`

**Actividades:**
1. Análisis de variables categóricas
2. One-hot encoding de variables nominales
3. Label encoding de variables ordinales
4. Binning de variables continuas
5. Creación de features de interacción
6. Escalado de variables numéricas
7. Validación de transformaciones

**Entregable:** Dataset procesado listo para ML

### Práctica 5: Dashboard de Visualización
**Objetivo:** Crear un dashboard visual completo con seaborn
**Dataset:** Conjunto de datos a elección
**Tiempo:** 2 horas
**Notebook:** `notebooks/practica-5-dashboard-viz.ipynb`

**Actividades:**
1. Diseñar layout del dashboard
2. Gráficos de distribución
3. Análisis temporal (si aplica)
4. Comparaciones categóricas
5. Heatmap de correlaciones
6. Insights visuales destacados
7. Exportar visualizaciones de alta calidad

**Entregable:** Dashboard completo con interpretaciones

## Ejercicios de Consolidación

### Ejercicio 1: EDA del Titanic
**Enunciado:** 
Realiza un EDA completo del famoso dataset del Titanic:
1. Carga el dataset (`ejercicios/titanic.csv`)
2. Analiza la estructura y tipos de datos
3. Trata valores faltantes (Age, Cabin, Embarked)
4. Crea visualizaciones que muestren:
   - Tasa de supervivencia por clase
   - Distribución de edad por género
   - Relación entre tarifa pagada y supervivencia
5. Genera un reporte ejecutivo con 5 insights clave

**Tiempo:** 2-3 horas

**Criterios de Corrección:**
- Exploración sistemática (20%)
- Tratamiento apropiado de valores faltantes (25%)
- Visualizaciones claras y etiquetadas (25%)
- Insights correctos y relevantes (20%)
- Código limpio y documentado (10%)

**Solución:** `ejercicios/solucion-ejercicio-1.ipynb`

### Ejercicio 2: Comparación de Estrategias de Imputación
**Enunciado:**
Usando el dataset `ejercicios/datos_faltantes.csv`:
1. Implementa 3 estrategias de imputación diferentes para cada columna numérica
2. Compara el impacto en:
   - Media y mediana
   - Desviación estándar
   - Distribución (KS test o visual)
3. Para cada estrategia, documenta:
   - Cuándo es apropiada
   - Ventajas y desventajas
   - Impacto observado
4. Recomienda la mejor estrategia para este dataset

**Tiempo:** 2 horas

**Criterios de Corrección:**
- Implementación correcta de 3 estrategias (30%)
- Comparación cuantitativa (30%)
- Análisis cualitativo (25%)
- Recomendación fundamentada (15%)

**Solución:** `ejercicios/solucion-ejercicio-2.ipynb`

### Ejercicio 3: Pipeline de Preprocesamiento
**Enunciado:**
Crea un pipeline reproducible que:
1. Cargue `ejercicios/raw_data.csv`
2. Limpie datos (nulos, duplicados, tipos)
3. Detecte y trate outliers
4. Realice feature engineering (encoding, scaling)
5. Exporte datos procesados
6. Genere un reporte automático de transformaciones

Requisitos técnicos:
- Usar funciones modulares
- Incluir logging de cada paso
- Guardar metadata de transformaciones
- Código reproducible (seeds si aplica)

**Tiempo:** 3-4 horas

**Criterios de Corrección:**
- Modularidad y organización del código (25%)
- Transformaciones correctas (25%)
- Reproducibilidad (20%)
- Logging y documentación (20%)
- Reporte automático (10%)

**Solución:** `ejercicios/solucion-ejercicio-3/`

### Ejercicio 4: Informe Ejecutivo con Visualizaciones
**Enunciado:**
Analiza el dataset `ejercicios/ecommerce_data.csv` y crea un informe ejecutivo (estilo Jupyter Notebook) que responda:
1. ¿Cuáles son los productos más vendidos?
2. ¿Cuál es el patrón temporal de ventas?
3. ¿Qué segmento de clientes es más rentable?
4. ¿Hay estacionalidad en las ventas?
5. ¿Qué variables predicen mejor la conversión?

Incluye al menos 6 visualizaciones de alta calidad.

**Tiempo:** 2-3 horas

**Criterios de Corrección:**
- Respuesta completa a cada pregunta (40%)
- Visualizaciones apropiadas y estéticas (30%)
- Narrativa clara y profesional (20%)
- Conclusiones accionables (10%)

**Solución:** `ejercicios/solucion-ejercicio-4.ipynb`

## Mini-Quiz

**Tiempo:** 15-20 minutos

### Preguntas Conceptuales

1. ¿Cuál es la diferencia entre `df.loc` y `df.iloc`?

2. Tienes un DataFrame con 1000 filas y una columna tiene 200 valores NaN (20%). ¿Eliminarías la columna, las filas, o imputarías? Justifica tu respuesta.

3. ¿Por qué One-Hot Encoding puede causar problemas si tienes muchas categorías únicas?

4. Explica qué es el "IQR" y cómo se usa para detectar outliers.

5. ¿Cuándo usarías `StandardScaler` vs `MinMaxScaler` vs `RobustScaler`?

### Preguntas de Código

6. Completa el código para imputar valores faltantes con la mediana:
```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, None, 4, None]})
# Tu código aquí
```

7. Crea un boxplot con seaborn para visualizar la distribución de 'price' por 'category':
```python
import seaborn as sns
# Tu código aquí
```

8. ¿Qué hace este código y cuál es el resultado?
```python
df.groupby('category')['sales'].agg(['mean', 'sum', 'count'])
```

**Soluciones:** `recursos/quiz-soluciones.md`

## Datasets del Módulo

Todos los datasets están en `recursos/datasets/`:
- `telco_churn.csv` - Práctica 1
- `sales_dirty.csv` - Práctica 2
- `house_prices.csv` - Práctica 3
- `bank_marketing.csv` - Práctica 4
- `titanic.csv` - Ejercicio 1
- `datos_faltantes.csv` - Ejercicio 2
- `raw_data.csv` - Ejercicio 3
- `ecommerce_data.csv` - Ejercicio 4

**Nota:** Si los datasets no están disponibles localmente, usa los scripts en `recursos/download_datasets.py` para descargarlos de Kaggle.

## Recursos Adicionales

### Documentación Oficial
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

### Tutoriales
- [Kaggle Learn: Pandas](https://www.kaggle.com/learn/pandas)
- [Real Python: Pandas Tutorials](https://realpython.com/learning-paths/pandas-data-science/)

### Libros
- "Python for Data Analysis" por Wes McKinney
- "Data Cleaning" por Ihab Ilyas y Xu Chu

### Videos
- StatQuest: "Data Cleaning"
- Corey Schafer: Pandas Series

## Checklist de Evaluación

Antes de pasar al siguiente módulo, asegúrate de:
- [ ] Completar las 5 prácticas guiadas
- [ ] Resolver al menos 3 de los 4 ejercicios
- [ ] Aprobar el mini-quiz con al menos 75%
- [ ] Ser capaz de explicar cuándo usar cada técnica de imputación
- [ ] Poder crear visualizaciones informativas sin consultar documentación
- [ ] Entender la importancia de prevenir data leakage

## Anti-Patrones a Evitar

❌ **Imputar antes de split train/test**
```python
# MAL
df['age'].fillna(df['age'].mean(), inplace=True)
train, test = train_test_split(df)
```

✅ **Imputar después de split**
```python
# BIEN
train, test = train_test_split(df)
mean_age = train['age'].mean()
train['age'].fillna(mean_age, inplace=True)
test['age'].fillna(mean_age, inplace=True)
```

❌ **Eliminar outliers sin análisis**
```python
# MAL - elimina automáticamente sin entender
df = df[df['price'] < df['price'].quantile(0.99)]
```

✅ **Analizar outliers primero**
```python
# BIEN - investiga antes de decidir
outliers = df[df['price'] > df['price'].quantile(0.99)]
print(f"Outliers: {len(outliers)}")
print(outliers.describe())
# Luego decide: eliminar, transformar, o mantener
```

## Siguientes Pasos

Una vez completado este módulo:
1. ✅ Verifica el checklist de evaluación
2. ✅ Completa el mini-quiz
3. ➡️ Continúa a **Módulo 2: Fundamentos de ML con Scikit-Learn**

---

**Nota:** Este módulo es la base de todo el curso. Asegúrate de dominarlo antes de avanzar.
