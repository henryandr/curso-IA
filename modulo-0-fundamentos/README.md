# Módulo 0: Fundamentos Matemáticos para ML (Opcional)

## Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- Calcular e interpretar estadísticas descriptivas (media, mediana, varianza, desviación estándar)
- Aplicar operaciones básicas de álgebra lineal (vectores, matrices, producto punto)
- Normalizar y estandarizar datos para ML
- Visualizar y analizar correlaciones entre variables

## Duración Estimada
**6-8 horas** (1 semana a ritmo relajado)

## Prerrequisitos
- Python básico
- Conocimientos matemáticos de secundaria

## Contenidos

### 1. Estadística Descriptiva (2 horas)
- Media, mediana, moda
- Varianza y desviación estándar
- Percentiles y cuartiles
- Distribuciones de probabilidad básicas (normal, uniforme)

### 2. Álgebra Lineal para ML (2 horas)
- Vectores y escalares
- Matrices y operaciones matriciales
- Producto punto y normas
- Proyecciones

### 3. Correlación y Covarianza (1.5 horas)
- Matriz de correlación
- Interpretación de correlaciones
- Visualización con heatmaps

### 4. Normalización y Escalado (1.5 horas)
- Min-Max scaling
- Z-score (estandarización)
- Cuándo usar cada uno

## Prácticas Guiadas

### Práctica 1: Estadísticas con NumPy
**Objetivo:** Calcular estadísticas descriptivas de un dataset
**Dataset:** Datos sintéticos de ventas
**Tiempo:** 45 min
**Notebook:** `notebooks/practica-1-estadisticas.ipynb`

**Actividades:**
1. Cargar datos con numpy
2. Calcular media, mediana, varianza
3. Visualizar distribuciones con histogramas
4. Interpretar resultados

### Práctica 2: Operaciones con Matrices
**Objetivo:** Realizar operaciones de álgebra lineal
**Tiempo:** 45 min
**Notebook:** `notebooks/practica-2-algebra-lineal.ipynb`

**Actividades:**
1. Crear vectores y matrices con numpy
2. Multiplicación de matrices
3. Calcular producto punto
4. Normas y distancias

### Práctica 3: Correlaciones
**Objetivo:** Analizar correlaciones entre variables
**Dataset:** Dataset de viviendas
**Tiempo:** 1 hora
**Notebook:** `notebooks/practica-3-correlaciones.ipynb`

**Actividades:**
1. Cargar dataset con pandas
2. Calcular matriz de correlación
3. Visualizar con seaborn heatmap
4. Identificar variables correlacionadas

### Práctica 4: Normalización
**Objetivo:** Implementar técnicas de escalado
**Tiempo:** 1 hora
**Notebook:** `notebooks/practica-4-normalizacion.ipynb`

**Actividades:**
1. Implementar min-max scaling desde cero
2. Implementar z-score desde cero
3. Comparar con sklearn StandardScaler y MinMaxScaler
4. Visualizar impacto del escalado

## Ejercicios de Consolidación

### Ejercicio 1: Análisis Estadístico
**Enunciado:** Dado el dataset `ejercicios/datos_ejercicio1.csv`, calcula:
- Media y mediana de todas las columnas numéricas
- Desviación estándar y coeficiente de variación
- Identifica outliers usando la regla 3-sigma
- Crea visualizaciones apropiadas

**Criterios de Corrección:**
- Cálculos correctos (40%)
- Identificación correcta de outliers (30%)
- Visualizaciones claras y etiquetadas (20%)
- Interpretación coherente (10%)

**Solución:** `ejercicios/solucion_ejercicio1.ipynb`

### Ejercicio 2: Transformaciones de Datos
**Enunciado:** Implementa tres funciones desde cero (sin sklearn):
1. `min_max_scale(X)`: normalización [0,1]
2. `standardize(X)`: z-score
3. `robust_scale(X)`: usando mediana y IQR

Prueba con el dataset `ejercicios/datos_ejercicio2.csv` y compara resultados.

**Criterios de Corrección:**
- Implementación correcta (50%)
- Pruebas con datos (25%)
- Comparación con sklearn (15%)
- Documentación del código (10%)

**Solución:** `ejercicios/solucion_ejercicio2.py`

### Ejercicio 3: Matriz de Correlación
**Enunciado:** Analiza el dataset `ejercicios/datos_ejercicio3.csv`:
- Calcula la matriz de correlación
- Visualiza con heatmap
- Identifica las 5 pares de variables más correlacionadas
- Explica qué implica cada correlación

**Criterios de Corrección:**
- Matriz correcta (30%)
- Visualización apropiada (30%)
- Identificación correcta de pares (20%)
- Interpretación sensata (20%)

**Solución:** `ejercicios/solucion_ejercicio3.ipynb`

## Mini-Quiz

**Tiempo:** 15-20 minutos

1. **Conceptual:** ¿Cuál es la diferencia entre media y mediana? ¿Cuándo preferirías una sobre la otra?

2. **Conceptual:** Si una variable tiene media=100 y desviación estándar=15, ¿qué porcentaje de datos esperarías entre 85 y 115? (asumiendo distribución normal)

3. **Código:** Completa el código para calcular la correlación entre dos variables:
```python
def correlation(x, y):
    # Tu código aquí
    pass
```

4. **Conceptual:** ¿Por qué es importante normalizar los datos antes de entrenar algunos modelos de ML?

5. **Aplicado:** Tienes dos variables: edad (rango 18-80) y salario (rango 20000-200000). ¿Qué problema podría surgir si no las escalas antes de usarlas en un modelo?

6. **Matemático:** Calcula el producto punto de [2, 3] y [4, 5].

7. **Conceptual:** ¿Qué indica una correlación de -0.9 entre dos variables?

8. **Aplicado:** ¿Cuándo usarías RobustScaler en lugar de StandardScaler?

**Soluciones:** `recursos/quiz-soluciones.md`

## Recursos Adicionales

### Lecturas
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas User Guide - Descriptive Statistics](https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics)
- Khan Academy: Statistics and Probability

### Videos
- 3Blue1Brown: "Essence of Linear Algebra" series
- StatQuest: "Statistics Fundamentals" series

### Libros
- "Naked Statistics" por Charles Wheelan (introducción no técnica)
- "Linear Algebra Done Right" por Sheldon Axler (más avanzado)

## Siguientes Pasos

Una vez completado este módulo:
1. ✅ Verifica que puedes resolver los ejercicios
2. ✅ Completa el mini-quiz
3. ➡️ Continúa a **Módulo 1: EDA con Pandas**

---

**Nota:** Este módulo es opcional si ya tienes sólidos conocimientos de estadística y álgebra lineal. Puedes hacer el quiz primero para evaluar si necesitas el repaso.
