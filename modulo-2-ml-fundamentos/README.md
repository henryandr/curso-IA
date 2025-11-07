# Módulo 2: Fundamentos de Machine Learning con Scikit-Learn

## Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- **Aplicar** algoritmos de regresión y clasificación con scikit-learn (Bloom 3)
- **Evaluar** modelos con métricas apropiadas (Bloom 5)
- **Implementar** validación cruzada correctamente (Bloom 3)
- **Crear** pipelines completos de ML (Bloom 6)
- **Prevenir** data leakage en tus flujos de trabajo (Bloom 4)

## Duración Estimada
**12-15 horas** (1.5 semanas)

## Contenidos Clave

### 1. Conceptos Fundamentales
- Aprendizaje supervisado vs no supervisado
- Overfitting, underfitting y bias-variance tradeoff
- Train/validation/test splits
- Validación cruzada (k-fold, stratified)

### 2. Regresión
- Linear Regression, Ridge, Lasso
- Métricas: MAE, MSE, RMSE, R², MAPE
- Interpretación de coeficientes

### 3. Clasificación
- Logistic Regression, KNN, Naive Bayes
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC
- Matriz de confusión
- Threshold tuning

### 4. Pipelines y Automatización
- sklearn.pipeline.Pipeline
- ColumnTransformer
- Prevención de data leakage
- Persistencia de modelos

## Prácticas Guiadas

1. **Regresión de Precios:** Predecir precios de casas (2h)
2. **Clasificación Binaria:** Predicción de churn (2h)
3. **Pipelines Completos:** Preprocesamiento + modelo (2h)
4. **Validación Cruzada:** Comparación robusta de modelos (1.5h)
5. **Análisis de Errores:** Learning curves y diagnóstico (1.5h)

## Ejercicios de Consolidación

1. Construir pipeline end-to-end para clasificación
2. Comparar 3 algoritmos con CV
3. Diagnosticar overfitting con learning curves
4. Optimizar threshold para maximizar F1-score

## Mini-Quiz (8 preguntas)

Conceptuales y de código sobre métricas, validación y pipelines.

## Anti-Patrones Críticos

❌ **Normalizar antes del split**
❌ **Evaluar repetidamente en test set**
❌ **Usar accuracy en datasets desbalanceados**

✅ **Usar pipelines**
✅ **Validación cruzada para comparar modelos**
✅ **Métricas apropiadas al problema de negocio**

## Recursos

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- "Hands-On Machine Learning" - Caps. 2-4
- [ML Glossary](https://ml-cheatsheet.readthedocs.io/)

## Siguientes Pasos

➡️ **Módulo 3:** Selección y Optimización de Modelos
