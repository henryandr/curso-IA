# Módulo 3: Selección y Optimización de Modelos

## 📚 Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- **Comparar** múltiples algoritmos de ML de forma sistemática (Bloom 5)
- **Aplicar** técnicas de feature selection (Bloom 3)
- **Optimizar** hiperparámetros con búsqueda en grilla y aleatoria (Bloom 3)  
- **Evaluar** modelos con experiment tracking (Bloom 5)
- **Implementar** estrategias para datos desbalanceados (Bloom 3)

## ⏱️ Duración Estimada
**12-15 horas** (1.5 semanas)

## 🗺️ Mapa Conceptual del Módulo

```
                    SELECCIÓN Y OPTIMIZACIÓN DE MODELOS
                                    │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
   ALGORITMOS                 OPTIMIZACIÓN              EVALUACIÓN
   AVANZADOS                                           AVANZADA
        │                          │                          │
  ┌─────┴─────┐           ┌────────┴────────┐        ┌────────┴────────┐
  │           │           │                 │        │                 │
Árboles    Gradient    Feature          Hyper-    Experiment      Imbalanced
Decision   Boosting   Selection        parameter   Tracking          Data
  │           │           │              Tuning        │                │
  • RF        • XGBoost   • RFE           • Grid      • MLflow         • SMOTE
  • Extra     • LightGBM  • SelectKBest   • Random    • W&B            • Weights
    Trees     • CatBoost  • Lasso         • Bayesian                   • Tuning
```

## 📖 Contenidos Detallados

### 1. Algoritmos Basados en Árboles

#### Decision Trees
Los árboles dividen el espacio de características mediante preguntas binarias.

**Criterios de División:**
- **Gini:** `Gini = 1 - Σ(pi²)`
- **Entropy:** `Entropy = -Σ(pi * log2(pi))`

**Ejemplo de código:**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

plot_tree(dt, filled=True, feature_names=feature_names)
plt.show()
```

#### Random Forest
Ensemble de árboles entrenados con bootstrap y feature sampling aleatorio.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

#### Gradient Boosting (XGBoost, LightGBM)

**XGBoost:**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
```

### 2. Feature Selection

**Métodos:**
- **Filter:** SelectKBest, Chi², ANOVA
- **Wrapper:** RFE, RFECV  
- **Embedded:** Lasso (L1), Tree importance

```python
from sklearn.feature_selection import SelectKBest, RFE, chi2

# Filter method
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Wrapper method
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)
```

### 4. Experiment Tracking con MLflow

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("model_comparison")

with mlflow.start_run(run_name="RandomForest"):
    model.fit(X_train, y_train)
    
    # Log parameters and metrics
    mlflow.log_params(model.get_params())
    mlflow.log_metric("roc_auc", roc_auc_score(y_val, y_pred))
    mlflow.sklearn.log_model(model, "model")
```

### 5. Datos Desbalanceados

**SMOTE:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

**Class Weights:**
```python
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
```

## 🎯 Prácticas Guiadas

1. **Comparación RF vs XGBoost** (2.5h) - Credit Card Fraud
2. **Hyperparameter Tuning** (2.5h) - Bank Marketing
3. **Feature Selection** (2h) - Wine Quality
4. **Imbalanced Data** (2h) - Loan Default
5. **Experiment Tracking** (1.5h) - MLflow setup

## ✏️ Ejercicios

1. Optimizar RandomForest con Grid/Random Search
2. Crear ensemble de 3 modelos diferentes
3. Feature engineering + selection con 100+ features

## 📚 Recursos

### Videos
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) (9 min)
- [StatQuest: Gradient Boost](https://www.youtube.com/watch?v=3CC4N4z3GJc) (10 min)
- [StatQuest: XGBoost](https://www.youtube.com/watch?v=OtD8wVaFm6E) (10 min)

### Lecturas
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [SMOTE Paper](https://arxiv.org/abs/1106.1813)

### Documentación
- [Scikit-learn Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)

## ✅ Checklist

- [ ] Entrenar y comparar 3+ algoritmos
- [ ] Aplicar hyperparameter tuning
- [ ] Usar experiment tracking
- [ ] Feature selection con 2+ métodos
- [ ] Manejar datos desbalanceados
- [ ] Documentar decisiones

## ➡️ Siguientes Pasos

**Módulo 4:** NLP Moderno y Embeddings
