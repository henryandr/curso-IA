# Estructura de Repositorio y Convenciones

## Estructura General del Repositorio

```
curso-IA/
├── README.md                                 # Inicio del curso
├── 00-SINTESIS-EJECUTIVA.md                 # Resumen ejecutivo
├── 01-PREGUNTAS-CALIBRACION.md              # Personalización
├── 02-SYLLABUS.md                           # Temario completo
├── 03-ANTI-PATRONES-Y-MEJORES-PRACTICAS.md  # Guía de calidad
├── requirements.txt                          # Dependencias Python
├── .gitignore                               # Archivos a ignorar
├── LICENSE                                  # Licencia del curso
│
├── modulo-0-fundamentos/                    # Repaso matemático (opcional)
│   ├── README.md                            # Guía del módulo
│   ├── notebooks/                           # Jupyter notebooks
│   │   ├── practica-1-estadisticas.ipynb
│   │   ├── practica-2-algebra-lineal.ipynb
│   │   ├── practica-3-correlaciones.ipynb
│   │   └── practica-4-normalizacion.ipynb
│   ├── ejercicios/                          # Ejercicios y soluciones
│   │   ├── datos_ejercicio1.csv
│   │   ├── solucion_ejercicio1.ipynb
│   │   └── ...
│   └── recursos/                            # Recursos adicionales
│       ├── quiz-soluciones.md
│       └── lecturas.md
│
├── modulo-1-eda/                            # EDA con pandas
│   ├── README.md
│   ├── notebooks/
│   │   ├── practica-1-eda-churn.ipynb
│   │   ├── practica-2-limpieza-ventas.ipynb
│   │   ├── practica-3-outliers-casas.ipynb
│   │   ├── practica-4-feature-engineering.ipynb
│   │   └── practica-5-dashboard-viz.ipynb
│   ├── ejercicios/
│   │   ├── titanic.csv
│   │   ├── solucion-ejercicio-1.ipynb
│   │   └── ...
│   └── recursos/
│       └── datasets/
│
├── modulo-2-ml-fundamentos/                 # ML con scikit-learn
├── modulo-3-seleccion-modelos/              # Optimización
├── modulo-4-nlp/                            # NLP moderno
├── modulo-5-deep-learning/                  # PyTorch
├── modulo-6-llms-rag/                       # LLMs y RAG
├── modulo-7-agentes-n8n/                    # Agentes y n8n
│
├── capstone-1/                              # Proyecto ML clásico
│   ├── README.md                            # Especificación y rúbrica
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_preprocessing.ipynb
│   │   ├── 03_model_comparison.ipynb
│   │   ├── 04_hyperparameter_tuning.ipynb
│   │   └── 05_final_evaluation.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── pipeline.py
│   │   ├── models.py
│   │   └── evaluation.py
│   ├── models/
│   ├── reports/
│   │   ├── REPORT.md
│   │   └── figures/
│   └── requirements.txt
│
├── capstone-2/                              # Proyecto NLP
│   ├── README.md
│   ├── data/
│   ├── notebooks/
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_baseline.ipynb
│   │   ├── 03_embeddings.ipynb
│   │   ├── 04_transformers.ipynb
│   │   ├── 05_experiments.ipynb
│   │   └── 06_final_evaluation.ipynb
│   ├── src/
│   ├── models/
│   ├── reports/
│   └── requirements.txt
│
├── capstone-3/                              # Proyecto RAG + Agente + n8n
│   ├── README.md
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── .env.example
│   ├── api/                                 # FastAPI
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routers/
│   │   └── models.py
│   ├── src/
│   │   ├── rag/
│   │   │   ├── chunking.py
│   │   │   ├── embeddings.py
│   │   │   └── vector_store.py
│   │   ├── agent/
│   │   │   ├── tools.py
│   │   │   ├── react_agent.py
│   │   │   └── memory.py
│   │   └── utils/
│   ├── n8n/
│   │   ├── workflow-ingestion.json
│   │   └── workflow-query.json
│   ├── data/
│   ├── tests/
│   │   ├── test_rag.py
│   │   ├── test_agent.py
│   │   └── test_api.py
│   ├── evaluation/
│   │   ├── test_cases.json
│   │   └── evaluate.py
│   └── requirements.txt
│
└── recursos/                                # Recursos compartidos
    ├── datasets/                            # Datasets del curso
    │   ├── README.md                        # Índice de datasets
    │   ├── download_datasets.py             # Script de descarga
    │   └── ...
    ├── guias-n8n/                           # Guías de n8n
    │   ├── guia-1-etl-basico.md
    │   ├── guia-2-orquestacion-ml.md
    │   └── guia-3-pipeline-rag.md
    ├── papers/                              # Papers de referencia
    │   ├── attention-is-all-you-need.pdf
    │   ├── bert.pdf
    │   ├── rag.pdf
    │   └── react.pdf
    └── templates/                           # Templates
        ├── notebook-template.ipynb
        ├── README-template.md
        └── docker-compose-template.yml
```

## Convenciones de Nomenclatura

### Archivos y Directorios

- **Módulos:** `modulo-N-nombre/` (minúsculas, guiones)
- **Capstones:** `capstone-N/` (donde N es 1, 2, o 3)
- **Notebooks:** `practica-N-descripcion.ipynb` o `0N_descripcion.ipynb`
- **Scripts Python:** `snake_case.py`
- **Datos:** `raw/`, `processed/`, `external/`
- **Modelos:** `models/nombre_modelo_vYYYYMMDD.pkl`

### Código Python

#### Imports
```python
# Estándar library
import os
import sys
from datetime import datetime

# Third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local
from src.preprocessing import clean_data
from src.models import train_model
```

#### Funciones
```python
def process_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Procesa y divide datos en train/test.
    
    Args:
        df: DataFrame con datos
        target_col: Nombre de la columna target
        test_size: Proporción del test set
        random_state: Seed para reproducibilidad
        
    Returns:
        Tuple con (X_train, X_test, y_train, y_test)
    """
    # Implementación
    pass
```

#### Clases
```python
class DataPreprocessor:
    """Preprocesador de datos para ML."""
    
    def __init__(self, strategy: str = 'mean'):
        """
        Inicializa el preprocesador.
        
        Args:
            strategy: Estrategia de imputación ('mean', 'median', 'mode')
        """
        self.strategy = strategy
        self.fitted_ = False
    
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """Ajusta el preprocesador a los datos."""
        # Implementación
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma los datos."""
        # Implementación
        pass
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ajusta y transforma en un paso."""
        return self.fit(X).transform(X)
```

### Notebooks Jupyter

#### Estructura Estándar
```markdown
# Título del Notebook

**Módulo:** X - Nombre del Módulo
**Práctica/Ejercicio:** N
**Duración Estimada:** X horas
**Objetivos:**
- Objetivo 1
- Objetivo 2

## 1. Setup e Imports
## 2. Carga de Datos
## 3. Análisis Exploratorio
## 4. Procesamiento
## 5. Modelado (si aplica)
## 6. Evaluación
## 7. Conclusiones

**Recursos:**
- Link 1
- Link 2
```

#### Celdas de Código
```python
# %% [markdown]
# ## 1. Setup e Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)

# Reproducibilidad
np.random.seed(42)

# %% [markdown]
# ## 2. Carga de Datos

# %%
df = pd.read_csv('../data/raw/dataset.csv')
print(f"Shape: {df.shape}")
df.head()
```

### Git y Control de Versiones

#### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data (large files)
data/raw/*.csv
data/raw/*.zip
data/processed/*.parquet
*.h5
*.hdf5

# Models
models/*.pkl
models/*.h5
models/*.pt
models/*.pth

# MLflow
mlruns/
mlartifacts/

# Environment
.env
.env.local
*.key
secrets/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# n8n
.n8n/

# Docker
docker-compose.override.yml
```

#### Commits
Formato:
```
<tipo>: <descripción breve>

<descripción detallada opcional>

<referencias opcionales>
```

Tipos:
- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Documentación
- `style`: Formato, sin cambios de código
- `refactor`: Refactorización
- `test`: Tests
- `chore`: Tareas de mantenimiento

Ejemplos:
```bash
git commit -m "feat: add baseline model for classification"
git commit -m "docs: update module 1 README with new exercises"
git commit -m "fix: correct data leakage in preprocessing pipeline"
```

## Documentación

### README.md de Módulos
Debe incluir:
1. **Objetivos de Aprendizaje** (claros y medibles)
2. **Duración Estimada**
3. **Prerrequisitos**
4. **Contenidos** (lista)
5. **Prácticas Guiadas** (con tiempo estimado)
6. **Ejercicios** (con criterios de corrección)
7. **Mini-Quiz**
8. **Recursos Adicionales**
9. **Checklist de Evaluación**
10. **Anti-Patrones a Evitar**
11. **Siguientes Pasos**

### README.md de Capstones
Debe incluir:
1. **Información General** (peso, duración, prerequisitos)
2. **Objetivos de Aprendizaje**
3. **Descripción del Proyecto**
4. **Opciones de Proyecto** (datasets sugeridos)
5. **Requisitos Técnicos** (detallados)
6. **Rúbrica de Evaluación** (tabla clara)
7. **Criterios de Aprobación**
8. **Extras (Puntos Bonus)**
9. **Timeline Sugerido**
10. **Checklist de Entrega**
11. **FAQs**
12. **Recursos de Ayuda**

### Docstrings
Usar formato Google:
```python
def train_model(X_train, y_train, model_type='logistic', **kwargs):
    """
    Entrena un modelo de clasificación.
    
    Args:
        X_train (pd.DataFrame): Features de entrenamiento
        y_train (pd.Series): Target de entrenamiento
        model_type (str): Tipo de modelo ('logistic', 'rf', 'xgboost')
        **kwargs: Hiperparámetros adicionales para el modelo
        
    Returns:
        object: Modelo entrenado
        
    Raises:
        ValueError: Si model_type no es válido
        
    Examples:
        >>> model = train_model(X_train, y_train, model_type='rf', n_estimators=100)
        >>> predictions = model.predict(X_test)
    """
    pass
```

## Estilo de Código

### Python (PEP 8)
- **Indentación:** 4 espacios
- **Líneas:** máximo 100 caracteres (preferible 79)
- **Imports:** agrupados y ordenados
- **Nombres:**
  - Variables/funciones: `snake_case`
  - Clases: `PascalCase`
  - Constantes: `UPPER_CASE`
  - Privados: `_leading_underscore`

### Linters y Formatters
```bash
# Black (formatter)
black src/ --line-length 100

# Flake8 (linter)
flake8 src/ --max-line-length 100 --ignore E203,W503

# mypy (type checker)
mypy src/ --ignore-missing-imports
```

## Testing

### Estructura de Tests
```
tests/
├── __init__.py
├── conftest.py              # Fixtures compartidos
├── test_preprocessing.py    # Tests de preprocesamiento
├── test_models.py           # Tests de modelos
└── test_evaluation.py       # Tests de evaluación
```

### Convenciones
```python
import pytest
from src.preprocessing import clean_data

def test_clean_data_removes_nulls():
    """Test que verifica eliminación de nulos."""
    # Arrange
    df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, 6]})
    
    # Act
    result = clean_data(df, strategy='drop')
    
    # Assert
    assert result.isnull().sum().sum() == 0
    assert len(result) == 2

def test_clean_data_imputes_mean():
    """Test que verifica imputación con media."""
    df = pd.DataFrame({'a': [1, None, 3]})
    result = clean_data(df, strategy='mean')
    assert result['a'].iloc[1] == 2.0
```

## Versionado de Modelos

### Nomenclatura
```
models/
├── baseline_v20250107.pkl
├── rf_optimized_v20250110.pkl
└── xgboost_final_v20250115.pkl
```

### Metadata
Guardar junto con el modelo:
```json
{
  "model_name": "xgboost_final",
  "version": "v20250115",
  "training_date": "2025-01-15T10:30:00",
  "features": ["age", "income", "score"],
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1
  },
  "metrics": {
    "accuracy": 0.85,
    "f1_score": 0.83,
    "roc_auc": 0.89
  },
  "dataset": {
    "train_size": 8000,
    "test_size": 2000,
    "stratified": true
  }
}
```

## Logging

### Configuración
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Uso
```python
logger.info("Starting training with 10000 samples")
logger.warning("Missing values detected in column 'age'")
logger.error("Failed to load model from disk")
```

## Buenas Prácticas Generales

1. **DRY (Don't Repeat Yourself):** Modulariza código repetido
2. **KISS (Keep It Simple, Stupid):** Prefiere soluciones simples
3. **YAGNI (You Aren't Gonna Need It):** No agregues funcionalidad innecesaria
4. **Separation of Concerns:** Separa lógica de negocio, datos, presentación
5. **Documentation:** Documenta decisiones, no código obvio
6. **Testing:** Tests primero para funciones críticas
7. **Version Control:** Commits pequeños y frecuentes
8. **Security:** Nunca commitear secretos

---

**Estas convenciones aseguran código limpio, mantenible y profesional en todo el curso.**
