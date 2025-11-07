# Módulo 4: NLP Moderno y Embeddings

## 📚 Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- **Preprocesar** texto para NLP de forma efectiva (Bloom 3)
- **Generar** y usar embeddings de palabras y documentos (Bloom 3)
- **Implementar** clasificación de texto con transformers (Bloom 6)
- **Realizar** búsqueda semántica con vectores (Bloom 3)
- **Evaluar** modelos de NLP con métricas apropiadas (Bloom 5)

## ⏱️ Duración Estimada
**12-15 horas** (1.5 semanas)

## 🗺️ Mapa Conceptual

```
                        NLP MODERNO Y EMBEDDINGS
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
  PREPROCESAMIENTO          REPRESENTACIONES            APLICACIONES
        │                         │                         │
  ┌─────┴─────┐           ┌───────┴───────┐         ┌──────┴──────┐
  │           │           │               │         │             │
Tokeniz   Limpieza    Clásicas      Modernas    Clasificación  Búsqueda
  │           │           │               │         │          Semántica
  • NLTK      • Stop      • BoW           • Word2Vec  • Text       │
  • spaCy       Words     • TF-IDF        • GloVe     Class      • FAISS
  • Hugging   • Lemma     • N-grams       • BERT      • Sent     • Chroma
    Face        tize                      • Sentence  Analysis   • Similarity
                                            BERT
```

## 📖 Contenidos Detallados

### 1. Preprocesamiento de Texto (2 horas)

#### Pipeline de Preprocesamiento

```
Texto Raw → Lowercasing → Remove HTML → Tokenización → Remove Stop Words
           → Lemmatization/Stemming → Remove Punctuation → Normalización
```

**Ejemplo Completo:**
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Pipeline completo de preprocesamiento"""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. Remove mentions y hashtags (opcional)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # 5. Tokenización
    tokens = word_tokenize(text)
    
    # 6. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # 7. Remove punctuation
    tokens = [t for t in tokens if t.isalnum()]
    
    # 8. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

# Ejemplo de uso
text = "I'm loving this <b>amazing</b> product! Check it out: http://example.com #awesome"
cleaned = preprocess_text(text)
print(cleaned)
# Output: "love amazing product check awesome"
```

**Alternativa con spaCy (más rápido y robusto):**
```python
import spacy

nlp = spacy.load('en_core_web_sm')

def preprocess_spacy(text):
    doc = nlp(text.lower())
    
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop 
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
    ]
    
    return ' '.join(tokens)
```

### 2. Representaciones Clásicas (2 horas)

#### Bag of Words (BoW)

**Concepto:** Representa cada documento como vector de frecuencias de palabras.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love programming"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("\nBoW Matrix:\n", X.toarray())
```

**Output:**
```
Vocabulary: ['amazing' 'is' 'learning' 'love' 'machine' 'programming']

BoW Matrix:
[[0 0 1 1 1 0]
 [1 1 1 0 1 0]
 [0 0 0 1 0 1]]
```

#### TF-IDF (Term Frequency - Inverse Document Frequency)

**Fórmula:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)

donde:
TF(t, d) = frecuencia del término t en documento d
IDF(t) = log(N / df(t))
N = número total de documentos
df(t) = número de documentos que contienen t
```

**Implementación:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,      # Top 5000 términos
    ngram_range=(1, 2),     # Unigramas y bigramas
    min_df=2,               # Mínimo en 2 documentos
    max_df=0.8              # Máximo en 80% documentos
)

X_tfidf = tfidf.fit_transform(corpus)

# Ver términos más importantes de un documento
feature_names = tfidf.get_feature_names_out()
doc_index = 0
tfidf_scores = X_tfidf[doc_index].toarray()[0]
top_indices = tfidf_scores.argsort()[-10:][::-1]

print("Top 10 términos del documento 0:")
for idx in top_indices:
    print(f"{feature_names[idx]}: {tfidf_scores[idx]:.3f}")
```

**Clasificación con TF-IDF:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Datos de ejemplo
texts = [...]  # Lista de textos
labels = [...]  # Lista de etiquetas

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Clasificador
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# Evaluación
y_pred = clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
```

### 3. Word Embeddings (3 horas)

#### Word2Vec: Conceptos

**Arquitecturas:**
1. **Skip-gram:** Predice contexto dado una palabra
2. **CBOW:** Predice palabra dado el contexto

**Diagrama Skip-gram:**
```
    Palabra: "learning"
         ↓
    [Embedding Layer]
         ↓
    Vector denso (300D)
         ↓
    [Softmax Layer]
         ↓
    Predice contexto: ["machine", "is", "fun"]
```

**Uso con Gensim:**
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Preparar datos (lista de frases tokenizadas)
sentences = [
    word_tokenize(text.lower())
    for text in corpus
]

# Entrenar Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Dimensión de embeddings
    window=5,             # Ventana de contexto
    min_count=2,          # Frecuencia mínima
    workers=4,
    sg=1                  # 1=Skip-gram, 0=CBOW
)

# Usar embeddings
vector = model.wv['machine']  # Vector de 100D

# Similitud
similar = model.wv.most_similar('machine', topn=5)
print("Palabras similares a 'machine':", similar)

# Operaciones con vectores
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man']
)
print("king - man + woman =", result[0])  # Debería ser "queen"
```

**Embeddings Pre-entrenados (GloVe):**
```python
# Descargar GloVe: https://nlp.stanford.edu/projects/glove/

import numpy as np

def load_glove(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

glove = load_glove('glove.6B.100d.txt')
print(f"Loaded {len(glove)} word vectors")

# Usar embedding
vector = glove['machine']  # Vector de 100D
```

### 4. Sentence Embeddings (3 horas)

#### Sentence-BERT

**Concepto:** Modifica BERT para generar embeddings de frases comparables.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo pre-entrenado
model = SentenceTransformer('all-MiniLM-L6-v2')  # Ligero y rápido

# Generar embeddings
sentences = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is a programming language",
    "Artificial intelligence mimics human intelligence"
]

embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")  # (4, 384)

# Calcular similitudes
similarities = cosine_similarity(embeddings)

print("\nSimilaridades:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:
            print(f"'{sent1[:30]}...' vs '{sent2[:30]}...': {similarities[i,j]:.3f}")
```

**Aplicación: Clasificación con Sentence Embeddings:**
```python
from sklearn.ensemble import RandomForestClassifier

# Generar embeddings de train/test
X_train_emb = model.encode(X_train)
X_test_emb = model.encode(X_test)

# Clasificador
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_emb, y_train)

# Evaluar
from sklearn.metrics import classification_report
y_pred = clf.predict(X_test_emb)
print(classification_report(y_test, y_pred))
```

### 5. Transformers con Hugging Face (4 horas)

#### Arquitectura Transformer

**Diagrama Simplificado:**
```
Input Text
    ↓
[Tokenizer] → Input IDs + Attention Mask
    ↓
[Embedding Layer]
    ↓
[Multi-Head Attention] × N layers
    ↓
[Feed Forward Network]
    ↓
[Pooling/Classification Head]
    ↓
Output (logits/embeddings)
```

**Pipeline de Hugging Face (Forma Más Fácil):**
```python
from transformers import pipeline

# Clasificación de sentimiento
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Zero-shot classification
classifier_zs = pipeline("zero-shot-classification")
result = classifier_zs(
    "This is a course about machine learning",
    candidate_labels=["education", "politics", "sports"]
)
print(result)
# {'sequence': '...', 'labels': ['education', 'politics', 'sports'],
#  'scores': [0.98, 0.01, 0.01]}
```

**Fine-tuning de BERT para Clasificación:**
```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
import numpy as np

# 1. Preparar datos
texts = [...]  # Lista de textos
labels = [...]  # Lista de labels (0, 1, 2, ...)

dataset = Dataset.from_dict({
    'text': texts,
    'label': labels
})

# Train/test split
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 2. Tokenizer
model_name = 'distilbert-base-uncased'  # Más ligero que BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# 3. Modelo
num_labels = len(set(labels))
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 5. Trainer
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

# 6. Entrenar
trainer.train()

# 7. Evaluar
results = trainer.evaluate()
print(results)

# 8. Predecir
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    pred_class = probs.argmax().item()
    confidence = probs[0, pred_class].item()
    return pred_class, confidence

text = "This product is absolutely terrible!"
pred, conf = predict(text)
print(f"Prediction: {pred}, Confidence: {conf:.3f}")
```

### 6. Búsqueda Semántica con FAISS (2 horas)

**FAISS:** Facebook AI Similarity Search - librería para búsqueda rápida de vecinos más cercanos.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Generar embeddings de documentos
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Machine learning is a branch of artificial intelligence",
    "Python is a popular programming language for data science",
    "Deep learning uses neural networks with many layers",
    # ... más documentos
]

doc_embeddings = model.encode(documents)
print(f"Embeddings shape: {doc_embeddings.shape}")  # (n_docs, embedding_dim)

# 2. Crear índice FAISS
dimension = doc_embeddings.shape[1]  # 384 para all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Agregar embeddings al índice
index.add(doc_embeddings.astype('float32'))
print(f"Total docs in index: {index.ntotal}")

# 3. Búsqueda
query = "What is deep learning?"
query_embedding = model.encode([query]).astype('float32')

# Buscar top-k documentos más similares
k = 3
distances, indices = index.search(query_embedding, k)

print(f"\nQuery: {query}")
print(f"\nTop {k} resultados:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. [Score: {1/(1+dist):.3f}] {documents[idx]}")
```

**Búsqueda Semántica con ChromaDB (alternativa más amigable):**
```python
import chromadb
from chromadb.utils import embedding_functions

# Cliente Chroma
client = chromadb.Client()

# Collection con embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="documents",
    embedding_function=sentence_transformer_ef
)

# Agregar documentos
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Consultar
results = collection.query(
    query_texts=["What is deep learning?"],
    n_results=3
)

print("Top 3 resultados:")
for doc, dist in zip(results['documents'][0], results['distances'][0]):
    print(f"[Score: {1-dist:.3f}] {doc}")
```

## 🎯 Prácticas Guiadas

1. **Clasificación de Sentiment (TF-IDF)** (2h) - IMDB Reviews
2. **Word2Vec y Similitud** (2h) - Corpus personalizado
3. **Sentence-BERT Classification** (2.5h) - AG News
4. **Fine-tuning BERT** (3h) - Text Classification
5. **Búsqueda Semántica** (2h) - FAISS/Chroma

## ✏️ Ejercicios

1. **Clasificador de Spam:** TF-IDF + embeddings + transformer
2. **Sistema de Búsqueda:** Implementar con FAISS
3. **Análisis de Similitud:** Comparar métodos de embedding

## 📚 Recursos Externos

### 📹 Videos Esenciales
- [3Blue1Brown: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) (20 min)
- [StatQuest: Word2Vec](https://www.youtube.com/watch?v=viZrOnJclY0) (20 min)
- [Hugging Face: Transformers Course](https://www.youtube.com/playlist?list=PLo2EIpI_JMQtNtKNFFSMNIZwspj8H7-sQ)

### 📖 Lecturas Obligatorias
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Paper original de Transformers
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)

### 🛠️ Documentación
- [Hugging Face Course](https://huggingface.co/learn/nlp-course/)
- [Sentence-Transformers Docs](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [NLTK Book](https://www.nltk.org/book/)

### 🎮 Herramientas Interactivas
- [Hugging Face Model Hub](https://huggingface.co/models) - Explorar modelos
- [Transformer Explainer](http://jalammar.github.io/illustrated-transformer/) - Visualización interactiva

## ✅ Checklist de Evaluación

- [ ] Preprocesar texto correctamente
- [ ] Implementar TF-IDF baseline
- [ ] Usar embeddings pre-entrenados
- [ ] Fine-tuning de transformer
- [ ] Búsqueda semántica funcional
- [ ] Evaluar con métricas de NLP

## ➡️ Siguientes Pasos

**Módulo 5:** Deep Learning con PyTorch
