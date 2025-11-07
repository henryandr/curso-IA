# Índice de Recursos del Curso

## Datasets Públicos Recomendados

### Clasificación y Regresión (Módulos 1-3, Capstone 1)

**Kaggle:**
- [Titanic](https://www.kaggle.com/c/titanic): Clasificación binaria (sobrevivencia)
- [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques): Regresión de precios
- [Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing): Clasificación (suscripción)
- [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): Clasificación desbalanceada
- [Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset): Predicción de rotación

**UCI ML Repository:**
- [Adult Income](https://archive.ics.uci.edu/ml/datasets/adult): Clasificación binaria
- [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality): Clasificación multi-clase
- [Student Performance](https://archive.ics.uci.edu/ml/datasets/student+performance): Regresión

### NLP (Módulos 4, 6, Capstone 2)

**Hugging Face Datasets:**
- [IMDB Reviews](https://huggingface.co/datasets/imdb): Sentiment analysis
- [AG News](https://huggingface.co/datasets/ag_news): Clasificación de noticias
- [SQuAD 2.0](https://huggingface.co/datasets/squad_v2): Question Answering
- [Amazon Reviews](https://huggingface.co/datasets/amazon_polarity): Reviews multi-dominio

**Otros:**
- [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/): Clasificación de textos
- [Reuters-21578](https://www.cs.umb.edu/~smimarog/textmining/datasets/): Noticias categorizadas
- [Yelp Reviews](https://www.yelp.com/dataset): Reviews de negocios

### Documentos para RAG (Módulo 6-7, Capstone 3)

- **Documentación técnica:** Python docs, scikit-learn docs
- **Wikipedia:** Artículos sobre IA/ML
- **ArXiv papers:** Papers de investigación
- **FAQs y manuales de usuario:** Crear propios o usar públicos

## Scripts de Descarga

Ver `recursos/datasets/download_datasets.py` para automatizar descargas desde Kaggle.

## Papers de Referencia

### Fundamentos
- "A Few Useful Things to Know About Machine Learning" (Domingos, 2012)

### NLP y Transformers
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- ["BERT: Pre-training of Deep Bidirectional Transformers"](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
- ["Sentence-BERT"](https://arxiv.org/abs/1908.10084) (Reimers & Gurevych, 2019)

### RAG y LLMs
- ["Retrieval-Augmented Generation"](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- ["Dense Passage Retrieval"](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020)

### Agentes
- ["ReAct: Synergizing Reasoning and Acting"](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- ["Toolformer"](https://arxiv.org/abs/2302.04761) (Schick et al., 2023)

## Libros (Referencias)

### Generales
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (Aurélien Géron)
- "Python for Data Analysis" (Wes McKinney)
- "Introduction to Statistical Learning" (James et al.) - [Gratis](https://www.statlearning.com/)

### Deep Learning y NLP
- "Deep Learning with PyTorch" (Stevens et al.)
- "Natural Language Processing with Transformers" (Tunstall et al.)
- "Speech and Language Processing" (Jurafsky & Martin) - [Gratis](https://web.stanford.edu/~jurafsky/slp3/)

### MLOps
- "Designing Machine Learning Systems" (Chip Huyen)
- "Building Machine Learning Powered Applications" (Emmanuel Ameisen)

## Cursos Online Complementarios

### Gratuitos
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Stanford CS229 - Machine Learning](https://cs229.stanford.edu/)

### De Pago
- Coursera: "Machine Learning Specialization" (Andrew Ng)
- DeepLearning.AI: "Deep Learning Specialization"
- Udacity: "Machine Learning Engineer Nanodegree"

## Documentación Oficial

### Librerías Core
- [NumPy](https://numpy.org/doc/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [PyTorch](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Herramientas
- [n8n Documentation](https://docs.n8n.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [FAISS](https://faiss.ai/)
- [ChromaDB](https://docs.trychroma.com/)

## Comunidades y Foros

### Reddit
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [r/learnmachinelearning](https://www.reddit.com/r/learnmachinelearning/)
- [r/LanguageTechnology](https://www.reddit.com/r/LanguageTechnology/)

### Discord/Slack
- [Hugging Face Discord](https://hf.co/join/discord)
- [n8n Community](https://community.n8n.io/)
- [Fast.ai Forums](https://forums.fast.ai/)

### Twitter/X
- Sigue: @karpathy, @ylecun, @goodfellow_ian, @HuggingFace, @weights_biases

## Blogs y Newsletters

- [Distill.pub](https://distill.pub/) - Explicaciones visuales de ML
- [Jay Alammar's Blog](https://jalammar.github.io/) - Visualizaciones de transformers
- [Sebastian Ruder's Blog](https://www.ruder.io/) - NLP
- [The Batch by DeepLearning.AI](https://www.deeplearning.ai/the-batch/)
- [Import AI](https://jack-clark.net/) - Newsletter semanal

## Herramientas Útiles

### Visualización y Debugging
- [Tensor Board](https://www.tensorflow.org/tensorboard)
- [Weights & Biases](https://wandb.ai/)
- [Netron](https://netron.app/) - Visualizador de modelos

### Anotación de Datos
- [Label Studio](https://labelstud.io/)
- [Prodigy](https://prodi.gy/)
- [Doccano](https://doccano.github.io/doccano/)

### Despliegue
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Streamlit](https://streamlit.io/)
- [Gradio](https://gradio.app/)

## Datasets para Práctica Adicional

### Computer Vision (Opcional)
- MNIST, Fashion-MNIST, CIFAR-10 (clásicos)
- ImageNet (grande, para transfer learning)

### Series Temporales
- [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

### Recomendación
- [MovieLens](https://grouplens.org/datasets/movielens/)
- [Book Crossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

## Licencias y Uso

⚠️ **Importante:** Verifica siempre la licencia de cada dataset antes de usarlo:
- **Educación:** La mayoría permite uso educativo
- **Comercial:** Algunos requieren licencia o atribución
- **Redistribución:** No siempre está permitida

## Cómo Contribuir

¿Encontraste un recurso útil? Abre un Pull Request añadiéndolo a esta lista con:
- Nombre y link
- Breve descripción
- Relevancia para qué módulo(s)
- Licencia/accesibilidad

---

**Última actualización:** 2025-11-07
