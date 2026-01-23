# Utilisation d'une image Python légère
FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système pour FAISS et le scraping
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie des besoins et installation
COPY rag/requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copie du code source et des données indexées
COPY rag/ ./rag/
COPY data/ ./data/

# Pré-téléchargement des modèles Sentence-Transformers pour éviter les téléchargements au runtime
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Port pour l'interface (Streamlit par exemple) ou l'API (FastAPI)
EXPOSE 8501

# Lancement de l'application
CMD ["python", "rag/rag_engine.py"]
