FROM python:3.10-slim

WORKDIR /app

# Dépendances système nécessaires à FAISS
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code uniquement
COPY rag/ ./rag/
COPY main.py .

# Pré-téléchargement des modèles
RUN python - <<EOF
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

EXPOSE 8501

CMD ["python", "main.py"]
