import faiss
import json
import os
import numpy as np
import logging
import torch
import time

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_index_cpu(data_dir="../data", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Version optimisée pour processeur (CPU) de la création de l'index FAISS.
    """
    chunks_path = os.path.join(data_dir, "chunks.json")
    index_path = os.path.join(data_dir, "faiss.index")

    # 1. Optimisation CPU : Utiliser tous les cœurs disponibles
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    logger.info(f"⚙️  Optimisation CPU : {num_threads} cœurs activés pour le calcul.")

    # 2. Chargement et validation
    if not os.path.exists(chunks_path):
        logger.error(f"Fichier {chunks_path} introuvable !")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Validation : on ne garde que les chunks qui ont du texte valide
    texts = [c["text"] for c in chunks if c.get("text") and len(c["text"]) > 0]
    
    if len(texts) != len(chunks):
        logger.warning(f"⚠️ {len(chunks) - len(texts)} chunks vides ou invalides ont été ignorés.")

    # 3. Encodage
    from sentence_transformers import SentenceTransformer
    logger.info(f"🧠 Chargement du modèle {model_name}...")
    model = SentenceTransformer(model_name, device="cpu")
    
    logger.info("⚡ Génération des vecteurs sur CPU (cela peut prendre quelques minutes)...")
    start_time = time.time()
    
    # batch_size=32 est souvent le point d'équilibre idéal sur CPU
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Libération immédiate de la mémoire des textes (optionnel mais recommandé sur portable)
    del texts
    
    # 4. Construction de l'index (compatible avec ton retriever actuel)
    logger.info("🏗️  Construction de l'index FAISS (L2)...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype("float32"))
    
    duration = time.time() - start_time

    # 5. Sauvegarde
    faiss.write_index(index, index_path)
    
    logger.info(f"✅ Indexation terminée en {duration:.2f}s")
    logger.info(f"💾 Index sauvegardé : {index_path} ({index.ntotal} vecteurs)")

if __name__ == "__main__":
    build_index_cpu()