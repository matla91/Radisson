import faiss
import json
import os
import numpy as np
import logging
import torch
import time

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_index_cpu(data_dir="../data", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    CPU-optimized version for building the FAISS index.
    """
    chunks_path = os.path.join(data_dir, "chunks.json")
    index_path = os.path.join(data_dir, "faiss.index")

    # 1. CPU optimization: use all available cores
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    logger.info(f"⚙️  CPU optimization: {num_threads} cores enabled for computation.")

    # 2. Loading and validation
    if not os.path.exists(chunks_path):
        logger.error(f"File {chunks_path} not found!")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Validation: keep only chunks containing valid text
    texts = [c["text"] for c in chunks if c.get("text") and len(c["text"]) > 0]
    
    if len(texts) != len(chunks):
        logger.warning(f"⚠️ {len(chunks) - len(texts)} empty or invalid chunks were ignored.")

    # 3. Encoding
    from sentence_transformers import SentenceTransformer
    logger.info(f"🧠 Loading model {model_name}...")
    model = SentenceTransformer(model_name, device="cpu")
    
    logger.info("⚡ Generating embeddings on CPU (this may take a few minutes)...")
    start_time = time.time()
    
    # batch_size=32 is often the best balance point on CPU
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Immediately free text memory (optional but recommended on laptops)
    del texts
    
    # 4. Index construction (compatible with your current retriever)
    logger.info("🏗️  Building FAISS index (L2)...")
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    
    duration = time.time() - start_time

    # 5. Saving
    faiss.write_index(index, index_path)
    
    logger.info(f"✅ Indexing completed in {duration:.2f}s")
    logger.info(f"💾 Index saved: {index_path} ({index.ntotal} vectors)")

if __name__ == "__main__":

    build_index_cpu()
