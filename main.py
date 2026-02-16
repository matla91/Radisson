import os
import subprocess
import sys

DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
FAISS_INDEX = os.path.join(DATA_DIR, "faiss.index")


def check_requirements():
    print("🔎 Vérification de l'environnement...")

    try:
        import sentence_transformers
        import faiss
        import streamlit
    except ImportError:
        print("❌ Dépendances manquantes.")
        print("👉 Lancez : pip install -r requirements.txt")
        sys.exit(1)

    print("✅ Environnement OK")


def build_data_if_needed():
    if os.path.exists(CHUNKS_FILE) and os.path.exists(FAISS_INDEX):
        print("✅ Données déjà construites.")
        return

    print("⚙️ Construction des données (premier lancement)...")

    subprocess.run([sys.executable, "scripts/scrape.py"], check=True)
    subprocess.run([sys.executable, "scripts/build_index.py"], check=True)

    print("✅ Index construit.")


def launch_app():
    print("🚀 Lancement du chatbot Radisson...")
    subprocess.run(
        ["streamlit", "run", "app/app.py"]
    )


if __name__ == "__main__":
    print("\n=== Radisson RAG Launcher ===\n")

    check_requirements()
    build_data_if_needed()
    launch_app()
