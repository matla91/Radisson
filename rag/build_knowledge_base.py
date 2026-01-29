# rag/build_knowledge_base.py
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

SCRAPE_SCRIPT = ROOT_DIR / "scraper" / "scrape_all.py"
CHUNK_SCRIPT = ROOT_DIR / "processing" / "chunk_documents.py"
FAISS_SCRIPT = ROOT_DIR / "embeddings" / "build_faiss_index.py"


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_step(label: str, script_path: Path) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"Script introuvable: {script_path}")

    banner(label)
    print(f"➡️  Running: {script_path.relative_to(ROOT_DIR)}")

    # On lance avec le même interpréteur Python que celui qui exécute ce script
    # et on se place à la racine du projet pour éviter les soucis de chemins relatifs.
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT_DIR),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Échec à l'étape: {label}\n"
            f"Commande: {sys.executable} {script_path}\n"
            f"Code retour: {result.returncode}"
        )


def main() -> None:
    start = time.time()

    banner("RADISSON — BUILD KNOWLEDGE BASE (offline pipeline)")

    DATA_DIR.mkdir(exist_ok=True)

    # 1) Scrape
    run_step("STEP 1 — Scraping documents", SCRAPE_SCRIPT)

    # 2) Processing + chunking
    run_step("STEP 2 — Processing / Chunking", CHUNK_SCRIPT)

    # 3) Embeddings + FAISS
    run_step("STEP 3 — Embeddings / FAISS indexing", FAISS_SCRIPT)

    # Check final
    banner("FINAL CHECK")
    required = [DATA_DIR / "chunks.json", DATA_DIR / "faiss.index"]
    missing = [p for p in required if not p.exists()]

    if missing:
        print("❌ Base de connaissances non générée correctement.")
        for p in missing:
            print(f"Missing: {p}")
        print("\n👉 Vérifie que tes scripts écrivent bien dans le dossier ./data")
        sys.exit(1)

    elapsed = time.time() - start
    banner("SUCCESS ✅")
    print("Knowledge base générée avec succès.")
    print(f"Temps total: {elapsed:.1f}s")
    print("\nTu peux maintenant lancer l'app :")
    print("  docker compose up --build")


if __name__ == "__main__":
    main()
