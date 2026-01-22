import json
import os
import re
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Paramètres optimisés pour des textes administratifs français
CHUNK_SIZE = 1000 
OVERLAP = 200  # Augmenté de 150 à 200 pour meilleure continuité

def extract_chapter(url: str) -> str:
    """Extrait le numéro de chapitre depuis l'URL."""
    match = re.search(r'chapitre-(\d+)', url)
    return f"Chapitre {match.group(1)}" if match else "Non classé"

def clean_chunk_text(text: str) -> str:
    """Nettoie le texte d'un chunk."""
    # Supprime les espaces multiples
    text = re.sub(r' {2,}', ' ', text)
    # Supprime les sauts de ligne multiples (max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def validate_chunk(chunk_text: str, min_length: int = 50) -> bool:
    """Vérifie qu'un chunk est valide (pas trop court, pas vide)."""
    return len(chunk_text.strip()) >= min_length

def main():
    print("\n" + "="*80)
    print("🔄 CHUNKING DES DOCUMENTS UQAC")
    print("="*80 + "\n")
    
    # 1. Chargement du JSON brut
    input_path = "../data/raw_texts.json"
    if not os.path.exists(input_path):
        print(f"❌ Erreur: {input_path} introuvable.")
        print(f"   Emplacement attendu: {os.path.abspath(input_path)}")
        return

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_documents = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Erreur de lecture JSON: {e}")
        return
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return

    print(f"✅ {len(raw_documents)} documents chargés")
    
    # 2. Conversion en objets "Document" LangChain avec TOUTES les métadonnées
    docs = []
    for doc in raw_documents:
        # AMÉLIORATION 1: Garder TOUTES les métadonnées, pas juste l'URL
        metadata = {
            "source": doc["metadata"]["source"],
            "title": doc["metadata"].get("title", "Sans titre"),
            "doc_type": doc["metadata"].get("doc_type", "Document"),
            "hierarchy": doc["metadata"].get("hierarchy", ""),
            "last_updated": doc["metadata"].get("last_updated", ""),
            "chapitre": extract_chapter(doc["metadata"]["source"]),  # NOUVEAU
        }
        
        # Ajoute page_count pour les PDFs
        if "page_count" in doc["metadata"]:
            metadata["page_count"] = doc["metadata"]["page_count"]
        
        docs.append(Document(
            page_content=doc["content"],
            metadata=metadata
        ))

    print(f"📄 Documents convertis avec métadonnées enrichies")

    # 3. Découpage intelligent (Recursive)
    # AMÉLIORATION 2: Séparateurs optimisés pour le français administratif
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        length_function=len,
        separators=[
            "\n\n\n",  # Séparations de sections majeures
            "\n\n",    # Paragraphes
            "\n",      # Lignes
            ". ",      # Phrases (avec espace après le point)
            "! ",      # Phrases exclamatives
            "? ",      # Questions
            ";",       # Points-virgules
            ",",       # Virgules (dernier recours)
            " ",       # Espaces
            ""         # Caractères individuels (vraiment dernier recours)
        ],
        is_separator_regex=False
    )

    print(f"🔪 Découpage en cours (taille: {CHUNK_SIZE}, overlap: {OVERLAP})...")
    all_chunks = text_splitter.split_documents(docs)

    # AMÉLIORATION 3: Enrichissement des métadonnées des chunks
    enriched_chunks = []
    chunk_stats = defaultdict(int)
    chunks_by_doc = defaultdict(int)
    
    for i, chunk in enumerate(all_chunks):
        # Nettoyage du texte
        cleaned_text = clean_chunk_text(chunk.page_content)
        
        # Validation
        if not validate_chunk(cleaned_text):
            print(f"⚠️  Chunk {i} trop court ({len(cleaned_text)} chars), ignoré")
            continue
        
        # Enrichissement des métadonnées
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(cleaned_text)
        
        # Statistiques
        chunk_stats[chunk.metadata["doc_type"]] += 1
        chunks_by_doc[chunk.metadata["source"]] += 1
        
        enriched_chunks.append({
            "id": f"chunk_{i}",  # ID unique pour ChromaDB
            "text": cleaned_text,
            "metadata": chunk.metadata
        })

    # 4. Statistiques détaillées
    print("\n" + "="*80)
    print("📊 STATISTIQUES DE CHUNKING")
    print("="*80 + "\n")
    
    print(f"📦 Total de chunks: {len(enriched_chunks)}")
    print(f"📄 Documents sources: {len(raw_documents)}")
    print(f"📈 Moyenne: {len(enriched_chunks) / len(raw_documents):.1f} chunks/document")
    
    # Distribution des longueurs
    lengths = [chunk["metadata"]["chunk_length"] for chunk in enriched_chunks]
    print(f"\n📏 Distribution des longueurs:")
    print(f"   - Min:     {min(lengths):,} caractères")
    print(f"   - Max:     {max(lengths):,} caractères")
    print(f"   - Moyenne: {sum(lengths) // len(lengths):,} caractères")
    print(f"   - Médiane: {sorted(lengths)[len(lengths)//2]:,} caractères")
    
    # Distribution par type de document
    print(f"\n📋 Chunks par type de document:")
    for doc_type, count in sorted(chunk_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(enriched_chunks)) * 100
        print(f"   - {doc_type:20s}: {count:4d} ({percentage:5.1f}%)")
    
    # Top documents avec le plus de chunks
    print(f"\n🔝 Top 5 documents avec le plus de chunks:")
    top_docs = sorted(chunks_by_doc.items(), key=lambda x: x[1], reverse=True)[:5]
    for url, count in top_docs:
        # Raccourcit l'URL pour l'affichage
        short_url = url.split('/')[-2] if len(url.split('/')) > 2 else url
        print(f"   - {count:3d} chunks: {short_url[:60]}...")
    
    # 5. Sauvegarde
    os.makedirs("../data", exist_ok=True)
    
    # Sauvegarde des chunks complets (pour ChromaDB)
    output_path = "../data/chunks.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_chunks, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Chunks sauvegardés: {output_path}")
    except Exception as e:
        print(f"\n❌ Erreur sauvegarde: {e}")
        return
    
    # AMÉLIORATION 4: Sauvegarde des métadonnées séparément (pour inspection)
    metadata_path = "../data/chunks_metadata.json"
    metadata_summary = {
        "total_chunks": len(enriched_chunks),
        "total_documents": len(raw_documents),
        "avg_chunks_per_doc": len(enriched_chunks) / len(raw_documents),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": OVERLAP,
        "stats_by_type": dict(chunk_stats),
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths) // len(lengths),
            "median": sorted(lengths)[len(lengths)//2]
        }
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Métadonnées sauvegardées: {metadata_path}")
    
    # 6. Échantillon pour vérification
    print("\n" + "="*80)
    print("📝 ÉCHANTILLON (3 premiers chunks)")
    print("="*80 + "\n")
    
    for i, chunk in enumerate(enriched_chunks[:3], 1):
        print(f"{i}. ID: {chunk['id']}")
        print(f"   Source: {chunk['metadata']['source'][-60:]}")
        print(f"   Type: {chunk['metadata']['doc_type']}")
        print(f"   Chapitre: {chunk['metadata']['chapitre']}")
        print(f"   Longueur: {chunk['metadata']['chunk_length']} caractères")
        preview = chunk['text'][:150].replace('\n', ' ')
        print(f"   Aperçu: {preview}...")
        print()
    
    print("="*80)
    print("✅ CHUNKING TERMINÉ AVEC SUCCÈS")
    print("="*80 + "\n")
    
    print("💡 Prochaines étapes:")
    print("   1. Vérifier chunks.json")
    print("   2. Générer les embeddings")
    print("   3. Charger dans ChromaDB")

if __name__ == "__main__":
    main()