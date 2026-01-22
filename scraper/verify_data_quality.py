#!/usr/bin/env python3
"""
Script de vérification rapide de la qualité des données scrapées.
Utilise les données du registre pour vérifier si elles sont utilisables.
"""

import json
import os

DATA_DIR = "../data"
REGISTRY_PATH = os.path.join(DATA_DIR, "scrape_registry.json")
DATA_PATH = os.path.join(DATA_DIR, "raw_texts.json")

def main():
    print("\n" + "="*80)
    print("🔍 VÉRIFICATION RAPIDE DES DONNÉES")
    print("="*80 + "\n")
    
    # 1. Vérifier le registre
    if not os.path.exists(REGISTRY_PATH):
        print("❌ Fichier scrape_registry.json introuvable")
        print(f"   Emplacement attendu: {os.path.abspath(REGISTRY_PATH)}")
        return
    
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    print(f"📋 Registre: {len(registry)} URLs enregistrées")
    
    # 2. Vérifier les données
    if not os.path.exists(DATA_PATH):
        print("❌ Fichier raw_texts.json introuvable")
        print(f"   Emplacement attendu: {os.path.abspath(DATA_PATH)}")
        return
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    print(f"📄 Documents: {len(documents)} documents chargés")
    
    if len(documents) == 0:
        print("\n⚠️  AUCUN DOCUMENT TROUVÉ!")
        print("Vous devez lancer le scraping:")
        print("   python3 scrape_all_diagnostic.py")
        return
    
    # 3. Analyse de qualité
    print("\n" + "="*80)
    print("📊 ANALYSE DE QUALITÉ")
    print("="*80 + "\n")
    
    total_chars = 0
    total_words = 0
    doc_types = {}
    short_docs = []
    long_docs = []
    
    for doc in documents:
        content = doc["content"]
        content_len = len(content)
        words = len(content.split())
        
        total_chars += content_len
        total_words += words
        
        # Comptage par type
        doc_type = doc["metadata"].get("doc_type", "Unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Détection de documents suspects
        if content_len < 500:
            short_docs.append((doc["metadata"]["source"], content_len))
        elif content_len > 50000:
            long_docs.append((doc["metadata"]["source"], content_len))
    
    # Affichage des statistiques
    print(f"📈 Volume total:")
    print(f"   - {total_chars:,} caractères")
    print(f"   - {total_words:,} mots")
    print(f"   - Moyenne: {total_chars // len(documents):,} caractères/document")
    print(f"   - Moyenne: {total_words // len(documents):,} mots/document")
    
    print(f"\n📑 Répartition par type:")
    for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(documents)) * 100
        print(f"   - {doc_type:20s}: {count:3d} ({percentage:5.1f}%)")
    
    # Alertes de qualité
    print("\n" + "="*80)
    print("⚠️  ALERTES DE QUALITÉ")
    print("="*80 + "\n")
    
    if len(short_docs) > 0:
        print(f"📏 {len(short_docs)} documents courts (< 500 caractères):")
        for url, length in short_docs[:5]:  # Limite à 5
            print(f"   - {length:4d} chars: {url[:70]}...")
        if len(short_docs) > 5:
            print(f"   ... et {len(short_docs) - 5} autres")
    else:
        print("✅ Aucun document anormalement court")
    
    if len(long_docs) > 0:
        print(f"\n📏 {len(long_docs)} documents très longs (> 50,000 caractères):")
        for url, length in long_docs[:5]:
            print(f"   - {length:,} chars: {url[:70]}...")
        if len(long_docs) > 5:
            print(f"   ... et {len(long_docs) - 5} autres")
    
    # Vérification des métadonnées
    print("\n" + "="*80)
    print("🏷️  MÉTADONNÉES")
    print("="*80 + "\n")
    
    docs_with_title = sum(1 for d in documents if d["metadata"].get("title") and d["metadata"]["title"] != "Sans titre")
    docs_with_hierarchy = sum(1 for d in documents if d["metadata"].get("hierarchy"))
    docs_with_date = sum(1 for d in documents if d["metadata"].get("last_updated"))
    
    print(f"Titres:      {docs_with_title:3d}/{len(documents)} ({docs_with_title/len(documents)*100:.1f}%)")
    print(f"Hiérarchie:  {docs_with_hierarchy:3d}/{len(documents)} ({docs_with_hierarchy/len(documents)*100:.1f}%)")
    print(f"Dates:       {docs_with_date:3d}/{len(documents)} ({docs_with_date/len(documents)*100:.1f}%)")
    
    # Échantillon
    print("\n" + "="*80)
    print("📝 ÉCHANTILLON (3 premiers documents)")
    print("="*80 + "\n")
    
    for i, doc in enumerate(documents[:3], 1):
        print(f"{i}. {doc['metadata']['title']}")
        print(f"   Type: {doc['metadata'].get('doc_type', 'N/A')}")
        print(f"   Longueur: {len(doc['content']):,} caractères")
        print(f"   Date: {doc['metadata'].get('last_updated', 'N/A')}")
        preview = doc['content'][:150].replace('\n', ' ')
        print(f"   Aperçu: {preview}...")
        print()
    
    # Verdict final
    print("="*80)
    print("🎯 VERDICT")
    print("="*80 + "\n")
    
    avg_length = total_chars // len(documents)
    
    if avg_length > 2000 and len(short_docs) < len(documents) * 0.1:
        print("✅ VOS DONNÉES SONT EXCELLENTES!")
        print("   - Longueur moyenne bonne (>2000 chars)")
        print("   - Peu de documents suspects (<10%)")
        print("   - Prêt pour intégration RAG!")
        print("\n💡 Action recommandée: Utilisez ces données directement")
        print("   Pas besoin de re-scraper!")
    elif avg_length > 1000:
        print("✅ VOS DONNÉES SONT UTILISABLES")
        print("   - Longueur moyenne acceptable (>1000 chars)")
        print("   - Quelques ajustements possibles")
        print("\n💡 Action recommandée: Utilisez ces données")
        print("   Corrigez les scrapers pour les futures mises à jour")
    else:
        print("⚠️  QUALITÉ DES DONNÉES DOUTEUSE")
        print("   - Longueur moyenne faible (<1000 chars)")
        print("   - Beaucoup de documents courts")
        print("\n💡 Action recommandée:")
        print("   1. Corrigez le scraper (voir GUIDE_CORRECTION.md)")
        print("   2. Re-scrapez: python3 scrape_all_diagnostic.py --force")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()