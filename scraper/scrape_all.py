import json
import os
import hashlib
import sys
import traceback
import logging
from datetime import datetime
from collections import defaultdict
from scrape_links import get_all_links, get_robust_session
from scrape_html import extract_html_content, HTMLExtractionError
from scrape_pdf import extract_pdf_text, PDFExtractionError

# Configuration
DATA_DIR = "../data"
REGISTRY_PATH = os.path.join(DATA_DIR, "scrape_registry.json")
DATA_PATH = os.path.join(DATA_DIR, "raw_texts.json")
LOG_PATH = os.path.join(DATA_DIR, "scrape_log.txt")
DEBUG_LOG_PATH = os.path.join(DATA_DIR, "scrape_debug.log")
ERROR_DETAILS_PATH = os.path.join(DATA_DIR, "error_details.json")
CHECKPOINT_INTERVAL = 10

# Configuration du logging
os.makedirs(DATA_DIR, exist_ok=True)

# Logger principal (INFO et plus)
main_logger = logging.getLogger()
main_logger.setLevel(logging.DEBUG)

# Handler pour le fichier de debug (TOUT)
debug_handler = logging.FileHandler(DEBUG_LOG_PATH, encoding='utf-8', mode='w')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
main_logger.addHandler(debug_handler)

# Handler pour la console (INFO et plus)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
main_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

def get_content_hash(text: str) -> str:
    """Calcule un hash MD5 du contenu"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_registry() -> dict:
    """Charge le registre des hashes de contenu"""
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ Erreur chargement registre: {e}")
            return {}
    return {}

def load_documents() -> list:
    """Charge les documents existants"""
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ Erreur chargement documents: {e}")
            return []
    return []

def save_checkpoint(documents: list, registry: dict):
    """Sauvegarde un checkpoint"""
    try:
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde checkpoint: {e}")
        return False

def save_error_details(error_details: dict):
    """Sauvegarde les détails des erreurs dans un fichier JSON"""
    try:
        with open(ERROR_DETAILS_PATH, "w", encoding="utf-8") as f:
            json.dump(error_details, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Détails des erreurs sauvegardés: {ERROR_DETAILS_PATH}")
    except Exception as e:
        logger.error(f"❌ Impossible de sauvegarder les détails d'erreur: {e}")

def categorize_error(exception) -> str:
    """Catégorise une erreur pour les statistiques"""
    if isinstance(exception, HTMLExtractionError):
        reason = exception.reason
        if "Content-Type invalide" in reason:
            return "Content-Type invalide (HTML attendu)"
        elif "Conteneur principal" in reason or "conteneur" in reason.lower():
            return "Structure HTML - Conteneur introuvable"
        elif "trop court" in reason.lower():
            return "Contenu trop court"
        elif "fantôme" in reason.lower():
            return "Contenu fantôme détecté"
        else:
            return f"Erreur HTML - {reason}"
    
    elif isinstance(exception, PDFExtractionError):
        reason = exception.reason
        if "trop petit" in reason.lower():
            return "PDF trop petit"
        elif "Impossible de lire" in reason:
            return "PDF corrompu/invalide"
        elif "Aucun texte" in reason:
            return "PDF sans texte (scan?)"
        elif "trop court" in reason.lower():
            return "Contenu PDF trop court"
        else:
            return f"Erreur PDF - {reason}"
    
    elif hasattr(exception, 'response'):
        # Erreur HTTP
        status_code = getattr(exception.response, 'status_code', 'Unknown')
        return f"Erreur HTTP {status_code}"
    
    else:
        return f"Erreur inattendue - {type(exception).__name__}"

def print_statistics(stats: dict, error_details: dict):
    """Affiche les statistiques finales avec détails des erreurs"""
    print("\n" + "="*80)
    print("📊 STATISTIQUES DE SCRAPING")
    print("="*80)
    print(f"Total de liens traités:     {stats['total']}")
    print(f"✅ Nouveaux documents:       {stats['new']}")
    print(f"🔄 Documents mis à jour:     {stats['updated']}")
    print(f"⏭️  Documents inchangés:      {stats['unchanged']}")
    print(f"❌ Échecs d'extraction:      {stats['failed']}")
    print(f"⏱️  Temps total:              {stats['duration']:.2f} secondes")
    print("="*80)
    
    if stats['failed'] > 0:
        print("\n" + "="*80)
        print("📋 VENTILATION DES ÉCHECS PAR CATÉGORIE")
        print("="*80)
        
        # Compte par catégorie
        category_counts = defaultdict(int)
        for url, details in error_details.items():
            category_counts[details['category']] += 1
        
        # Affichage trié par fréquence
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['failed']) * 100
            print(f"  {count:3d} ({percentage:5.1f}%) - {category}")
        
        print("\n" + "="*80)
        print("🔍 EXEMPLES D'ERREURS PAR CATÉGORIE")
        print("="*80)
        
        # Affiche quelques exemples pour chaque catégorie
        examples_shown = defaultdict(int)
        max_examples_per_category = 2
        
        for url, details in error_details.items():
            category = details['category']
            if examples_shown[category] < max_examples_per_category:
                print(f"\n📌 {category}:")
                print(f"   URL: {url}")
                print(f"   Détails: {details.get('error_details', 'Aucun détail')}")
                examples_shown[category] += 1
        
        print("\n" + "="*80)
        print(f"💡 RECOMMANDATIONS:")
        print("="*80)
        
        # Recommandations basées sur les erreurs
        if category_counts.get("Structure HTML - Conteneur introuvable", 0) > 0:
            print("  🔧 Beaucoup d'erreurs de conteneur HTML:")
            print("     → Vérifiez le fichier scrape_debug.log pour voir la structure des pages")
            print("     → La structure du site a peut-être changé")
        
        if category_counts.get("Content-Type invalide (HTML attendu)", 0) > 0:
            print("  🔧 Content-Type invalides détectés:")
            print("     → Certains liens ne pointent pas vers du HTML")
            print("     → Vérifiez les filtres dans scrape_links.py")
        
        if any("PDF" in cat for cat in category_counts.keys()):
            print("  🔧 Problèmes PDF détectés:")
            print("     → Certains PDF sont peut-être scannés (besoin d'OCR)")
            print("     → Vérifiez si pypdf peut les lire correctement")
        
        print(f"\n📄 Fichiers de diagnostic:")
        print(f"   - Logs détaillés:    {DEBUG_LOG_PATH}")
        print(f"   - Détails d'erreurs: {ERROR_DETAILS_PATH}")
    
    print("\n" + "="*80)

def main(force_update: bool = False, verify_ssl: bool = True):
    """
    Pipeline principal de scraping avec diagnostic avancé.
    
    Args:
        force_update: Force la mise à jour même si le hash n'a pas changé
        verify_ssl: Active/désactive la vérification SSL
    """
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info(f"🚀 DÉBUT DU SCRAPING - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info(f"Mode: {'Mise à jour forcée' if force_update else 'Incrémental'}")
    logger.info(f"SSL Verification: {verify_ssl}")
    
    # Récupération des liens
    logger.info("🔍 Récupération de la liste des liens...")
    links = get_all_links(verify_ssl=verify_ssl)
    
    if not links:
        logger.error("❌ Aucun lien trouvé. Abandon.")
        return
    
    logger.info(f"✅ {len(links)} liens à traiter")
    
    # Chargement de l'existant
    registry = load_registry()
    documents = load_documents()
    current_docs = {doc["metadata"]["source"]: doc for doc in documents}
    
    # Initialisation de la session
    session = get_robust_session(verify_ssl=verify_ssl)
    
    # Structures pour le nouveau scraping
    final_docs = []
    stats = {
        'total': len(links),
        'new': 0,
        'updated': 0,
        'unchanged': 0,
        'failed': 0,
        'duration': 0
    }
    
    # Dictionnaire pour stocker les détails d'erreur
    error_details = {}
    
    # Traitement de chaque lien
    print(f"\n{'='*80}")
    print("🔄 TRAITEMENT DES LIENS")
    print(f"{'='*80}\n")
    
    for i, url in enumerate(links, start=1):
        progress = f"[{i}/{len(links)}]"
        print(f"{progress} {url}")
        logger.info(f"{progress} Traitement: {url}")
        
        try:
            # Vérification du Content-Type AVANT extraction
            try:
                head_response = session.head(url, timeout=10, allow_redirects=True)
                content_type = head_response.headers.get('Content-Type', 'Unknown').lower()
                logger.debug(f"Content-Type pré-vérification: {content_type}")
            except Exception as e:
                logger.warning(f"Impossible de pré-vérifier Content-Type: {e}")
                content_type = None
            
            # Extraction selon le type
            if url.endswith(".pdf"):
                result = extract_pdf_text(url, session)
            else:
                result = extract_html_content(url, session)
            
            if result:
                new_hash = get_content_hash(result["content"])
                old_hash = registry.get(url)
                
                # Stratégie de mise à jour
                if old_hash == new_hash and not force_update:
                    print(f"      ⏭️  Inchangé")
                    logger.info(f"{progress} Inchangé: {url}")
                    final_docs.append(current_docs[url])
                    stats['unchanged'] += 1
                    
                elif old_hash and old_hash != new_hash:
                    print(f"      🔄 Mis à jour")
                    logger.info(f"{progress} Mis à jour: {url}")
                    final_docs.append(result)
                    registry[url] = new_hash
                    stats['updated'] += 1
                    
                else:
                    print(f"      ✨ Nouveau")
                    logger.info(f"{progress} Nouveau: {url}")
                    final_docs.append(result)
                    registry[url] = new_hash
                    stats['new'] += 1
            else:
                print(f"      ❌ Échec (résultat None)")
                logger.error(f"{progress} Échec: résultat None pour {url}")
                stats['failed'] += 1
                error_details[url] = {
                    'category': 'Résultat None inattendu',
                    'error_details': 'La fonction a retourné None sans lever d\'exception'
                }
                if url in current_docs:
                    final_docs.append(current_docs[url])
        
        except (HTMLExtractionError, PDFExtractionError) as e:
            # Erreur structurée de nos scrapers
            print(f"      ❌ {e.reason}")
            logger.error(f"{progress} {e.reason}")
            
            # Catégorisation de l'erreur
            category = categorize_error(e)
            stats['failed'] += 1
            
            # Sauvegarde des détails
            error_details[url] = {
                'category': category,
                'reason': e.reason,
                'error_details': e.details
            }
            
            # Log du traceback complet
            logger.debug("Traceback complet:")
            logger.debug(traceback.format_exc())
            
            # On garde l'ancien si disponible
            if url in current_docs:
                final_docs.append(current_docs[url])
                logger.info(f"{progress} Document ancien conservé")
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Interruption utilisateur détectée!")
            print("\n⚠️ Sauvegarde du checkpoint avant arrêt...")
            save_checkpoint(final_docs, registry)
            save_error_details(error_details)
            logger.info("✅ Checkpoint sauvegardé. Vous pouvez relancer le script.")
            sys.exit(0)
            
        except Exception as e:
            # Erreur inattendue
            print(f"      ❌ Erreur inattendue: {type(e).__name__}")
            logger.error(f"{progress} Erreur inattendue: {url}")
            logger.error(f"Type: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            
            # Traceback complet dans le log
            logger.error("Traceback complet:")
            logger.error(traceback.format_exc())
            
            category = categorize_error(e)
            stats['failed'] += 1
            
            error_details[url] = {
                'category': category,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            
            if url in current_docs:
                final_docs.append(current_docs[url])
        
        # Checkpoint périodique
        if i % CHECKPOINT_INTERVAL == 0:
            print(f"\n💾 Checkpoint automatique ({i}/{len(links)} traités)...")
            if save_checkpoint(final_docs, registry):
                logger.info(f"💾 Checkpoint {i}/{len(links)} sauvegardé")
                # Sauvegarde aussi les erreurs
                save_error_details(error_details)
            else:
                print("⚠️ Échec sauvegarde checkpoint")
    
    # Sauvegarde finale
    print(f"\n{'='*80}")
    print("💾 SAUVEGARDE FINALE")
    print(f"{'='*80}\n")
    
    if save_checkpoint(final_docs, registry):
        logger.info(f"✅ {len(final_docs)} documents sauvegardés dans {DATA_PATH}")
    else:
        logger.error("❌ Échec de la sauvegarde finale!")
        return
    
    # Sauvegarde des détails d'erreur
    save_error_details(error_details)
    
    # Statistiques
    end_time = datetime.now()
    stats['duration'] = (end_time - start_time).total_seconds()
    
    print_statistics(stats, error_details)
    
    logger.info("="*80)
    logger.info(f"✅ FIN DU SCRAPING - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏱️  Durée totale: {stats['duration']:.2f} secondes")
    logger.info("="*80)

def verify_data():
    """Vérifie l'intégrité des données sauvegardées"""
    print("\n🔍 VÉRIFICATION DE L'INTÉGRITÉ DES DONNÉES\n")
    
    if not os.path.exists(DATA_PATH):
        print("❌ Fichier de données introuvable.")
        return
    
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            documents = json.load(f)
        
        print(f"✅ {len(documents)} documents chargés")
        
        sources = set()
        doc_types = {}
        total_chars = 0
        
        for doc in documents:
            source = doc["metadata"]["source"]
            if source in sources:
                print(f"⚠️ Doublon détecté: {source}")
            sources.add(source)
            
            doc_type = doc["metadata"].get("doc_type", "Unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            total_chars += len(doc["content"])
        
        print(f"\n📊 Répartition par type:")
        for doc_type, count in sorted(doc_types.items()):
            print(f"   {doc_type}: {count}")
        
        print(f"\n📈 Volume total: {total_chars:,} caractères")
        print(f"📈 Moyenne: {total_chars // len(documents):,} caractères/document")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scraper du manuel de gestion UQAC - Version Diagnostic")
    parser.add_argument("--force", action="store_true", help="Force la mise à jour de tous les documents")
    parser.add_argument("--verify", action="store_true", help="Vérifie l'intégrité des données sans scraper")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Désactive la vérification SSL")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_data()
    else:
        main(force_update=args.force, verify_ssl=not args.no_ssl_verify)