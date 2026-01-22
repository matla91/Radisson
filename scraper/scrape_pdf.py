import os
import tempfile
import re
import logging
from datetime import datetime
from pypdf import PdfReader

# Configuration du logger
logger = logging.getLogger(__name__)

class PDFExtractionError(Exception):
    """Exception personnalisée pour les erreurs d'extraction PDF"""
    def __init__(self, url, reason, details=None):
        self.url = url
        self.reason = reason
        self.details = details or {}
        super().__init__(f"{reason}: {url}")

def clean_pdf_text(text: str) -> str:
    """Nettoie le texte extrait d'un PDF"""
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    replacements = {
        'ï¬': 'fi', 'ï¬‚': 'fl', 'â€™': "'", 'â€œ': '"',
        'â€': '"', 'â€"': '—', 'â€"': '–', 'Â': ' ',
        '\uf0b7': '•',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    
    return text.strip()

def parse_pdf_date(date_str: str | None) -> str | None:
    """Parse les dates des métadonnées PDF"""
    if not date_str:
        return None
    
    try:
        match = re.match(r'D:(\d{4})(\d{2})(\d{2})', str(date_str))
        if match:
            year, month, day = match.groups()
            months_fr = {
                '01': 'janvier', '02': 'février', '03': 'mars',
                '04': 'avril', '05': 'mai', '06': 'juin',
                '07': 'juillet', '08': 'août', '09': 'septembre',
                '10': 'octobre', '11': 'novembre', '12': 'décembre'
            }
            month_name = months_fr.get(month, month)
            return f"{int(day)} {month_name} {year}"
    except Exception:
        pass
    
    return None

def extract_title_from_pdf(reader: PdfReader, url: str) -> str:
    """Extrait le titre du PDF"""
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()
        if title and title.lower() != 'untitled':
            logger.debug(f"Titre depuis métadonnées: {title}")
            return title
    
    try:
        first_page = reader.pages[0]
        first_text = first_page.extract_text()
        if first_text:
            lines = [line.strip() for line in first_text.split('\n') if line.strip()]
            if lines:
                potential_title = lines[0][:150]
                if len(potential_title.split()) >= 3:
                    logger.debug(f"Titre depuis première ligne: {potential_title}")
                    return potential_title
    except Exception as e:
        logger.warning(f"Impossible d'extraire le titre de la première page: {e}")
    
    filename = url.split("/")[-1]
    title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    logger.debug(f"Titre depuis nom de fichier: {title}")
    return title.strip() or "Document PDF"

def extract_hierarchy_from_url(url: str) -> str:
    """Construit une hiérarchie basée sur la structure de l'URL"""
    parts = url.split('/')
    hierarchy_parts = []
    
    for part in parts:
        if 'chapitre' in part.lower():
            match = re.search(r'chapitre[_-]?(\d+)', part, re.I)
            if match:
                hierarchy_parts.append(f"Chapitre {match.group(1)}")
        elif 'section' in part.lower():
            match = re.search(r'section[_-]?(\d+)', part, re.I)
            if match:
                hierarchy_parts.append(f"Section {match.group(1)}")
    
    if hierarchy_parts:
        return " > ".join(hierarchy_parts)
    
    return "Document PDF"

def extract_pdf_text(url: str, session, min_length: int = 100) -> dict | None:
    """
    Extrait le contenu d'un PDF avec logging diagnostic complet.
    
    Lève PDFExtractionError avec des détails spécifiques en cas d'échec.
    """
    tmp_path = None
    logger.info(f"📕 Extraction PDF: {url}")
    
    try:
        # 1. Téléchargement du PDF
        response = session.get(url, timeout=30)
        
        # Log détaillé de la réponse HTTP
        logger.debug(f"Status Code: {response.status_code}")
        logger.debug(f"Content-Type: {response.headers.get('Content-Type', 'Non spécifié')}")
        logger.debug(f"Content-Length: {response.headers.get('Content-Length', 'Non spécifié')}")
        
        # Vérification du Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
            logger.warning(f"Content-Type inattendu: {content_type}")
            # On continue quand même, certains serveurs ne configurent pas correctement le Content-Type
        
        response.raise_for_status()
        
        # Vérification de la taille du fichier
        content_length = len(response.content)
        logger.debug(f"Taille du fichier téléchargé: {content_length:,} octets ({content_length/1024:.2f} KB)")
        
        if content_length < 100:
            raise PDFExtractionError(
                url,
                "Fichier PDF trop petit",
                {
                    'size_bytes': content_length,
                    'content_type': content_type
                }
            )
        
        # 2. Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        logger.debug(f"PDF sauvegardé temporairement: {tmp_path}")
        
        # 3. Lecture du PDF
        try:
            reader = PdfReader(tmp_path)
        except Exception as e:
            raise PDFExtractionError(
                url,
                "Impossible de lire le PDF",
                {
                    'error': str(e),
                    'file_size': content_length,
                    'possible_cause': 'Fichier corrompu ou format invalide'
                }
            )
        
        page_count = len(reader.pages)
        logger.debug(f"PDF chargé avec succès: {page_count} pages")
        
        # 4. Extraction des métadonnées
        title = extract_title_from_pdf(reader, url)
        hierarchy = extract_hierarchy_from_url(url)
        
        # Log des métadonnées PDF
        if reader.metadata:
            logger.debug("Métadonnées PDF:")
            for key, value in reader.metadata.items():
                logger.debug(f"  {key}: {value}")
        
        # 5. Extraction de la date
        meta_date = None
        if reader.metadata:
            mod_date = reader.metadata.get("/ModDate") or reader.metadata.get("/CreationDate")
            meta_date = parse_pdf_date(mod_date)
            if meta_date:
                logger.debug(f"Date extraite: {meta_date}")
        
        # 6. Extraction du texte de toutes les pages
        text_parts = []
        pages_with_errors = []
        pages_with_no_text = []
        
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
                    logger.debug(f"  Page {i+1}: {len(page_text)} caractères extraits")
                else:
                    pages_with_no_text.append(i+1)
                    logger.debug(f"  Page {i+1}: Aucun texte")
            except Exception as e:
                pages_with_errors.append((i+1, str(e)))
                logger.warning(f"  Page {i+1}: Erreur - {e}")
                continue
        
        # Log du résumé d'extraction
        logger.debug(f"Résumé extraction:")
        logger.debug(f"  Pages avec texte: {len(text_parts)}/{page_count}")
        if pages_with_no_text:
            logger.debug(f"  Pages sans texte: {pages_with_no_text}")
        if pages_with_errors:
            logger.debug(f"  Pages avec erreurs: {[p[0] for p in pages_with_errors]}")
        
        if not text_parts:
            raise PDFExtractionError(
                url,
                "Aucun texte extractible",
                {
                    'total_pages': page_count,
                    'pages_with_errors': len(pages_with_errors),
                    'possible_cause': 'PDF scanné sans OCR ou protégé',
                    'pages_with_errors_detail': pages_with_errors[:5]  # Limite à 5 pour éviter trop de logs
                }
            )
        
        # 7. Assemblage et nettoyage du texte
        full_text = "\n\n".join(text_parts)
        full_text = clean_pdf_text(full_text)
        
        logger.debug(f"Texte total après nettoyage: {len(full_text)} caractères")
        
        # 8. Détection de PDF scanné
        avg_text_per_page = len(full_text) / page_count
        logger.debug(f"Moyenne de texte par page: {avg_text_per_page:.2f} caractères")
        
        if avg_text_per_page < 50:
            logger.warning(f"⚠️ Possible PDF scanné (faible densité de texte: {avg_text_per_page:.2f} char/page)")
        
        # 9. Validation de la longueur
        if len(full_text) < min_length:
            preview = full_text[:100] if full_text else "(vide)"
            raise PDFExtractionError(
                url,
                "Contenu trop court",
                {
                    'length': len(full_text),
                    'min_required': min_length,
                    'preview': preview,
                    'pages': page_count,
                    'avg_per_page': avg_text_per_page,
                    'title': title
                }
            )
        
        # 10. Détermination du type de document
        doc_type = "Document PDF"
        lower_title = title.lower()
        lower_url = url.lower()
        
        if "politique" in lower_title or "politique" in lower_url:
            doc_type = "Politique PDF"
        elif "règlement" in lower_title or "reglement" in lower_url:
            doc_type = "Règlement PDF"
        elif "formulaire" in lower_title or "formulaire" in lower_url:
            doc_type = "Formulaire PDF"
        elif "procédure" in lower_title or "procedure" in lower_url:
            doc_type = "Procédure PDF"
        elif "guide" in lower_title:
            doc_type = "Guide PDF"
        
        logger.debug(f"Type de document: {doc_type}")
        
        # 11. Construction du résultat
        logger.info(f"✅ Extraction réussie: {len(full_text)} caractères, {page_count} pages")
        
        return {
            "content": full_text,
            "metadata": {
                "source": url,
                "title": title,
                "hierarchy": hierarchy,
                "last_updated": meta_date,
                "doc_type": doc_type,
                "page_count": page_count
            }
        }
        
    except PDFExtractionError as e:
        # Log l'erreur structurée
        logger.error(f"❌ {e.reason}")
        for key, value in e.details.items():
            logger.error(f"   {key}: {value}")
        raise
        
    except Exception as e:
        logger.exception(f"❌ Erreur inattendue lors de l'extraction PDF")
        raise PDFExtractionError(
            url,
            f"Erreur inattendue: {type(e).__name__}",
            {'error_message': str(e)}
        )
        
    finally:
        # Nettoyage du fichier temporaire
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"Fichier temporaire supprimé: {tmp_path}")
            except Exception as e:
                logger.warning(f"Impossible de supprimer le fichier temporaire: {e}")

if __name__ == "__main__":
    from scrape_links_diagnostic import get_robust_session
    
    session = get_robust_session()
    test_url = "https://www.uqac.ca/mgestion/exemple.pdf"
    
    try:
        result = extract_pdf_text(test_url, session)
        if result:
            print(f"✅ Extraction réussie")
            print(f"Titre: {result['metadata']['title']}")
            print(f"Pages: {result['metadata']['page_count']}")
            print(f"Longueur: {len(result['content'])} caractères")
    except PDFExtractionError as e:
        print(f"❌ Échec: {e.reason}")
        print(f"Détails: {e.details}")