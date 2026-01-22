import requests
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import urllib3

# Désactiver les warnings SSL si nécessaire
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://www.uqac.ca/mgestion/"

# Configuration du logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/scrape_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_robust_session(verify_ssl: bool = True):
    """
    Crée une session requests robuste avec:
    - User-Agent pour éviter les blocages
    - Stratégie de retry automatique en cas d'erreur serveur
    - Timeout par défaut
    - Option pour désactiver la vérification SSL
    
    Args:
        verify_ssl: Si False, désactive la vérification SSL (utile pour certains serveurs)
    """
    session = requests.Session()
    
    # Headers pour simuler un navigateur réel
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    })
    
    # Stratégie de reconnexion: 5 tentatives avec délai exponentiel
    retries = Retry(
        total=5,
        backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Configuration SSL
    session.verify = verify_ssl
    
    if not verify_ssl:
        logger.warning("⚠️ Vérification SSL DÉSACTIVÉE - Utilisez avec précaution!")
    
    return session

def is_relevant(url: str) -> bool:
    """
    Filtre les URLs pour ne garder que les documents finaux pertinents.
    Exclut:
    - Les pages racines et de navigation
    - Les ancres (#)
    - Les pages de chapitre (trop générales)
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    parts = path.split("/")
    
    logger.debug(f"Analyse pertinence: {url}")
    logger.debug(f"  - Parties du chemin: {parts}")
    
    # Filtre les pages racines et les chapitres généraux
    if len(parts) < 4:
        logger.debug(f"  - REJETÉ: Trop peu de parties ({len(parts)} < 4)")
        return False
    
    # Exclut les pages de navigation de type "chapitre-X"
    if parts[-1].startswith("chapitre"):
        logger.debug(f"  - REJETÉ: Page de chapitre ({parts[-1]})")
        return False
    
    # Exclut les ancres
    if "#" in url:
        logger.debug(f"  - REJETÉ: Contient une ancre")
        return False
    
    # Exclut les fichiers non pertinents
    excluded_extensions = ['.jpg', '.png', '.gif', '.css', '.js', '.zip']
    if any(url.lower().endswith(ext) for ext in excluded_extensions):
        logger.debug(f"  - REJETÉ: Extension non pertinente")
        return False
    
    logger.debug(f"  - ACCEPTÉ ✓")
    return True

def get_all_links(verify_ssl: bool = True):
    """
    Récupère tous les liens pertinents depuis la page d'accueil du manuel de gestion.
    Retourne une liste triée d'URLs uniques.
    
    Args:
        verify_ssl: Si False, désactive la vérification SSL
    """
    session = get_robust_session(verify_ssl=verify_ssl)
    
    try:
        logger.info(f"🔍 Récupération des liens depuis {BASE_URL}")
        logger.debug(f"SSL Verification: {verify_ssl}")
        
        response = session.get(BASE_URL, timeout=30)
        
        # Log des informations de réponse
        logger.debug(f"Status Code: {response.status_code}")
        logger.debug(f"Content-Type: {response.headers.get('Content-Type')}")
        logger.debug(f"Content-Length: {response.headers.get('Content-Length')}")
        
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, "lxml")
        links = set()
        
        # Extraction de tous les liens
        all_links = soup.find_all("a", href=True)
        logger.debug(f"Total de balises <a> trouvées: {len(all_links)}")
        
        for a in all_links:
            full_url = urljoin(BASE_URL, a["href"])
            
            # Filtre: doit commencer par BASE_URL et être pertinent
            if full_url.startswith(BASE_URL) and is_relevant(full_url):
                links.add(full_url)
        
        sorted_links = sorted(list(links))
        logger.info(f"✅ {len(sorted_links)} liens pertinents trouvés")
        
        # Log des premiers liens pour inspection
        if sorted_links:
            logger.debug("Exemples de liens trouvés:")
            for link in sorted_links[:5]:
                logger.debug(f"  - {link}")
        
        return sorted_links
        
    except requests.exceptions.SSLError as e:
        logger.error(f"❌ Erreur SSL: {e}")
        logger.info("💡 Essayez avec verify_ssl=False")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur lors de la récupération des liens: {e}")
        return []
    except Exception as e:
        logger.exception(f"❌ Erreur inattendue: {e}")
        return []

if __name__ == "__main__":
    # Test du module
    links = get_all_links()
    print(f"\n📋 Exemples de liens trouvés:")
    for link in links[:10]:
        print(f"  - {link}")