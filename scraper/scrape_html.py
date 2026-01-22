import re
import logging
from bs4 import BeautifulSoup
from datetime import datetime

# Configuration du logger
logger = logging.getLogger(__name__)

class HTMLExtractionError(Exception):
    """Exception personnalisée pour les erreurs d'extraction HTML"""
    def __init__(self, url, reason, details=None):
        self.url = url
        self.reason = reason
        self.details = details or {}
        super().__init__(f"{reason}: {url}")

def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant espaces multiples et caractères de contrôle"""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    return text.strip()

def table_to_markdown(table) -> str:
    """Convertit une balise <table> HTML en tableau Markdown"""
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["td", "th"]):
            cell_text = clean_text(cell.get_text())
            cell_text = cell_text.replace("|", "\\|")
            cells.append(cell_text)
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    
    if not rows:
        return ""
    
    first_row_has_th = bool(table.find("tr").find_all("th"))
    if first_row_has_th and len(rows) > 0:
        num_columns = rows[0].count("|") - 1
        separator = "|" + " --- |" * num_columns
        rows.insert(1, separator)
    
    return "\n\n" + "\n".join(rows) + "\n\n"

def extract_date(text: str) -> str | None:
    """Recherche des motifs de date dans le texte"""
    patterns = [
        r'(?:mis[e]?\s+à\s+jour|date|modifié|publié|en\s+vigueur)[:\s]+(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})',
        r'(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})',
        r'((?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})',
        r'(\d{4}-\d{2}-\d{2})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def extract_hierarchy(soup) -> str:
    """Extrait le fil d'Ariane (breadcrumb)"""
    breadcrumb = soup.find("div", class_=re.compile(r"breadcrumb", re.I))
    if breadcrumb:
        return clean_text(breadcrumb.get_text(" > ", strip=True))
    
    breadcrumb = soup.find("nav", {"aria-label": "breadcrumb"})
    if breadcrumb:
        return clean_text(breadcrumb.get_text(" > ", strip=True))
    
    breadcrumb = soup.find(["ol", "ul"], class_=re.compile(r"breadcrumb", re.I))
    if breadcrumb:
        items = [li.get_text(strip=True) for li in breadcrumb.find_all("li")]
        return " > ".join(items)
    
    return ""

def extract_title(soup) -> str:
    """Extrait le titre du document"""
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text())
    
    title_tag = soup.find("title")
    if title_tag:
        title_text = clean_text(title_tag.get_text())
        title_text = re.sub(r'\s*[-|]\s*UQAC.*$', '', title_text, flags=re.I)
        return title_text
    
    h2 = soup.find("h2")
    if h2:
        return clean_text(h2.get_text())
    
    return "Sans titre"

def remove_navigation_noise(soup):
    """
    Supprime SÉLECTIVEMENT les éléments de navigation.
    VERSION CORRIGÉE: Ne supprime QUE les éléments de navigation, 
    PAS le contenu principal.
    """
    # Liste des éléments à supprimer de manière ciblée
    # NE PAS supprimer les divs de manière générique!
    
    # Supprime les <nav> (sauf s'ils contiennent du contenu principal)
    for nav in soup.find_all("nav"):
        # Garde les nav si elles ont beaucoup de contenu (> 500 chars)
        if len(nav.get_text(strip=True)) < 500:
            nav.decompose()
    
    # Supprime header/footer seulement s'ils sont petits
    for tag in soup.find_all(["header", "footer"]):
        if len(tag.get_text(strip=True)) < 500:
            tag.decompose()
    
    # Supprime les éléments avec des IDs/classes spécifiques de navigation
    navigation_patterns = [
        {"id": re.compile(r"^(nav|menu|sidebar|header|footer)$", re.I)},
        {"class_": lambda x: x and any(
            re.search(r"^(nav|menu|sidebar|breadcrumb)$", cls, re.I) 
            for cls in (x if isinstance(x, list) else [x])
        )}
    ]
    
    for selector in navigation_patterns:
        for element in soup.find_all(**selector):
            # Vérification de sécurité: ne supprime que si petit
            if len(element.get_text(strip=True)) < 500:
                element.decompose()

def inspect_page_structure(soup, url):
    """
    Inspecte et log la structure de la page pour diagnostic.
    """
    logger.warning(f"🔍 INSPECTION DE LA STRUCTURE: {url}")
    
    # Liste toutes les divs avec leurs classes et IDs
    divs = soup.find_all("div", limit=50)
    logger.debug(f"Total de <div> trouvées: {len(divs)}")
    
    if divs:
        divs_with_class = [(div.get('class'), div.get('id'), len(div.get_text(strip=True))) 
                          for div in divs if div.get('class') or div.get('id')]
        if divs_with_class:
            logger.debug("Classes, IDs et longueur des 20 premières divs:")
            for i, (classes, div_id, length) in enumerate(divs_with_class[:20], 1):
                class_str = '.'.join(classes) if classes else 'pas de classe'
                id_str = f"#{div_id}" if div_id else 'pas d\'id'
                logger.debug(f"  {i}. {class_str} {id_str} ({length} chars)")
    
    # Liste les balises principales
    main_tags = {
        'main': soup.find_all('main'),
        'article': soup.find_all('article'),
        'section': soup.find_all('section'),
        'body': [soup.body] if soup.body else []
    }
    
    for tag_name, tags in main_tags.items():
        if tags:
            logger.debug(f"Balises <{tag_name}> trouvées: {len(tags)}")
            for tag in tags[:3]:  # Limite à 3
                if tag:
                    text_len = len(tag.get_text(strip=True))
                    logger.debug(f"  - class: {tag.get('class')}, id: {tag.get('id')}, contenu: {text_len} chars")

def extract_html_content(url: str, session, min_length: int = 100) -> dict | None:
    """
    Extrait le contenu structuré d'une page HTML avec logging diagnostic complet.
    """
    logger.info(f"📄 Extraction HTML: {url}")
    
    try:
        # 1. Requête HTTP
        response = session.get(url, timeout=30)
        
        logger.debug(f"Status Code: {response.status_code}")
        logger.debug(f"Content-Type: {response.headers.get('Content-Type', 'Non spécifié')}")
        logger.debug(f"Content-Length: {response.headers.get('Content-Length', 'Non spécifié')}")
        logger.debug(f"Encoding détecté: {response.encoding}")
        
        # Vérification du Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            raise HTMLExtractionError(
                url, 
                "Content-Type invalide",
                {
                    'content_type': content_type,
                    'status_code': response.status_code
                }
            )
        
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        # 2. Parsing HTML
        soup = BeautifulSoup(response.text, "lxml")
        logger.debug(f"HTML parsé avec succès, longueur: {len(response.text)} caractères")
        
        # 3. Extraction des métadonnées AVANT nettoyage
        title = extract_title(soup)
        hierarchy = extract_hierarchy(soup)
        logger.debug(f"Titre: {title}")
        logger.debug(f"Hiérarchie: {hierarchy if hierarchy else 'Non trouvée'}")
        
        # 4. Nettoyage SÉLECTIF du DOM
        # IMPORTANT: Copie du soup pour préserver l'original si besoin
        remove_navigation_noise(soup)
        
        # 5. Sélection du conteneur principal avec diagnostic
        container = None
        attempted_selectors = []
        
        # NOUVEAU: D'abord essayer de trouver le body (doit toujours exister)
        if soup.body:
            # Chercher dans le body
            logger.debug("Body trouvé, recherche du conteneur de contenu...")
            
            # Essai 1: div avec classe content/main
            container = soup.body.find("div", class_=re.compile(r"(entry-content|main-content|content|page-content|post-content|article-content)", re.I))
            attempted_selectors.append(("div avec classe content", container is not None))
            
            # Essai 2: main
            if not container:
                container = soup.body.find("main")
                attempted_selectors.append(("main", container is not None))
            
            # Essai 3: article
            if not container:
                container = soup.body.find("article")
                attempted_selectors.append(("article", container is not None))
            
            # Essai 4: div avec id content/main
            if not container:
                container = soup.body.find("div", id=re.compile(r"(content|main|primary)", re.I))
                attempted_selectors.append(("div avec id content/main", container is not None))
            
            # Essai 5: DERNIER RECOURS - utiliser le body lui-même
            if not container:
                container = soup.body
                attempted_selectors.append(("body (fallback)", True))
                logger.debug("Utilisation du body comme conteneur (aucun conteneur spécifique trouvé)")
        
        logger.debug("Sélecteurs tentés:")
        for selector, found in attempted_selectors:
            status = "✓" if found else "✗"
            logger.debug(f"  {status} {selector}")
        
        if not container:
            # Inspection détaillée de la structure
            inspect_page_structure(soup, url)
            raise HTMLExtractionError(
                url,
                "Aucun conteneur principal trouvé (body absent)",
                {
                    'attempted_selectors': [s[0] for s in attempted_selectors],
                    'body_present': soup.body is not None
                }
            )
        
        logger.debug(f"Conteneur sélectionné: {container.name} (class: {container.get('class')}, id: {container.get('id')})")
        
        # 6. Conversion des tableaux
        tables_found = container.find_all("table")
        logger.debug(f"Tableaux trouvés: {len(tables_found)}")
        
        for i, table in enumerate(tables_found, 1):
            md_table = table_to_markdown(table)
            if md_table:
                logger.debug(f"  Tableau {i} converti en Markdown ({len(md_table)} caractères)")
                table.replace_with(BeautifulSoup(md_table, "html.parser"))
        
        # 7. Extraction du texte
        full_text = container.get_text("\n", strip=True)
        full_text = clean_text(full_text)
        
        logger.debug(f"Texte extrait: {len(full_text)} caractères")
        
        # 8. Extraction de la date
        update_date = extract_date(full_text)
        if update_date:
            logger.debug(f"Date trouvée: {update_date}")
        
        # 9. Détermination du type de document
        doc_type = "Document"
        if "politique" in url.lower() or "politique" in title.lower():
            doc_type = "Politique"
        elif "reglement" in url.lower() or "règlement" in title.lower():
            doc_type = "Règlement"
        elif "formulaire" in url.lower() or "formulaire" in title.lower():
            doc_type = "Formulaire"
        elif "procedure" in url.lower() or "procédure" in title.lower():
            doc_type = "Procédure"
        
        logger.debug(f"Type de document: {doc_type}")
        
        # 10. Validation de la longueur
        if len(full_text) < min_length:
            preview = full_text[:100] if full_text else "(vide)"
            raise HTMLExtractionError(
                url,
                "Contenu trop court",
                {
                    'length': len(full_text),
                    'min_required': min_length,
                    'preview': preview,
                    'title': title
                }
            )
        
        # 11. Détection de contenu "fantôme"
        ghost_patterns = [
            (r"javascript\s+est\s+désactivé", "JavaScript désactivé"),
            (r"page\s+non\s+trouvée", "Page non trouvée"),
            (r"erreur\s+404", "Erreur 404"),
            (r"access\s+denied", "Accès refusé")
        ]
        
        for pattern, description in ghost_patterns:
            if re.search(pattern, full_text, re.I):
                raise HTMLExtractionError(
                    url,
                    f"Contenu fantôme détecté: {description}",
                    {
                        'pattern_matched': pattern,
                        'preview': full_text[:200]
                    }
                )
        
        # 12. Construction du résultat
        logger.info(f"✅ Extraction réussie: {len(full_text)} caractères, {len(tables_found)} tableaux")
        
        return {
            "content": full_text,
            "metadata": {
                "source": url,
                "title": title,
                "hierarchy": hierarchy,
                "last_updated": update_date,
                "doc_type": doc_type
            }
        }
        
    except HTMLExtractionError as e:
        logger.error(f"❌ {e.reason}")
        for key, value in e.details.items():
            logger.error(f"   {key}: {value}")
        raise
        
    except Exception as e:
        logger.exception(f"❌ Erreur inattendue lors de l'extraction HTML")
        raise HTMLExtractionError(
            url,
            f"Erreur inattendue: {type(e).__name__}",
            {'error_message': str(e)}
        )