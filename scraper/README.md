# UQAC Management Manual Scraping Pipeline

## Overview

This directory contains a production-grade **ETL (Extract-Transform-Load) pipeline** designed to systematically harvest and structure content from the UQAC (Université du Québec à Chicoutimi) management manual website. The pipeline transforms an unstructured web corpus into a well-formatted JSON knowledge base, ready for downstream processing such as RAG (Retrieval-Augmented Generation) systems, semantic search, or document analysis.

**Primary Objective:** Convert a complex, multi-format web resource (`https://www.uqac.ca/mgestion/`) into a single, validated, and incrementally maintainable `raw_texts.json` dataset with enriched metadata.

**Key Design Principles:**
- **Robustness:** Resilient to network failures, malformed documents, and server errors
- **Incrementality:** Only processes modified content through MD5 hash comparison
- **Traceability:** Multi-level diagnostic logging for production monitoring and debugging
- **Data Integrity:** Comprehensive error handling and checkpoint-based recovery

---

## Key Features

### 1. Robustness

The pipeline implements enterprise-grade reliability mechanisms to handle real-world web scraping challenges:

#### HTTP Layer Resilience
- **Exponential Backoff Retry:** Automatic retry with exponential delays (1s → 2s → 4s → 8s → 16s)
- **Status Code Handling:** Intelligent retry on transient failures (429, 500, 502, 503, 504)
- **SSL Flexibility:** Configurable SSL verification for environments with certificate issues
- **User-Agent Spoofing:** Mimics genuine browser requests to avoid bot detection
- **Connection Pooling:** Persistent HTTP sessions with keep-alive for efficiency

#### Content Extraction Resilience
- **Multi-Strategy Parsing:** Cascading selectors for HTML content containers (main → article → div.content → body)
- **Graceful Degradation:** Falls back to body element if semantic containers are absent
- **PDF Error Tolerance:** Continues extraction even if individual pages fail
- **Encoding Auto-Detection:** Handles UTF-8, Latin-1, and malformed character encodings

#### Recovery Mechanisms
- **Checkpoint System:** Periodic saves every 10 documents to prevent data loss
- **Keyboard Interrupt Handling:** Graceful shutdown with state preservation on Ctrl+C
- **Old Content Preservation:** Retains previous versions when new extraction fails

```python
# Example: Robust session configuration
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
```

### 2. Incremental Scraping

The pipeline employs **content-addressable storage** via MD5 hashing to minimize redundant processing and network load:

#### Hash-Based Change Detection
- **Registry Persistence:** Maintains `scrape_registry.json` mapping URLs to MD5 hashes
- **Content Fingerprinting:** Compares new content hash against registry before update
- **Differential Updates:** Only overwrites documents when content has genuinely changed
- **Network Optimization:** Skips re-download of unchanged resources

#### Update Strategies
1. **NEW:** Document URL not in registry → Full extraction + hash storage
2. **UPDATED:** Hash mismatch → Re-extraction + hash update
3. **UNCHANGED:** Hash match → Reuse existing content (no network call)

**Force Mode:** `--force` flag bypasses hash checks for complete refresh

```python
new_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
old_hash = registry.get(url)

if old_hash == new_hash and not force_update:
    print("⏭️ Unchanged")
    final_docs.append(current_docs[url])
elif old_hash != new_hash:
    print("🔄 Updated")
    registry[url] = new_hash
```

### 3. Diagnostic Logging

Multi-tier logging system provides comprehensive observability for production environments:

#### Logging Hierarchy
1. **DEBUG Level (`scrape_debug.log`):**
   - HTTP headers (Content-Type, Content-Length, Status Codes)
   - HTML structure inspection (div classes, IDs, selector attempts)
   - PDF page-by-page extraction details
   - Regex pattern matching results
   - Full exception tracebacks

2. **INFO Level (Console):**
   - Progress indicators (`[42/150]`)
   - Document status (NEW/UPDATED/UNCHANGED)
   - Checkpoint confirmations
   - Final statistics summary

3. **ERROR Level:**
   - Extraction failures with categorization
   - Registry/checkpoint save failures
   - Structured error details in `error_details.json`

#### Error Categorization
The pipeline classifies failures into actionable categories:

```python
# Example categories
- "Structure HTML - Conteneur introuvable"  # HTML parsing issues
- "PDF sans texte (scan?)"                   # Scanned PDFs without OCR
- "Contenu trop court"                       # Validation failures
- "Erreur HTTP 404"                          # Network errors
```

**Error Analysis Report:** Automatically generated at completion with:
- Total failures by category
- Representative examples per category
- Remediation recommendations
- Diagnostic file references

---

## Technical Architecture

### System Design

The pipeline follows a **phased extraction model** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. Link Discovery Phase                       │
│  scrape_links.py → Crawls base URL → Filters relevant links     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│               2. Content Type Identification                     │
│  scrape_all.py → HEAD request → Content-Type check              │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│  3a. HTML Path   │    │   3b. PDF Path   │
│ scrape_html.py   │    │  scrape_pdf.py   │
│   ├─ Parse DOM   │    │   ├─ Download    │
│   ├─ Extract     │    │   ├─ Parse       │
│   ├─ Clean       │    │   ├─ Extract     │
│   └─ Metadata    │    │   └─ Metadata    │
└──────────┬───────┘    └──────────┬───────┘
           │                       │
           └───────────┬───────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               4. Validation & Persistence                        │
│  scrape_all.py → Hash check → Registry update → JSON save       │
└─────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

#### `scrape_links.py` - Link Discovery Module
**Purpose:** Identify all relevant document URLs from the management manual homepage

**Core Functions:**
- `get_robust_session(verify_ssl: bool) → requests.Session`
  - Creates HTTP session with retry logic and browser headers
- `is_relevant(url: str) → bool`
  - Filters out navigation pages, anchors, and non-content resources
  - Criteria: Path depth ≥ 4 segments, excludes chapter indices
- `get_all_links(verify_ssl: bool) → List[str]`
  - Returns sorted, deduplicated list of scrapable URLs

**Filtering Logic:**
```python
# Exclusion criteria
- Path depth < 4 segments          # Reject: /mgestion/chapitre-5
- Ends with "chapitre"              # Reject: /mgestion/chapitre-5/chapitre
- Contains anchor (#)               # Reject: /page.html#section2
- Binary extensions (.jpg, .zip)    # Reject non-textual content
```

#### `scrape_html.py` - HTML Content Extractor
**Purpose:** Extract clean, structured text from HTML pages with metadata enrichment

**Key Functions:**
- `extract_html_content(url, session, min_length=100) → dict | None`
  - Main extraction pipeline
  - Returns: `{"content": str, "metadata": dict}` or None on failure

**Extraction Pipeline:**
1. **HTTP Fetch:** GET request with timeout (30s)
2. **Validation:** Verify `Content-Type: text/html`
3. **Metadata Extraction:**
   - `extract_title(soup)` → H1, title tag, or "Sans titre"
   - `extract_hierarchy(soup)` → Breadcrumb trail (e.g., "Chapitre 5 > Section 2")
   - `extract_date(text)` → Last updated date (French month names)
4. **DOM Cleaning:** `remove_navigation_noise(soup)` → Removes nav, header, footer
5. **Content Selection:** Cascading search for main content container
6. **Table Conversion:** `table_to_markdown(table)` → Markdown table syntax
7. **Text Extraction:** Get cleaned text with newline preservation
8. **Quality Checks:**
   - Length validation (min 100 chars)
   - Ghost content detection (404 errors, JavaScript disabled warnings)

**Exception Handling:**
```python
class HTMLExtractionError(Exception):
    def __init__(self, url, reason, details=None):
        self.url = url
        self.reason = reason
        self.details = details or {}
```

Raises structured exceptions with diagnostic details:
- `Content-Type invalide`: Not HTML/XHTML
- `Conteneur introuvable`: No main content container
- `Contenu trop court`: Below minimum length threshold
- `Contenu fantôme`: Error page detected

#### `scrape_pdf.py` - PDF Content Extractor
**Purpose:** Extract text from PDF documents with comprehensive error handling

**Key Functions:**
- `extract_pdf_text(url, session, min_length=100) → dict | None`
  - Downloads PDF to temporary file
  - Extracts text with pypdf.PdfReader
  - Returns structured document with metadata

**Extraction Pipeline:**
1. **Download:** HTTP GET → Temporary file storage
2. **Validation:** 
   - Content-Type check (application/pdf)
   - Minimum size verification (> 100 bytes)
3. **Parsing:** `PdfReader(tmp_path)` with error handling
4. **Metadata Extraction:**
   - `extract_title_from_pdf(reader, url)` → From metadata or first page
   - `extract_hierarchy_from_url(url)` → From URL structure
   - `parse_pdf_date(metadata)` → From ModDate or CreationDate
5. **Page-by-Page Extraction:** 
   - Iterate through all pages
   - Log extraction success/failure per page
   - Continue on individual page errors
6. **Text Cleaning:** `clean_pdf_text(text)`
   - Remove control characters
   - Fix common PDF encoding artifacts (fi, fl ligatures)
   - Normalize whitespace
7. **Quality Checks:**
   - Total length validation
   - Scanned PDF detection (< 50 chars/page average)

**Exception Handling:**
```python
class PDFExtractionError(Exception):
    # Similar structure to HTMLExtractionError
```

Error categories:
- `PDF trop petit`: File size < 100 bytes
- `Impossible de lire`: Corrupted or encrypted PDF
- `Aucun texte extractible`: Scanned PDF without OCR
- `Contenu trop court`: Below minimum threshold

#### `scrape_all.py` - Orchestration & Persistence Layer
**Purpose:** Main entry point coordinating all modules with state management

**Core Responsibilities:**
1. **Link Discovery:** Call `get_all_links()` to enumerate targets
2. **State Management:**
   - Load existing documents from `raw_texts.json`
   - Load hash registry from `scrape_registry.json`
3. **Extraction Loop:**
   - Pre-check Content-Type with HEAD request
   - Route to HTML or PDF extractor
   - Hash comparison for incremental updates
4. **Error Handling:** Categorize and log all extraction failures
5. **Persistence:**
   - Periodic checkpoints every 10 documents
   - Final save with statistics
   - Error detail export to `error_details.json`

**State Transitions:**
```python
# For each URL
if not in registry:
    → NEW: Extract + Save + Hash
elif hash_changed:
    → UPDATED: Extract + Save + Update Hash
elif hash_unchanged and not force:
    → UNCHANGED: Reuse existing
```

**Statistics Tracking:**
```python
stats = {
    'total': len(links),
    'new': 0,
    'updated': 0,
    'unchanged': 0,
    'failed': 0,
    'duration': 0.0
}
```

---

## Data Structure

### Output Format: `raw_texts.json`

The pipeline produces a **JSON array of document objects**, each following this schema:

```json
[
  {
    "content": "Full extracted text content...",
    "metadata": {
      "source": "https://www.uqac.ca/mgestion/chapitre-5/section-2/politique-voyage.html",
      "title": "Politique de Remboursement des Frais de Déplacement",
      "hierarchy": "Chapitre 5 > Section 2 > Politiques",
      "last_updated": "15 septembre 2024",
      "doc_type": "Politique",
      "page_count": 12  // PDF only
    }
  }
]
```

### Field Specifications

#### `content` (string)
- **Description:** Full text content extracted from the document
- **Processing:**
  - Whitespace normalized (multiple spaces → single space)
  - Line breaks preserved for paragraph structure
  - HTML tables converted to Markdown format
  - PDF ligatures corrected (fi, fl, etc.)
  - Control characters removed
- **Validation:** Minimum length 100 characters

#### `metadata` Object

##### `source` (string, required)
- **Description:** Canonical URL of the source document
- **Format:** Fully qualified URL
- **Example:** `https://www.uqac.ca/mgestion/chapitre-5/politique-voyage.html`

##### `title` (string, required)
- **Description:** Document title
- **Extraction Priority (HTML):**
  1. `<h1>` tag content
  2. `<title>` tag (cleaned of "- UQAC" suffixes)
  3. `<h2>` tag content
  4. Fallback: "Sans titre"
- **Extraction Priority (PDF):**
  1. PDF metadata `/Title` field
  2. First line of first page (if ≥ 3 words)
  3. Filename (cleaned)
  4. Fallback: "Document PDF"

##### `hierarchy` (string, optional)
- **Description:** Breadcrumb navigation path
- **Extraction (HTML):** From breadcrumb navigation elements (aria-label, class patterns)
- **Extraction (PDF):** Inferred from URL structure (e.g., "Chapitre 5 > Section 2")
- **Example:** `"Ressources Humaines > Politiques > Congés"`

##### `last_updated` (string, optional)
- **Description:** Last modification date
- **Format:** French date format (e.g., "15 septembre 2024")
- **Extraction:**
  - HTML: Regex search for date patterns in text
  - PDF: `/ModDate` or `/CreationDate` metadata
- **Patterns Matched:**
  ```python
  "mis à jour: 15 septembre 2024"
  "15 septembre 2024"
  "septembre 2024"
  "2024-09-15"
  ```

##### `doc_type` (string, required)
- **Description:** Classification of document type
- **Values:**
  - `"Politique"` - Policy documents
  - `"Règlement"` - Regulations
  - `"Formulaire"` - Forms
  - `"Procédure"` - Procedures
  - `"Guide"` - Guides (PDF only)
  - `"Document"` - Default/unclassified
- **Detection:** Keyword matching in URL and title (case-insensitive)

##### `page_count` (integer, PDF only)
- **Description:** Number of pages in the PDF document
- **Example:** `12`

### Auxiliary Files

#### `scrape_registry.json` - Hash Registry
Maps URLs to content MD5 hashes for incremental scraping:

```json
{
  "https://www.uqac.ca/mgestion/politique-1.html": "5d41402abc4b2a76b9719d911017c592",
  "https://www.uqac.ca/mgestion/formulaire.pdf": "7d793037a0760186574b0282f2f435e7"
}
```

#### `error_details.json` - Error Diagnostic Report
Structured error information for failed extractions:

```json
{
  "https://www.uqac.ca/mgestion/broken.html": {
    "category": "Structure HTML - Conteneur introuvable",
    "reason": "Aucun conteneur principal trouvé (body absent)",
    "error_details": {
      "attempted_selectors": ["div avec classe content", "main", "article"],
      "body_present": true
    }
  }
}
```

---

## Error Handling

### Exception Hierarchy

The pipeline defines **custom exception classes** for structured error reporting:

#### `HTMLExtractionError`
```python
class HTMLExtractionError(Exception):
    def __init__(self, url, reason, details=None):
        self.url = url           # Source URL
        self.reason = reason     # Human-readable error message
        self.details = details   # Dict of diagnostic information
```

**Common Reasons:**
- `"Content-Type invalide"` - Server returned non-HTML content
- `"Aucun conteneur principal trouvé"` - DOM structure unrecognized
- `"Contenu trop court"` - Text length below minimum threshold
- `"Contenu fantôme détecté: Page non trouvée"` - 404 error page

**Example Details:**
```python
{
    'content_type': 'application/json',
    'status_code': 200,
    'preview': 'First 100 chars of content...'
}
```

#### `PDFExtractionError`
```python
class PDFExtractionError(Exception):
    # Identical structure to HTMLExtractionError
```

**Common Reasons:**
- `"Fichier PDF trop petit"` - File size < 100 bytes
- `"Impossible de lire le PDF"` - Corrupted or encrypted file
- `"Aucun texte extractible"` - Scanned PDF without OCR layer
- `"Contenu trop court"` - Extracted text below threshold

**Example Details:**
```python
{
    'total_pages': 8,
    'pages_with_errors': 2,
    'possible_cause': 'PDF scanné sans OCR',
    'avg_per_page': 12.5
}
```

### Error Categorization System

The `categorize_error()` function maps exceptions to **actionable categories** for analysis:

```python
def categorize_error(exception) -> str:
    if isinstance(exception, HTMLExtractionError):
        if "Content-Type invalide" in exception.reason:
            return "Content-Type invalide (HTML attendu)"
        elif "conteneur" in exception.reason.lower():
            return "Structure HTML - Conteneur introuvable"
        # ... more patterns
    elif isinstance(exception, PDFExtractionError):
        if "Aucun texte" in exception.reason:
            return "PDF sans texte (scan?)"
        # ... more patterns
```

### Error Reporting

At completion, the pipeline generates a **comprehensive error analysis report**:

```
========================================
📋 VENTILATION DES ÉCHECS PAR CATÉGORIE
========================================
  15 (45.5%) - Structure HTML - Conteneur introuvable
   8 (24.2%) - Contenu trop court
   6 (18.2%) - PDF sans texte (scan?)
   4 (12.1%) - Erreur HTTP 404

========================================
🔍 EXEMPLES D'ERREURS PAR CATÉGORIE
========================================

📌 Structure HTML - Conteneur introuvable:
   URL: https://www.uqac.ca/mgestion/page-1.html
   Détails: {'attempted_selectors': ['main', 'article']}

========================================
💡 RECOMMANDATIONS:
========================================
  🔧 Beaucoup d'erreurs de conteneur HTML:
     → Vérifiez le fichier scrape_debug.log
     → La structure du site a peut-être changé
  
📄 Fichiers de diagnostic:
   - Logs détaillés:    ../data/scrape_debug.log
   - Détails d'erreurs: ../data/error_details.json
```

### Fault Tolerance Strategies

1. **Graceful Degradation:**
   - On extraction failure, preserve old version if available
   - Continue processing remaining URLs instead of halting

2. **Checkpoint Recovery:**
   - Periodic saves every 10 documents
   - Keyboard interrupt (Ctrl+C) triggers immediate checkpoint
   - Restart resumes from last checkpoint

3. **Error Context Preservation:**
   - Full traceback logged to `scrape_debug.log`
   - Structured details saved to `error_details.json`
   - Human-readable summary in console output

---

## Reproducibility

### Command-Line Interface

The pipeline provides a **comprehensive CLI** for various execution modes:

```bash
python scrape_all.py [OPTIONS]
```

#### Available Arguments

##### `--force`
**Purpose:** Force re-extraction of all documents, bypassing hash comparison

**Use Cases:**
- Complete data refresh after schema changes
- Re-extraction with updated parsing logic
- Validation of incremental scraping accuracy

**Behavior:**
```python
if old_hash == new_hash and not force_update:
    print("⏭️ Unchanged")  # Normal behavior
else:
    print("🔄 Updated")     # With --force
```

**Example:**
```bash
python scrape_all.py --force
```

##### `--verify`
**Purpose:** Validate integrity of existing `raw_texts.json` without scraping

**Checks Performed:**
- JSON syntax validation
- Document count and statistics
- Duplicate URL detection
- Document type distribution
- Content volume analysis

**Output:**
```
🔍 VÉRIFICATION DE L'INTÉGRITÉ DES DONNÉES

✅ 142 documents chargés

📊 Répartition par type:
   Document: 45
   Politique: 38
   Règlement: 32
   Formulaire: 27

📈 Volume total: 1,248,573 caractères
📈 Moyenne: 8,791 caractères/document
```

**Example:**
```bash
python scrape_all.py --verify
```

##### `--no-ssl-verify`
**Purpose:** Disable SSL certificate verification (use with caution)

**Use Cases:**
- Corporate networks with SSL interception
- Development environments with self-signed certificates
- Temporary workaround for certificate issues

**Security Note:** Only use in trusted environments

**Example:**
```bash
python scrape_all.py --no-ssl-verify
```

### Execution Modes

#### 1. Initial Full Scrape
```bash
python scrape_all.py
```
- Discovers all links via `get_all_links()`
- Extracts content from all URLs
- Builds complete `raw_texts.json`
- Creates `scrape_registry.json` with MD5 hashes
- Duration: ~10-30 minutes depending on corpus size

#### 2. Incremental Update
```bash
python scrape_all.py
```
(Same command, different behavior when registry exists)
- Loads existing registry and documents
- Only re-extracts documents with changed hashes
- Preserves bandwidth and processing time
- Duration: ~1-5 minutes for typical update rate

#### 3. Force Full Refresh
```bash
python scrape_all.py --force
```
- Bypasses all hash checks
- Re-extracts every document
- Updates all hashes in registry
- Use after code changes or schema updates

#### 4. Data Integrity Check
```bash
python scrape_all.py --verify
```
- No network requests
- Analyzes existing `raw_texts.json`
- Reports duplicates, statistics, and anomalies
- Quick validation (~1 second)

### Output Files

All output files are written to `../data/`:

```
../data/
├── raw_texts.json           # Main output: Extracted documents
├── scrape_registry.json     # MD5 hash registry for incremental updates
├── scrape_debug.log         # Debug-level logs (all events)
├── scrape_log.txt           # Info-level logs (deprecated)
├── error_details.json       # Structured error diagnostics
└── (checkpoint files)       # Temporary checkpoints during execution
```

### Performance Characteristics

**Typical Execution Metrics (150 documents):**
- **Full Scrape:** 15-25 minutes
- **Incremental (10% changed):** 2-4 minutes
- **Force Refresh:** 20-30 minutes
- **Verification:** < 1 second

**Resource Usage:**
- **Network:** ~5-15 MB download (full scrape)
- **Memory:** ~50-100 MB peak (in-memory document store)
- **Disk:** ~2-5 MB output files

**Concurrency:** Single-threaded sequential processing (by design for debugging)

### Debugging Workflow

1. **Run with Default Settings:**
   ```bash
   python scrape_all.py
   ```

2. **Check Summary Statistics:**
   - Review console output for failure counts
   - Note categorized error types

3. **Examine Detailed Logs:**
   ```bash
   tail -n 100 ../data/scrape_debug.log
   ```
   - Search for specific URLs
   - Review HTML structure inspection logs

4. **Analyze Error Details:**
   ```bash
   cat ../data/error_details.json | jq '.[] | select(.category=="Structure HTML - Conteneur introuvable")'
   ```
   - Filter by error category
   - Identify patterns in failed extractions

5. **Validate Output:**
   ```bash
   python scrape_all.py --verify
   ```
   - Ensure no duplicate URLs
   - Confirm expected document count

---

## Dependencies

### Required Python Packages

```python
# Core Dependencies
beautifulsoup4>=4.12.0    # HTML parsing
lxml>=5.0.0               # XML/HTML parser backend
requests>=2.31.0          # HTTP client
urllib3>=2.0.0            # HTTP library (requests dependency)
pypdf>=4.0.0              # PDF text extraction

# Standard Library (no installation needed)
hashlib                   # MD5 hashing
json                      # JSON serialization
logging                   # Diagnostic logging
argparse                  # CLI argument parsing
traceback                 # Error stack traces
tempfile                  # Temporary PDF storage
```

**Installation:**
```bash
pip install beautifulsoup4 lxml requests pypdf
```

---

## Best Practices

### Production Deployment

1. **Scheduled Execution:**
   - Run incrementally (weekly/monthly) to capture updates
   - Use `--verify` before critical downstream processing

2. **Monitoring:**
   - Set up alerts for high failure rates (> 10%)
   - Track execution time trends to detect performance degradation

3. **Error Analysis:**
   - Regularly review `error_details.json` for new error patterns
   - Investigate "Structure HTML - Conteneur introuvable" spikes (indicates site redesign)

4. **Version Control:**
   - Commit `scrape_registry.json` alongside code
   - Track `raw_texts.json` with Git LFS if repository allows

### Maintenance

1. **After Site Redesign:**
   - Review `scrape_debug.log` for failed selector patterns
   - Update `extract_html_content()` container selection logic
   - Run `--force` to rebuild entire corpus

2. **After Pipeline Updates:**
   - Use `--force` to ensure consistency
   - Run `--verify` to validate output integrity
   - Compare document counts before/after

3. **Error Triage:**
   - Priority 1: "Structure HTML - Conteneur introuvable" → Update selectors
   - Priority 2: "PDF sans texte" → Manual OCR or exclusion
   - Priority 3: "Contenu trop court" → Lower threshold or manual review

**Last Updated:** January 2025  
**Version:** 2.0  
