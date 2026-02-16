# app.py
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# --- Robust imports (peu importe d'où tu lances streamlit) ---
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rag_engine import RAGEngineV2  # noqa: E402


# Ajoute cet import en haut si tu ne l'as pas déjà
import ollama 

def get_ollama_models():
    """Récupère la liste des modèles disponibles localement via Ollama."""
    try:
        models_info = ollama.list()
        # ollama.list() renvoie un dictionnaire avec une clé 'models'
        # On extrait juste les noms (ex: 'llama3.2:latest')
        return [m['model'] for m in models_info['models']]
    except Exception as e:
        # Si Ollama est éteint ou erreur, on renvoie une liste de secours
        print(f"Erreur connexion Ollama: {e}")
        return ["mistral:latest", "llama3.2:latest"]
    
# -----------------------------
# Helpers UI
# -----------------------------
def _guess_url(meta: Dict[str, Any]) -> Optional[str]:
    # essaie plusieurs clés possibles
    for k in ("url", "source_url", "source", "link", "href"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip().startswith("http"):
            return v.strip()
    return None


def _render_sources(contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return ""

    items: List[str] = []
    seen = set()

    for c in contexts[:10]:  # on prend un peu plus, puis on déduplique
        meta = c.get("metadata", {}) or {}
        title = str(meta.get("title", "Document")).strip()
        doc_type = str(meta.get("doc_type", "N/A")).strip()
        chapitre = str(meta.get("chapitre", "N/A")).strip()
        url = _guess_url(meta)

        key = (title, doc_type, chapitre, url)
        if key in seen:
            continue
        seen.add(key)

        left = (
            f"{html.escape(title)} "
            f"<span style='opacity:.75'>({html.escape(doc_type)})</span>"
            f" — {html.escape(chapitre)}"
        )

        if url:
            url_e = html.escape(url)
            items.append(f"<li>{left} : <a href='{url_e}' target='_blank'>{url_e}</a></li>")
        else:
            items.append(f"<li>{left}</li>")

        if len(items) >= 6:
            break

    return (
        "<div class='sources'>"
        "<div class='sources-title'>Sources</div>"
        "<ul class='sources-list'>"
        + "".join(items)
        + "</ul>"
        "</div>"
    )


def _render_message(role: str, content: str, contexts: Optional[List[Dict[str, Any]]] = None) -> str:
    # role: "user" ou "assistant"
    role_class = "user" if role == "user" else "ai"

    # ✅ On échappe uniquement le TEXTE (jamais la structure HTML)
    safe_text = html.escape(content).replace("\n", "<br>")

    sources_html = ""
    if role == "assistant" and contexts:
        sources_html = _render_sources(contexts)

    # ⚠️ IMPORTANT : pas d'indentation ici (sinon Markdown => code block)
    return (
        f"<div class='msg-row {role_class}'>"
        f"  <div class='bubble {role_class}'>"
        f"    {safe_text}"
        f"    {sources_html}"
        f"  </div>"
        f"</div>"
    )


# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(page_title="Radisson — ChatBot UQAC", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
/* Empêche le scroll de la page (scroll uniquement dans la zone chat) */
html, body { height: 100%; overflow: hidden; }
.stApp { height: 100vh; overflow: hidden; }

/* garde le bouton pour ré-ouvrir la sidebar si tu la replis */
[data-testid="collapsedControl"] { display: flex !important; }

/* Container principal plein écran */
div.block-container {
  padding-top: 1.2rem;
  padding-bottom: 0.6rem;
  height: 100vh;
  overflow: hidden;
}

/* Zone chat scrollable */
.chat-scroll {
  height: calc(100vh - 9.5rem);
  overflow-y: auto;
  padding-right: .5rem;
  padding-bottom: .5rem;
}

/* Messages */
.msg-row { display: flex; width: 100%; margin: .45rem 0; }
.msg-row.user { justify-content: flex-end; }
.msg-row.ai { justify-content: flex-start; }

.bubble {
  max-width: 72%;
  padding: .85rem 1rem;
  border-radius: 18px;
  line-height: 1.45;
  font-size: 1rem;
  border: 1px solid rgba(0,0,0,.08);
  color: inherit;
  word-wrap: break-word;
  overflow-wrap: anywhere;
}

.bubble.user {
  background: rgba(0,0,0,.06);
  border-top-right-radius: 8px;
}

.bubble.ai {
  background: rgba(0,0,0,.025);
  border-top-left-radius: 8px;
}

/* Sources sous la réponse */
.sources {
  margin-top: .7rem;
  padding-top: .55rem;
  border-top: 1px dashed rgba(0,0,0,.18);
}
.sources-title { font-weight: 600; font-size: .9rem; opacity: .85; margin-bottom: .25rem; }
.sources-list { margin: 0; padding-left: 1.1rem; }
.sources-list li { margin: .2rem 0; font-size: .9rem; opacity: .9; }
.sources a { text-decoration: none; }
.sources a:hover { text-decoration: underline; }

/* Petit polish */
h1, h2, h3 { margin-bottom: .4rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = (THIS_DIR.parent / "data").resolve()

# ✅ Changement demandé : si la base n'est pas générée, on affiche un message clair et on stop
REQUIRED_FILES = [DATA_DIR / "faiss.index", DATA_DIR / "chunks.json"]
missing = [p for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error("Base de connaissances non générée.")
    st.markdown(
        """
### Avant d'utiliser le chatbot
Les fichiers nécessaires n'ont pas été trouvés dans `data/`.

1. Génère la base de connaissances (scraping → processing → embeddings → FAISS) :
```bash
python rag/build_knowledge_base.py
```

2. Puis relance l'application :
```bash
streamlit run app.py
```

> Si tu utilises Docker : génère d'abord la base (sur ta machine, pour remplir `./data`), puis lance `docker compose up`.
"""
    )
    st.stop()

# -----------------------------
# Engine init (par session)
# -----------------------------
# 1. On récupère la liste des modèles DISPONIBLES immédiatement
available_models = get_ollama_models()

# 2. On choisit un modèle par défaut sécurisé
# Si llama3.2 est là, on le prend. Sinon, on prend le tout premier de la liste.
if "llama3.2:latest" in available_models:
    starting_model = "llama3.2:latest"
else:
    starting_model = available_models[0] if available_models else "mistral:latest"

# 3. On initialise l'uniquement si ce n'est pas déjà fait
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngineV2(
        data_dir=str(DATA_DIR),
        retrieval_method="hybrid",
        max_history=5,
        use_ollama=True,
        ollama_model=starting_model, # <--- On utilise le modèle détecté
    )

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = starting_model

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, Any]] = []

engine: RAGEngineV2 = st.session_state.engine

# -----------------------------
# Sidebar (réglages)
# -----------------------------
with st.sidebar:
    st.title("⚙️ Réglages")

    retrieval_method = st.selectbox("Méthode de retrieval", ["hybrid", "dense", "bm25"], index=0)
    top_k = st.slider("Top-k documents", min_value=2, max_value=10, value=5, step=1)

    engine.retrieval_method = retrieval_method

    st.divider()
    st.caption("LLM local (Ollama)")
    
    default_index = 0
    if "llama3.2:latest" in available_models:
        default_index = available_models.index("llama3.2:latest")
    
    st.selectbox(
        "Modèle Ollama",
        options=available_models,
        index=default_index,
        key="ollama_model",
        help="Sélectionne un modèle installé sur ta machine via 'ollama pull <nom>'",
    )
    
    if not available_models or available_models == ["mistral:latest", "llama3.2:latest"]:
        st.warning("⚠️ Impossible de lister les modèles (Ollama est lancé ?). Mode secours activé.")

    st.divider()
    st.caption("Backend")
    st.write(f"**Data dir :** `{DATA_DIR}`")

    index_ok = (DATA_DIR / "faiss.index").exists()
    chunks_ok = (DATA_DIR / "chunks.json").exists()
    st.write(f"**faiss.index :** {'✅' if index_ok else '❌'}")
    st.write(f"**chunks.json :** {'✅' if chunks_ok else '❌'}")

    st.divider()
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        engine.clear_history()
        st.rerun()

engine.ollama_model = st.session_state.ollama_model

# -----------------------------
# Header
# -----------------------------
st.title("Radisson — ChatBot UQAC")

# -----------------------------
# Render chat (HTML rendu correctement)
# -----------------------------
chat_html = "".join(
    _render_message(m["role"], m["content"], m.get("contexts")) for m in st.session_state.messages
)

st.markdown(f"<div class='chat-scroll'>{chat_html}</div>", unsafe_allow_html=True)

# -----------------------------
# Input + backend call
# -----------------------------
user_text = st.chat_input("Pose ta question sur le guide de gestion…")

if user_text:
    # 1) message user
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 2) backend
    with st.spinner("Radisson réfléchit…"):
        try:
            answer, contexts = engine.ask(user_text, k=top_k)
        except FileNotFoundError as e:
            answer = f"Erreur : {e}"
            contexts = []
        except Exception as e:
            answer = f"Erreur backend : {e}"
            contexts = []

    # 3) message assistant + contexts
    st.session_state.messages.append({"role": "assistant", "content": answer, "contexts": contexts})

    # 4) refresh
    st.rerun()
