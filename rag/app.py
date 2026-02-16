# rag/app.py
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# --- Robust imports (no matter where you run Streamlit from) ---
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rag_engine import RAGEngineV2  # noqa: E402


# -----------------------------
# UI helpers
# -----------------------------
def _guess_url(meta: Dict[str, Any]) -> Optional[str]:
    # Try multiple possible keys
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

    for c in contexts[:10]:  # Take a bit more, then deduplicate
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
        + "".join(items) +
        "</ul>"
        "</div>"
    )


def _render_message(role: str, content: str, contexts: Optional[List[Dict[str, Any]]] = None) -> str:
    # role: "user" or "assistant"
    role_class = "user" if role == "user" else "ai"

    # ✅ Escape only the TEXT (never the HTML structure)
    safe_text = html.escape(content).replace("\n", "<br>")

    sources_html = ""
    if role == "assistant" and contexts:
        sources_html = _render_sources(contexts)

    # ⚠️ IMPORTANT: no indentation here (otherwise Markdown => code block)
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
# Paths & Engine init (per session)
# -----------------------------
DATA_DIR = (THIS_DIR.parent / "data").resolve()

if "engine" not in st.session_state:
    st.session_state.engine = RAGEngineV2(
        data_dir=str(DATA_DIR),
        retrieval_method="hybrid",
        max_history=5,
        use_ollama=True,
        ollama_model="mistral:latest",
    )

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "mistral:latest"

if "messages" not in st.session_state:
    # messages: [{"role": "user"/"assistant", "content": "...", "contexts": [...]}]
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

    st.text_input(
        "Modèle Ollama",
        key="ollama_model",
        help="Ex: mistral:latest, llama3, qwen2.5"
    )

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
# Render chat (HTML rendered properly)
# -----------------------------
chat_html = "".join(
    _render_message(m["role"], m["content"], m.get("contexts"))
    for m in st.session_state.messages
)

st.markdown(f"<div class='chat-scroll'>{chat_html}</div>", unsafe_allow_html=True)

# -----------------------------
# Input + backend call
# -----------------------------
user_text = st.chat_input("Pose ta question sur le guide de gestion…")

if user_text:
    # 1) User message
    st.session_state.messages.append({"role": "user", "content": user_text})

    # 2) Backend
    with st.spinner("Radisson réfléchit…"):
        try:
            answer, contexts = engine.ask(user_text, k=top_k)
        except FileNotFoundError as e:
            answer = f"Erreur : {e}"
            contexts = []
        except Exception as e:
            answer = f"Erreur backend : {e}"
            contexts = []

    # 3) Assistant message + contexts
    st.session_state.messages.append({"role": "assistant", "content": answer, "contexts": contexts})

    # 4) Refresh
    st.rerun()
