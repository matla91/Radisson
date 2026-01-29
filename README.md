# Radisson

<img width="385" height="239" alt="image" src="https://github.com/user-attachments/assets/44f29f36-9085-4ac5-82c6-09ac7c22e2cf" />

In the spirit of **Pierre-Esprit Radisson**, the French explorer who ventured beyond the known maps of the 17th century, this system was built to navigate the unknown.

Where the explorer once crossed the lands of the North and the Saguenay, Radisson explores a different kind of territory: the institutional documentary ecosystem of the Université du Québec à Chicoutimi.

This landscape is composed of policies, regulations, procedures, and administrative frameworks — dense, fragmented, and difficult to traverse. Information exists, but is often buried beneath structure, formal language, and cross-references.

Radisson acts as a **methodical explorer**.  
It maps documents, follows the right paths, and returns with **precise, verifiable, source-grounded answers**.

Radisson does not invent knowledge.  
It uncovers it.

**Core Mission:** Provide factual, verifiable answers to administrative questions by retrieving and synthesizing information from a structured knowledge base, never inventing or extrapolating beyond source documents.

---

## What is Radisson?

Radisson is a document exploration system designed to help users find reliable information inside large institutional knowledge bases.

It is built for environments where documents are numerous, structured, and written in formal language — such as university regulations, administrative manuals, internal policies, or procedural guides.

Instead of manually browsing dozens of documents, users can ask a question in natural language and receive a clear answer based solely on official sources.

Radisson does not rely on general knowledge or assumptions.  
It works exclusively with the documents provided to it.

---

## What problem does it solve?

In institutional contexts, information is rarely missing — it is **hard to access**.

Answers are often:
- scattered across multiple documents  
- hidden behind complex wording  
- difficult to cross-reference  
- time-consuming to verify  

This makes simple questions unnecessarily difficult to resolve.

Radisson simplifies this process by exploring the document base on behalf of the user and returning information that is:
- clear  
- directly supported by official texts  
- easy to verify  

---

## What Radisson focuses on

Radisson was designed with a single priority: **reliability**.

The system aims to:
- reduce time spent searching through documents  
- limit misinterpretation of institutional rules  
- provide answers that can be traced back to their sources  
- support users in administrative or regulatory contexts  

It is not designed to be creative, conversational, or opinion-based.

Its role is simple:  
**help users access institutional knowledge with confidence.**

## 🚀 Getting Started

Radisson works in **two clearly separated phases**:

1. **Knowledge base construction (offline)** — executed once
2. **Interactive usage (runtime)** — via Streamlit interface

This separation ensures reliability, performance, and reproducibility.

---

## ✅ Prerequisites

Before running Radisson, make sure you have the following installed:

* **Python 3.10+**
* **Docker** and **Docker Compose**
* **Ollama** (local LLM server)

Install Ollama:
[https://ollama.com](https://ollama.com)

Then download a model (example):

```bash
ollama pull mistral
```

---

## 📥 Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/radisson.git
cd radisson
```

---

## 🧱 Step 2 — Build the knowledge base (mandatory)

Radisson does **not** include pre-generated data.

The document base must be constructed locally.
This step performs the complete offline pipeline:

* document scraping
* text cleaning and normalization
* chunking
* embedding generation
* FAISS vector index creation

Run the unified build script:

```bash
python rag/build_knowledge_base.py
```

⏱️ This step may take several minutes depending on the number of documents.

After completion, the following files must exist:

```
data/
├── chunks.json
└── faiss.index
```

These files are generated locally and are **not versioned in Git**.

---

## 💬 Step 3 — Launch the application

Once the knowledge base has been generated, start the application using Docker:

```bash
docker compose up --build
```

Then open your browser at:

```
http://localhost:8501
```

The Streamlit interface will load the existing knowledge base and allow users to ask questions.

---

## ⚠️ Important note

The Streamlit application **does not perform scraping or embedding generation**.

If the required files are missing, the interface will display an explicit message indicating that the knowledge base must be generated first.

This design reflects real-world Retrieval-Augmented Generation (RAG) systems, where:

* heavy data processing is performed offline
* runtime inference remains fast and stable

---

## 🧠 Execution summary

```bash
# 1. Clone the project
git clone https://github.com/your-username/radisson.git
cd radisson

# 2. Build knowledge base (once)
python rag/build_knowledge_base.py

# 3. Launch application
docker compose up --build
```

Radisson is now ready to explore institutional knowledge with reliable, source-grounded answers.

**System Version:** 2.0  
**Last Updated:** January 2025  
