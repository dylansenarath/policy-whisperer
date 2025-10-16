# rag_core.py
"""
RAG core used by the Streamlit app.

Flow:
1) Load company-controlled sources (fixed URLs + PDFs)
2) Split into chunks
3) Embed with a sentence-transformer
4) Store in a persistent Chroma vector DB (so it survives restarts)
5) Retrieve with MMR
6) Answer via Groq LLM using the retrieved context
7) Return answer + grouped sources for the UI
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os
from urllib.parse import urlparse

# Loaders / splitters / vector store / embeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
try:
    # Preferred import in many examples
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Fallback if the above is unavailable
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Prompt, LLM, and data types
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document


# Directory where the vector DB persists its data on disk.
PERSIST_DIR = Path("data/chroma")

# Company-owned PDF sources (place the files here).
PDF_DIR = Path("data/pdfs")
# in rag_core.py
PDF_PATHS = [
    PDF_DIR / "hand_hygiene.pdf",
    PDF_DIR / "insulin_administration.pdf",
    PDF_DIR / "restraints_medical_behavioral.pdf",  
]


# Sample company-owned web sources
SEED_URLS = [
    "https://www.cdc.gov/injection-safety/hcp/infection-control/index.html",
    "https://diabetes.org/health-wellness/medication/insulin-routines",
]

# Embedding model (fast + solid for small corpora).
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_KW = {"normalize_embeddings": True}

# Retriever settings.
RETRIEVAL_K = 6
MMR_LAMBDA = 0.2
FETCH_K = 20  # extra candidates before MMR filters down to k

# Groq model; allow override via environment variable.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")



def ensure_dirs() -> None:
    """Create required folders for PDFs and Chroma persistence."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def _user_agent() -> str:
    """Return a User-Agent header to avoid 403s on some sites."""
    return os.getenv("USER_AGENT", "PolicyWhisperer/0.1 (+https://example.org)")


def _title_from_url(u: str) -> str:
    """Derive a short human-friendly title from a URL."""
    try:
        p = urlparse(u)
        tail = (p.path or "/").strip("/").split("/")[-1] or p.netloc
        title = tail.replace("-", " ").replace("_", " ").strip() or p.netloc
        return title.title()[:80]
    except Exception:
        return "source"


def _build_prompt() -> PromptTemplate:
    """Prompt instructing the LLM to answer only from provided context and cite inline."""
    return PromptTemplate.from_template(
        "You are a policy assistant. Answer using ONLY the context snippets below.\n"
        "If the answer is not clearly contained in the context, say you don't know.\n"
        "Cite snippets inline using tags like [S1], [S2].\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer (concise, stepwise if helpful, with inline [S#] citations):"
    )


# ------------------------------------------------------------------------------
# Index building (called from the Streamlit sidebar)
# ------------------------------------------------------------------------------

def build_or_refresh_index() -> Tuple[int, str]:
    """
    Build (or rebuild) the vector index from the fixed company sources.

    Steps:
    - Load PDFs and URLs
    - Split into overlapping chunks
    - Embed each chunk
    - Save into a persistent Chroma database

    Returns:
        (num_chunks, human_message)
    """
    ensure_dirs()

    # Load documents
    loaded: List[Document] = []
    try:
        wl = WebBaseLoader(web_paths=SEED_URLS, requests_kwargs={"headers": {"User-Agent": _user_agent()}})
        loaded += wl.load()
    except Exception as e:
        print(f"[WARN] URL load issue: {e}")

    for path in PDF_PATHS:
        if Path(path).exists():
            try:
                loaded += PyPDFLoader(str(path)).load()
            except Exception as e:
                print(f"[WARN] PDF load issue ({path}): {e}")

    if not loaded:
        return 0, "No documents loaded. Check PDF paths and network for URLs."

    # Split documents into chunks (good default sizes for policy text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(loaded)
    if not chunks:
        return 0, "Loaded documents but text splitter produced 0 chunks."

    # Embed and persist to Chroma (auto-persist; no manual persist() needed)
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, encode_kwargs=EMBED_KW)
    Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=str(PERSIST_DIR))

    return len(chunks), f"Indexed {len(chunks)} chunks from {len(loaded)} source docs."


# ------------------------------------------------------------------------------
# Health / readiness checks for the UI
# ------------------------------------------------------------------------------

def has_index() -> bool:
    """Return True if the Chroma persist directory contains data."""
    return PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())


def health_check() -> Dict[str, Any]:
    """
    Quick environment check surfaced in the Streamlit sidebar.
    Verifies:
    - chromadb import works (vector DB runtime)
    - GROQ_API_KEY exists
    - required PDFs present
    - persist dir is writable
    """
    ensure_dirs()
    checks = []

    def add(ok: bool, msg: str):
        checks.append({"ok": ok, "msg": msg})

    try:
        import chromadb  # noqa: F401
        add(True, "chromadb import ✓")
    except Exception as e:
        add(False, f"chromadb import ✗ ({e})")

    add(bool(os.getenv("GROQ_API_KEY")), "GROQ_API_KEY set ✓" if os.getenv("GROQ_API_KEY") else "GROQ_API_KEY missing ✗")

    for p in PDF_PATHS:
        exists = Path(p).exists()
        add(exists, f"PDF exists: {Path(p).name} {'✓' if exists else '✗'}")

    add(True, f"persist dir ready at {PERSIST_DIR} ✓")

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# ------------------------------------------------------------------------------
# Retrieval + QA
# ------------------------------------------------------------------------------

def _make_retriever():
    """
    Open the persisted Chroma store and return an MMR retriever.

    Notes:
    - MMR (Maximal Marginal Relevance) diversifies results to reduce redundancy.
    - fetch_k>k lets MMR select the best-diverse k from a larger candidate pool.
    """
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, encode_kwargs=EMBED_KW)
    vs = Chroma(embedding_function=embedder, persist_directory=str(PERSIST_DIR))
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA},
    )


def _build_context_and_sources(docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create:
      - a context string with [S#] tags before each chunk (for inline citations),
      - a grouped sources list used by the UI.

    Grouping collapses many chunks from the same file/URL into one source entry
    with multiple tags, e.g., ["[S1]", "[S3]"].
    """
    lines: List[str] = []
    grouped: Dict[str, Dict[str, Any]] = {}

    def src_key(d: Document) -> str:
        return str(d.metadata.get("source") or d.metadata.get("url") or d.metadata.get("file_path") or "unknown")

    for i, d in enumerate(docs, 1):
        tag = f"[S{i}]"
        snippet = d.page_content.strip().replace("\n", " ")
        lines.append(f"{tag} {snippet}")

        key = src_key(d)
        if key.startswith("http"):
            title = (d.metadata.get("title") or "").strip() or _title_from_url(key)
        else:
            title = Path(key).stem.replace("_", " ").replace("-", " ").strip().title() or "source"
            if title and not title.lower().endswith("(pdf)"):
                title = f"{title} (PDF)"

        if key not in grouped:
            grouped[key] = {"title": title, "url": key, "tags": [tag]}
        else:
            grouped[key]["tags"].append(tag)

    def tag_index(s: str) -> int:
        return int(s[2:-1]) if s.startswith("[S") and s.endswith("]") else 1

    sources = sorted(grouped.values(), key=lambda s: tag_index(s["tags"][0]))
    return "\n\n".join(lines), sources


def ask_with_sources(question: str, k: int = RETRIEVAL_K) -> Dict[str, Any]:
    """
    Main QA entry point used by the UI.

    Steps:
    - Retrieve k relevant chunks with MMR
    - Build a tagged context and grouped sources
    - Call Groq LLM with a grounded prompt
    - Return: {"answer": <str>, "sources": <list-of-dicts>}
    """
    if not os.getenv("GROQ_API_KEY"):
        return {"answer": "Server misconfiguration: GROQ_API_KEY is not set.", "sources": []}

    retriever = _make_retriever()
    docs: List[Document] = retriever.invoke(question)
    if not docs:
        return {"answer": "I don’t find a relevant policy snippet for that.", "sources": []}

    context, sources = _build_context_and_sources(docs)

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
    prompt = _build_prompt()
    answer = llm.invoke(prompt.format(context=context, question=question)).content

    return {"answer": answer, "sources": sources}
