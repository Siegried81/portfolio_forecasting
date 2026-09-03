"""
Lightweight retrieval-augmented generation (RAG) over the news/filings corpus
collected in the AI Analyst tab — a genuine index-then-retrieve pipeline, kept
deliberately separate from the context-injection approach used for portfolio
metrics elsewhere in `ai_features.py` (that distinction is the point: metrics
are a small, fixed, MUST-be-complete set of numbers where full injection is
correct; news/filings are unstructured, can grow arbitrarily large, and only
the query-relevant subset should reach the prompt — that's what RAG is for).

Why TF-IDF instead of neural embeddings: this corpus is small (a handful of
headlines/filings per session, rebuilt fresh every run — no persistence, no
need for semantic nuance across a large corpus) and the queries are the kind
of keyword-driven questions a user actually types in a chat box ("what's the
news on NVDA earnings"). A sentence-transformers model would add a large,
slow-to-install dependency and a multi-hundred-MB model download for retrieval
quality gains that don't materially matter at this scale. TF-IDF + cosine
similarity is deterministic, fast, dependency-light (scikit-learn is already
pulled in transitively by PyPortfolioOpt's `cvxpy` stack), and sufficient here.
If this corpus grew into hundreds of documents with persistence across
sessions, a real vector store (Chroma/FAISS) with neural embeddings would be
the right upgrade — noted in the README as the natural next step, not built
here because it would be unjustified complexity for this corpus size.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    """One retrievable unit: a news headline/summary or a filing entry."""
    text: str
    source: str    # display name, e.g. "Reuters", "Bloomberg", "SEC EDGAR"
    provider: str  # which fetcher produced it: "NewsAPI" / "Finnhub" / "SEC EDGAR"
    ticker: str
    url: str


def build_chunks(articles: list[dict]) -> list[Chunk]:
    """
    Turn the raw article/filing dicts already collected by news_data.py (via
    ai_features.generate_news_digest) into retrievable chunks. Title + summary
    are concatenated into one chunk per article — granular enough for a small
    corpus like this without the overhead of further splitting.
    """
    return [
        Chunk(
            text=f"{a.get('title', '')}. {a.get('description', '')}".strip(),
            source=a.get("source", "unknown"),
            provider=a.get("provider", "unknown"),
            ticker=a.get("ticker", ""),
            url=a.get("url", ""),
        )
        for a in articles
        if a.get("title")
    ]


def retrieve(query: str, chunks: list[Chunk], top_k: int = 4) -> list[Chunk]:
    """
    Return up to `top_k` chunks most relevant to `query` by TF-IDF cosine
    similarity, filtering out zero-similarity matches (a chunk sharing no
    vocabulary with the query is not relevant, regardless of rank). Returns []
    if there's nothing to retrieve from, or if the query/corpus share no
    vocabulary at all — callers should treat that as "no relevant news found"
    and fall back gracefully, never crash on it.
    """
    if not chunks or not query.strip():
        return []

    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts + [query])
    except ValueError:
        # e.g. every chunk + the query reduce to nothing but stopwords after
        # cleaning — genuinely nothing to retrieve on, not a bug to raise on.
        return []

    query_vector = matrix[-1]
    doc_vectors = matrix[:-1]
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    ranked_indices = sorted(range(len(chunks)), key=lambda i: similarities[i], reverse=True)
    return [chunks[i] for i in ranked_indices[:top_k] if similarities[i] > 0]


def format_retrieved_chunks(chunks: list[Chunk]) -> str:
    """Render retrieved chunks into a text block for prompt injection, tagged
    by provider/source so the LLM (and the README's cross-referencing framing)
    can distinguish primary filings from media coverage."""
    if not chunks:
        return ""
    lines = [f"  - [{c.provider}/{c.source}, {c.ticker}] {c.text}" for c in chunks]
    return "RETRIEVED NEWS/FILINGS CONTEXT (top matches for this question):\n" + "\n".join(lines)