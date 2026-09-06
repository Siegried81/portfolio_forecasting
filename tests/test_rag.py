"""
Unit tests for src/rag.py (TF-IDF retrieval over the news/filings corpus).

Priority: this module had NO test coverage before this pass, despite being
the actual retrieval step behind the chatbot's "answers about news" path in
ai_features.answer_portfolio_question. Focus on the contract callers rely on
(retrieve() never raises, ranks by relevance, drops zero-similarity noise)
rather than TF-IDF internals, which sklearn already tests upstream.
"""
import pytest

from src.rag import Chunk, build_chunks, format_retrieved_chunks, retrieve


# ---------------------------------------------------------------------------
# build_chunks
# ---------------------------------------------------------------------------

def test_build_chunks_concatenates_title_and_description():
    articles = [{"title": "AAPL beats estimates", "description": "Strong iPhone sales.",
                 "source": "Reuters", "provider": "NewsAPI", "ticker": "AAPL", "url": "http://x"}]
    chunks = build_chunks(articles)
    assert len(chunks) == 1
    assert chunks[0].text == "AAPL beats estimates. Strong iPhone sales."
    assert chunks[0].ticker == "AAPL"


def test_build_chunks_skips_articles_without_a_title():
    articles = [{"title": "", "description": "no title here"}, {"title": "Real headline", "description": ""}]
    chunks = build_chunks(articles)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Real headline")


def test_build_chunks_empty_input_returns_empty_list():
    assert build_chunks([]) == []


# ---------------------------------------------------------------------------
# retrieve — the actual TF-IDF ranking
# ---------------------------------------------------------------------------

def _chunk(text: str, ticker: str = "AAPL") -> Chunk:
    return Chunk(text=text, source="Reuters", provider="NewsAPI", ticker=ticker, url="http://x")


def test_retrieve_ranks_the_most_relevant_chunk_first():
    chunks = [
        _chunk("Quarterly earnings beat analyst expectations on strong iPhone demand"),
        _chunk("The weather in Cupertino was sunny this week"),
        _chunk("Apple announces new retail store opening in Tokyo"),
    ]
    results = retrieve("earnings iPhone demand", chunks, top_k=2)
    assert len(results) >= 1
    assert results[0].text.startswith("Quarterly earnings")


def test_retrieve_filters_out_zero_similarity_chunks():
    # Query shares no vocabulary at all with either chunk -> nothing should be relevant.
    chunks = [_chunk("Quarterly earnings beat expectations"), _chunk("New product launch event")]
    results = retrieve("xyzxyz nonword qqqqq", chunks, top_k=5)
    assert results == []


def test_retrieve_empty_chunks_returns_empty_list():
    assert retrieve("any query", [], top_k=4) == []


def test_retrieve_empty_query_returns_empty_list():
    chunks = [_chunk("Some headline about earnings")]
    assert retrieve("   ", chunks, top_k=4) == []


def test_retrieve_respects_top_k():
    chunks = [_chunk(f"Earnings report number {i} beats expectations") for i in range(10)]
    results = retrieve("earnings report expectations", chunks, top_k=3)
    assert len(results) <= 3


def test_retrieve_never_raises_on_stopword_only_corpus():
    # Every chunk + the query reduce to nothing but stopwords after cleaning —
    # TfidfVectorizer raises ValueError internally; retrieve() must swallow it.
    chunks = [_chunk("the a an of")]
    assert retrieve("the a an", chunks, top_k=4) == []


# ---------------------------------------------------------------------------
# format_retrieved_chunks
# ---------------------------------------------------------------------------

def test_format_retrieved_chunks_empty_list_returns_empty_string():
    assert format_retrieved_chunks([]) == ""


def test_format_retrieved_chunks_tags_provider_and_source():
    chunks = [_chunk("Earnings beat expectations")]
    formatted = format_retrieved_chunks(chunks)
    assert "[NewsAPI/Reuters, AAPL]" in formatted
    assert "Earnings beat expectations" in formatted