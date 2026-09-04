"""
Unit tests for the news-sentiment cascade in src/news_data.py.

Priority: the FALLBACK behavior, since that's what makes "not available" an
honest signal rather than a silently-wrong neutral score.
"""
import pytest

import src.news_data as news_data
from src.news_data import compute_local_sentiment, get_ticker_sentiment


# ---------------------------------------------------------------------------
# compute_local_sentiment (VADER)
# ---------------------------------------------------------------------------

def test_compute_local_sentiment_none_when_no_articles():
    assert compute_local_sentiment([]) is None


def test_compute_local_sentiment_positive_for_clearly_positive_headlines():
    articles = [
        {"title": "Company smashes earnings expectations, stock soars", "description": "Record profits and a raised outlook."},
        {"title": "Analysts upgrade rating after outstanding quarter", "description": "Strong growth across every segment."},
    ]
    result = compute_local_sentiment(articles)
    assert result is not None
    assert result["provider"].startswith("VADER")
    assert result["score"] > 0.1
    assert result["n_articles"] == 2


def test_compute_local_sentiment_negative_for_clearly_negative_headlines():
    articles = [
        {"title": "Company misses estimates, shares plunge", "description": "Disappointing results and a bleak outlook."},
        {"title": "Regulators launch investigation into fraud allegations", "description": "Shares tumble on the news."},
    ]
    result = compute_local_sentiment(articles)
    assert result is not None
    assert result["score"] < -0.1


# ---------------------------------------------------------------------------
# get_ticker_sentiment — the Finnhub -> VADER -> None cascade
# ---------------------------------------------------------------------------

def test_get_ticker_sentiment_prefers_finnhub_when_available(monkeypatch):
    fake_finnhub_result = {"provider": "Finnhub (aggregated)", "score": 0.4, "bullish_pct": 70, "bearish_pct": 30, "n_articles": 50}
    monkeypatch.setattr(news_data, "fetch_finnhub_sentiment", lambda ticker: fake_finnhub_result)
    result = get_ticker_sentiment("AAPL", articles=[{"title": "irrelevant, should not be used"}])
    assert result == fake_finnhub_result


def test_get_ticker_sentiment_falls_back_to_vader_when_finnhub_unavailable(monkeypatch):
    monkeypatch.setattr(news_data, "fetch_finnhub_sentiment", lambda ticker: None)
    articles = [{"title": "Great news for shareholders as profits jump", "description": "A very strong quarter."}]
    result = get_ticker_sentiment("AAPL", articles)
    assert result is not None
    assert result["provider"].startswith("VADER")


def test_get_ticker_sentiment_none_when_both_sources_have_nothing(monkeypatch):
    monkeypatch.setattr(news_data, "fetch_finnhub_sentiment", lambda ticker: None)
    result = get_ticker_sentiment("AAPL", articles=[])
    assert result is None
