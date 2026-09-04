"""
News & filings acquisition from three complementary sources — deliberately not
just one, so results can be cross-referenced rather than relying on a single
provider's coverage and rate limits:

1. NewsAPI (general news aggregator, 100 req/day free) — broad media coverage.
2. Finnhub (dedicated financial-news API, 60 req/MIN free) — company-news
   endpoint purpose-built for tickers, much more generous quota than NewsAPI.
3. SEC EDGAR full-text search (free, no key) — 8-K filings ("material event"
   disclosures), a genuine PRIMARY source rather than journalism about a
   company. Distinct in kind from the other two, not just a third news feed.

Kept deliberately dumb (no sentiment scoring here): the LLM does the qualitative
read in `ai_features.py`. Every fetch function fails SOFTLY - a missing/expired
key or a down provider must never crash the app, since this is enrichment, not
core to the portfolio maths. `generate_news_digest` in ai_features.py merges
whatever came back from however many of the three sources succeeded.
"""
from __future__ import annotations

import datetime as dt
import logging

import requests
import streamlit as st

from src.config import LLM_SETTINGS

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
SEC_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Free NewsAPI dev tier only returns articles from the last ~30 days.
NEWS_LOOKBACK_DAYS = 14

# SEC requires a descriptive User-Agent identifying the requester (their policy,
# not optional) — a generic/missing one gets throttled or blocked outright.
SEC_USER_AGENT = "portfolio-forecasting-bootcamp-project research@example.com"


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_ticker_headlines(ticker: str, company_name: str | None = None, max_articles: int = 5) -> list[dict]:
    """
    NewsAPI: general-media headlines mentioning a ticker (or its company name,
    which gives much better recall than the raw ticker symbol — "AAPL" barely
    appears in prose, "Apple" does). Returns [] (never raises) on any failure.
    """
    if not LLM_SETTINGS.newsapi_key:
        return []

    query = company_name or ticker
    since = (dt.datetime.utcnow() - dt.timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    params = {
        "q": query, "from": since, "sortBy": "relevancy", "language": "en",
        "pageSize": max_articles, "apiKey": LLM_SETTINGS.newsapi_key,
    }
    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("NewsAPI fetch failed for %s: %s", ticker, exc)
        return []

    articles = payload.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "description": a.get("description", "") or "",
            "source": (a.get("source") or {}).get("name", "unknown"),
            "published_at": a.get("publishedAt", ""),
            "url": a.get("url", ""),
            "provider": "NewsAPI",
        }
        for a in articles
        if a.get("title") and a.get("title") != "[Removed]"
    ]


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_finnhub_news(ticker: str, max_articles: int = 5) -> list[dict]:
    """
    Finnhub: dedicated company-news endpoint (not a general search — it's
    purpose-built per-ticker), 60 req/min free tier. Returns [] on any failure,
    including a missing FINNHUB_API_KEY.
    """
    if not LLM_SETTINGS.finnhub_api_key:
        return []

    since = (dt.datetime.utcnow() - dt.timedelta(days=NEWS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    params = {"symbol": ticker, "from": since, "to": today, "token": LLM_SETTINGS.finnhub_api_key}

    try:
        response = requests.get(FINNHUB_URL, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Finnhub fetch failed for %s: %s", ticker, exc)
        return []

    if not isinstance(articles, list):  # Finnhub returns {} with an error message on bad symbol/key
        return []

    return [
        {
            "title": a.get("headline", ""),
            "description": a.get("summary", "") or "",
            "source": a.get("source", "unknown"),
            "published_at": dt.datetime.fromtimestamp(a["datetime"]).isoformat() if a.get("datetime") else "",
            "url": a.get("url", ""),
            "provider": "Finnhub",
        }
        for a in articles[:max_articles]
        if a.get("headline")
    ]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sec_filings(company_name: str, ticker: str, max_filings: int = 3) -> list[dict]:
    """
    SEC EDGAR full-text search: recent 8-K filings ("material event" disclosures
    — earnings, executive changes, M&A, major agreements) mentioning the company.
    No API key required, but SEC requires a descriptive User-Agent (enforced —
    a missing/generic one gets throttled). This is a PRIMARY regulatory source,
    genuinely different in kind from the news aggregators above, not a third
    copy of the same headlines.

    Returns [] on any failure. Field-shape caveat, same spirit as the Twelve
    Data fundamentals parser: EDGAR's full-text search response shape was
    implemented from public documentation, not verified against a live query in
    this environment (network access here is restricted to package registries)
    — if results look wrong, inspect the raw JSON directly before assuming the
    parser is exhaustive.
    """
    params = {"q": f'"{company_name}"', "forms": "8-K", "dateRange": "custom",
              "startdt": (dt.date.today() - dt.timedelta(days=30)).isoformat(),
              "enddt": dt.date.today().isoformat()}
    headers = {"User-Agent": SEC_USER_AGENT}

    try:
        response = requests.get(SEC_FULLTEXT_SEARCH_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("SEC EDGAR fetch failed for %s: %s", ticker, exc)
        return []

    hits = (payload.get("hits") or {}).get("hits") or []
    results = []
    for hit in hits[:max_filings]:
        source = hit.get("_source", {})
        cik = str(source.get("ciks", [""])[0]).lstrip("0") if source.get("ciks") else ""
        accession = (hit.get("_id", "") or "").split(":")[0]
        results.append({
            "title": f"8-K filing: {source.get('display_names', [company_name])[0]}",
            "description": f"Filed {source.get('file_date', 'unknown date')}",
            "source": "SEC EDGAR",
            "published_at": source.get("file_date", ""),
            "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}" if cik else "https://www.sec.gov/edgar/search/",
            "provider": "SEC EDGAR",
        })
    return results


# ---------------------------------------------------------------------------
# News sentiment — added 2026-09-04. Cascade: Finnhub's own aggregated
# sentiment (PRIMARY, wider corpus than what this app fetches itself) -> VADER
# computed locally from the headlines already fetched above (FALLBACK, free,
# offline, no key) -> None if there's truly nothing to score. Callers must
# show an explicit "not available" message on None rather than silently
# omitting the section — a blank sentiment section reads as "neutral", which
# is a claim, not an absence of one.
# ---------------------------------------------------------------------------

FINNHUB_SENTIMENT_URL = "https://finnhub.io/api/v1/news-sentiment"


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_finnhub_sentiment(ticker: str) -> dict | None:
    """
    Finnhub's own aggregated news-sentiment endpoint: bullish/bearish % across
    a wider article set than the ~5 headlines this app fetches per ticker, plus
    a weekly article-volume ("buzz") figure. Tried FIRST because it's a genuine
    independent aggregation, not a second opinion computed from the same small
    sample already on screen.

    CAVEAT: this specific endpoint has been reported plan-restricted on some
    Finnhub free-tier accounts (inconsistent with their published free-tier
    feature list, similar in spirit to the Twelve Data /statistics situation
    documented elsewhere in this app) — not verified against a live key in
    this environment. Fails soft (returns None) on ANY error, including a 403,
    so `get_ticker_sentiment` below always has the local VADER fallback to
    fall back to rather than surfacing a raw API error to the user.
    """
    if not LLM_SETTINGS.finnhub_api_key:
        return None
    try:
        response = requests.get(
            FINNHUB_SENTIMENT_URL, params={"symbol": ticker, "token": LLM_SETTINGS.finnhub_api_key}, timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Finnhub sentiment fetch failed for %s: %s", ticker, exc)
        return None

    sentiment = payload.get("sentiment") or {}
    bullish = sentiment.get("bullishPercent")
    bearish = sentiment.get("bearishPercent")
    if bullish is None or bearish is None:
        return None  # Finnhub returns an empty/zeroed shape for tickers it has no coverage for

    return {
        "provider": "Finnhub (aggregated)",
        "bullish_pct": bullish * 100,
        "bearish_pct": bearish * 100,
        # Normalised to [-1, 1] so it's directly comparable to VADER's compound
        # score below — callers apply the SAME +/-0.1 threshold either way.
        "score": bullish - bearish,
        "n_articles": (payload.get("buzz") or {}).get("articlesInLastWeek"),
    }


_vader_analyzer = None  # lazy singleton — building the lexicon has a small fixed cost


def _get_vader_analyzer():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def compute_local_sentiment(articles: list[dict]) -> dict | None:
    """
    Fallback sentiment scored locally from the headlines/descriptions already
    fetched above, when Finnhub's own endpoint isn't available. VADER (Valence
    Aware Dictionary and sEntiment Reasoner) was chosen for the same reason
    TF-IDF was chosen over neural embeddings in rag.py: this is a small, cheap,
    deterministic scoring pass on a handful of short texts — not worth burning
    LLM budget or adding latency for. It's free, fully offline (a bundled
    lexicon, no model download, no API key), and specifically tuned for short,
    informal text, which headlines are closer to than to long-form prose.

    Returns None if there's nothing to score (no articles at all) — that case
    is distinct from "computed a neutral score" and callers must not conflate
    the two.
    """
    if not articles:
        return None
    analyzer = _get_vader_analyzer()
    scores = [
        analyzer.polarity_scores(f"{a.get('title', '')}. {a.get('description', '')}")["compound"]
        for a in articles
    ]
    avg_score = sum(scores) / len(scores)
    bullish_count = sum(1 for s in scores if s > 0.05)   # VADER's own documented neutral band
    bearish_count = sum(1 for s in scores if s < -0.05)
    return {
        "provider": "VADER (computed locally from fetched headlines)",
        "bullish_pct": 100 * bullish_count / len(scores),
        "bearish_pct": 100 * bearish_count / len(scores),
        "score": avg_score,
        "n_articles": len(scores),
    }


def get_ticker_sentiment(ticker: str, articles: list[dict]) -> dict | None:
    """
    Single entry point for sentiment on one ticker: Finnhub's aggregated score
    if available, else VADER computed from `articles` (already fetched by the
    caller — no extra network call), else None. Callers show an explicit
    "sentiment not available" message on None, never a silent blank.
    """
    sentiment = fetch_finnhub_sentiment(ticker)
    if sentiment is not None:
        return sentiment
    return compute_local_sentiment(articles)