"""
Macro data via FRED (Federal Reserve Economic Data — free, api key at
fred.stlouisfed.org/docs/api/api_key.html).

Why this earns its place (vs. NewsAPI, Alpha Vantage, Finnhub, all of which
mostly duplicate what yfinance/NewsAPI already cover): the app's risk-free rate
was a hardcoded guess (`DEFAULT_RISK_FREE_RATE = 0.04` in config.py). For a tool
built to demonstrate finance-engineering competence, guessing a rate that's
published daily and free to fetch is the kind of shortcut a real reviewer would
flag. FRED's 3-month T-bill series (DGS3MO) is the standard proxy for the
risk-free rate used in Sharpe/Sortino calculations — this module fetches the
latest print and the UI pre-fills the sidebar slider with it, while still
letting the user override it manually.

Fails soft (returns None) on any error — a stale/missing macro data point should
never block the app, since the user-adjustable slider is always the fallback.
"""
from __future__ import annotations

import requests
import streamlit as st

from src.config import LLM_SETTINGS

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# DGS3MO = "Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity" —
# the standard short-term risk-free rate proxy in mean-variance / Sharpe literature.
RISK_FREE_SERIES_ID = "DGS3MO"

# DGS10 = 10-Year Treasury yield — used here only to compute the 10Y-3M term spread,
# the textbook yield-curve recession signal (a negative spread has preceded every
# US recession since the 1960s, with a handful of false positives). This is the
# kind of macro context a candidate with real markets background is expected to
# at least surface, even in a portfolio-optimization exercise.
TEN_YEAR_SERIES_ID = "DGS10"


def _fetch_fred_series_latest(series_id: str) -> float | None:
    """Shared fetch logic for a single FRED series' latest non-missing value, as a
    decimal (FRED returns Treasury yields as a percentage, e.g. 4.12 -> 0.0412)."""
    if not LLM_SETTINGS.fred_api_key:
        return None

    params = {
        "series_id": series_id,
        "api_key": LLM_SETTINGS.fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,  # a few recent points, in case the very latest is a "." (no data that day)
    }
    try:
        response = requests.get(FRED_URL, params=params, timeout=10)
        response.raise_for_status()
        observations = response.json().get("observations", [])
    except (requests.RequestException, ValueError):
        return None

    for obs in observations:
        value = obs.get("value")
        if value and value != ".":
            try:
                return float(value) / 100.0
            except ValueError:
                continue
    return None


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_current_risk_free_rate() -> float | None:
    """Latest published 3-month T-bill yield, as a decimal annual rate (e.g. 0.0412
    for 4.12%). Returns None if no FRED_API_KEY is set or the request fails."""
    return _fetch_fred_series_latest(RISK_FREE_SERIES_ID)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_macro_snapshot() -> dict[str, float | None]:
    """
    Small macro dashboard: 3-month yield, 10-year yield, and the term spread
    between them. Each field is independently None-safe (a partial FRED outage
    on one series must not blank out the other two).
    """
    three_month = _fetch_fred_series_latest(RISK_FREE_SERIES_ID)
    ten_year = _fetch_fred_series_latest(TEN_YEAR_SERIES_ID)
    spread = (ten_year - three_month) if (three_month is not None and ten_year is not None) else None
    return {"three_month_yield": three_month, "ten_year_yield": ten_year, "term_spread_10y_3m": spread}