"""
Central configuration: default universe, financial constants, and env var loading.

Keeping every "magic number" and API key lookup in ONE place (instead of scattered
across the app) is a deliberate design choice: it's the first place a reviewer -
or future-you - will look, and the only place that needs to change if a default
moves (e.g. risk-free rate, default tickers).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=True)  # override=True: .env values win over any pre-existing shell-level
# env vars of the same name (even an empty/stale one) — the default (override=False) would
# silently keep a blank shell variable and ignore .env, which is exactly the kind of "the key
# is definitely in .env but the app still sees nothing" bug this project hit in practice.
# No-op in prod (Render/Streamlit Cloud inject env vars directly, no .env file exists there).

# --- Default investable universe -------------------------------------------------
# Individual equities: liquid, well-covered US large caps (matches the brief's examples).
DEFAULT_EQUITY_TICKERS: list[str] = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOG"]

# Optional ETF/index sleeve: lets the user add diversification across asset classes
# (equities, bonds, gold, silver, oil, broad commodities, real estate) without
# hand-typing tickers - a request a real client would make in a second-round
# interview case study, and a genuinely different risk/return profile from the
# 5 US tech-heavy equities in the default universe (i.e. NOT just decoration).
OPTIONAL_ETF_TICKERS: list[str] = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "TLT",   # 20+yr US Treasuries (rate/duration exposure)
    "GLD",   # Gold (inflation / crisis hedge)
    "SLV",   # Silver (industrial + precious-metal hybrid, higher beta than gold)
    "USO",   # WTI crude oil (energy/inflation exposure, distinct driver from equities)
    "DBC",   # Broad commodities basket (energy + metals + agriculture)
    "UUP",   # US Dollar Index bull fund (FX/currency exposure — ETF proxy, not raw
             # spot FX, kept consistent with USO/SLV/GLD above being ETFs rather
             # than raw futures: cleanly optimisable long-only, no margin/roll quirks)
    "VNQ",   # US REITs (real estate)
]

# --- Broader S&P 500 universe, organised by GICS sector -----------------------------
# A curated ~45-name subset (not all 500 — see README for why: covariance estimation
# degrades badly with hundreds of names and a few years of daily data; real buy-side
# desks use factor models or sector-constrained universes for exactly this reason,
# not a raw 500x500 mean-variance optimisation). Each sector has enough names to
# build a genuinely diversified sub-portfolio, not just one or two tokens.
SP500_SECTOR_UNIVERSE: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "PM"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA"],
    "Healthcare": ["UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "TMO"],
    "Industrials": ["CAT", "HON", "UNP", "BA", "GE", "RTX"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Materials": ["LIN", "SHW", "APD"],
    "Real Estate": ["PLD", "AMT", "EQIX"],
    "Utilities": ["NEE", "DUK", "SO"],
}

# Quick-select preset: largest-cap, most-recognised names across sectors — the
# "just give me something sensible fast" option, as opposed to picking sectors by hand.
MEGA_CAP_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "JPM", "LLY", "V", "WMT", "UNH", "XOM", "MA",
]

# Every ticker the sidebar could possibly offer, deduplicated — used to build the
# multiselect's option list regardless of which preset/sectors are active.
ALL_KNOWN_TICKERS: list[str] = sorted(set(
    DEFAULT_EQUITY_TICKERS
    + OPTIONAL_ETF_TICKERS
    + MEGA_CAP_TICKERS
    + [t for tickers in SP500_SECTOR_UNIVERSE.values() for t in tickers]
))

# Used as the market proxy for beta calculations and as an LLM-commentary reference point.
BENCHMARK_TICKER: str = "SPY"

# CBOE Volatility Index — the standard market "fear gauge" (implied 30-day S&P 500
# volatility). Shown as macro risk context, not part of the optimisable universe
# (it isn't an investable asset with a price return in the usual sense).
VIX_TICKER: str = "^VIX"

# --- Finance constants -------------------------------------------------------------
DEFAULT_RISK_FREE_RATE: float = 0.04  # ~US T-bill yield, override in the UI if needed

# Max weight any single asset can take in an optimized portfolio. Without this,
# unconstrained mean-variance optimization on a short/volatile window (especially
# the hindsight "realized-optimal" benchmark) will happily put 100% into whatever
# one name got lucky — mathematically correct, but not how any real portfolio is
# actually run, and it makes the three-portfolio comparison table look absurd
# rather than informative. Position limits are standard institutional practice,
# not a workaround.
DEFAULT_MAX_WEIGHT_PER_ASSET: float = 0.35
TRADING_DAYS_PER_YEAR: int = 252
MONTHS_PER_YEAR: int = 12

# Transaction cost charged (as turnover × this rate) each time a portfolio
# rebalances — i.e. at every walk-forward window boundary, and once for the
# initial trade into the single-window comparison. 10 bps (0.10%) is a
# reasonable retail/liquid-ETF assumption; institutional desks on large-cap
# names can be lower, illiquid names higher. Set to 0 in the UI to see the
# frictionless (textbook) comparison.
DEFAULT_TRANSACTION_COST_BPS: float = 10.0

FREQUENCY_TO_PERIODS_PER_YEAR: dict[str, int] = {
    "daily": TRADING_DAYS_PER_YEAR,
    "weekly": 52,
    "monthly": MONTHS_PER_YEAR,
}

# --- Forecasting -------------------------------------------------------------------
MIN_HISTORY_POINTS_FOR_FORECAST: int = 30  # below this, ARIMA/ETS fits are unreliable
DEFAULT_FORECAST_HORIZON_DAYS: int = 30

# --- Quick date-range presets (sidebar UX) ------------------------------------------
QUICK_DATE_RANGES: dict[str, int] = {
    "1 an": 365,
    "3 ans": 3 * 365,
    "5 ans": 5 * 365,
    "10 ans": 10 * 365,
    "Max (15 ans)": 15 * 365,
}

# --- Walk-forward (multi-window) backtesting ----------------------------------------
# A single train/test split can be luck or bad luck for that one window. Walk-forward
# repeats the historical/forecast/realized comparison across several EXPANDING windows
# (each refit uses all data available up to that point, then is tested on the next
# horizon-sized slice) so the comparison becomes a distribution, not one data point —
# standard practice in any real backtest, and the difference between a demo and a
# credible one.
DEFAULT_WALK_FORWARD_WINDOWS: int = 5
MIN_WALK_FORWARD_WINDOWS: int = 3
MAX_WALK_FORWARD_WINDOWS: int = 8
# Longer than MIN_HISTORY_POINTS_FOR_FORECAST: the FIRST window's training set needs
# enough history for a stable initial fit, not just the bare statmodels minimum.
WALK_FORWARD_MIN_TRAIN_PERIODS: int = 90

# --- LLM / news / macro / market-data-fallback config (secrets from env, never hardcoded) ---
def _load_groq_keys() -> list[str]:
    """
    Collect up to 5 Groq keys from the environment: GROQ_API_KEY (primary) plus
    GROQ_API_KEY_2 .. GROQ_API_KEY_5. Multiple keys exist to spread free-tier
    rate limits across accounts — same pattern already proven on the Innovation
    Radar project. Empty/unset slots are skipped, so this also works fine with
    just one key configured.
    """
    keys = []
    primary = os.getenv("GROQ_API_KEY")
    if primary:
        keys.append(primary)
    for i in range(2, 6):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)
    return keys


@dataclass(frozen=True)
class LLMSettings:
    groq_api_keys: list[str] = field(default_factory=_load_groq_keys)
    # Default updated 2026-09: Groq deprecated llama-3.3-70b-versatile (decommissioned
    # 2026-08-16). openai/gpt-oss-120b is Groq's official recommended replacement —
    # if this breaks again later, check https://console.groq.com/docs/deprecations
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))
    newsapi_key: str | None = field(default_factory=lambda: os.getenv("NEWSAPI_KEY") or None)
    finnhub_api_key: str | None = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY") or None)
    fred_api_key: str | None = field(default_factory=lambda: os.getenv("FRED_API_KEY") or None)
    twelvedata_api_key: str | None = field(default_factory=lambda: os.getenv("TWELVEDATA_API_KEY") or None)

    @property
    def groq_api_key(self) -> str | None:
        """Convenience accessor for callers that only care about "is Groq configured
        at all" — returns the first key, or None if the list is empty."""
        return self.groq_api_keys[0] if self.groq_api_keys else None


LLM_SETTINGS = LLMSettings()

# Hard cap on tokens sent to the LLM as "context" (news articles, metrics dump).
# Keeps prompts cheap and avoids truncation errors on smaller models (Ollama local models
# often have much smaller context windows than Groq's hosted ones).
MAX_CONTEXT_TOKENS: int = 3000