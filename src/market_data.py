"""
Market data acquisition via yfinance (free, no API key required), with automatic
retry and a Twelve Data fallback for when Yahoo Finance itself is unreachable.

Why the fallback exists: as of late 2026, Yahoo Finance's anti-bot layer (the
cookie/crumb handshake yfinance depends on) has become noticeably unreliable —
this is a widely reported, ongoing issue across the yfinance community, not
specific to any one network or account. A finance tool that goes dark whenever
Yahoo has a bad day is a real reliability gap, so a second source is used as a
fallback rather than just failing.

Twelve Data (not Stooq) was chosen for the fallback after live testing: Stooq's
CSV export now runs a JavaScript proof-of-work anti-bot challenge (confirmed by
inspecting its raw response — a `crypto.subtle.digest` SHA-256 puzzle that a
plain HTTP client cannot solve), so a `requests.get()`-based fallback against it
is a dead end, not a fix. Twelve Data is a genuine key-based REST API rather
than a scraped endpoint, which sidesteps that entire arms race. Free tier: 800
requests/day, 8/minute — and it accepts multiple tickers in ONE call, which
matters for staying inside that budget with a 5+ ticker portfolio.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from src.config import LLM_SETTINGS, VIX_TICKER

logger = logging.getLogger(__name__)


def _redact_api_key(text: str) -> str:
    """
    Strip any `apikey=...` query param value out of a string before it's ever
    logged or, critically, raised as an exception message that Streamlit's
    `st.error()` displays verbatim on screen. `requests`' HTTPError includes the
    full request URL (query string and all) in its default __str__ — without
    this, a failed Twelve Data call leaks the API key straight into the UI.
    """
    return re.sub(r"apikey=[^&\s]+", "apikey=***REDACTED***", text)

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
TWELVEDATA_MAX_ATTEMPTS = 3
TWELVEDATA_BACKOFF_SECONDS = 20.0  # doubles each retry: 20s, 40s — 429 free-tier limits reset per-minute
YFINANCE_MAX_ATTEMPTS = 3
YFINANCE_BACKOFF_SECONDS = 2.0  # doubles each retry: 2s, 4s


class MarketDataError(RuntimeError):
    """Raised when price data cannot be retrieved from ANY source."""


class TwelveDataPlanRestricted(RuntimeError):
    """Raised when Twelve Data returns a 403 indicating an endpoint (typically
    /statistics for non-demo tickers) requires a paid plan — distinct from a
    transient failure, so callers can surface the real reason to the user."""


def _download_yfinance(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    One yfinance attempt with retry-with-backoff for transient failures (mainly
    429 rate-limiting on Yahoo's cookie/crumb endpoint). Retrying helps with a
    momentary block; it does nothing for a sustained one, which is what the
    Twelve Data fallback in `fetch_adjusted_close` is for.
    """
    last_error: Exception | None = None
    for attempt in range(1, YFINANCE_MAX_ATTEMPTS + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end + dt.timedelta(days=1),  # yfinance's `end` is exclusive
                auto_adjust=True,  # returns already-adjusted "Close" (splits + dividends)
                progress=False,
                group_by="ticker",
            )
            if not raw.empty:
                return raw
            last_error = MarketDataError("Empty response")
        except Exception as exc:  # yfinance raises a mix of requests/JSON/custom errors
            last_error = exc
            logger.warning("yfinance attempt %d/%d failed: %s", attempt, YFINANCE_MAX_ATTEMPTS, exc)

        if attempt < YFINANCE_MAX_ATTEMPTS:
            time.sleep(YFINANCE_BACKOFF_SECONDS * attempt)  # 2s, then 4s

    raise MarketDataError(f"Yahoo Finance unreachable after {YFINANCE_MAX_ATTEMPTS} attempts: {last_error}")


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _download_yahoo_direct(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Direct call to Yahoo Finance's own REST API (`v8/finance/chart`) — bypassing
    the `yfinance` library entirely. Added per the brief's literal wording ("the
    Yahoo Finance API through the yfinance package"): this is what "the Yahoo
    Finance API" actually is underneath the library.

    Honest expectation, stated plainly rather than left implicit: this endpoint
    sits behind the SAME cookie/crumb anti-bot layer that already blocks the
    `yfinance` library above (see `_download_yfinance`'s docstring) — a raw
    unauthenticated request here is very likely to fail identically. It's kept
    as a single, fast, no-retry attempt specifically because it's cheap insurance
    (Yahoo's anti-bot posture does fluctuate) with no cost when it doesn't pay
    off, not because it's expected to reliably succeed where the library fails.
    """
    period1 = int(dt.datetime.combine(start, dt.time.min).timestamp())
    period2 = int(dt.datetime.combine(end + dt.timedelta(days=1), dt.time.min).timestamp())
    headers = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-forecasting/1.0)"}

    columns = {}
    for ticker in tickers:
        try:
            response = requests.get(
                YAHOO_CHART_URL.format(ticker=ticker),
                params={"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"},
                headers=headers, timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Direct Yahoo API failed for %s: %s", ticker, exc)
            continue

        result = (payload.get("chart") or {}).get("result")
        if not result:
            continue
        result = result[0]
        timestamps = result.get("timestamp")
        closes = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
        if not timestamps or not closes:
            continue

        series = pd.Series(closes, index=pd.to_datetime(timestamps, unit="s").normalize(), name=ticker)
        columns[ticker] = series.dropna()

    if not columns:
        raise MarketDataError("Direct Yahoo Finance API also returned no data for any ticker.")
    return pd.concat(columns, axis=1)


def _download_twelvedata(tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fallback source: Twelve Data's time_series endpoint, one batched call for
    every ticker (comma-separated `symbol` param) — deliberately NOT one call
    per ticker, to conserve the free tier's 800/day, 8/min budget.

    Response shape differs by ticker count: a single symbol returns
    {"meta": ..., "values": [...]} directly; multiple symbols return a dict
    KEYED by symbol, each holding that same shape — both are handled here.
    """
    if not LLM_SETTINGS.twelvedata_api_key:
        raise MarketDataError("No TWELVEDATA_API_KEY configured — cannot use the fallback source.")

    params = {
        "symbol": ",".join(tickers),
        "interval": "1day",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "apikey": LLM_SETTINGS.twelvedata_api_key,
        "order": "ASC",
    }
    try:
        response = requests.get(TWELVEDATA_URL, params=params, timeout=15)
        payload = None
        for attempt in range(1, TWELVEDATA_MAX_ATTEMPTS + 1):
            if response.status_code == 429:
                if attempt == TWELVEDATA_MAX_ATTEMPTS:
                    raise MarketDataError(
                        "Twelve Data rate limit hit (free tier: 8 req/min, 800/day). "
                        "Wait ~60s and retry — this is usually transient, not a quota exhaustion."
                    )
                logger.warning("Twelve Data 429, retrying in %.0fs (attempt %d/%d)", TWELVEDATA_BACKOFF_SECONDS * attempt, attempt, TWELVEDATA_MAX_ATTEMPTS)
                time.sleep(TWELVEDATA_BACKOFF_SECONDS * attempt)
                response = requests.get(TWELVEDATA_URL, params=params, timeout=15)
                continue
            response.raise_for_status()
            payload = response.json()
            break
    except (requests.RequestException, ValueError) as exc:
        raise MarketDataError(f"Twelve Data request failed: {_redact_api_key(str(exc))}") from exc

    if payload.get("status") == "error" or "code" in payload and "message" in payload and "values" not in payload:
        raise MarketDataError(f"Twelve Data error: {payload.get('message', payload)}")

    # Normalise both response shapes into {ticker: {"values": [...]}}
    per_ticker = payload if len(tickers) > 1 and all(t in payload for t in tickers) else {tickers[0]: payload}

    columns = {}
    for ticker in tickers:
        entry = per_ticker.get(ticker)
        values = entry.get("values") if entry else None
        if not values:
            logger.warning("Twelve Data returned no values for %s", ticker)
            continue
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"])
        columns[ticker] = df.set_index("datetime")["close"].sort_index()

    if not columns:
        raise MarketDataError("Twelve Data fallback also returned no data for any ticker.")
    return pd.concat(columns, axis=1)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_adjusted_close(
    tickers: list[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for one or more tickers. Three-step
    fallback chain: (1) the `yfinance` library, with retry-with-backoff — this
    is "the Yahoo Finance API" per the brief; (2) a direct, unauthenticated call
    to Yahoo's own REST endpoint, bypassing the library — same underlying source,
    different code path, in case the library's cookie/crumb handling specifically
    (not Yahoo itself) is what's failing; (3) Twelve Data, a genuinely different
    provider, as the real fallback once both Yahoo-based attempts are exhausted.

    Returns a DataFrame indexed by date, one column per ticker, forward-filled for
    isolated missing sessions (holidays that differ slightly across exchanges/ETFs)
    but NOT filled at the edges - leading/trailing NaNs are dropped so every column
    only spans dates where it actually traded.
    """
    if not tickers:
        raise MarketDataError("No tickers provided.")

    source = "yfinance"
    try:
        raw = _download_yfinance(tickers, start, end)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = pd.concat(
                {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)},
                axis=1,
            )
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    except MarketDataError as yfinance_error:
        logger.warning("yfinance library failed, trying direct Yahoo API: %s", yfinance_error)
        try:
            prices = _download_yahoo_direct(tickers, start, end)
            source = "yahoo direct API"
        except MarketDataError as yahoo_direct_error:
            logger.warning("Direct Yahoo API also failed, falling back to Twelve Data: %s", yahoo_direct_error)
            source = "twelvedata (yahoo unavailable)"
            prices = _download_twelvedata(tickers, start, end)

    missing = set(tickers) - set(prices.columns)
    if missing:
        raise MarketDataError(f"No data for: {', '.join(sorted(missing))} (source: {source}). Check the ticker symbols.")

    prices = prices.ffill().dropna(how="all")
    if prices.empty:
        raise MarketDataError(f"Downloaded data is empty after cleaning (source: {source}) — widen the date range.")

    prices.attrs["source"] = source
    return prices


def resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample daily adjusted-close prices to weekly/monthly, taking the LAST
    observation of each period (standard convention for price series — unlike
    returns, prices should never be averaged or summed across a period).
    """
    freq_map = {"daily": None, "weekly": "W-FRI", "monthly": "ME"}
    rule = freq_map.get(frequency)
    if rule is None:
        return prices
    source = prices.attrs.get("source")  # .resample() drops .attrs — carry it through
    resampled = prices.resample(rule).last().dropna(how="all")
    if source:
        resampled.attrs["source"] = source
    return resampled


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_vix_snapshot(lookback_days: int = 90) -> pd.Series | None:
    """
    Recent CBOE VIX levels (not an "adjusted close" — it's an index, not a
    tradeable asset, but VIX_TICKER still resolves to a valid `Close` series on
    every source below). Returns None on failure rather than raising, since the
    VIX panel is contextual risk information, not core to the app's maths.

    Reuses `fetch_adjusted_close`'s full 3-step fallback chain (yfinance ->
    direct Yahoo API -> Twelve Data) instead of a separate yfinance-only path —
    the VIX panel showing "n/a" whenever Yahoo alone is blocked (which is often,
    per the rest of this module's docstrings) was a real gap, not a deliberate
    trade-off worth keeping.
    """
    start = dt.date.today() - dt.timedelta(days=lookback_days)
    end = dt.date.today()
    try:
        prices = fetch_adjusted_close([VIX_TICKER], start, end)
    except MarketDataError:
        return None

    if VIX_TICKER not in prices.columns:
        return None
    return prices[VIX_TICKER].dropna()


TWELVEDATA_STATISTICS_URL = "https://api.twelvedata.com/statistics"


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_fundamentals(ticker: str) -> dict | None:
    """
    Per-ticker fundamentals (P/E, market cap, dividend yield, beta) via Twelve
    Data's /statistics endpoint.

    IMPORTANT CAVEAT (be upfront about this, don't just silently degrade): Twelve
    Data's docs list some fundamentals fields as premium-plan-only, and the exact
    free-tier field availability isn't confirmed without a live key — this
    function is written defensively (every field access is a best-effort .get()
    with a fallback to None) so a partially-populated or fully-missing response
    degrades to "field unavailable" in the UI rather than crashing. If your key
    returns a different shape than expected, the raw response is worth
    inspecting directly (`curl` the endpoint) rather than assuming this parser
    is exhaustive — it was built without the ability to test against a live key.

    Returns None (not an exception) if the endpoint is unreachable, requires a
    higher plan, or the key isn't configured — fundamentals are an enrichment,
    not something that should ever block the rest of the app.
    """
    if not LLM_SETTINGS.twelvedata_api_key:
        return None

    for attempt in range(1, TWELVEDATA_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                TWELVEDATA_STATISTICS_URL,
                params={"symbol": ticker, "apikey": LLM_SETTINGS.twelvedata_api_key},
                timeout=10,
            )
        except requests.RequestException as exc:
            logger.warning("Twelve Data fundamentals request failed for %s: %s", ticker, _redact_api_key(str(exc)))
            return None

        if response.status_code == 429:
            if attempt == TWELVEDATA_MAX_ATTEMPTS:
                logger.warning("Twelve Data fundamentals rate-limited for %s after %d attempts", ticker, attempt)
                return None
            time.sleep(TWELVEDATA_BACKOFF_SECONDS * attempt)
            continue

        try:
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Twelve Data fundamentals request failed for %s: %s", ticker, _redact_api_key(str(exc)))
            return None
        break
    else:
        return None

    if payload.get("status") == "error" or "code" in payload and "message" in payload:
        message = payload.get("message", str(payload))
        if payload.get("code") == 403:
            # Confirmed via live testing (2026-09-03): Twelve Data's free tier only
            # grants /statistics access on their public demo symbol (AAPL) — every
            # other ticker returns this 403, regardless of retry or pacing. This is
            # a plan restriction, not a transient failure, so it's raised distinctly
            # rather than folded into the generic "return None" path — the caller
            # should tell the user WHY, not just show a blank dash that looks like
            # a bug.
            raise TwelveDataPlanRestricted(message)
        logger.warning("Twelve Data fundamentals unavailable for %s: %s", ticker, message)
        return None

    stats = payload.get("statistics", {})
    valuations = stats.get("valuations_metrics", {}) or {}
    stock_stats = stats.get("stock_statistics", {}) or {}
    dividends = stats.get("dividends_and_splits", {}) or {}

    result = {
        "name": (payload.get("meta") or {}).get("name"),
        "market_cap": valuations.get("market_capitalization"),
        "pe_ratio": valuations.get("trailing_pe"),
        "forward_pe": valuations.get("forward_pe"),
        "peg_ratio": valuations.get("peg_ratio"),
        "price_to_book": valuations.get("price_to_book_mrq"),
        "beta": stock_stats.get("beta"),
        "dividend_yield": dividends.get("forward_annual_dividend_yield"),
        "52w_high": stock_stats.get("52_week_high"),
        "52w_low": stock_stats.get("52_week_low"),
    }
    # Not worth returning a dict of all-None values — treat as unavailable.
    return result if any(v is not None for k, v in result.items() if k != "name") else None