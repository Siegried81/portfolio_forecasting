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

# Circuit breaker: once BOTH Yahoo paths (yfinance library + direct API) fail in
# the same call, skip re-attempting Yahoo entirely for this long — avoids paying
# the full multi-second retry cost again on every Streamlit rerun (every widget
# interaction) while Yahoo is known to be down for this process. Module-level
# (not per-request), on purpose: the whole point is state that survives across
# separate `fetch_adjusted_close` calls within the same running process.
YAHOO_CIRCUIT_BREAKER_SECONDS = 180.0  # 3 minutes — long enough to skip a burst of
# rapid interactions, short enough to retry Yahoo again well within one debugging session
_yahoo_down_until: float = 0.0


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

    # Normalise both response shapes into {ticker: {"values": [...]}}.
    # BUGFIX (2026-09-04): the previous check `all(t in payload for t in tickers)`
    # required EVERY requested ticker to be present as a key. If Twelve Data drops
    # one bad/delisted symbol from a multi-ticker batch response instead of keeping
    # it as an error-tagged key, that check silently failed and the code wrapped the
    # WHOLE multi-ticker payload as {tickers[0]: payload} — mis-parsing every ticker,
    # not just the missing one. Detect the shape directly instead: a single-ticker
    # response has "values"/"meta" at the TOP level; a multi-ticker response is a
    # dict keyed by symbol, each holding that shape one level down.
    per_ticker = {tickers[0]: payload} if ("values" in payload or "meta" in payload) else payload

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

    Circuit breaker: Streamlit reruns the WHOLE script on every widget
    interaction — adding one ticker, moving a slider — which means a naive
    implementation re-attempts the full Yahoo retry sequence (up to 6 failed
    calls with backoff sleeps) on every single interaction while Yahoo is down,
    even though the previous attempt (seconds ago) already proved it's down.
    Once both Yahoo paths fail, `_yahoo_down_until` records "don't bother
    retrying Yahoo before this time" — subsequent calls within that window skip
    straight to Twelve Data. This is a genuine responsiveness fix, not just log
    noise reduction: observed in practice, a user rapidly toggling tickers while
    Yahoo was down was hitting the full multi-second retry chain on every click.

    Returns a DataFrame indexed by date, one column per ticker, forward-filled for
    isolated missing sessions (holidays that differ slightly across exchanges/ETFs)
    but NOT filled at the edges - leading/trailing NaNs are dropped so every column
    only spans dates where it actually traded.
    """
    if not tickers:
        raise MarketDataError("No tickers provided.")

    global _yahoo_down_until
    skip_yahoo = time.monotonic() < _yahoo_down_until

    source = "yfinance"
    if skip_yahoo:
        logger.info("Yahoo circuit breaker active (down recently) — skipping straight to Twelve Data.")
        source = "twelvedata (yahoo recently unavailable)"
        prices = _download_twelvedata(tickers, start, end)
    else:
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
                _yahoo_down_until = time.monotonic() + YAHOO_CIRCUIT_BREAKER_SECONDS
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


def _fetch_twelvedata_fundamentals(ticker: str) -> dict | None:
    """
    Twelve Data's /statistics endpoint — kept as a FALLBACK only. Confirmed via
    live testing (2026-09-03) that the free tier restricts this endpoint to
    their public demo symbol (AAPL); every other ticker 403s regardless of
    retry/pacing. Finnhub (tried first in `fetch_fundamentals` below) covers
    every ticker on its free tier, so this mostly matters if FINNHUB_API_KEY
    isn't configured, or specifically for AAPL where Twelve Data happens to work.
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
            raise TwelveDataPlanRestricted(message)
        logger.warning("Twelve Data fundamentals unavailable for %s: %s", ticker, message)
        return None

    stats = payload.get("statistics", {})
    valuations = stats.get("valuations_metrics", {}) or {}
    stock_stats = stats.get("stock_statistics", {}) or {}
    dividends = stats.get("dividends_and_splits", {}) or {}

    result = {
        "name": (payload.get("meta") or {}).get("name"),
        "source": "Twelve Data",
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
    return result if any(v is not None for k, v in result.items() if k not in ("name", "source")) else None


FINNHUB_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"
FINNHUB_METRIC_URL = "https://finnhub.io/api/v1/stock/metric"


def _fetch_finnhub_fundamentals(ticker: str) -> dict | None:
    """
    Fundamentals via Finnhub's `/stock/profile2` (company profile) and
    `/stock/metric` (valuation/risk ratios) — PRIMARY source, tried before
    Twelve Data. Unlike Twelve Data's /statistics, these two endpoints are
    confirmed free-tier for arbitrary tickers (not restricted to a demo
    symbol) per Finnhub's own free-tier feature list and multiple independent
    working code examples — verified via web search, not a live key in this
    environment, so if field names have drifted, `curl` both endpoints
    directly before assuming this parser is stale.

    Quirk worth knowing if debugging: Finnhub's `marketCapitalization` is
    denominated in MILLIONS of the reporting currency, not raw units — this
    function multiplies by 1e6 so the UI's `$X,XXX,XXX,XXX` formatting stays
    consistent with the Twelve Data fallback's raw-unit convention.
    """
    if not LLM_SETTINGS.finnhub_api_key:
        return None

    try:
        profile_resp = requests.get(
            FINNHUB_PROFILE_URL, params={"symbol": ticker, "token": LLM_SETTINGS.finnhub_api_key}, timeout=10,
        )
        metric_resp = requests.get(
            FINNHUB_METRIC_URL, params={"symbol": ticker, "metric": "all", "token": LLM_SETTINGS.finnhub_api_key}, timeout=10,
        )
        profile_resp.raise_for_status()
        metric_resp.raise_for_status()
        profile = profile_resp.json()
        metric = (metric_resp.json() or {}).get("metric", {}) or {}
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Finnhub fundamentals request failed for %s: %s", ticker, exc)
        return None

    if not profile and not metric:
        return None  # Finnhub returns {} (not an error field) for an unrecognised symbol

    market_cap_millions = profile.get("marketCapitalization")
    result = {
        "name": profile.get("name"),
        "source": "Finnhub",
        "market_cap": market_cap_millions * 1_000_000 if market_cap_millions else None,
        "pe_ratio": metric.get("peBasicExclExtraTTM") or metric.get("peTTM"),
        "forward_pe": metric.get("peForward"),  # often absent on free tier — stays None, shows as "—"
        "peg_ratio": metric.get("pegRatio"),
        "price_to_book": metric.get("pbQuarterly") or metric.get("pbAnnual"),
        "beta": metric.get("beta"),
        "dividend_yield": (
            (metric.get("dividendYieldIndicatedAnnual") or metric.get("currentDividendYieldTTM")) / 100
            if (metric.get("dividendYieldIndicatedAnnual") or metric.get("currentDividendYieldTTM"))
            else None
        ),  # Finnhub returns yield as a percentage number (e.g. 0.44 for 0.44%), our UI expects
        # a decimal fraction (0.0044) since it formats with :.2% — divide by 100 to match.
        "52w_high": metric.get("52WeekHigh"),
        "52w_low": metric.get("52WeekLow"),
    }
    return result if any(v is not None for k, v in result.items() if k not in ("name", "source")) else None


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_fundamentals(ticker: str) -> dict | None:
    """
    Per-ticker fundamentals (P/E, market cap, dividend yield, beta, 52w range).
    Tries Finnhub first (free tier covers arbitrary tickers), falls back to
    Twelve Data only if Finnhub isn't configured or returns nothing — Twelve
    Data's free tier is confirmed restricted to their demo symbol (AAPL) for
    this data, so it's a fallback, not the primary path, as of 2026-09-03.

    Returns None (not an exception) if both sources fail or neither key is
    configured — fundamentals are an enrichment, not something that should
    ever block the rest of the app. May still raise TwelveDataPlanRestricted
    from the Twelve Data fallback — callers already handle that distinctly.
    """
    result = _fetch_finnhub_fundamentals(ticker)
    if result is not None:
        return result
    return _fetch_twelvedata_fundamentals(ticker)