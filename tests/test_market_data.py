"""
Unit tests for src/market_data.py.

Fully mocked — no real network calls, no dependency on a live TWELVEDATA_API_KEY.
Priorities, per the code review:
1. Regression test for the 2026-09-04 fix: a multi-ticker Twelve Data response
   that drops a bad/delisted symbol (instead of keeping it as an error-tagged
   key) must not corrupt the tickers that DID come back.
2. The Yahoo circuit breaker actually skips yfinance/direct-Yahoo once tripped.
3. _redact_api_key never leaks a key into a message that could reach st.error().
"""
import datetime as dt
import dataclasses

import pandas as pd
import pytest
import streamlit as st

import src.market_data as market_data
from src.market_data import (
    MarketDataError,
    _download_twelvedata,
    _redact_api_key,
    fetch_adjusted_close,
)


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Isolate every test from the module-level circuit breaker and Streamlit's
    process-wide cache — both are shared state that would otherwise leak
    between tests depending on execution order."""
    st.cache_data.clear()
    monkeypatch.setattr(market_data, "_yahoo_down_until", 0.0)
    # LLM_SETTINGS is a frozen dataclass (config.py, deliberately immutable) —
    # can't monkeypatch a single field on it directly (raises
    # FrozenInstanceError). Replace the whole object with a copy instead,
    # scoped to market_data's own imported name so other modules are unaffected.
    fake_settings = dataclasses.replace(market_data.LLM_SETTINGS, twelvedata_api_key="fake-test-key")
    monkeypatch.setattr(market_data, "LLM_SETTINGS", fake_settings)
    yield
    st.cache_data.clear()


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise __import__("requests").HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


# ---------------------------------------------------------------------------
# _redact_api_key
# ---------------------------------------------------------------------------

def test_redact_api_key_strips_the_value():
    text = "GET https://api.twelvedata.com/time_series?symbol=AAPL&apikey=sk-super-secret-123"
    redacted = _redact_api_key(text)
    assert "sk-super-secret-123" not in redacted
    assert "apikey=***REDACTED***" in redacted


# ---------------------------------------------------------------------------
# _download_twelvedata — regression test for the partial-batch parsing bug
# ---------------------------------------------------------------------------

def test_download_twelvedata_partial_batch_does_not_lose_good_tickers(monkeypatch):
    """
    Simulates Twelve Data DROPPING one requested ticker entirely from a
    multi-symbol response (instead of keeping it as an error-tagged key) —
    the exact scenario that silently corrupted the whole batch before the fix.
    """
    good_ticker_payload = {
        "meta": {"symbol": "AAPL"},
        "values": [
            {"datetime": "2024-01-02", "close": "100.0"},
            {"datetime": "2024-01-03", "close": "101.5"},
        ],
    }
    # Only "AAPL" comes back — "BADTICKER" is silently absent, not even as an
    # error-tagged key, which is the real-world case that broke the old logic.
    response_payload = {"AAPL": good_ticker_payload}

    monkeypatch.setattr(
        market_data.requests, "get",
        lambda *a, **k: _FakeResponse(response_payload),
    )

    result = _download_twelvedata(["AAPL", "BADTICKER"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    # AAPL must still be parsed correctly — this is what broke before the fix.
    assert "AAPL" in result.columns
    assert list(result["AAPL"].values) == [100.0, 101.5]
    # BADTICKER legitimately has no data and should simply be absent, not raise.
    assert "BADTICKER" not in result.columns


def test_download_twelvedata_single_ticker_shape_still_works(monkeypatch):
    """Single-ticker requests return the flat {"meta":..., "values":[...]} shape
    directly (no outer ticker key) — must still be handled after the fix."""
    payload = {
        "meta": {"symbol": "AAPL"},
        "values": [{"datetime": "2024-01-02", "close": "100.0"}],
    }
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse(payload))

    result = _download_twelvedata(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert list(result.columns) == ["AAPL"]
    assert result["AAPL"].iloc[0] == 100.0


def test_download_twelvedata_raises_when_nothing_comes_back(monkeypatch):
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _FakeResponse({"AAPL": {"values": []}}))
    with pytest.raises(MarketDataError):
        _download_twelvedata(["AAPL"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))


# ---------------------------------------------------------------------------
# Circuit breaker — once both Yahoo paths fail, skip straight to Twelve Data
# ---------------------------------------------------------------------------

def test_circuit_breaker_skips_yahoo_after_a_prior_failure(monkeypatch):
    yfinance_calls = {"count": 0}
    direct_calls = {"count": 0}

    def _failing_yfinance(*a, **k):
        yfinance_calls["count"] += 1
        raise MarketDataError("Yahoo down (simulated)")

    def _failing_direct(*a, **k):
        direct_calls["count"] += 1
        raise MarketDataError("Yahoo direct also down (simulated)")

    fake_twelvedata_df = pd.DataFrame(
        {"X1": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    monkeypatch.setattr(market_data, "_download_yfinance", _failing_yfinance)
    monkeypatch.setattr(market_data, "_download_yahoo_direct", _failing_direct)
    monkeypatch.setattr(market_data, "_download_twelvedata", lambda *a, **k: fake_twelvedata_df.copy())

    # Call #1: both Yahoo paths fail -> trips the breaker, falls back to Twelve Data.
    fetch_adjusted_close(["X1"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert yfinance_calls["count"] == 1
    assert direct_calls["count"] == 1
    assert market_data._yahoo_down_until > 0

    # Call #2, DIFFERENT tickers (so Streamlit's cache can't just return the
    # first call's result) but WITHIN the breaker window: must skip Yahoo
    # entirely and go straight to Twelve Data.
    fake_twelvedata_df2 = pd.DataFrame(
        {"X2": [50.0, 51.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    monkeypatch.setattr(market_data, "_download_twelvedata", lambda *a, **k: fake_twelvedata_df2.copy())
    result = fetch_adjusted_close(["X2"], dt.date(2024, 1, 1), dt.date(2024, 1, 5))

    assert yfinance_calls["count"] == 1  # unchanged — Yahoo was never retried
    assert direct_calls["count"] == 1
    assert "X2" in result.columns
    assert result.attrs["source"] == "twelvedata (yahoo recently unavailable)"