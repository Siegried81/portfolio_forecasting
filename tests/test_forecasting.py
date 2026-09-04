"""
Unit tests for src/forecasting.py.

Priority: the FALLBACK branches, not the happy path — a silently-skipped fallback
is invisible in the UI (the app just looks like it's using ARIMA when it's
actually degraded to naive) unless it's covered by a test that forces the
trigger condition directly.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import MIN_HISTORY_POINTS_FOR_FORECAST
from src.forecasting import (
    FORECAST_MODELS,
    arima_forecast,
    ets_forecast,
    forecast_all_assets,
    naive_forecast,
)


def _price_series(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    # Geometric-ish random walk, always positive
    returns = rng.normal(0.0005, 0.01, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=idx, name="PRICE")


# ---------------------------------------------------------------------------
# naive_forecast
# ---------------------------------------------------------------------------

def test_naive_forecast_shape_and_index():
    prices = _price_series(60)
    result = naive_forecast(prices, horizon=10)
    assert len(result["forecast"]) == 10
    assert (result["upper"] >= result["forecast"]).all()
    assert (result["lower"] <= result["forecast"]).all()
    # future index must continue strictly after the last observed date
    assert result["forecast"].index[0] > prices.index[-1]


# ---------------------------------------------------------------------------
# ets_forecast — short-history fallback
# ---------------------------------------------------------------------------

def test_ets_forecast_falls_back_to_naive_on_short_history():
    short_prices = _price_series(MIN_HISTORY_POINTS_FOR_FORECAST - 5)
    result = ets_forecast(short_prices, horizon=10)
    expected = naive_forecast(short_prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_ets_forecast_runs_on_sufficient_history():
    prices = _price_series(120)
    result = ets_forecast(prices, horizon=15)
    assert len(result["forecast"]) == 15
    assert not result["forecast"].isna().any()


# ---------------------------------------------------------------------------
# arima_forecast — short-history fallback + non-convergence fallback
# ---------------------------------------------------------------------------

def test_arima_forecast_falls_back_to_naive_on_short_history():
    short_prices = _price_series(MIN_HISTORY_POINTS_FOR_FORECAST - 5)
    result = arima_forecast(short_prices, horizon=10)
    expected = naive_forecast(short_prices, horizon=10)
    pd.testing.assert_series_equal(result["forecast"], expected["forecast"])


def test_arima_forecast_falls_back_to_naive_when_no_order_converges():
    # A perfectly constant series makes every ARIMA(p,1,q) order degenerate
    # (zero variance after differencing) — this must fall back to naive
    # rather than raising, since forecast_all_assets has no other safety net.
    idx = pd.bdate_range("2024-01-01", periods=MIN_HISTORY_POINTS_FOR_FORECAST + 10)
    constant_prices = pd.Series([100.0] * len(idx), index=idx)
    result = arima_forecast(constant_prices, horizon=10, max_p=1, max_q=1)
    assert len(result["forecast"]) == 10
    assert not result["forecast"].isna().any()


# ---------------------------------------------------------------------------
# forecast_all_assets — shape contract every optimizer call relies on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", list(FORECAST_MODELS.keys()))
def test_forecast_all_assets_returns_one_column_per_ticker(model_name):
    prices = pd.DataFrame({
        "AAA": _price_series(80, seed=1).values,
        "BBB": _price_series(80, seed=2).values,
    }, index=_price_series(80, seed=1).index)
    forecast = forecast_all_assets(prices, horizon=10, model_name=model_name)
    assert list(forecast.columns) == ["AAA", "BBB"]
    assert len(forecast) == 10
    assert not forecast.isna().any().any()
