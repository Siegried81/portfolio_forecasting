"""
Price forecasting models — statsmodels-based (NOT Kats: Kats has been effectively
unmaintained since 2021 and conflicts with modern pandas/numpy, which would burn
hours of a 2-day deadline on dependency resolution instead of on the actual
analysis). statsmodels is the industry-standard, well-maintained alternative and
is what most quant/finance teams actually reach for.

Three models, increasing sophistication, all returning the SAME shape so the
optimization layer doesn't need to know which one produced a forecast:

- `naive_forecast`   : random-walk-with-drift baseline. Always include this - if a
                        fancier model can't beat it, the fancier model isn't adding value.
- `ets_forecast`      : Exponential Smoothing (Holt's linear trend). Robust, few
                        assumptions, good default for noisy financial series.
- `arima_forecast`    : ARIMA with a small grid search over (p,d,q) by AIC. Can
                        capture autocorrelation structure the other two miss, at
                        the cost of being slower and more prone to overfitting on
                        short series.

IMPORTANT, and stated explicitly in the UI/README: stock price forecasting from
price history alone has very weak predictive power (near random-walk territory
per the efficient market hypothesis). These models exist to demonstrate the
forecast -> optimize -> compare methodology the brief asks for, not because
short-horizon price forecasts should be trusted for real allocation decisions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import MIN_HISTORY_POINTS_FOR_FORECAST

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

ForecastResult = dict[str, pd.Series]  # {"forecast": Series, "lower": Series, "upper": Series}


def _future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Business-day index continuing directly after the last observed date."""
    return pd.bdate_range(start=history.index[-1], periods=horizon + 1, freq="B")[1:]


def naive_forecast(prices: pd.Series, horizon: int) -> ForecastResult:
    """Random walk with drift: tomorrow's price = last price + average historical daily change."""
    daily_changes = prices.diff().dropna()
    drift = daily_changes.mean()
    last_price = prices.iloc[-1]
    steps = np.arange(1, horizon + 1)
    point_forecast = last_price + drift * steps

    # Widening confidence band ~ sqrt(t) x historical daily std dev (standard random-walk property)
    std = daily_changes.std(ddof=1)
    band = std * np.sqrt(steps)
    idx = _future_index(prices, horizon)
    return {
        "forecast": pd.Series(point_forecast, index=idx, name="naive"),
        "lower": pd.Series(point_forecast - 1.96 * band, index=idx),
        "upper": pd.Series(point_forecast + 1.96 * band, index=idx),
    }


def ets_forecast(prices: pd.Series, horizon: int) -> ForecastResult:
    """Holt's linear-trend Exponential Smoothing."""
    if len(prices) < MIN_HISTORY_POINTS_FOR_FORECAST:
        return naive_forecast(prices, horizon)
    model = ExponentialSmoothing(prices, trend="add", damped_trend=True, initialization_method="estimated")
    fitted = model.fit(optimized=True)
    point_forecast = fitted.forecast(horizon)

    resid_std = fitted.resid.std(ddof=1)
    steps = np.arange(1, horizon + 1)
    band = resid_std * np.sqrt(steps)
    idx = _future_index(prices, horizon)
    return {
        "forecast": pd.Series(point_forecast.values, index=idx, name="ets"),
        "lower": pd.Series(point_forecast.values - 1.96 * band, index=idx),
        "upper": pd.Series(point_forecast.values + 1.96 * band, index=idx),
    }


def arima_forecast(prices: pd.Series, horizon: int, max_p: int = 3, max_q: int = 3) -> ForecastResult:
    """
    ARIMA(p,1,q) with a small AIC grid search. d=1 is fixed (first-differencing)
    since price levels are non-stationary by construction — searching d as well
    would mostly just re-discover d=1 at extra compute cost for no real benefit here.
    """
    if len(prices) < MIN_HISTORY_POINTS_FOR_FORECAST:
        return naive_forecast(prices, horizon)

    best_aic, best_order, best_fit = np.inf, (1, 1, 0), None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(prices, order=(p, 1, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order, best_fit = fit.aic, (p, 1, q), fit
            except Exception:
                continue  # non-convergent order — skip, don't crash the whole forecast

    if best_fit is None:
        return naive_forecast(prices, horizon)

    result = best_fit.get_forecast(steps=horizon)
    idx = _future_index(prices, horizon)
    conf_int = result.conf_int(alpha=0.05)
    return {
        "forecast": pd.Series(result.predicted_mean.values, index=idx, name="arima"),
        "lower": pd.Series(conf_int.iloc[:, 0].values, index=idx),
        "upper": pd.Series(conf_int.iloc[:, 1].values, index=idx),
    }


FORECAST_MODELS = {
    "Naive (random walk)": naive_forecast,
    "ETS (Holt linear trend)": ets_forecast,
    "ARIMA (auto order)": arima_forecast,
}


def forecast_all_assets(
    prices: pd.DataFrame,
    horizon: int,
    model_name: str = "ETS (Holt linear trend)",
) -> pd.DataFrame:
    """Run the chosen model independently per asset (no cross-asset correlation in
    the forecast itself — that's handled downstream by the covariance matrix used
    in optimization) and return a single DataFrame of forecasted prices."""
    model_fn = FORECAST_MODELS[model_name]
    forecasts = {}
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        forecasts[ticker] = model_fn(series, horizon)["forecast"]
    return pd.DataFrame(forecasts)