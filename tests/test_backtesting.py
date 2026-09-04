"""
Walk-forward (multi-window) backtesting.

Rationale (why this exists on top of the single train/test split in app.py's
"Forecast & Compare" tab): one held-out window tells you whether forecasting
helped THAT ONE TIME — which can easily be luck. Walk-forward repeats the exact
same historical/forecast/realized comparison across several EXPANDING windows
(window k's training data = everything up to that point; test data = the next
`horizon` periods), producing a distribution of outcomes instead of one number.
This is standard practice in any credible backtest and is what separates
"I built an optimizer" from "I validated that it actually adds value" — the
second is what a quant/portfolio-management interviewer is really probing for.

Kept in its own module (not bolted onto app.py) because it's pure orchestration
over the SAME pure functions used by the single-window comparison — metrics.py,
optimization.py, forecasting.py — so the two comparisons can never silently
compute things differently.
"""
from __future__ import annotations

import pandas as pd

from src.forecasting import forecast_all_assets
from src.metrics import apply_transaction_cost, compute_returns, compute_turnover, portfolio_returns, summarise_performance
from src.config import DEFAULT_COV_METHOD, DEFAULT_PCA_FACTORS
from src.optimization import forecast_mu, historical_mu_cov, optimize_max_sharpe, resolve_weight_bounds


def generate_expanding_windows(
    n_periods: int, horizon: int, min_train_periods: int, max_windows: int,
) -> list[tuple[int, int]]:
    """
    Build (train_end, test_end) index pairs for expanding-window walk-forward.
    Window 1 trains on [0, min_train_periods), tests on the next `horizon`
    periods; window 2 trains on [0, min_train_periods + horizon), and so on —
    each window sees strictly more training data than the last, which mirrors
    how a real strategy would actually be refit over time (using everything
    known so far), rather than an artificially fixed-size rolling window.
    """
    windows = []
    train_end = min_train_periods
    while train_end + horizon <= n_periods and len(windows) < max_windows:
        windows.append((train_end, train_end + horizon))
        train_end += horizon
    return windows


def run_walk_forward(
    prices: pd.DataFrame,
    tickers: list[str],
    horizon: int,
    n_windows: int,
    forecast_model: str,
    risk_free_rate: float,
    periods_per_year: int,
    min_train_periods: int,
    max_weight_per_asset: float = 1.0,
    allow_short_selling: bool = False,
    transaction_cost_bps: float = 0.0,
    cov_method: str = DEFAULT_COV_METHOD,
    n_factors: int = DEFAULT_PCA_FACTORS,
) -> pd.DataFrame:
    """
    Run the historical / forecast / realized-optimal comparison across several
    expanding windows. Returns a tidy long-format DataFrame — one row per
    (window, portfolio type) — ready for both a summary table and a box plot.

    Columns: window, window_end (date), portfolio, annual_return,
    annual_volatility, sharpe_ratio, sortino_ratio, max_drawdown, var_95, cvar_95.

    Transaction costs: `transaction_cost_bps` charges turnover × rate at EVERY
    window boundary for every portfolio type, tracked independently (each
    portfolio type rebalances against its OWN previous window's weights, not a
    shared reference) — this is also where "rebalancing frequency" shows up in
    practice: a shorter `horizon` means more windows over the same history, i.e.
    more frequent rebalancing, i.e. more cumulative cost drag realised here.

    `cov_method`/`n_factors` (added 2026-09) are passed straight through to
    every `historical_mu_cov` call below (both the historical-training-window
    fit and the realized/hindsight fit) — so a walk-forward run over a wide
    universe uses the SAME covariance estimator throughout, not Ledoit-Wolf in
    one place and PCA in another. Defaults preserve the original
    Ledoit-Wolf-everywhere behaviour for any existing caller that doesn't pass
    these.
    """
    windows = generate_expanding_windows(len(prices), horizon, min_train_periods, n_windows)
    records: list[dict] = []
    previous_weights: dict[str, pd.Series] = {}

    for window_idx, (train_end, test_end) in enumerate(windows, start=1):
        train_prices = prices.iloc[:train_end]
        test_prices = prices.iloc[train_end:test_end]
        test_returns = compute_returns(test_prices[tickers])

        weight_bounds = resolve_weight_bounds(max_weight_per_asset, allow_short_selling)

        # 1. Historical-based — fit on this window's training slice only
        mu_hist, cov_hist = historical_mu_cov(
            train_prices[tickers], periods_per_year, cov_method=cov_method, n_factors=n_factors,
        )
        w_hist = optimize_max_sharpe(mu_hist, cov_hist, risk_free_rate, weight_bounds)

        # 2. Forecast-based — forecast over this window's test horizon, cov stays historical
        forecasted = forecast_all_assets(train_prices[tickers], horizon, forecast_model)
        mu_fcst = forecast_mu(train_prices.iloc[-1][tickers], forecasted, horizon, periods_per_year)
        w_fcst = optimize_max_sharpe(mu_fcst, cov_hist, risk_free_rate, weight_bounds)

        # 3. Realized-optimal — hindsight fit on this window's actual test-period returns
        mu_real, cov_real = historical_mu_cov(
            test_prices[tickers], periods_per_year, cov_method=cov_method, n_factors=n_factors,
        )
        w_real = optimize_max_sharpe(mu_real, cov_real, risk_free_rate, weight_bounds)

        for label, weights in [("Historical-based", w_hist), ("Forecast-based", w_fcst), ("Realized-optimal", w_real)]:
            port_returns = portfolio_returns(test_returns, weights)
            if transaction_cost_bps > 0:
                turnover = compute_turnover(weights, previous_weights.get(label))
                port_returns = apply_transaction_cost(port_returns, turnover, transaction_cost_bps)
            previous_weights[label] = weights
            perf = summarise_performance(port_returns, risk_free_rate, periods_per_year)
            records.append({"window": window_idx, "window_end": test_prices.index[-1], "portfolio": label, **perf})

    return pd.DataFrame(records)


def summarise_walk_forward(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std of each metric per portfolio type across all windows —
    the distribution summary that answers "does forecasting help ON AVERAGE,
    and how consistently?", not just "did it help once"."""
    return results.groupby("portfolio")[
        ["annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
    ].agg(["mean", "std"])


def forecast_win_rate(results: pd.DataFrame) -> float:
    """Fraction of windows where Forecast-based Sharpe beat Historical-based Sharpe —
    a simple, interview-friendly headline number to pair with the full distribution."""
    pivot = results.pivot(index="window", columns="portfolio", values="sharpe_ratio")
    if "Forecast-based" not in pivot.columns or "Historical-based" not in pivot.columns:
        return float("nan")
    return float((pivot["Forecast-based"] > pivot["Historical-based"]).mean())