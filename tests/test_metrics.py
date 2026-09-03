"""
Unit tests for the pure finance calculations in src/metrics.py.

These are worth having precisely BECAUSE this is a finance app: a silently wrong
Sharpe/drawdown formula is much more damaging (and much harder to spot by eye)
than a UI bug. Tests use small hand-checkable synthetic series rather than real
market data, so expected values can be verified by hand, not just "does it run".
"""
import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    information_ratio,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    summarise_performance,
    treynor_ratio,
    value_at_risk,
)


def test_annualised_return_matches_hand_calculation():
    # 2 periods of +10% each, compounded: 1.1 * 1.1 - 1 = 0.21
    returns = pd.Series([0.10, 0.10])
    result = annualised_return(returns, periods_per_year=2)
    assert result == pytest.approx(0.21, abs=1e-9)


def test_annualised_return_zero_for_flat_series():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert annualised_return(returns, periods_per_year=252) == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_on_known_path():
    # Wealth path: 1.0 -> 1.2 -> 0.9 -> 1.05  => trough 0.9 vs peak 1.2 => -25%
    returns = pd.Series([0.20, -0.25, 0.1667])
    dd = max_drawdown(returns)
    assert dd == pytest.approx(-0.25, abs=1e-3)


def test_max_drawdown_is_zero_for_monotonic_gains():
    returns = pd.Series([0.01, 0.02, 0.015])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)


def test_sortino_ignores_upside_volatility():
    # Same mean/vol profile as Sharpe would see, but Sortino should differ because
    # only the negative deviations count towards the denominator.
    calm_upside = pd.Series([0.05, -0.01, 0.05, -0.01])  # big upside swings, small downside
    volatile_upside = pd.Series([0.20, -0.01, 0.20, -0.01])  # even bigger upside swings, same downside

    sharpe_calm = sharpe_ratio(calm_upside, risk_free_rate=0.0, periods_per_year=4)
    sharpe_volatile = sharpe_ratio(volatile_upside, risk_free_rate=0.0, periods_per_year=4)
    sortino_calm = sortino_ratio(calm_upside, risk_free_rate=0.0, periods_per_year=4)
    sortino_volatile = sortino_ratio(volatile_upside, risk_free_rate=0.0, periods_per_year=4)

    # Both series share the EXACT SAME downside observations (-0.01, -0.01), only
    # the upside differs. Sharpe's denominator (total std dev) grows with the bigger
    # upside swings, while Sortino's denominator (downside deviation only) does not.
    # So Sortino should reward the bigger upside relatively MORE than Sharpe does -
    # i.e. the Sortino/Sharpe ratio should be higher for the more upside-volatile series.
    ratio_calm = sortino_calm / sharpe_calm
    ratio_volatile = sortino_volatile / sharpe_volatile
    assert ratio_volatile > ratio_calm


def test_value_at_risk_and_cvar_ordering():
    # CVaR (average of the tail) must always be <= VaR (the tail threshold itself)
    # for a left-tail loss convention, since the tail average includes worse outcomes.
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    var_95 = value_at_risk(returns, confidence=0.95)
    cvar_95 = conditional_value_at_risk(returns, confidence=0.95)
    assert cvar_95 <= var_95


def test_annualised_volatility_scales_with_sqrt_time():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01])
    vol_daily = annualised_volatility(returns, periods_per_year=252)
    vol_monthly = annualised_volatility(returns, periods_per_year=12)
    # Same raw series, higher periods_per_year -> higher annualised vol (sqrt scaling)
    assert vol_daily > vol_monthly


def test_calmar_ratio_manual_calculation():
    # Wealth path 1.0 -> 1.2 -> 0.9 -> 1.05 => max drawdown -25% (same series as
    # the max_drawdown test above). Annualised return over 3 periods/year=3 is
    # the compounded growth itself: 1.05 - 1 = 0.05 -> Calmar = 0.05 / 0.25 = 0.2
    returns = pd.Series([0.20, -0.25, 0.1667])
    calmar = calmar_ratio(returns, periods_per_year=3)
    expected = annualised_return(returns, periods_per_year=3) / 0.25
    assert calmar == pytest.approx(expected, rel=1e-6)


def test_calmar_ratio_nan_when_no_drawdown():
    returns = pd.Series([0.01, 0.02, 0.015])  # monotonic gains -> max_drawdown = 0
    assert np.isnan(calmar_ratio(returns))


def test_omega_ratio_above_one_for_upside_skewed_returns():
    # Big consistent gains, small consistent losses -> Omega should clearly exceed 1
    skewed = pd.Series([0.03, 0.03, 0.03, -0.01, -0.01])
    omega = omega_ratio(skewed, threshold=0.0)
    assert omega > 1.0


def test_omega_ratio_below_one_for_downside_skewed_returns():
    skewed = pd.Series([0.01, 0.01, -0.03, -0.03, -0.03])
    omega = omega_ratio(skewed, threshold=0.0)
    assert omega < 1.0


def test_information_ratio_nan_when_tracking_error_zero():
    # Portfolio identical to its benchmark -> zero tracking error -> undefined IR
    benchmark = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    portfolio = benchmark.copy()
    assert np.isnan(information_ratio(portfolio, benchmark, periods_per_year=252))


def test_information_ratio_positive_when_portfolio_outperforms():
    np.random.seed(11)
    benchmark = pd.Series(np.random.normal(0.0004, 0.01, 300))
    portfolio = benchmark + 0.0003  # small but real, noisy-free outperformance each period
    ir = information_ratio(portfolio, benchmark, periods_per_year=252)
    assert ir > 0


def test_treynor_ratio_manual_calculation():
    # Single-period annual return 15%, risk-free 4%, beta 1.5 -> (0.15-0.04)/1.5
    treynor = treynor_ratio(pd.Series([0.15]), beta=1.5, risk_free_rate=0.04, periods_per_year=1)
    assert treynor == pytest.approx((0.15 - 0.04) / 1.5, rel=1e-9)


def test_treynor_ratio_nan_for_zero_beta():
    assert np.isnan(treynor_ratio(pd.Series([0.10]), beta=0.0, periods_per_year=1))


def test_summarise_performance_includes_benchmark_metrics_only_when_provided():
    np.random.seed(12)
    returns = pd.Series(np.random.normal(0.0005, 0.012, 200))
    benchmark = pd.Series(np.random.normal(0.0003, 0.010, 200))

    without_benchmark = summarise_performance(returns, periods_per_year=252)
    assert "calmar_ratio" in without_benchmark and "omega_ratio" in without_benchmark
    assert "information_ratio" not in without_benchmark and "beta" not in without_benchmark

    with_benchmark = summarise_performance(returns, periods_per_year=252, benchmark_returns=benchmark)
    assert "information_ratio" in with_benchmark and "treynor_ratio" in with_benchmark and "beta" in with_benchmark
