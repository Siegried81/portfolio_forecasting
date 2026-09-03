"""
Portfolio Forecasting & Optimization — Streamlit app.

Layout: a sidebar to configure the universe/dates/model, and four tabs:
  1. Overview          — prices, returns, correlation of the selected universe
  2. Efficient Frontier — historical mean-variance optimization
  3. Forecast & Compare — the brief's core ask: historical vs forecast-based vs
                           realized-optimal portfolios, side by side
  4. AI Analyst         — LLM commentary, news digest, and a grounded Q&A chatbot

Design choice: this file is the ORCHESTRATION layer only — it wires UI widgets to
the pure functions in src/. No financial formula and no LLM-prompting logic lives
here, so the calculation logic stays independently testable (see tests/) and the
UI stays skimmable.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ai_features import answer_portfolio_question, build_results_context, generate_commentary, generate_news_digest
from src.rag import build_chunks
from src.backtesting import forecast_win_rate, generate_expanding_windows, run_walk_forward
from src.config import (
    ALL_KNOWN_TICKERS,
    BENCHMARK_TICKER,
    DEFAULT_EQUITY_TICKERS,
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_MAX_WEIGHT_PER_ASSET,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_WALK_FORWARD_WINDOWS,
    FREQUENCY_TO_PERIODS_PER_YEAR,
    LLM_SETTINGS,
    MAX_WALK_FORWARD_WINDOWS,
    MEGA_CAP_TICKERS,
    MIN_WALK_FORWARD_WINDOWS,
    QUICK_DATE_RANGES,
    SP500_SECTOR_UNIVERSE,
    WALK_FORWARD_MIN_TRAIN_PERIODS,
)
from src.forecasting import FORECAST_MODELS, forecast_all_assets
from src.llm_client import LLMUnavailableError
from src.macro_data import fetch_current_risk_free_rate, fetch_macro_snapshot
from src.market_data import (
    MarketDataError,
    TwelveDataPlanRestricted,
    fetch_adjusted_close,
    fetch_fundamentals,
    fetch_vix_snapshot,
    resample_prices,
)
from src.metrics import apply_transaction_cost, compute_returns, compute_turnover, portfolio_returns, summarise_performance
from src.optimization import (
    efficient_frontier_points,
    forecast_mu,
    historical_mu_cov,
    optimize_max_sharpe,
    portfolio_performance,
    resolve_weight_bounds,
)

st.set_page_config(page_title="Portfolio Forecasting & Optimization", layout="wide", page_icon="📈")


# ----------------------------------------------------------------------------------
# Sidebar — all user inputs live here, nowhere else, so the rest of the app can
# just read st.session_state / the returned tuple without re-deriving inputs.
# ----------------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.header("Portfolio configuration")

    def _apply_universe_preset() -> None:
        preset = st.session_state["universe_preset_select"]
        if preset == "Brief default (5)":
            st.session_state["assets_select"] = DEFAULT_EQUITY_TICKERS.copy()
        elif preset == "Mega Caps (15)":
            st.session_state["assets_select"] = MEGA_CAP_TICKERS.copy()
        # "Custom / sector picker" leaves the current selection untouched — the
        # sector multiselect below is what drives it instead.

    st.sidebar.selectbox(
        "Universe preset", options=["Brief default (5)", "Mega Caps (15)", "Custom / sector picker"],
        key="universe_preset_select", on_change=_apply_universe_preset,
        help="Presets replace your current asset selection. Pick 'Custom / sector picker' to "
             "build a universe from GICS sectors instead.",
    )

    sector_tickers: list[str] = []
    if st.session_state.get("universe_preset_select") == "Custom / sector picker":
        selected_sectors = st.sidebar.multiselect(
            "GICS sectors", options=list(SP500_SECTOR_UNIVERSE.keys()),
            help="Adds every ticker from the selected sector(s) to the Assets list below — "
                 "still editable by hand afterwards.",
        )
        for sector in selected_sectors:
            sector_tickers.extend(SP500_SECTOR_UNIVERSE[sector])
        if sector_tickers and st.sidebar.button("Apply sector selection to Assets"):
            st.session_state["assets_select"] = sorted(set(sector_tickers))

    tickers = st.sidebar.multiselect(
        "Assets", options=ALL_KNOWN_TICKERS, default=DEFAULT_EQUITY_TICKERS, key="assets_select",
        help="Equities, sector picks, ETFs, commodities and FX proxies — all in one list. "
             "Add custom tickers below if you don't see what you're after.",
    )
    custom = st.sidebar.text_input("Add custom tickers (comma-separated)", value="")
    if custom.strip():
        tickers = list(dict.fromkeys(tickers + [t.strip().upper() for t in custom.split(",") if t.strip()]))

    today = dt.date.today()

    def _apply_quick_range() -> None:
        label = st.session_state["quick_range_select"]
        if label != "Custom":
            st.session_state["start_date_input"] = today - dt.timedelta(days=QUICK_DATE_RANGES[label])
            st.session_state["end_date_input"] = today

    st.sidebar.selectbox(
        "Quick range", options=["Custom"] + list(QUICK_DATE_RANGES.keys()),
        index=3,  # defaults to "5 ans" on first load, matching the previous hardcoded default
        key="quick_range_select", on_change=_apply_quick_range,
        help="Picks a preset window. Switch to 'Custom' to set exact dates below.",
    )
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input(
        "Start date", value=today - dt.timedelta(days=QUICK_DATE_RANGES["5 ans"]), max_value=today, key="start_date_input",
    )
    end_date = col2.date_input("End date", value=today, max_value=today, key="end_date_input")

    frequency = st.sidebar.selectbox("Frequency", options=["daily", "weekly", "monthly"], index=0)

    fred_rate = fetch_current_risk_free_rate()
    rf_default = fred_rate if fred_rate is not None else DEFAULT_RISK_FREE_RATE
    rf_label = "Risk-free rate (annual)" + (" — live 3M T-bill via FRED" if fred_rate is not None else "")
    risk_free_rate = st.sidebar.slider(rf_label, 0.0, 0.10, rf_default, 0.0025, format="%.2f%%")
    max_weight_per_asset = st.sidebar.slider(
        "Max weight per asset", 0.10, 1.00, DEFAULT_MAX_WEIGHT_PER_ASSET, 0.05, format="%.0f%%",
        help="Position-size cap applied to every optimized portfolio (including the hindsight "
             "'realized-optimal' benchmark). Standard institutional practice — without it, "
             "optimizers happily put 100% into whichever single name got lucky, which is "
             "mathematically valid but not how any real portfolio is actually run. Automatically "
             "relaxed if it's tighter than 1/(number of assets selected).",
    )
    allow_short_selling = st.sidebar.checkbox(
        "Allow short selling", value=False,
        help="Long-only by default (weights ≥ 0). Enabling this lets the optimizer take negative "
             "(short) positions up to the same magnitude as the max-weight cap — e.g. with a 35% "
             "cap, weights can range -35% to +35% per asset. Portfolio stays fully invested "
             "(weights still sum to 100%); this does NOT add gross leverage beyond that. "
             "Note: with few assets selected, a tight cap can leave no real room to short — e.g. "
             "3 assets at a 35% cap can push a losing asset down to about +30% at most, never "
             "negative, since the other two can only fund 70% of the portfolio between them. "
             "More assets or a higher cap gives shorts more room to actually bind.",
    )
    transaction_cost_bps = st.sidebar.slider(
        "Transaction cost (bps per rebalance)", 0, 50, int(DEFAULT_TRANSACTION_COST_BPS), 1,
        help="Charged as turnover × this rate every time a portfolio rebalances — once for the "
             "initial trade in the comparison above, and at every walk-forward window boundary "
             "below (which is also, in effect, your rebalancing frequency: a shorter forecast "
             "horizon means more windows, i.e. more frequent rebalancing, i.e. more cumulative "
             "cost — try shortening the horizon to see this drag show up). Set to 0 for the "
             "frictionless textbook comparison.",
    )
    horizon_unit = {"daily": "trading days", "weekly": "weeks", "monthly": "months"}[frequency]
    forecast_horizon = st.sidebar.slider(
        f"Forecast horizon ({horizon_unit})", min_value=10, max_value=90, value=DEFAULT_FORECAST_HORIZON_DAYS, step=5,
        help="This slice of history at the END of your date range is held out and forecasted — "
             "the actual prices in that slice are then used to compute the 'realized-optimal' benchmark. "
             f"Counted in periods at the selected frequency ({frequency}), not calendar days.",
    )
    forecast_model = st.sidebar.selectbox("Forecast model", options=list(FORECAST_MODELS.keys()), index=1)

    st.sidebar.divider()
    walk_forward_windows = st.sidebar.slider(
        "Walk-forward windows", min_value=MIN_WALK_FORWARD_WINDOWS, max_value=MAX_WALK_FORWARD_WINDOWS,
        value=DEFAULT_WALK_FORWARD_WINDOWS,
        help="Repeats the historical/forecast/realized comparison across this many expanding "
             "windows instead of just one, so the result is a distribution, not a single lucky "
             "(or unlucky) outcome. ARIMA is noticeably slower here than ETS/naive.",
    )

    return dict(
        tickers=tickers, start_date=start_date, end_date=end_date, frequency=frequency,
        risk_free_rate=risk_free_rate, forecast_horizon=forecast_horizon, forecast_model=forecast_model,
        walk_forward_windows=walk_forward_windows, max_weight_per_asset=max_weight_per_asset,
        allow_short_selling=allow_short_selling, transaction_cost_bps=transaction_cost_bps,
    )


# ----------------------------------------------------------------------------------
# Data loading — one shared call, cached inside market_data.py, so every tab
# reuses the same downloaded prices instead of re-fetching per tab.
# ----------------------------------------------------------------------------------
def load_data(config: dict) -> pd.DataFrame | None:
    tickers_with_benchmark = list(dict.fromkeys(config["tickers"] + [BENCHMARK_TICKER]))
    try:
        daily_prices = fetch_adjusted_close(tickers_with_benchmark, config["start_date"], config["end_date"])
    except MarketDataError as exc:
        st.error(f"Could not load market data: {exc}")
        return None
    return resample_prices(daily_prices, config["frequency"])


def render_macro_panel() -> dict:
    """
    Macro & risk context: 3M / 10Y Treasury yields, the term spread (classic
    recession signal when negative), and the VIX level (market "fear gauge").
    Returns a plain dict so it can also be folded into the AI commentary context —
    an analyst's read of portfolio results should account for the macro backdrop,
    not just the numbers in isolation.
    """
    st.subheader("Macro & risk context")
    macro = fetch_macro_snapshot()
    vix_series = fetch_vix_snapshot()
    vix_level = float(vix_series.iloc[-1]) if vix_series is not None and not vix_series.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("3M T-bill (FRED)", f"{macro['three_month_yield']:.2%}" if macro["three_month_yield"] is not None else "n/a")
    col2.metric("10Y Treasury (FRED)", f"{macro['ten_year_yield']:.2%}" if macro["ten_year_yield"] is not None else "n/a")

    spread = macro["term_spread_10y_3m"]
    spread_label = f"{spread:+.2%}" if spread is not None else "n/a"
    col3.metric(
        "10Y-3M spread", spread_label,
        delta="Inverted — recession signal" if (spread is not None and spread < 0) else None,
        delta_color="inverse",
    )

    if vix_level is not None:
        vix_regime = "Calm" if vix_level < 15 else "Normal" if vix_level < 25 else "Elevated"
        col4.metric("VIX (fear gauge)", f"{vix_level:.1f}", delta=vix_regime, delta_color="off")
    else:
        col4.metric("VIX (fear gauge)", "n/a")

    if macro["three_month_yield"] is None or macro["ten_year_yield"] is None:
        st.caption("Set FRED_API_KEY in .env for live Treasury yields — see README.")
        with st.expander("Debug: what this app process actually sees"):
            key = LLM_SETTINGS.fred_api_key
            if key:
                st.write(f"FRED_API_KEY: configured, length={len(key)}, starts with `{key[:4]}...`")
            else:
                st.write("FRED_API_KEY: **not set / empty** in this running process's environment.")
            st.write(f"Working directory: `{os.getcwd()}`")
            st.write(f".env exists at that path: `{os.path.exists(os.path.join(os.getcwd(), '.env'))}`")
    st.caption(
        "The 10Y-3M spread turning negative has preceded every US recession since the 1960s "
        "(with some false positives) — shown here as context, not a trading signal in itself."
    )

    return {"macro": macro, "vix_level": vix_level}


def render_overview_tab(prices: pd.DataFrame, tickers: list[str], periods_per_year: int) -> None:
    source = prices.attrs.get("source", "yfinance")
    if "twelvedata" in source:
        st.caption(f"⚠️ Data source: **{source}** — Yahoo Finance was unreachable, using the Twelve Data fallback.")
    elif "yahoo direct" in source:
        st.caption(f"ℹ️ Data source: **{source}** — the yfinance library failed, but a direct Yahoo API call succeeded.")

    st.subheader("Price history (adjusted close)")
    fig = go.Figure()
    for ticker in tickers:
        # Rebase to 100 at the start so assets with very different price levels
        # (e.g. AAPL ~$200 vs GOOG ~$150) are visually comparable on one chart.
        rebased = prices[ticker] / prices[ticker].iloc[0] * 100
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased.values, name=ticker, mode="lines"))
    fig.update_layout(yaxis_title="Rebased to 100", height=420, legend_title="Ticker")
    st.plotly_chart(fig, width='stretch')

    returns = compute_returns(prices[tickers])
    # BENCHMARK_TICKER is always fetched in load_data() regardless of the user's
    # selection, specifically so beta/Information Ratio are always computable here
    # even if the user didn't add SPY to their own asset list.
    benchmark_returns = compute_returns(prices[[BENCHMARK_TICKER]])[BENCHMARK_TICKER] if BENCHMARK_TICKER in prices.columns else None

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Per-asset annualised metrics")
        rows = []
        for ticker in tickers:
            asset_benchmark = benchmark_returns if ticker != BENCHMARK_TICKER else None  # a benchmark vs itself is meaningless
            m = summarise_performance(returns[ticker], periods_per_year=periods_per_year, benchmark_returns=asset_benchmark)
            rows.append({
                "Ticker": ticker,
                "Ann. return": f"{m['annual_return']:.1%}",
                "Ann. volatility": f"{m['annual_volatility']:.1%}",
                "Sharpe": f"{m['sharpe_ratio']:.2f}",
                "Sortino": f"{m['sortino_ratio']:.2f}",
                "Calmar": f"{m['calmar_ratio']:.2f}" if not pd.isna(m['calmar_ratio']) else "—",
                "Omega": f"{m['omega_ratio']:.2f}" if not pd.isna(m['omega_ratio']) else "—",
                "Beta (vs SPY)": f"{m.get('beta', float('nan')):.2f}" if 'beta' in m and not pd.isna(m['beta']) else "—",
                "Max drawdown": f"{m['max_drawdown']:.1%}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), width='stretch')

    with col_right:
        st.subheader("Correlation matrix")
        corr = returns.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
            colorscale="RdBu", reversescale=True, text=corr.round(2).values, texttemplate="%{text}",
        ))
        fig_corr.update_layout(height=380)
        st.plotly_chart(fig_corr, width='stretch')
        st.caption(
            "Low/negative correlation between holdings is what actually reduces "
            "portfolio-level risk below the average of the individual assets' risk — "
            "this matrix is the reason diversification works, not just a decorative chart."
        )

    st.divider()
    st.subheader("Fundamentals")
    st.caption(
        "Via Twelve Data (`/statistics`) — some fields may be unavailable on the free tier "
        "depending on the exchange/plan; unavailable fields show as '—' rather than failing "
        "the whole table. Fetched on demand (button, not automatic) to conserve API quota."
    )
    if not LLM_SETTINGS.twelvedata_api_key:
        st.info("Set `TWELVEDATA_API_KEY` in `.env` to enable this section.")
    elif st.button("Fetch fundamentals for selected assets"):
        if len(tickers) > 8:
            st.caption(
                f"⏳ {len(tickers)} tickers on an 8 req/min free-tier limit — this will take "
                f"~{len(tickers) * 8 // 60 + 1} min. Paced deliberately to avoid hitting the rate limit."
            )
        progress = st.progress(0.0, text="Fetching fundamentals...")
        rows = []
        plan_restricted = False
        for i, ticker in enumerate(tickers):
            if i > 0:
                time.sleep(7.5)  # keeps us under Twelve Data's free-tier 8 req/min, proactively
            if plan_restricted:
                # Already confirmed this plan can't use /statistics beyond the demo
                # symbol — no point burning quota (or the user's time) retrying it
                # per ticker; every subsequent call would just 403 identically.
                f = None
            else:
                try:
                    f = fetch_fundamentals(ticker)
                except TwelveDataPlanRestricted:
                    plan_restricted = True
                    f = None
            progress.progress((i + 1) / len(tickers), text=f"Fetching fundamentals... ({i + 1}/{len(tickers)})")
            rows.append({
                "Ticker": ticker,
                "Name": (f or {}).get("name") or "—",
                "Market cap": f"${f['market_cap']:,.0f}" if f and f.get("market_cap") else "—",
                "P/E (trailing)": f"{f['pe_ratio']:.1f}" if f and f.get("pe_ratio") else "—",
                "Forward P/E": f"{f['forward_pe']:.1f}" if f and f.get("forward_pe") else "—",
                "Beta": f"{f['beta']:.2f}" if f and f.get("beta") else "—",
                "Div. yield": f"{f['dividend_yield']:.2%}" if f and f.get("dividend_yield") else "—",
                "52w range": (
                    f"${f['52w_low']:.0f} – ${f['52w_high']:.0f}"
                    if f and f.get("52w_low") and f.get("52w_high") else "—"
                ),
            })
        progress.empty()
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), width='stretch')
        if plan_restricted:
            st.warning(
                "Your Twelve Data plan only includes `/statistics` for their public demo symbol "
                "(AAPL) — every other ticker returns a 403 'requires pro/ultra/venture/enterprise "
                "plan' error, confirmed directly against the API. This is a plan limitation, not "
                "a bug: see [twelvedata.com/pricing](https://twelvedata.com/pricing) if you want "
                "fundamentals for more than one demo ticker."
            )
        elif all(r["Market cap"] == "—" and r["P/E (trailing)"] == "—" for r in rows):
            st.warning(
                "No fundamentals fields came back for any ticker — this likely means your "
                "Twelve Data plan doesn't include the `/statistics` endpoint's fields. "
                "Check your plan at twelvedata.com/pricing."
                )


def render_frontier_tab(
    prices: pd.DataFrame, tickers: list[str], risk_free_rate: float, periods_per_year: int, max_weight_per_asset: float,
    allow_short_selling: bool,
) -> pd.Series:
    st.subheader("Mean-variance efficient frontier (historical)")
    mu, cov = historical_mu_cov(prices[tickers], periods_per_year)
    weight_bounds = resolve_weight_bounds(max_weight_per_asset, allow_short_selling)
    weights = optimize_max_sharpe(mu, cov, risk_free_rate=risk_free_rate, weight_bounds=weight_bounds)
    perf = portfolio_performance(mu, cov, weights, risk_free_rate)
    frontier = efficient_frontier_points(mu, cov, weight_bounds=weight_bounds)

    col_chart, col_weights = st.columns([2, 1])
    with col_chart:
        fig = go.Figure()
        if not frontier.empty:
            fig.add_trace(go.Scatter(
                x=frontier["volatility"], y=frontier["return"], mode="lines",
                name="Efficient frontier", line=dict(color="#0E4C92"),
            ))
        # Individual assets, for context — shows why diversifying beats holding any single name
        for ticker in tickers:
            fig.add_trace(go.Scatter(
                x=[cov.loc[ticker, ticker] ** 0.5], y=[mu[ticker]], mode="markers+text",
                text=[ticker], textposition="top center", marker=dict(size=9), name=ticker, showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=[perf["expected_volatility"]], y=[perf["expected_return"]], mode="markers",
            marker=dict(size=16, color="red", symbol="star"), name="Max-Sharpe portfolio",
        ))
        fig.update_layout(xaxis_title="Annualised volatility", yaxis_title="Annualised expected return", height=460)
        st.plotly_chart(fig, width='stretch')

    with col_weights:
        st.metric("Expected annual return", f"{perf['expected_return']:.1%}")
        st.metric("Expected annual volatility", f"{perf['expected_volatility']:.1%}")
        st.metric("Expected Sharpe ratio", f"{perf['expected_sharpe']:.2f}")
        st.write("**Optimal weights (max Sharpe):**")
        weights_display = weights[weights > 0.001].sort_values(ascending=False)
        st.dataframe(weights_display.map(lambda w: f"{w:.1%}").rename("Weight"), width='stretch')

    st.caption(
        "Note: 'expected' figures above come from the optimizer's own mu/covariance inputs "
        "(historical, Ledoit-Wolf shrinkage) — they are the optimizer's target, not a guarantee. "
        "See the Forecast & Compare tab for how this historical optimum performs OUT of sample."
    )
    return weights


def render_forecast_compare_tab(
    prices: pd.DataFrame, tickers: list[str], config: dict,
) -> tuple[dict, dict, dict] | None:
    """
    The heart of the brief: split history into a training window and a held-out
    window of length `forecast_horizon`. Build THREE portfolios and compare them
    on the SAME held-out window, so it's a fair, apples-to-apples comparison:

      1. Historical-optimal : weights from mu/cov estimated on the TRAINING window only
      2. Forecast-optimal   : weights from mu forecasted FOR the held-out window
                               (cov still historical — see optimization.py docstring)
      3. Realized-optimal   : weights from mu/cov estimated on the held-out window's
                               ACTUAL returns — the hindsight benchmark

    All three portfolios' weights are then applied to the SAME actual realized
    returns of the held-out window, so the bar chart below answers the brief's
    exact question: "did forecasting help, and by how much vs. doing nothing
    fancy, vs. the best that was theoretically achievable?"
    """
    horizon = config["forecast_horizon"]
    if len(prices) <= horizon + 30:
        st.warning(
            f"Not enough history for a {horizon}-day held-out window — widen the date "
            "range or shorten the forecast horizon in the sidebar."
        )
        return None

    train_prices = prices.iloc[:-horizon]
    test_prices = prices.iloc[-horizon:]
    test_returns = compute_returns(test_prices[tickers])
    periods_per_year = FREQUENCY_TO_PERIODS_PER_YEAR[config["frequency"]]
    rf = config["risk_free_rate"]

    with st.spinner(f"Fitting {config['forecast_model']} per asset..."):
        forecasted_prices = forecast_all_assets(train_prices[tickers], horizon, config["forecast_model"])

    weight_bounds = resolve_weight_bounds(config["max_weight_per_asset"], config["allow_short_selling"])

    # --- 1. Historical-optimal (trained on train_prices only) ---
    mu_hist, cov_hist = historical_mu_cov(train_prices[tickers], periods_per_year)
    w_historical = optimize_max_sharpe(mu_hist, cov_hist, rf, weight_bounds)

    # --- 2. Forecast-optimal (mu from the forecast, cov still historical) ---
    mu_fcst = forecast_mu(train_prices.iloc[-1][tickers], forecasted_prices, horizon, periods_per_year)
    w_forecast = optimize_max_sharpe(mu_fcst, cov_hist, rf, weight_bounds)

    # --- 3. Realized-optimal (hindsight: fitted on the actual held-out returns) ---
    mu_real, cov_real = historical_mu_cov(test_prices[tickers], periods_per_year)
    w_realized = optimize_max_sharpe(mu_real, cov_real, rf, weight_bounds)

    # Apply all three weight vectors to the SAME actual realized returns
    portfolios = {"Historical-based": w_historical, "Forecast-based": w_forecast, "Realized-optimal": w_realized}
    test_benchmark_returns = (
        compute_returns(test_prices[[BENCHMARK_TICKER]])[BENCHMARK_TICKER] if BENCHMARK_TICKER in prices.columns else None
    )
    cost_bps = config["transaction_cost_bps"]
    realized_metrics = {}
    for name, w in portfolios.items():
        port_returns = portfolio_returns(test_returns, w)
        if cost_bps > 0:
            turnover = compute_turnover(w, None)  # starting from cash — establishing this position from scratch
            port_returns = apply_transaction_cost(port_returns, turnover, cost_bps)
        realized_metrics[name] = summarise_performance(port_returns, rf, periods_per_year, test_benchmark_returns)

    st.subheader(f"Out-of-sample comparison — last {horizon} periods")
    st.caption(
        "All three portfolios are weighted differently, but evaluated on the exact SAME actual "
        "price moves below — this isolates how much the ALLOCATION choice mattered."
    )

    metrics_df = pd.DataFrame(realized_metrics).T
    display_df = metrics_df.copy()
    for col in ["period_return", "annual_return", "annual_volatility", "max_drawdown", "var_95", "cvar_95"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1%}")
    for col in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio", "beta", "information_ratio", "treynor_ratio"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if not pd.isna(x) and np.isfinite(x) else "—")
    display_df = display_df.drop(columns=["n_periods"], errors="ignore")
    display_df = display_df.rename(columns={
        "period_return": f"Return ({horizon} periods, not annualised)",
        "annual_return": "Ann. return", "annual_volatility": "Ann. vol.", "sharpe_ratio": "Sharpe",
        "sortino_ratio": "Sortino", "calmar_ratio": "Calmar", "omega_ratio": "Omega",
        "max_drawdown": "Max DD", "var_95": "VaR 95%", "cvar_95": "CVaR 95%",
        "beta": "Beta (vs SPY)", "information_ratio": "Info. Ratio", "treynor_ratio": "Treynor",
    })
    st.dataframe(display_df, width='stretch')
    st.caption(
        "**Read the 'Ann. return' column with the raw window return next to it in mind**: "
        f"this comparison covers only {horizon} periods — annualising a short window's return "
        "amplifies it mathematically (a strong run over a few months can annualise to a triple-"
        "digit number that never actually occurred over a full year). The raw, non-annualised "
        "return shows what actually happened in this specific window."
    )
    st.caption(
        "Calmar = return per unit of worst drawdown lived through. Omega = full-distribution "
        "gain/loss ratio (catches skew Sharpe/Sortino can miss). Information Ratio = consistency "
        "of the edge over SPY. Treynor = return per unit of market (systematic) risk, vs Sharpe's "
        "total-risk denominator."
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(realized_metrics.keys()), y=[m["sharpe_ratio"] for m in realized_metrics.values()], name="Sharpe"))
    fig.add_trace(go.Bar(x=list(realized_metrics.keys()), y=[m["sortino_ratio"] for m in realized_metrics.values()], name="Sortino"))
    fig.update_layout(barmode="group", height=380, yaxis_title="Ratio")
    st.plotly_chart(fig, width='stretch')

    gap = realized_metrics["Forecast-based"]["sharpe_ratio"] - realized_metrics["Historical-based"]["sharpe_ratio"]
    ceiling = realized_metrics["Realized-optimal"]["sharpe_ratio"] - realized_metrics["Historical-based"]["sharpe_ratio"]
    st.info(
        f"Forecasting moved realised Sharpe by **{gap:+.2f}** vs. the historical-based portfolio "
        f"(the theoretical ceiling, if the forecast had been perfect, was **{ceiling:+.2f}**). "
        "A small or negative gap here is the expected, honest result for short-horizon price "
        "forecasting — see the caveat below."
    )
    st.caption(
        "⚠️ Per the efficient market hypothesis, short-horizon equity price forecasts carry very "
        "limited genuine predictive power — most of a forecast model's apparent skill on any single "
        "backtest window can be noise. Treat the 'Forecast-based' column as a methodology demonstration, "
        "not a signal to act on."
    )

    st.divider()
    render_walk_forward_section(prices, tickers, config)

    return realized_metrics["Historical-based"], realized_metrics["Forecast-based"], realized_metrics["Realized-optimal"]


def render_walk_forward_section(prices: pd.DataFrame, tickers: list[str], config: dict) -> None:
    """
    Multi-window validation of the single comparison above. Separated into its
    own function (rather than inlined) because it's conceptually a second,
    independent analysis — same three portfolio types, but judged across several
    expanding windows instead of one — and keeping it separate makes that clear
    in the code, not just in the UI.
    """
    st.subheader("Walk-forward validation (multi-window)")
    st.caption(
        "The comparison above uses ONE held-out window — which can be lucky or unlucky. "
        "This repeats it across several expanding windows (each refit uses all data available "
        "up to that point) to check whether any edge holds up, or was a one-off."
    )

    horizon = config["forecast_horizon"]
    n_windows = config["walk_forward_windows"]
    periods_per_year = FREQUENCY_TO_PERIODS_PER_YEAR[config["frequency"]]

    possible_windows = generate_expanding_windows(len(prices), horizon, WALK_FORWARD_MIN_TRAIN_PERIODS, n_windows)
    if len(possible_windows) < MIN_WALK_FORWARD_WINDOWS:
        st.warning(
            f"Not enough history for {MIN_WALK_FORWARD_WINDOWS}+ walk-forward windows at this "
            f"horizon ({len(possible_windows)} possible) — widen the date range or shorten the "
            "forecast horizon in the sidebar."
        )
        return

    if config["forecast_model"] == "ARIMA (auto order)" and len(tickers) * len(possible_windows) > 20:
        st.caption("⏳ ARIMA across many windows/assets can take 20-30s — ETS is much faster if you're iterating.")

    with st.spinner(f"Running {len(possible_windows)} walk-forward windows..."):
        results = run_walk_forward(
            prices, tickers, horizon, n_windows, config["forecast_model"],
            config["risk_free_rate"], periods_per_year, WALK_FORWARD_MIN_TRAIN_PERIODS,
            config["max_weight_per_asset"], config["allow_short_selling"], config["transaction_cost_bps"],
        )

    fig = go.Figure()
    for portfolio in ["Historical-based", "Forecast-based", "Realized-optimal"]:
        subset = results[results["portfolio"] == portfolio]
        fig.add_trace(go.Box(y=subset["sharpe_ratio"], name=portfolio, boxpoints="all"))
    fig.update_layout(yaxis_title=f"Sharpe ratio across {len(possible_windows)} windows", height=380)
    st.plotly_chart(fig, width='stretch')

    summary = results.groupby("portfolio")[["sharpe_ratio", "sortino_ratio", "annual_return"]].agg(["mean", "std"])
    summary.columns = [f"{a} ({b})" for a, b in summary.columns]
    st.dataframe(summary.style.format("{:.2f}"), width='stretch')

    win_rate = forecast_win_rate(results)
    if not pd.isna(win_rate):
        st.info(
            f"Forecast-based beat Historical-based on Sharpe in **{win_rate:.0%}** of the "
            f"{len(possible_windows)} windows tested. Close to 50% is the honest expectation for "
            "short-horizon price forecasting (see the caveat above) — a rate consistently well "
            "above 50% across many windows would be the real signal that the forecasting step "
            "is adding value, not just noise from a single lucky split."
        )


def render_ai_analyst_tab(
    weights: pd.Series, hist_metrics: dict, fcst_metrics: dict | None, real_metrics: dict | None,
    tickers: list[str], macro_context: dict | None = None,
) -> None:
    st.subheader("AI Portfolio Analyst")

    if not LLM_SETTINGS.groq_api_key:
        st.warning(
            "No GROQ_API_KEY configured — this tab will try a local Ollama fallback "
            "(`ollama serve` running on this machine). Set GROQ_API_KEY in your .env for "
            "the hosted, faster path. See README.md."
        )

    results_context = build_results_context(weights, hist_metrics, fcst_metrics, real_metrics, macro_context)
    st.session_state.setdefault("results_context", results_context)
    st.session_state["results_context"] = results_context  # always keep the latest run's numbers

    col_commentary, col_news = st.columns(2)

    with col_commentary:
        st.markdown("**Analyst commentary**")
        if st.button("Generate commentary", key="gen_commentary"):
            with st.spinner("Thinking..."):
                try:
                    text, backend = generate_commentary(results_context)
                    st.session_state["commentary"] = (text, backend)
                except LLMUnavailableError as exc:
                    st.error(f"LLM unavailable: {exc}")
        if "commentary" in st.session_state:
            text, backend = st.session_state["commentary"]
            st.write(text)
            st.caption(f"Generated by: {backend}")

    with col_news:
        st.markdown("**News digest**")
        if st.button("Fetch & summarise news", key="gen_news"):
            with st.spinner("Fetching headlines..."):
                # No reliable free "ticker -> company name" API — a small static map
                # covers the brief's example universe; unmapped tickers just use the symbol.
                company_names = {
                    "AAPL": "Apple", "MSFT": "Microsoft", "TSLA": "Tesla",
                    "AMZN": "Amazon", "GOOG": "Google", "SPY": "S&P 500",
                    "QQQ": "Nasdaq", "TLT": "Treasury bonds", "GLD": "Gold", "VNQ": "REIT real estate",
                }
                digest, backend, articles = generate_news_digest(tickers, company_names)
                st.session_state["news"] = (digest, backend, articles)
                st.session_state["news_chunks"] = build_chunks(articles)
        if "news" in st.session_state:
            digest, backend, articles = st.session_state["news"]
            st.write(digest)
            if articles:
                provider_counts = pd.Series([a["provider"] for a in articles]).value_counts()
                breakdown = ", ".join(f"{n} {p}" for p, n in provider_counts.items())
                st.caption(f"Generated by: {backend} · {len(articles)} items across {len(provider_counts)} sources ({breakdown})")
                with st.expander("Sources"):
                    for a in articles:
                        st.markdown(f"- [{a['provider']}] [{a['title']}]({a['url']}) — *{a['source']}* ({a['ticker']})")

    st.divider()
    st.markdown("**Ask about your results**")
    st.caption("Grounded in the numbers computed above — won't invent figures that aren't there.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("e.g. Why is the Sortino ratio higher than the Sharpe ratio here?")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, backend = answer_portfolio_question(
                        question, st.session_state["results_context"], st.session_state["chat_history"][:-1],
                        st.session_state.get("news_chunks"),
                    )
                    st.write(answer)
                    st.caption(f"Generated by: {backend}")
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                except LLMUnavailableError as exc:
                    st.error(f"LLM unavailable: {exc}")


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main() -> None:
    st.title("📈 Portfolio Forecasting & Optimization")
    st.caption(
        "Mean-variance optimization, forecast-driven allocation, and an AI analyst — "
        "built as a technical case study (see README.md for methodology & scope)."
    )

    config = render_sidebar()
    if not config["tickers"]:
        st.info("Pick at least one asset in the sidebar to get started.")
        return

    prices = load_data(config)
    if prices is None:
        return

    periods_per_year = FREQUENCY_TO_PERIODS_PER_YEAR[config["frequency"]]
    tab_overview, tab_frontier, tab_compare, tab_ai = st.tabs(
        ["Overview", "Efficient Frontier", "Forecast & Compare", "AI Analyst"]
    )

    with tab_overview:
        macro_context = render_macro_panel()
        st.divider()
        render_overview_tab(prices, config["tickers"], periods_per_year)

    with tab_frontier:
        weights = render_frontier_tab(
            prices, config["tickers"], config["risk_free_rate"], periods_per_year, config["max_weight_per_asset"],
            config["allow_short_selling"],
        )

    with tab_compare:
        result = render_forecast_compare_tab(prices, config["tickers"], config)
        hist_m, fcst_m, real_m = result if result else (None, None, None)

    with tab_ai:
        if hist_m is None:
            st.info("Run the Forecast & Compare tab first (needs a valid date range/horizon) to unlock AI commentary.")
        else:
            render_ai_analyst_tab(weights, hist_m, fcst_m, real_m, config["tickers"], macro_context)


if __name__ == "__main__":
    main()