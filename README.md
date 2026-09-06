# Portfolio Forecasting & Optimization

I built this as a technical case study for BeCode's AI & Data Science bootcamp (GenAI Developer
track): an interactive Streamlit app that builds and compares three portfolios —
**historical-based**, **forecast-based**, and **realized-optimal (hindsight)** — using
mean-variance optimization, plus an AI analyst layer I added on top (LLM commentary, news digest,
grounded Q&A chatbot).

## Live demo
- Streamlit Community Cloud: [portfolio-forecasting-sieg.streamlit.app](https://portfolio-forecasting-sieg.streamlit.app/)
- Render: [portfolio-forecasting.onrender.com](https://portfolio-forecasting.onrender.com)

## What it does, in one paragraph

You pick a universe of stocks/ETFs, a date range, and a frequency. I compute historical returns,
volatility, correlation, and the max-Sharpe efficient-frontier portfolio. Then I hold out the last
N periods, forecast each asset's price over that window (ARIMA / Exponential Smoothing / naive
random walk — see "Why not Kats" below), build an optimal portfolio from the *forecasted* returns,
and compare its **actual, realized** out-of-sample performance against (a) the historical-based
portfolio and (b) the hindsight-optimal portfolio built from the *actual* returns of that same
window. An AI Analyst tab (Groq, with local Ollama fallback) narrates the results and answers
questions about them, grounded in the computed numbers — plus a news digest per ticker
cross-referencing NewsAPI, Finnhub, and SEC EDGAR filings.

## Contents

- [Screenshots](#screenshots)
- [Understanding the KPIs — formulas & how to read them](#understanding-the-kpis--formulas--how-to-read-them)
- [Repository structure](#repository-structure)
- [Key design decisions vs. the original brief](#key-design-decisions-vs-the-original-brief)
- [The three-portfolio comparison, precisely](#the-three-portfolio-comparison-precisely)
- [Walk-forward validation](#walk-forward-validation-why-one-comparison-isnt-enough)
- [Expanded universe & macro context](#expanded-universe--macro-context)
- [Setup](#setup)
- [Deployment](#deployment)
- [Sources (from the brief)](#sources-from-the-brief)
- [Key challenges](#key-challenges)
- [Known limitations](#known-limitations)
- [Next steps](#next-steps)

## Screenshots

**Overview — Macro & Risk context**
![Macro and risk context panel: FRED yields, term spread, VIX, Sahm Rule](images/overview_macrp_risks.png)

**Overview — Prices & Analytics**
![Rebased price history and time-series diagnostics table](images/prices_analysis.png)

**Efficient Frontier**
![Mean-variance efficient frontier with optimal weights, diversification ratio and HHI](images/efficient_frontier.png)

**Forecast & Compare — the three-portfolio comparison**
![Historical vs forecast-based vs realized-optimal portfolio comparison table and chart](images/forecast_compare.png)

**Walk-forward validation**
![Box plot of realized Sharpe ratio per portfolio type across expanding windows](images/walk_forward_validation.png)

**AI Analyst — commentary & sentiment**
![LLM-generated portfolio commentary](images/AI_portfolio_analyst.png)
![News sentiment by ticker with source attribution](images/sentiment_sources.png)

**Chatbot — grounded Q&A**
![Chatbot answering a question about portfolio allocation, grounded in computed numbers](images/chatbot_1.png)
![Chatbot follow-up question with context retained](images/chatbot_2.png)

## Understanding the KPIs — formulas & how to read them

Computed in `src/metrics.py` (unit-tested in `tests/test_metrics.py`) unless noted otherwise.
`r` = period return series (daily/weekly/monthly per the sidebar), `rf` = risk-free rate,
`n` = periods per year (252/52/12).

### 1. Return & risk — the building blocks

| Metric | Formula | How to read it |
|---|---|---|
| Annualised Return | `(∏(1+r))^(n/periods) − 1` | Geometric, not arithmetic mean × n — a ±50% sequence is 0%, not the naive average (classic trap). |
| Annualised Volatility | `std(r) × √n` | Total risk (upside + downside count equally). Not inherently "bad" — see Sortino/Omega for asymmetric views. |
| Max Drawdown | `min[W(t)/max(W(0..t)) − 1]`, `W(t)=∏(1+r)` | Worst peak-to-trough loss — correlates most with an investor actually panic-selling. |
| Ulcer Index (2026-09) | RMS of the % drawdown at *every* point (Peter Martin, 1987) | Captures *duration* underwater, not just depth. Lower is better; 0 = never dipped below a prior peak. |
| VaR 95% | 5th percentile of the return distribution (non-parametric) | "Worst 1-in-20 period loss exceeded X%" — says nothing about how much worse those periods got. |
| CVaR 95% | `mean(r \| r ≤ VaR_95)` | Average loss beyond VaR — always at least as bad; a large VaR/CVaR gap flags a fat, dangerous tail. |

### 2. Risk-adjusted return ratios — reward per unit of risk

| Metric | Formula | How to read it |
|---|---|---|
| Sharpe | `(mean(r−rf_period)/std(r−rf_period)) × √n` | Above 1 generally good, above 2 very good. Penalises upside volatility exactly as much as downside. |
| Sharpe SE (2026-09) | `√((1+0.5×SR_period²)/n)`, annualised (Lo, 2002) | Standard error of the Sharpe estimate itself — a rough 95% range is `Sharpe ± 1.96×SE`. Puts a real number behind "a Sharpe of 5.89 on 30 periods isn't reliable" instead of just an appeal to intuition. |
| Sortino | Same, denominator = downside deviation only (`√(mean((r−rf_period)² \| r−rf_period<0))`) | Sortino ≥ Sharpe is normal for equities — only downside swings count against it. |
| Calmar | `annual_return / \|max_drawdown\|` | Penalises only the single *worst* outcome lived through, not the whole spread — the number a risk committee asks for. |
| Omega | `Σ(r−threshold \| r>threshold) / \|Σ(r−threshold \| r<threshold)\|` | Uses the *entire* empirical distribution, so it diverges from Sharpe/Sortino exactly when returns are skewed/fat-tailed. `∞` (shown `—`) = zero losing periods in the sample. |
| Skewness / Kurtosis (2026-09) | 3rd / 4th standardised moments of `r` | Skew: 0 symmetric, negative = fatter *left* tail (large losses — the typical equity shape). Kurtosis (pandas excess convention): 0 = normal tails, positive = fatter than normal. |

### 3. Benchmark-relative metrics — vs. SPY

Computed whenever a benchmark series is available (SPY is always fetched in the background
regardless of your ticker selection, specifically so these are always computable).

| Metric | Formula | How to read it |
|---|---|---|
| Beta | `Cov(r, r_benchmark) / Var(r_benchmark)` | 1.0 moves with the market; >1.0 amplifies; <1.0 dampens; negative = hedge-like (rare). |
| Information Ratio | `annualised(mean(r−r_benchmark)) / (std(r−r_benchmark)×√n)` | *Consistency* of outperformance vs. SPY, not just its size — what active-mandate reviews lead with, not Sharpe. |
| Treynor | `(annual_return − rf) / beta` | Return per unit of *systematic* (market) risk, vs. Sharpe's total-risk denominator. |
| Jensen's Alpha | `annual_return − [rf + beta×(annual_benchmark_return − rf)]` | Return above what CAPM predicts given the portfolio's own beta — what a CAPM-literate interviewer asks for the moment Beta is on screen. |

### 4. Optimizer outputs — expected, not realized (Efficient Frontier tab)

These come from the optimizer's own inputs (historical mean/covariance, Ledoit-Wolf shrinkage) —
the optimizer's *target*, not a guarantee. Compare against the *realized* metrics in Forecast &
Compare to see the gap between expectation and outcome.

- **Expected return** `w·μ`, **expected volatility** `√(w·Σ·w)`, **expected Sharpe** `(expected_return−rf)/expected_volatility`.
- **Diversification ratio** (2026-09) = weighted-average individual asset volatility ÷ actual
  portfolio volatility. >1 whenever correlations are below 1 (the normal case) — the numeric
  version of the Overview tab's correlation matrix. =1 means diversification buys nothing.
- **Concentration / HHI** (2026-09) = `Σ w_i²`. Ranges `1/N` (equal-weighted) to `1.0` (single
  asset) — a quick check that the sidebar's max-weight cap is actually doing its job.

### 5. Forecast validation

**Forecast win rate** (walk-forward section, Forecast & Compare tab) — across every expanding
window, the fraction where the forecast-based portfolio's *realized* Sharpe beat the
historical-based one's. Near 50% across many windows is the honest, expected result for
short-horizon price forecasting (consistent with the efficient market hypothesis) — a rate
consistently well above 50% would be the real signal of a genuine edge. Don't read much into a
single window (see "Walk-forward validation" below for why).

### 6. Macro context (FRED — shapes interpretation, not portfolio-specific)

| Series | What it signals |
|---|---|
| VIX | "Fear gauge": <15 calm, 15–25 normal, >25 elevated stress. |
| 10Y–3M Treasury term spread | Negative has preceded every US recession since the 1960s (some false positives) — context, not a trading signal. |
| Risk-free rate (3M T-bill) | `rf` in every ratio above that needs one — overridable by hand in the sidebar. |
| CPI YoY, unemployment rate, Fed funds rate (2026-09) | The rest of the Fed's dual mandate plus the actual policy rate — the yield curve and VIX alone don't cover this. |
| Sahm Rule recession indicator (2026-09, `SAHMREALTIME`) | ≥0.50 (3-month avg. unemployment 0.50pt above its 12-month low) has coincided with every US recession's start since 1970, no false positives to date. |
| Baa corporate credit spread vs 10Y (2026-09, `BAA10Y`) | Corporate credit-risk appetite — a different stress channel from the yield curve or VIX. |
| Real GDP growth (2026-09, quarterly, `A191RL1Q225SBEA`) | Headline "GDP grew at X% annualized" — updates far less often than the rest of the panel. |
| Industrial Production YoY (2026-09, `INDPRO`) | Proxy for ISM Manufacturing PMI (paid/proprietary, no free API) — real output data capturing the same manufacturing-momentum signal. |

Every series above links to its FRED source page from a "Sources" expander under the macro panel.

### 7. Time-series diagnostics (Overview tab, per asset — series structure, not performance)

The kind of check done *before* trusting a forecast, not after.

- **ADF stationarity test** (on returns) — validates `forecasting.py`'s own ARIMA `d=1`
  first-differencing choice. "Yes" (p<0.05) is the expected, textbook result for returns (unlike
  price *levels*, non-stationary by construction).
- **Hurst exponent** (on prices, variance-of-lagged-differences estimator — a documented
  simplification, not full rescaled-range analysis) — >0.55 trending/momentum, <0.45
  mean-reverting, ≈0.5 random walk. Most liquid large-caps sit close to 0.5.
- **Rolling Sharpe ratio** (window auto-sized to sample length) — a single end-of-sample Sharpe can
  hide a regime change (calm-then-crisis or the reverse); the rolling version shows how it actually
  evolved.

### 8. Fundamentals (data points, not ratios)

Per-ticker (Overview tab, fetched on demand): market cap, trailing/forward P/E, beta (the
provider's own calculation — may differ slightly from this app's `beta_vs_benchmark` above due to
lookback/methodology differences), dividend yield, 52-week range. Finnhub tried first, Twelve Data
as fallback (see the source-chain note further below) — the table's **Source** column shows which
one actually answered for each ticker.



## Repository structure

```
portfolio-forecasting/
├── app.py                    # Streamlit UI — orchestration only, no finance/LLM logic
├── src/
│   ├── __init__.py
│   ├── config.py              # single source of truth: defaults, env vars, constants
│   ├── market_data.py         # yfinance fetch + cache + frequency resampling + fallback chain
│   ├── news_data.py           # NewsAPI/Finnhub/SEC EDGAR headlines per ticker (fails soft if no key)
│   ├── macro_data.py          # FRED: live 3-month T-bill rate, pre-fills the risk-free-rate slider
│   ├── metrics.py             # pure finance math: returns, Sharpe, Sortino, VaR, CVaR, drawdown, beta
│   ├── forecasting.py         # naive / ETS / ARIMA price forecasting (statsmodels), parallel across tickers
│   ├── optimization.py        # PyPortfolioOpt wrapper: mean-variance, efficient frontier
│   ├── backtesting.py         # walk-forward (multi-window) validation of the 3-portfolio comparison
│   ├── llm_client.py          # Groq (multi-key rotation) + Ollama local fallback, one call site for both
│   ├── ai_features.py         # commentary / news digest (parallel per ticker) / chatbot — prompt logic lives here
│   ├── rag.py                  # TF-IDF retrieval over the news/filings corpus for chatbot Q&A
│   ├── timeseries_diagnostics.py  # ADF stationarity, Hurst exponent, rolling Sharpe
│   └── factor_models.py       # PCA statistical factor model (covariance for wide universes)
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py         # unit tests for the finance formulas (hand-checkable synthetic data)
│   ├── test_optimization.py    # negative bounds, infeasible cap, degenerate mu<rf fallback
│   ├── test_forecasting.py     # short-history and non-convergence fallback paths
│   ├── test_backtesting.py     # expanding-window edge cases, end-to-end walk-forward smoke test
│   ├── test_market_data.py     # fully mocked: Twelve Data partial-batch parsing, Yahoo circuit breaker
│   ├── test_macro_data.py      # FRED divide_by pitfall (Sahm Rule), YoY calculation, GDP/PMI-proxy fetch
│   ├── test_news_sentiment.py  # Finnhub -> VADER sentiment cascade, explicit "not available" case
│   ├── test_timeseries_diagnostics.py  # ADF stationary vs. random walk, Hurst trending vs. mean-reverting
│   ├── test_factor_models.py   # PCA cov: symmetric/PSD, known factor-structure recovery, clamping
│   ├── test_rag.py             # TF-IDF retrieval: ranking, zero-similarity filtering, never-raises contract
│   ├── test_llm_client.py      # Groq->Ollama fallback cascade, token-budget truncation without network
│   └── test_ai_features.py     # build_results_context formatting (shared by commentary + chatbot)
├── .github/workflows/ci.yml  # pytest (blocking) + mypy (non-blocking) on every push/PR
├── requirements.txt
├── Dockerfile                 # containerised run — also the base for Render's `env: docker`
├── docker-compose.yml         # local Docker run with healthcheck
├── build-docker.sh / deploy.sh / deploy.bat  # local build/run helpers (WSL, Linux/macOS, Windows)
├── render.yaml                # Render Blueprint (Infrastructure as Code)
├── .streamlit/config.toml     # theme/UI config — tracked (unlike secrets.toml, see .gitignore)
├── .dockerignore
├── .gitignore
├── .env                       # your local keys — NEVER committed
└── .env.example                # copy to .env and fill in your keys
```

**Why this layout:** `src/metrics.py`, `src/forecasting.py`, and `src/optimization.py` have zero
Streamlit or LLM dependency — they're plain functions on DataFrames/Series, independently testable
and reusable outside the app (a notebook, a batch job). `app.py` only wires UI widgets to these
functions. `llm_client.py` is the single seam that talks to an LLM provider, which is what makes
the Groq→Ollama fallback (and future provider swaps) a one-file change.

## Key design decisions vs. the original brief

The brief names some tools I found to be a poor fit for this scope — documented here rather
than silently swapped:

| Brief suggests | Used instead | Why |
|---|---|---|
| **Kats** for forecasting | **statsmodels** (ARIMA, Holt-Winters ETS) + a naive random-walk baseline | Kats' last PyPI release was **0.2.0 on 15 March 2022** — [pypi.org/project/kats/#history](https://pypi.org/project/kats/#history) — nothing published since, so it predates ~3.5 years of pandas/numpy releases. statsmodels is the actively-maintained, industry-standard alternative. |
| **Riskfolio-Lib** for optimization | **PyPortfolioOpt** | Two alternatives were evaluated against PyPortfolioOpt, both rejected for the same reason: real capability this project's scope doesn't need, at a heavier dependency cost. **Riskfolio-Lib** ([pypi.org/project/Riskfolio-Lib](https://pypi.org/project/Riskfolio-Lib/)) is built on `cvxpy` with 12 convex risk measures, Black-Litterman, risk factors, tracking-error/turnover constraints. **skfolio** ([skfolio.org](https://skfolio.org/)) is newer (2026) and scikit-learn-native (`fit`/`predict`, `GridSearchCV`-compatible) — Mean-Risk, Risk Budgeting, Hierarchical Risk Parity, Black-Litterman, Ledoit-Wolf/Gerber/denoising covariance estimators, walk-forward cross-validation built in — genuinely closer to this project's own PCA-factor-model and walk-forward additions than Riskfolio-Lib is, but still pulls in `cvxpy` + `joblib` + its own `plotly` pin, and its `fit`/`predict` API is a full rewrite of `optimization.py`, not a drop-in swap. PyPortfolioOpt ([pypi.org/project/pyportfolioopt](https://pypi.org/project/pyportfolioopt/)) covers exactly max-Sharpe/min-vol/efficient-frontier with the lightest footprint of the three. **If the brief literally requires a feature none of PyPortfolioOpt's surface has** (e.g. Black-Litterman by name, or a specific convex risk measure like CVaR-native optimization) **skfolio is the one to migrate to**, not Riskfolio-Lib — same reasoning as the PCA-factor-model addition already in this codebase: skfolio's built-in walk-forward cross-validation and Ledoit-Wolf/Gerber covariance estimators overlap directly with `backtesting.py`/`factor_models.py`'s own hand-rolled versions, so a migration would consolidate rather than duplicate. Absent that literal requirement, staying on PyPortfolioOpt keeps the dependency surface and `optimization.py`'s API exactly as documented above — no functional gain from switching without a concrete need driving it. |
| **GitHub Pages** for deployment | **Streamlit Community Cloud** (primary) + **Render** (backup, via `render.yaml` + `Dockerfile`) | GitHub Pages serves static files only — "GitHub Pages does not support server-side languages such as PHP, Ruby, or Python" ([official GitHub Docs](https://docs.github.com/articles/creating-project-pages-manually)) — it cannot run a Streamlit server process. |
| **ISM Manufacturing PMI** for macro context | **Industrial Production Index (FRED `INDPRO`)** | ISM's PMI is a paid, proprietary survey-based series — not available on FRED or any free API. Industrial Production is the closest legitimate free alternative: real output data rather than a survey diffusion index, but it captures the same underlying signal (manufacturing-sector momentum). |
| **Airflow** for orchestration | **None — on-demand fetch inside Streamlit's own request-response cycle** | This app is interactive and synchronous (change a sidebar parameter, it recomputes), not a scheduled batch pipeline — there's no recurring DAG to orchestrate. Airflow would mean deploying a scheduler + webserver + metadata DB for zero present need, the same over-engineering trap Riskfolio-Lib was avoided for above. It would become the right tool if this moved from live per-request fetches to nightly pre-materialised data — noted as a real future option, not dismissed outright. |
| **LangChain / LangGraph** for the LLM layer | **Custom `llm_client.py`** (one call site, Groq→Ollama fallback, multi-key rotation) | Every LLM use in this app (commentary, news digest, chatbot) is a single, well-defined call with context injection — no multi-step agent deciding which tool to call next, no complex cross-session memory to manage. LangChain/LangGraph earn their weight when an agent genuinely orchestrates multiple tools/steps dynamically; here it would be a heavy dependency hiding a simpler fallback/rotation mechanism behind an abstraction layer, for no functional gain. |

## The three-portfolio comparison, precisely

1. **Historical-based**: max-Sharpe weights from mean/covariance estimated on the *training*
   window only (everything before the held-out forecast horizon).
2. **Forecast-based**: same optimizer, but the expected-return vector (μ) comes from the chosen
   forecasting model's predicted prices over the held-out window; the covariance matrix stays
   historical (forecasting a full covariance matrix reliably is a much harder problem, and using
   the historical covariance here is standard practice even in forecast-driven allocation).
3. **Realized-optimal**: max-Sharpe weights fitted on the *actual* returns of the held-out
   window — the hindsight benchmark the other two are judged against.

All three weight vectors are then applied to the **same actual realized returns** of the held-out
window, so the comparison isolates the effect of the allocation choice alone.

## Walk-forward validation (why one comparison isn't enough)

A single train/test split can be a lucky or unlucky draw — it says "forecasting helped this one
time," not "forecasting helps in general." The **Walk-forward validation** section (below the
main comparison, same tab) repeats the exact same historical/forecast/realized comparison across
several **expanding windows**: window *k*'s training set is everything known up to that point,
tested on the next `horizon`-sized slice — each refit uses strictly more data than the last, the
way a real strategy would actually be re-run over time.

Output: a box plot of realized Sharpe per portfolio type across all windows (a spread, not one
number), a mean/std summary table, and a **forecast win rate** — the fraction of windows where the
forecast-based portfolio beat the historical-based one on Sharpe. A win rate hovering near 50%
across many windows is the honest, expected result for short-horizon price forecasting; a rate
consistently well above 50% would be the actual signal that the forecasting step adds value rather
than noise from one convenient split. `src/backtesting.py` implements this on top of the exact same
`metrics.py` / `optimization.py` / `forecasting.py` functions the single-window comparison uses, so
the two views can never silently disagree on how a metric is computed.

Note: ARIMA across many windows/assets is noticeably slower (grid search × every window) —
ETS or naive are the practical default for iterating with walk-forward turned up.

## Expanded universe & macro context

Beyond the brief's 5 default equities, the optional ETF sleeve now spans multiple asset classes
so the tool demonstrates real cross-asset thinking, not just a bigger stock list — each was picked
to represent a genuinely different risk driver, not to pad the list:

| Ticker | Asset class | Why it's here |
|---|---|---|
| `SPY` / `QQQ` | US equity indices | Broad-market and tech-heavy benchmarks |
| `TLT` | Long-duration US Treasuries | Rate/duration exposure, typically negatively correlated with equities in risk-off moves |
| `GLD` / `SLV` | Gold / Silver | Inflation & crisis hedge; silver has a higher-beta, more industrial profile than gold |
| `USO` / `DBC` | Oil / broad commodities basket | Energy & inflation exposure, a different driver from equities or rates |
| `UUP` | US Dollar Index | FX/currency exposure — an ETF proxy rather than raw spot FX or futures, so it optimises cleanly long-only without margin/roll complications |
| `VNQ` | US REITs | Real estate, a distinct cash-flow driver from both equities and bonds |

The **Overview tab** also now opens with a **macro & risk context panel**: the live 3-month and
10-year Treasury yields (FRED), the **10Y-3M term spread** (a negative spread has preceded every
US recession since the 1960s, with some false positives — shown as context, not a signal to trade
on), and the **VIX** level with a simple calm/normal/elevated read. These feed into the AI
Analyst's commentary too, so the narrative accounts for the macro backdrop, not just the portfolio
numbers in isolation.

**Honest caveat, stated in the app itself**: short-horizon equity price forecasting from price
history alone has weak genuine predictive power (consistent with the efficient market hypothesis).
The forecast-based portfolio here demonstrates the required methodology; it is not a claim that
the forecast should be trusted for real allocation decisions. This mirrors the hallucination/
overconfidence caveats from the bootcamp's ethics module — a model producing a confident number is
not the same as that number being reliable.

## Setup

```bash
git clone <your-repo-url> && cd portfolio-forecasting
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then fill in your keys, see below
streamlit run app.py
```

### API keys (all free tiers)

| Key | Where to get it | Required? |
|---|---|---|
| `GROQ_API_KEY` (+ optional `_2` .. `_5`) | [console.groq.com](https://console.groq.com) | Optional — AI Analyst tab falls back to local Ollama without it |
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org/register) (100 req/day free) | Optional — one of three news/filing sources; digest still works with any subset configured |
| `FINNHUB_API_KEY` | [finnhub.io](https://finnhub.io/register) (free, 60 req/**min**) | Optional — dedicated financial-news API, cross-referenced alongside NewsAPI and SEC EDGAR; also the primary source for per-ticker fundamentals |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Optional — risk-free rate slider and macro panel fall back to a fixed 4% default / n/a without it |
| `TWELVEDATA_API_KEY` | [twelvedata.com](https://twelvedata.com) (free, 800 req/day) | Optional — fallback market data source if Yahoo Finance is unreachable, and fallback fundamentals source (mainly covers `AAPL` on the free tier — see caveat below) |
| Ollama (local fallback) | `ollama serve` + `ollama pull llama3.1` — [ollama.com](https://ollama.com) | Optional — only used if Groq fails/is unset |

**News digest cross-referencing.** The AI Analyst's news digest pulls from three genuinely
different source types rather than one: NewsAPI (general media), Finnhub (dedicated financial
news, far more generous free-tier quota), and SEC EDGAR full-text search (free, no key —
primary-source 8-K "material event" filings, not journalism about the company). Each headline
in the UI's "Sources" expander is tagged with which provider it came from; the LLM prompt
explicitly treats SEC filings as more authoritative than media coverage of the same event and
notes when multiple sources corroborate the same story. Works with any subset of the three news
keys configured — missing one just means fewer sources, not a broken feature.

Market data (`yfinance`) needs no key. The app runs fully without *any* key — you just lose the
AI Analyst tab's content and the live risk-free rate, not the core optimization/forecasting/
comparison functionality.

**Groq key rotation.** Free-tier Groq accounts hit daily/per-minute rate limits fast, especially
during a demo. Up to 5 keys can be set (`GROQ_API_KEY`, `GROQ_API_KEY_2` .. `GROQ_API_KEY_5`) —
`llm_client.py` tries them in order and advances to the next one **only on a 429 rate-limit
error**. A non-rate-limit error (bad key, deprecated model) fails immediately to the Ollama
fallback instead of burning time cycling through keys that all share the same problem. Same
pattern already proven on the Innovation Radar project's `llm_client.py`.

**Live risk-free rate (FRED).** Rather than a hardcoded guess, the sidebar's risk-free rate slider
pre-fills with the actual current 3-month T-bill yield (FRED series `DGS3MO`) when `FRED_API_KEY`
is set — still fully overridable by hand. `Alpha Vantage` and `Finnhub` were considered too, but
both mostly duplicate what `yfinance` (prices) and `NewsAPI` (headlines) already cover; FRED adds
a genuinely new, finance-relevant data point (a real macro rate) instead of a redundant one.

### Run the tests

```bash
pytest tests/ -v
```

### Troubleshooting: `pandas` fails to build from source

If `pip install -r requirements.txt` fails while compiling `pandas` (a wall of Cython/C++
compiler output ending in a `meson`/`ninja` error), you're very likely on **Python 3.14 or
newer** on a system where pip has no prebuilt wheel to fall back on for an older `pandas` pin.
`requirements.txt` uses minimum-version pins (`pandas>=3.0.5`, not `==2.2.2`) specifically to
avoid this — pandas only shipped working Python 3.14 wheels starting at 3.0.5 (3.0.0-3.0.4 have
a confirmed segfault regression on 3.14). If you still hit a build error:

```bash
python --version              # confirm which Python the venv is actually using
pip install --upgrade pandas numpy scipy statsmodels streamlit   # force the latest wheels
```

If that still fails, your Python is newer than every dependency has wheels for yet — the
reliable fix is a slightly older interpreter (3.12 is the safest bet as of 2026) via `pyenv`,
`uv python install 3.12`, or your distro's package manager, then recreating the venv with it.

### Troubleshooting: Yahoo Finance returns no data / `crumb = 'Edge: Too Many Requests'`

As of late 2026, Yahoo Finance's anti-bot cookie/crumb handshake (which `yfinance` depends on)
has become noticeably unreliable — this is a widely reported issue across the `yfinance`
community, not specific to this app or your network. The app handles this with a three-step
chain, each step only running if the previous one actually failed:

1. **`yfinance` library** — with retry-with-backoff (3 attempts). This is "the Yahoo Finance API"
   as named in the brief.
2. **Direct Yahoo Finance REST API** — bypasses the `yfinance` library entirely, in case its
   cookie/crumb handling specifically (not Yahoo itself) is the point of failure. Same underlying
   source, different code path. Honest expectation: this sits behind the same anti-bot layer, so
   it's cheap insurance rather than a reliable fix — included because it's literally what the
   brief specifies, not because it's expected to outperform the library.
3. **Twelve Data** — a genuinely different provider (free key, 800 req/day), only reached once
   both Yahoo-based attempts are exhausted.

The Overview tab shows a caption indicating which source actually served the data. Set
`TWELVEDATA_API_KEY` in `.env` to enable step 3 — without it, the app surfaces Yahoo's error.

If you still see no data after all three:
```bash
rm -rf ~/.cache/py-yfinance   # clears a possibly-stale cached cookie
```
then retry, or wait a few minutes — Yahoo's blocks are usually transient, not permanent.

### Docker (local)

```bash
docker build -t portfolio-forecasting .
docker run -p 8501:8501 --env-file .env portfolio-forecasting
```

## Deployment

**Streamlit Community Cloud** (primary): push to GitHub → [share.streamlit.io](https://share.streamlit.io)
→ New app → point at this repo, `app.py` as the entrypoint → add `GROQ_API_KEY` / `NEWSAPI_KEY`
under app Settings → Secrets (TOML format, same keys as `.env.example`).

**Render** (backup): push to GitHub → Render dashboard → New → Blueprint → point at this repo.
`render.yaml` provisions the service from the `Dockerfile` automatically; set `GROQ_API_KEY` and
`NEWSAPI_KEY` in the Render dashboard's environment variables (they're marked `sync: false` in
`render.yaml` so they're never committed).

**Before pushing:** run `git diff --cached` to check nothing sensitive slipped into a config
file, and keep a local backup of the repo before pulling/pushing on a shared/team remote.

## Expanded universe & advanced KPIs

Beyond the brief's 5 default equities and the ETF/commodity sleeve:

**Universe presets** — sidebar "Universe preset" selector:
- *Brief default (5)*: AAPL/MSFT/TSLA/AMZN/GOOG, as specified.
- *Mega Caps (15)*: the largest, most-recognised S&P 500 names across sectors — a one-click
  "give me something sensible" option.
- *Custom / sector picker*: build a universe from 11 GICS sectors (7-12 liquid names each, 104
  tickers total — expanded 2026-09 from an initial ~59). Deliberately NOT all ~500 S&P
  constituents — see the callout below.

**Why not all 500 S&P constituents:** covariance estimation degrades badly with hundreds of
names and only a few years of daily history (the classic "more parameters than data" problem);
real buy-side desks handle this with factor models (Barra, Fama-French) or sector-constrained
universes, not a raw 500×500 mean-variance optimisation on a handful of return observations per
pair. A curated, sector-organised universe is the professionally correct choice here, not a
scope-limited shortcut. **The 104-ticker universe already sits at the point where this matters in
practice** — which is exactly why the covariance estimator below exists.

**Covariance estimation: Ledoit-Wolf shrinkage vs. PCA factor model** (added 2026-09, sidebar
"Covariance estimator")

The math, precisely: with N assets, a covariance matrix has N(N+1)/2 free parameters — 5,460 for
this app's full 104-ticker universe, against at best a few thousand daily observations. Ledoit-Wolf
shrinkage (the long-standing default here) is a real fix for a *small* universe's noisy sample
covariance, but it doesn't remove the underlying degrees-of-freedom problem — it just shrinks
toward a target. A **PCA statistical factor model** does remove it: assume returns are driven by a
small number of common factors plus asset-specific noise, and the parameter count collapses from
N(N+1)/2 down to roughly N × k (k = factor count) + N. This is the SAME core idea real buy-side
desks use (Barra, APT) for exactly this problem — implemented here as a *statistical* factor model
(PCA extracts factors directly from the return data, orthogonal by construction), not a
*fundamental* one like Barra (pre-specified style factors — value, size, momentum — fit by
cross-sectional regression against company characteristics), which this project has neither the
fundamentals data nor the scope to build. Documented as that honest simplification, not oversold
as a Barra reimplementation.

- **When to use which**: Ledoit-Wolf for the default small universes (5-15 tickers); PCA once you
  build a wide custom universe (~40+ tickers via the sector picker).
- **Where it applies**: one dispatch point (`optimization.historical_mu_cov`'s `cov_method`
  parameter), threaded through every place a covariance is estimated — Efficient Frontier,
  Forecast & Compare, and the walk-forward validation all use the same choice, never Ledoit-Wolf
  in one tab and PCA in another.
- **Transparency**: the Efficient Frontier tab shows the actual cumulative explained variance for
  the chosen factor count — a factor model explaining 35% of variance is a materially weaker
  covariance estimate than one explaining 85%, and that number is surfaced directly rather than
  left implicit in a black-box matrix.
- See `src/factor_models.py`'s docstring for the full derivation and the orthogonal-factor-model
  assumption (idiosyncratic risk uncorrelated across assets) that makes the parameter-count
  reduction work.

**Advanced risk-adjusted metrics** (Calmar, Omega, Information Ratio, Treynor, Beta) — full
formulas and interpretation guidance in **"Understanding the KPIs"** near the top of this README.

**Per-ticker fundamentals** (Overview tab, fetched on demand via a button — not automatic, to
conserve API quota): market cap, trailing P/E, beta, dividend yield, price-to-book, 52-week range.

**Source chain, updated 2026-09-03 after live testing exposed a real limitation:** Twelve Data's
`/statistics` endpoint turned out to be restricted on the free tier to their public demo symbol
(`AAPL`) — every other ticker returned a `403 {"message": "/statistics is available exclusively
with pro or ultra or venture or enterprise plans"}`, confirmed directly via `curl`. Rather than
ship a feature that only works for one hardcoded ticker, fundamentals now try **Finnhub first**
(`/stock/profile2` + `/stock/metric`) — confirmed free-tier for arbitrary tickers, and you likely
already have `FINNHUB_API_KEY` configured for the news digest — falling back to Twelve Data only
if Finnhub isn't configured or returns nothing (which still covers `AAPL` via the demo-symbol
path). Two Finnhub-specific parsing quirks worth knowing if this needs debugging later: market cap
is returned in **millions**, not raw units (multiplied by 1e6 in code to match the display format),
and dividend yield is a **percentage number** (e.g. `0.72` for 0.72%), not a decimal fraction
(divided by 100 in code to match). Finnhub's field names were confirmed via public documentation
and multiple independent working code examples, not a live key in this environment — if a field
looks wrong, `curl` both endpoints directly before assuming the parser is stale.

**News sentiment (AI Analyst tab, next to the news digest):** same source-chain philosophy as
fundamentals — try the best genuinely-free source first, fall back rather than fail. Finnhub's
own `/news-sentiment` endpoint (an aggregation over a much wider article set than the ~5
headlines this app fetches per ticker) is tried first; if it's unavailable (no key, or
plan-restricted on some accounts — unverified against a live key here), sentiment is computed
**locally** with VADER (a free, offline, lexicon-based scorer — no API key, no model download,
tuned for short informal text) on the headlines already fetched for the digest. If neither
source has anything, the UI shows an explicit "sentiment not available" message rather than a
silent blank, which would otherwise read as a false "neutral" claim. Every sentiment reading is
tagged with which of the two computed it.

## Sources (from the brief)

Tools and references the original brief pointed to. Kept as-is unless noted otherwise (see
"Key design decisions" table above for the two swaps and why):

| Source | Role in this project |
|---|---|
| [Yahoo Finance / `yfinance`](https://github.com/ranaroussi/yfinance) | Market data acquisition — adjusted close prices, used as specified |
| [Portfolio Visualizer](https://www.portfoliovisualizer.com/) | UX/feature reference cited in the brief for inspiration (efficient frontier, allocation view) |
| [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/en/latest/index.html) | Brief's suggested optimization library — **not used**, see decisions table |
| [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) | Brief's suggested alternative — **used** for mean-variance optimization + efficient frontier |
| [Kats](https://facebookresearch.github.io/Kats/) | Brief's suggested forecasting library — **not used**, see decisions table |
| [PyCaret](https://pycaret.org/) | Brief's suggested forecasting alternative — considered, `statsmodels` chosen instead for a smaller, more predictable dependency footprint on a 2-day deadline |
| [Streamlit](https://streamlit.io/) | App framework, used as specified |
| [Streamlit Community Cloud](https://streamlit.io/cloud) | Primary deployment target |

## Key challenges

1. **Fair out-of-sample comparison design.** The trap in a "forecast vs. realized" comparison is
   letting any forward-looking information leak into the historical/forecast portfolios' training
   window. I solved this with a strict train/test split: μ and Σ for portfolios 1 and 2 are
   estimated *only* on data before the held-out window, and all three weight vectors are evaluated
   on the identical realized returns of that window — so the bar chart genuinely isolates
   allocation skill from lucky market conditions.

2. **PyPortfolioOpt's `max_sharpe()` fails hard, not soft, when every asset's expected return is
   below the risk-free rate** (a realistic case — pick a bear-market date range, or a forecast
   that goes negative). It raises a plain `ValueError`, not its own `OptimizationError`, so a
   naive `except OptimizationError` silently misses it and crashes the app on a very plausible
   user input. I caught it explicitly (see `optimization.py`) with a min-volatility fallback, and
   verified it with a synthetic bear-market test case, not just the happy path.

3. **Keeping the LLM grounded to avoid hallucinated numbers.** A finance tool that confidently
   states a wrong Sharpe ratio is worse than a tool that says nothing. I solved this with an
   explicit context-injection pattern (`ai_features.build_results_context`): every computed number
   the LLM is allowed to reference is serialised into the prompt, and the system prompt explicitly
   instructs the model to say "not available" rather than estimate a figure not present there.

4. **Per-ticker forecasting and news fetching were the two biggest wall-clock costs** (ARIMA
   fitting across many windows/assets; three sequential HTTP calls per ticker for the news
   digest). I parallelised both with `concurrent.futures.ThreadPoolExecutor`, capped at 8 workers
   to avoid oversubscribing a small container. I used threads rather than processes for the ARIMA
   case specifically: the fitting is dominated by numpy/scipy linear algebra, which releases the
   GIL during BLAS calls, so threads give a real speedup without the pickling/Streamlit-context
   fragility subprocesses would add. I kept column/ticker order explicit (`executor.map` for
   forecasting, original-order iteration over a completion-order results dict for news) so
   parallelising never makes output order depend on which network call happened to finish first.

## Short selling, transaction costs, and RAG

Three things previously listed as deliberately out of scope for the 2-day window, added once
there was time to do them properly rather than as an afterthought:

**Short selling / leverage.** Sidebar toggle "Allow short selling" — off by default (long-only,
weights ≥ 0), on makes bounds symmetric (`-cap` to `+cap` around the same max-weight setting).
The portfolio stays fully invested (weights still sum to 100%); this does not add gross leverage
beyond that. `resolve_weight_bounds()` in `optimization.py` centralises the long-only vs.
long-short logic so every optimizer call site (frontier, single-window comparison, walk-forward)
stays consistent.

**Transaction costs & rebalancing frequency.** Sidebar slider "Transaction cost (bps per
rebalance)", default 10 bps. Charged as `turnover × cost rate` at every point a portfolio
actually rebalances: once for the initial trade in the single-window comparison, and at every
walk-forward window boundary — tracked independently per portfolio type against ITS OWN previous
weights, not a shared reference. "Rebalancing frequency" surfaces through the existing forecast
horizon control rather than a separate parameter: a shorter horizon means more walk-forward
windows over the same history, i.e. more frequent rebalancing, i.e. more cumulative cost drag —
shortening the horizon is how to see this effect directly. Set the cost to 0 for the frictionless
textbook comparison. `compute_turnover()` / `apply_transaction_cost()` in `metrics.py`.

**RAG for the chatbot.** A genuine index-then-retrieve pipeline (`src/rag.py`) over the news/
filings corpus collected in the AI Analyst tab, kept deliberately separate from the context-
injection approach still used for portfolio metrics (that distinction is the point — metrics are
a small, fixed, must-be-complete set of numbers where full injection is correct; news/filings are
unstructured and only the query-relevant subset should reach the prompt). TF-IDF + cosine
similarity was chosen over neural embeddings: this corpus is small and rebuilt fresh every
session (no persistence), so a sentence-transformers model would add a large, slow-to-install
dependency for retrieval-quality gains that don't matter at this scale — scikit-learn (already a
transitive dependency via PyPortfolioOpt's `cvxpy` stack) is sufficient and dependency-light. Only
the top-4 chunks relevant to the user's specific question are retrieved and injected — verified
directly: a question about NVDA earnings pulls in the NVDA news chunk but not an unrelated Apple
one, and a pure concept question ("what is Sharpe ratio") retrieves nothing and injects no RAG
block at all, rather than force-fitting irrelevant news into every answer. If this corpus grew
into hundreds of persisted documents across sessions, a real vector store (Chroma/FAISS) with
neural embeddings would be the right upgrade — noted here as the natural next step, not built
because it would be unjustified complexity at the current corpus size.

## Known limitations

Things I'm aware of and chose not to fix within this project's scope, rather than gaps I missed:

- **Short-horizon price forecasting genuinely has weak predictive power.** The whole app is built
  around this honestly (win-rate near 50% is the expected result, not a bug) — but it means the
  "Forecast-based" portfolio should never be read as an actionable signal, only as a methodology
  demonstration. I say this explicitly in three places in the UI so it can't be missed.
- **The forecasted covariance matrix is still historical**, not forecasted (see
  `optimization.py`'s docstring) — standard practice even in forecast-driven allocation, but worth
  stating plainly: only μ comes from the forecast, Σ never does.
- **The Yahoo circuit breaker is process-wide, not per-session.** On a multi-user deployment
  (Streamlit Community Cloud, Render), if Yahoo fails for one user it's skipped for everyone for
  the next 3 minutes. I think this is the right trade-off (Yahoo being down is a server-side fact,
  not a per-user one), but it's a deliberate simplification worth flagging, not an oversight.
- **RAG (`rag.py`) has no persistence** — the news/filings corpus is rebuilt fresh every session,
  so TF-IDF retrieval quality resets each time and can't learn from accumulated history.
- **CI's `mypy` step is non-blocking for now** (`continue-on-error: true` in `ci.yml`) — the
  codebase wasn't written under mypy from day one, so a first strict run surfaces a backlog I
  haven't triaged yet. `pytest` is the blocking gate today.
- **`.dockerignore`/local `.venv` aren't tracked in git** by design.
- ~~Dockerfile pins `python:3.12-slim` while `requirements.txt` was written against Python 3.14
  compatibility testing — never reconciled.~~ **Resolved (2026-09-04):** the full test suite
  (105 tests, including the 3 new files above) was run against Python 3.12.3 with the exact
  pinned `requirements.txt` — 105/105 pass. The version-floor pins (`pandas>=3.0.5`,
  `statsmodels>=0.14.4,<0.15.0`, etc.) were written to cover 3.12 through 3.14 in the same range,
  not 3.14-only as this bullet used to imply; `python:3.12-slim` in the Dockerfile was always a
  safe choice, it just hadn't been verified end-to-end until now. 3.12 stays the recommended
  local interpreter (see the troubleshooting section above) since it has the widest, most mature
  wheel availability of the two as of 2026.
- **Two bugs found and fixed during this pass (2026-09-04):** (1) the sidebar's "Risk-free rate"
  and "Max weight per asset" sliders stored fractions (e.g. `0.35`) but formatted them with a
  literal `%` suffix (`format="%.0f%%"`) — `st.slider`'s `format` only controls display, it never
  multiplies by 100, so "35%" was rendering as "0%". Fixed by running both sliders in
  percentage-point units and converting to a fraction right after, so every downstream function's
  input is unchanged. (2) `llm_client.truncate_to_token_budget`'s truncation branch called
  `tiktoken.get_encoding()` unguarded — if that download is blocked (a restricted container
  network, confirmed directly in this project's own test sandbox), it raised an uncaught
  `HTTPError` instead of degrading, breaking this module's own "fails soft" contract. Both now
  have regression tests (`test_llm_client.py`, and the slider fix is a UI-only change covered by
  manual verification — see the Deployment section for how to re-check it after a `streamlit run`).

## Next steps

Roughly in the order I'd tackle them:

1. **Blocking `mypy --strict`** once the current backlog is triaged — the codebase is already
   fully type-hinted (`from __future__ import annotations` everywhere), so the marginal cost of
   turning this on for real is low relative to the safety net it adds.
2. **Session-scoped Yahoo circuit breaker** if this ever moves to genuine multi-tenant use, so one
   user's Yahoo failure doesn't affect another's.
3. **A real vector store (Chroma/FAISS) for RAG** if the news/filings corpus ever gets persisted
   across sessions instead of rebuilt fresh each time — noted in `rag.py`'s own docstring as the
   natural upgrade path, not built now because it would be unjustified complexity at the current
   scale.
4. **Reconcile the Dockerfile's Python version** with what `requirements.txt` was actually tested
   against, so local dev and the containerised deploy can't silently diverge.
5. **LangGraph, if the AI Analyst ever becomes a real multi-step agent** — e.g. deciding on its own
   to pull fresh news, recompute a scenario, then compare it against the current portfolio, instead
   of the three fixed, hand-wired features (commentary/digest/chatbot) it is today. Not needed for
   the current scope (see "Key design decisions" above), but the natural next step if the AI layer
   grows from "answers questions about fixed numbers" into "decides what to compute next."