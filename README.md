# Portfolio Forecasting & Optimization

Interactive Streamlit app that builds and compares three portfolios — **historical-based**,
**forecast-based**, and **realized-optimal (hindsight)** — using mean-variance optimization,
and adds an AI analyst layer (LLM commentary, news digest, grounded Q&A chatbot).

Built as a technical case study (BeCode AI & Data Science bootcamp — GenAI Developer track).

## Live demo
- Streamlit Community Cloud: _add URL after deploying_
- Render (backup): _add URL after deploying_

## What this does, in one paragraph

You pick a universe of stocks/ETFs, a date range, and a frequency. The app computes historical
returns, volatility, correlation, and the max-Sharpe efficient-frontier portfolio. It then holds
out the last N periods, forecasts each asset's price over that window (ARIMA / Exponential
Smoothing / naive random walk — see "Why not Kats" below), builds an optimal portfolio from the
*forecasted* returns, and compares its **actual, realized** out-of-sample performance against
(a) the historical-based portfolio and (b) the hindsight-optimal portfolio built from the *actual*
returns of that same window. An AI Analyst tab (Groq, with local Ollama fallback) narrates the
results and answers questions about them, grounded in the computed numbers — plus a NewsAPI-based
news digest per ticker.

## Contents

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

## Understanding the KPIs — formulas & how to read them

Every metric below is computed in `src/metrics.py` (pure functions, unit-tested in
`tests/test_metrics.py`) unless noted otherwise. `r` = period return series (daily/weekly/monthly
depending on the sidebar), `rf` = risk-free rate, `n` = periods per year (252/52/12).

### 1. Return & risk — the building blocks

**Annualised Return**
- What it measures: the compounded (geometric) growth rate, scaled to a 1-year horizon.
- Formula: `(∏(1 + r)) ^ (n / periods) − 1`
- How to read it: the "headline" return. Geometric, not arithmetic mean × n — arithmetic mean
  overstates the true return of a volatile series (a +50%/−50% sequence has 0% geometric return,
  not the naive +0% either — check the actual math if this surprises you, it's a classic trap).
- Where: every metrics table (Overview, Efficient Frontier, Forecast & Compare).

**Annualised Volatility**
- What it measures: the standard deviation of returns, annualised.
- Formula: `std(r) × √n`
- How to read it: total risk — both upside and downside swings count equally. Higher isn't
  automatically "bad" (see Sortino/Omega below for asymmetric views), but it's the risk figure
  every other ratio on this page normalises against in some form.
- Where: same tables as Annualised Return.

**Max Drawdown**
- What it measures: the single largest peak-to-trough decline in cumulative wealth, over the
  whole period shown.
- Formula: build the wealth index `W(t) = ∏(1+r)` up to `t`, then `min[ W(t) / max(W(0..t)) − 1 ]`
- How to read it: "if you'd invested at the worst possible moment and sold at the worst possible
  moment after, this is what you'd have lost." A negative percentage, e.g. −35%. This is the
  number that correlates most with an investor actually panic-selling — bigger drawdowns are
  harder to sit through than volatility alone suggests.
- Where: every metrics table; also the input to Calmar (below).

**VaR 95% (Value at Risk)**
- What it measures: the loss threshold that historical returns crossed in the worst 5% of periods.
- Formula: the 5th percentile of the historical return distribution (non-parametric — no normal-
  distribution assumption, since equity returns are fat-tailed).
- How to read it: "on the worst 1-in-20 periods historically, the loss exceeded X%." It says
  nothing about how much worse than X% those periods got — that's what CVaR is for.
- Where: every metrics table.

**CVaR 95% (Conditional VaR / Expected Shortfall)**
- What it measures: the *average* loss across only the periods that were worse than VaR.
- Formula: `mean(r | r ≤ VaR_95)`
- How to read it: the more informative tail-risk number — always at least as bad as VaR (it's an
  average of the tail beyond the VaR threshold). A large gap between VaR and CVaR flags a fat,
  dangerous tail that VaR alone would understate.
- Where: every metrics table.

### 2. Risk-adjusted return ratios — reward per unit of risk

**Sharpe Ratio**
- What it measures: excess return (over the risk-free rate) per unit of *total* volatility.
- Formula: `(mean(r − rf_period) / std(r − rf_period)) × √n`, where `rf_period` is the annual `rf`
  converted to a per-period rate — never subtract the annual rate directly from period returns,
  it massively understates excess return.
- How to read it: the standard "risk-adjusted return" headline number. Higher is better; above 1
  is generally considered good, above 2 very good, in the context of realistic equity strategies.
  Penalises upside volatility exactly as much as downside — its main limitation, which Sortino
  fixes.
- Where: every metrics table; the walk-forward box plot's y-axis.

**Sortino Ratio**
- What it measures: like Sharpe, but the denominator only counts *downside* deviation (returns
  below the risk-free target).
- Formula: `(mean(r − rf_period) / downside_deviation) × √n`, where
  `downside_deviation = √(mean((r − rf_period)² | r − rf_period < 0))`
- How to read it: two strategies can share a Sharpe ratio while one has "many small gains, rare
  big losses" and the other "steady symmetric swings" — Sortino tells them apart, since only the
  first strategy's big losses hurt this ratio. Sortino ≥ Sharpe is normal for most real return
  series (upside vol usually exceeds downside vol for equities over time).
- Where: every metrics table.

**Calmar Ratio**
- What it measures: annualised return per unit of the *single worst* drawdown lived through.
- Formula: `annualised_return / |max_drawdown|`
- How to read it: where Sharpe/Sortino penalise the whole distribution's spread, Calmar penalises
  only the worst outcome an investor actually experienced — the number a risk committee asks for
  when "volatility looks fine on average, but that one drawdown was brutal." No universal
  good/bad threshold; compare across the three portfolios in the same tab, not against a memorised
  number.
- Where: Overview (per-asset), Forecast & Compare (per-portfolio).

**Omega Ratio**
- What it measures: the ratio of total gains to total losses above/below a threshold (0% by
  default), using the *entire* empirical return distribution — not just its mean and variance.
- Formula: `sum(r − threshold | r > threshold) / |sum(r − threshold | r < threshold)|`
- How to read it: > 1 means gains outweighed losses in total magnitude; < 1 means the reverse.
  Because it uses the full distribution shape, Omega will diverge from Sharpe/Sortino exactly when
  returns are skewed or fat-tailed — precisely the case where a mean-variance summary is most
  likely to mislead. Treat it as a second opinion alongside Sharpe/Sortino, not a replacement.
  A value of `∞` (shown as `—` in the app) means there were literally no losing periods in the
  sample — check the sample size before trusting that.
- Where: every metrics table.

### 3. Benchmark-relative metrics — vs. SPY

Computed only when a benchmark return series is available (SPY is always fetched in the
background regardless of your ticker selection specifically so these are always computable).

**Beta**
- What it measures: sensitivity to the benchmark's moves — CAPM beta.
- Formula: `Cov(r, r_benchmark) / Var(r_benchmark)`
- How to read it: beta of 1.0 moves with the market; > 1.0 amplifies market moves (more
  aggressive); < 1.0 dampens them (more defensive); negative beta moves opposite the market (rare,
  usually a hedge-like asset e.g. some gold/vol exposure in stress periods).
- Where: Overview (per-asset), Forecast & Compare (per-portfolio).

**Information Ratio**
- What it measures: how *consistent* the portfolio's outperformance over SPY has been, not just
  its size.
- Formula: `annualised(mean(r − r_benchmark)) / (std(r − r_benchmark) × √n)` — active return over
  tracking error.
- How to read it: this is the metric active-mandate performance reviews lead with, not Sharpe —
  Sharpe judges a portfolio in isolation, IR judges it against the mandate it's meant to beat. A
  small, very *consistent* edge over SPY gives a high IR even with modest absolute returns; a big
  edge that's erratic gives a lower IR than you'd expect from the headline number alone.
- Where: Forecast & Compare (per-portfolio).

**Treynor Ratio**
- What it measures: excess return per unit of *systematic* (market/beta) risk, instead of Sharpe's
  *total*-risk denominator.
- Formula: `(annualised_return − rf) / beta`
- How to read it: two portfolios can share a Sharpe ratio while one carries far more market
  exposure (higher beta) than the other — Treynor surfaces that difference. Use it alongside
  Sharpe when you specifically want to know how much of the return came from being exposed to the
  market at all, versus genuine diversification or security selection.
- Where: Forecast & Compare (per-portfolio).

### 4. Optimizer outputs — expected, not realized

Shown in the **Efficient Frontier** tab. These come from the optimizer's own inputs (historical
mean/covariance, Ledoit-Wolf shrinkage) — they are the optimizer's *target*, not a guarantee of
what will actually happen. Compare them against the *realized* metrics in Forecast & Compare to
see the gap between expectation and outcome.

- **Expected annual return** = `w · μ` (weights dotted with the expected-return vector)
- **Expected annual volatility** = `√(w · Σ · w)` (portfolio variance from the covariance matrix)
- **Expected Sharpe** = `(expected_return − rf) / expected_volatility`

### 5. Forecast validation

**Forecast win rate** (walk-forward section, Forecast & Compare tab)
- What it measures: across every walk-forward window, the fraction where the forecast-based
  portfolio's *realized* Sharpe beat the historical-based portfolio's.
- How to read it: a rate hovering near 50% across many windows is the *expected, honest* result
  for short-horizon equity price forecasting (consistent with the efficient market hypothesis) —
  it means the forecast isn't reliably adding value beyond noise. A rate consistently well above
  50% across many independent windows would be the actual signal of a genuine edge. Don't read
  much into the result from a single window (see "Walk-forward validation" further below for why).

### 6. Macro context (not portfolio-specific, but shapes interpretation)

- **VIX** — CBOE Volatility Index, the market's "fear gauge" (implied 30-day S&P 500 volatility).
  App's read: < 15 = calm, 15–25 = normal, > 25 = elevated stress. Useful context for reading
  every other metric above: the same Sharpe ratio means something different achieved during a
  VIX-12 calm stretch vs. a VIX-30 stress period.
- **10Y–3M Treasury term spread** — a negative spread (short rates above long rates) has preceded
  every US recession since the 1960s, with some false positives. Shown as context, not a trading
  signal.
- **Risk-free rate** — live 3-month T-bill yield (FRED), used as `rf` in every ratio above that
  needs one (Sharpe, Sortino, Treynor). Overridable by hand in the sidebar.

### 7. Fundamentals (data points, not ratios)

Per-ticker, via Twelve Data (Overview tab, fetched on demand): market capitalization, trailing/
forward P/E, beta (Twelve Data's own calculation — may differ slightly from the app's own
`beta_vs_benchmark` above due to differing lookback windows/methodology), dividend yield, 52-week
price range. See the fundamentals caveat further below (free-tier field availability).

## Repository structure

```
portfolio-forecasting/
├── app.py                 # Streamlit UI — orchestration only, no finance/LLM logic
├── src/
│   ├── config.py           # single source of truth: defaults, env vars, constants
│   ├── market_data.py      # yfinance fetch + cache + frequency resampling
│   ├── news_data.py        # NewsAPI headlines per ticker (fails soft if no key)
│   ├── macro_data.py       # FRED: live 3-month T-bill rate, pre-fills the risk-free-rate slider
│   ├── metrics.py          # pure finance math: returns, Sharpe, Sortino, VaR, CVaR, drawdown, beta
│   ├── forecasting.py      # naive / ETS / ARIMA price forecasting (statsmodels)
│   ├── optimization.py     # PyPortfolioOpt wrapper: mean-variance, efficient frontier
│   ├── backtesting.py      # walk-forward (multi-window) validation of the 3-portfolio comparison
│   ├── llm_client.py       # Groq primary + Ollama local fallback, one call site for both
│   └── ai_features.py      # commentary / news digest / chatbot — prompt logic lives here
├── tests/
│   └── test_metrics.py     # unit tests for the finance formulas (hand-checkable synthetic data)
├── requirements.txt
├── Dockerfile               # containerised run — also the base for Render's `env: docker`
├── render.yaml               # Render Blueprint (Infrastructure as Code)
├── .streamlit/config.toml
└── .env.example              # copy to .env and fill in your keys (never commit .env)
```

**Why this layout:** `src/metrics.py`, `src/forecasting.py`, and `src/optimization.py` have zero
Streamlit or LLM dependency — they're plain functions on DataFrames/Series, independently testable
and reusable outside the app (a notebook, a batch job). `app.py` only wires UI widgets to these
functions. `llm_client.py` is the single seam that talks to an LLM provider, which is what makes
the Groq→Ollama fallback (and future provider swaps) a one-file change.

## Key design decisions vs. the original brief

The brief names some tools that are a poor fit for a 2-day solo delivery — documented here rather
than silently swapped, since a technical interviewer will ask "why":

| Brief suggests | Used instead | Why |
|---|---|---|
| **Kats** for forecasting | **statsmodels** (ARIMA, Holt-Winters ETS) + a naive random-walk baseline | Kats has been effectively unmaintained since 2021 and conflicts with current pandas/numpy — the install alone would burn hours. statsmodels is the actively-maintained, industry-standard alternative. |
| **Riskfolio-Lib** for optimization | **PyPortfolioOpt** | Lighter dependency footprint, actively maintained, covers exactly what's needed (max-Sharpe, min-vol, efficient frontier) without Riskfolio's heavier convex-optimization stack. |
| **GitHub Pages** for deployment | **Streamlit Community Cloud** (primary) + **Render** (backup, via `render.yaml` + `Dockerfile`) | GitHub Pages only serves static sites — it cannot run a Streamlit server process. |

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
| `NEWSAPI_KEY` | [newsapi.org](https://newsapi.org/register) (100 req/day free) | Optional — news digest shows "unavailable" without it |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Optional — risk-free rate slider falls back to a fixed 4% default without it |
| `TWELVEDATA_API_KEY` | [twelvedata.com](https://twelvedata.com) (free, 800 req/day) | Optional — used only as a fallback if Yahoo Finance is unreachable |
| Ollama (local fallback) | `ollama serve` + `ollama pull llama3.1` — [ollama.com](https://ollama.com) | Optional — only used if Groq fails/is unset |

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
- *Custom / sector picker*: build a universe from 11 GICS sectors (~5-8 liquid names each, ~59
  tickers total). Deliberately NOT all ~500 S&P constituents — see the callout below.

**Why not all 500 S&P constituents:** covariance estimation degrades badly with hundreds of
names and only a few years of daily history (the classic "more parameters than data" problem);
real buy-side desks handle this with factor models (Barra, Fama-French) or sector-constrained
universes, not a raw 500×500 mean-variance optimisation on a handful of return observations per
pair. A curated, sector-organised universe is the professionally correct choice here, not a
scope-limited shortcut.

**Advanced risk-adjusted metrics** (Calmar, Omega, Information Ratio, Treynor, Beta) — full
formulas and interpretation guidance in **"Understanding the KPIs"** near the top of this README.

**Per-ticker fundamentals** (Overview tab, fetched on demand via a button — not automatic, to
conserve Twelve Data's daily quota): market cap, trailing/forward P/E, beta, dividend yield,
52-week range. Requires `TWELVEDATA_API_KEY`. **Caveat, stated plainly:** Twelve Data's
`/statistics` endpoint has fields documented as premium-plan-only, and exact free-tier field
availability wasn't verifiable without a live key during development — the parser is defensive
(every field is a best-effort lookup with a `—` fallback), but if your key returns an unexpected
shape, `curl` the endpoint directly to inspect the real response rather than assuming the parser
covers it.

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
   window. Solved by a strict train/test split: μ and Σ for portfolios 1 and 2 are estimated
   *only* on data before the held-out window, and all three weight vectors are evaluated on the
   identical realized returns of that window — so the bar chart genuinely isolates allocation
   skill from lucky market conditions.

2. **PyPortfolioOpt's `max_sharpe()` fails hard, not soft, when every asset's expected return is
   below the risk-free rate** (a realistic case — pick a bear-market date range, or a forecast
   that goes negative). It raises a plain `ValueError`, not its own `OptimizationError`, so a
   naive `except OptimizationError` silently misses it and crashes the app on a very plausible
   user input. Caught explicitly (see `optimization.py`) with a min-volatility fallback, and
   verified with a synthetic bear-market test case, not just the happy path.

3. **Keeping the LLM grounded to avoid hallucinated numbers.** A finance tool that confidently
   states a wrong Sharpe ratio is worse than a tool that says nothing. Solved with an explicit
   context-injection pattern (`ai_features.build_results_context`): every computed number the LLM
   is allowed to reference is serialised into the prompt, and the system prompt explicitly
   instructs the model to say "not available" rather than estimate a figure not present there.

## What's out of scope (by design, given the 2-day window)

- Short selling / leverage (long-only optimization; `weight_bounds` in `optimization.py` can be
  opened to `(-1, 1)` if needed later).
- Transaction costs and rebalancing frequency in the backtest.
- Full RAG for the chatbot — the current approach (context injection of the computed results) is
  the right-sized solution for a small, fixed set of numbers; a vector store would be unjustified
  overhead here, but is the natural next step if this grew to cover a large news/filings corpus.
# portfolio_forecasting
