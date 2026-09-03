"""
AI-powered features layered on top of the finance engine: a narrative commentary
on portfolio results, a news digest per ticker, and a grounded Q&A chatbot.

Grounding strategy (explicit, and worth stating for anyone reviewing this against
the ethics/RAG material): this is CONTEXT INJECTION, not full RAG. The computed
metrics/weights/news are serialised into the system prompt so the model answers
from the actual numbers on screen rather than from memorised training data — this
removes most hallucination risk for THIS narrow use case (a fixed, small set of
facts) without the complexity of a vector store, which would be overkill for a
handful of numbers that already fit in a prompt.
"""
from __future__ import annotations

import pandas as pd

from src.llm_client import chat, truncate_to_token_budget
from src.news_data import fetch_finnhub_news, fetch_sec_filings, fetch_ticker_headlines

SYSTEM_PERSONA = (
    "You are a CFA-level portfolio analyst writing for an informed but non-technical "
    "reader. Be precise, quantitative, and honest about uncertainty. Never invent "
    "numbers that are not given to you in the context — if something isn't in the "
    "context, say so explicitly rather than guessing.\n\n"
    "CRITICAL: sanity-check annualised figures before reporting them at face value. "
    "An annualised return is calculated by compounding a short window's actual return "
    "out to a full year — over a short window (a handful of weeks or months), this "
    "can turn an ordinary result into a triple-digit annualised number that never "
    "actually occurred and is not a repeatable rate. If annualised return exceeds "
    "roughly 50-80% or Sharpe exceeds roughly 3, and the context shows the underlying "
    "window is short (well under a year, i.e. few periods), you MUST say so explicitly "
    "— e.g. 'this annualises a strong N-period run into an inflated headline figure; "
    "the raw period return of X% is the more honest read of what actually happened.' "
    "Never describe volatility as 'modest' or a portfolio as low-risk purely because "
    "its annualised return is very high — a sky-high annualised return is itself a "
    "signal to look at the raw window return, not evidence of a great risk-adjusted "
    "outcome."
)


def _format_metrics_block(label: str, metrics: dict[str, float]) -> str:
    lines = [f"{label}:"]
    if "period_return" in metrics and "n_periods" in metrics:
        lines.append(f"  Raw return over the {metrics['n_periods']}-period window (NOT annualised): {metrics['period_return']:.2%}")
    lines += [
        f"  Annual return (compounded from the window above — see caveat if window is short): {metrics['annual_return']:.2%}",
        f"  Annual volatility: {metrics['annual_volatility']:.2%}",
        f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}",
        f"  Sortino ratio: {metrics['sortino_ratio']:.2f}",
        f"  Max drawdown: {metrics['max_drawdown']:.2%}",
        f"  95% VaR (period): {metrics['var_95']:.2%}",
        f"  95% CVaR (period): {metrics['cvar_95']:.2%}",
    ]
    return "\n".join(lines) + "\n"


def build_results_context(
    weights: pd.Series,
    historical_metrics: dict[str, float],
    forecast_metrics: dict[str, float] | None,
    realized_metrics: dict[str, float] | None,
    macro_context: dict | None = None,
) -> str:
    """Serialise the app's computed results into a compact text block reused as
    grounding context for BOTH the commentary generator and the chatbot — one
    source of truth, so the two features can never disagree with each other."""
    lines = ["PORTFOLIO WEIGHTS (optimal, max-Sharpe):"]
    for ticker, w in weights.items():
        if w > 0.001:
            lines.append(f"  {ticker}: {w:.1%}")

    lines.append("")
    lines.append(_format_metrics_block("HISTORICAL-BASED PORTFOLIO (realised, in-sample)", historical_metrics))
    if forecast_metrics:
        lines.append(_format_metrics_block("FORECAST-BASED PORTFOLIO (realised out-of-sample)", forecast_metrics))
    if realized_metrics:
        lines.append(_format_metrics_block("REALIZED-OPTIMAL PORTFOLIO (hindsight benchmark)", realized_metrics))

    if macro_context:
        macro = macro_context.get("macro", {})
        vix = macro_context.get("vix_level")
        macro_lines = ["MACRO & RISK BACKDROP (current, not historical to the portfolio period):"]
        if macro.get("three_month_yield") is not None:
            macro_lines.append(f"  3-month T-bill yield: {macro['three_month_yield']:.2%}")
        if macro.get("ten_year_yield") is not None:
            macro_lines.append(f"  10-year Treasury yield: {macro['ten_year_yield']:.2%}")
        if macro.get("term_spread_10y_3m") is not None:
            spread = macro["term_spread_10y_3m"]
            note = " (INVERTED — a historical recession signal)" if spread < 0 else ""
            macro_lines.append(f"  10Y-3M term spread: {spread:+.2%}{note}")
        if vix is not None:
            regime = "calm" if vix < 15 else "normal" if vix < 25 else "elevated"
            macro_lines.append(f"  VIX (market fear gauge): {vix:.1f} ({regime})")
        if len(macro_lines) > 1:  # only include if we actually got at least one live figure
            lines.append("")
            lines.append("\n".join(macro_lines))

    return "\n".join(lines)


def generate_commentary(results_context: str) -> tuple[str, str]:
    """Narrative, analyst-style commentary on the computed portfolio results.
    Returns (text, backend_used)."""
    messages = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {
            "role": "user",
            "content": (
                "Write a concise (150-200 words) portfolio commentary based ONLY on the "
                "data below. Cover: (1) what drove the historical allocation, (2) how the "
                "forecast-based portfolio compares to the realized-optimal benchmark and "
                "what that gap implies about forecast reliability, (3) one concrete risk "
                "to flag (drawdown, concentration, or volatility), and (4) if a macro "
                "backdrop is provided, one line on how it colours the picture (e.g. an "
                "inverted yield curve or an elevated VIX warrants more caution than the "
                "portfolio numbers alone would suggest). Do not give investment advice or "
                "recommendations to buy/sell.\n\n" + results_context
            ),
        },
    ]
    return chat(messages, temperature=0.4, max_tokens=400)


def generate_news_digest(tickers: list[str], company_names: dict[str, str]) -> tuple[str, str, list[dict]]:
    """
    Fetch recent headlines/filings per ticker from THREE sources — NewsAPI
    (general media), Finnhub (dedicated financial news), SEC EDGAR (primary
    regulatory filings) — and ask the LLM for a short digest that cross-
    references them rather than relying on a single provider's coverage.

    Returns (digest_text, backend_used, raw_articles) — raw_articles are kept so
    the UI can render clickable source links (the LLM output alone should never
    be the only trace of a claim; always show the reader where it came from).
    """
    all_articles: list[dict] = []
    news_block_parts = []
    for ticker in tickers:
        company = company_names.get(ticker)
        articles = (
            fetch_ticker_headlines(ticker, company)
            + fetch_finnhub_news(ticker)
            + (fetch_sec_filings(company, ticker) if company else [])
        )
        for a in articles:
            a["ticker"] = ticker
        all_articles.extend(articles)
        if articles:
            headlines = "\n".join(f"  - [{a['provider']}] {a['title']} ({a['source']})" for a in articles)
            news_block_parts.append(f"{ticker}:\n{headlines}")

    if not news_block_parts:
        return (
            "No recent news/filings available from any source (NewsAPI, Finnhub, SEC EDGAR) — "
            "check that at least one of NEWSAPI_KEY / FINNHUB_API_KEY is configured, or the "
            "free tier's quota may be exhausted.",
            "n/a",
            [],
        )

    news_block = truncate_to_token_budget("\n\n".join(news_block_parts))
    messages = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {
            "role": "user",
            "content": (
                "Summarise the recent news/filings below in 3-5 bullet points, grouped by "
                "ticker if relevant. Each item is tagged with its source in brackets "
                "([NewsAPI]/[Finnhub] = media coverage, [SEC EDGAR] = a primary regulatory "
                "8-K filing). Treat SEC EDGAR items as more authoritative than media coverage "
                "of the same event, and note explicitly when multiple sources corroborate the "
                "same story (a stronger signal than a single outlet). Focus on anything that "
                "could plausibly move the stock (earnings, guidance, litigation, product "
                "launches, macro exposure). Stay factual - do not speculate beyond what's "
                "stated.\n\n" + news_block
            ),
        },
    ]
    text, backend = chat(messages, temperature=0.3, max_tokens=350)
    return text, backend, all_articles


def answer_portfolio_question(
    question: str, results_context: str, chat_history: list[dict], news_chunks: list | None = None,
) -> tuple[str, str]:
    """
    Grounded Q&A: answers are constrained to the computed results context plus
    ordinary finance knowledge for explaining concepts (e.g. "what is Sortino
    ratio") — but the system prompt explicitly forbids fabricating numbers not
    present in the context, which is the main hallucination risk for a finance
    tool (a confidently wrong Sharpe ratio is worse than no answer).

    `news_chunks` (from src.rag.build_chunks, built once after fetching the news
    digest) enables genuine RAG for questions ABOUT news/filings rather than the
    portfolio numbers: only the top-k chunks relevant to THIS specific question
    are retrieved and injected, not the entire news corpus on every turn. This
    is a real index-then-retrieve pipeline, not a rebrand of the metrics
    context-injection above — see src/rag.py's docstring for why the two use
    different approaches (small fixed structured data vs. a growable
    unstructured corpus).
    """
    retrieved_block = ""
    if news_chunks:
        from src.rag import format_retrieved_chunks, retrieve
        relevant = retrieve(question, news_chunks, top_k=4)
        retrieved_block = format_retrieved_chunks(relevant)

    system_content = (
        SYSTEM_PERSONA
        + "\n\nYou may explain general finance concepts (e.g. what Sharpe ratio "
        "means) from your own knowledge. But any SPECIFIC NUMBER about THIS "
        "portfolio must come from the context below - if asked about a number "
        "not present there, say it isn't available rather than estimating it.\n\n"
        "CRITICAL — if the user pastes a bare sequence of numbers (e.g. '16.7% "
        "28.1% 0.55 0.56...') without saying which metric each one is, DO NOT "
        "guess an order or say a number 'is likely to represent' some metric. "
        "That is hallucination by another name — matching numbers to labels by "
        "assumed position is exactly the kind of unstated guess this app exists "
        "to avoid. Instead: (1) check whether the pasted numbers match a row in "
        "the context above for the ticker/portfolio they named — if they match "
        "exactly, confirm the mapping explicitly ('these correspond to X, Y, Z ' "
        "from the table below') rather than silently assuming it; (2) if they "
        "don't clearly match, or there are more numbers than you can confidently "
        "map, ask the user to state which number is which metric before explaining "
        "any of them."
        "\n\n" + results_context
    )
    if retrieved_block:
        system_content += (
            "\n\nIf the question is about news/filings, use ONLY the retrieved items below — "
            "if nothing relevant was retrieved, say so rather than inventing news.\n\n" + retrieved_block
        )

    messages = [
        {"role": "system", "content": system_content},
        *chat_history,
        {"role": "user", "content": question},
    ]
    return chat(messages, temperature=0.2, max_tokens=350)