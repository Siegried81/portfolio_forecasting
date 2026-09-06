"""
Unit tests for src/ai_features.py. Had NO test coverage before this pass.

Priority: `build_results_context` — the pure text-formatting function that is
the SAME grounding context fed to both the commentary generator and the
chatbot (per its own docstring, "one source of truth"). A formatting bug here
silently propagates into every LLM feature at once, so it's worth testing on
its own rather than only indirectly via a mocked chat() call. The LLM-calling
functions themselves are tested by mocking src.llm_client.chat — never a real
network call.
"""
import pandas as pd
import pytest

import src.ai_features as ai_features
from src.ai_features import build_results_context, generate_commentary


def _metrics(**overrides) -> dict:
    base = {
        "annual_return": 0.12, "annual_volatility": 0.18, "sharpe_ratio": 0.67,
        "sortino_ratio": 0.90, "max_drawdown": -0.22, "var_95": -0.03, "cvar_95": -0.05,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# build_results_context — pure formatting, shared by every LLM feature
# ---------------------------------------------------------------------------

def test_build_results_context_includes_only_nonzero_weights():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.0005, "TSLA": 0.40})
    context = build_results_context(weights, _metrics(), None, None)
    assert "AAPL: 60.0%" in context
    assert "TSLA: 40.0%" in context
    assert "MSFT" not in context  # below the 0.1% display threshold


def test_build_results_context_omits_forecast_and_realized_sections_when_none():
    weights = pd.Series({"AAPL": 1.0})
    context = build_results_context(weights, _metrics(), None, None)
    assert "HISTORICAL-BASED PORTFOLIO" in context
    assert "FORECAST-BASED PORTFOLIO" not in context
    assert "REALIZED-OPTIMAL PORTFOLIO" not in context


def test_build_results_context_includes_all_three_sections_when_provided():
    weights = pd.Series({"AAPL": 1.0})
    context = build_results_context(weights, _metrics(), _metrics(), _metrics())
    assert "HISTORICAL-BASED PORTFOLIO" in context
    assert "FORECAST-BASED PORTFOLIO" in context
    assert "REALIZED-OPTIMAL PORTFOLIO" in context


def test_build_results_context_shows_period_return_only_when_present():
    weights = pd.Series({"AAPL": 1.0})
    with_period = build_results_context(weights, _metrics(period_return=0.05, n_periods=30), None, None)
    assert "Raw return over the 30-period window" in with_period
    without_period = build_results_context(weights, _metrics(), None, None)
    assert "Raw return over the" not in without_period


def test_build_results_context_flags_inverted_yield_curve():
    weights = pd.Series({"AAPL": 1.0})
    macro_context = {"macro": {"term_spread_10y_3m": -0.005}, "vix_level": None}
    context = build_results_context(weights, _metrics(), None, None, macro_context)
    assert "INVERTED" in context


def test_build_results_context_flags_sahm_rule_recession_signal():
    weights = pd.Series({"AAPL": 1.0})
    macro_context = {"macro": {"sahm_rule_indicator": 0.6}, "vix_level": None}
    context = build_results_context(weights, _metrics(), None, None, macro_context)
    assert "0.60" in context and "recession" in context


def test_build_results_context_omits_macro_section_when_nothing_came_back():
    weights = pd.Series({"AAPL": 1.0})
    # Every macro field None -> the block should never be appended (an empty
    # "MACRO & RISK BACKDROP" header with nothing under it would read as a
    # bug, not an absence of data).
    macro_context = {"macro": {}, "vix_level": None}
    context = build_results_context(weights, _metrics(), None, None, macro_context)
    assert "MACRO & RISK BACKDROP" not in context


def test_build_results_context_classifies_vix_regime():
    weights = pd.Series({"AAPL": 1.0})
    calm = build_results_context(weights, _metrics(), None, None, {"macro": {}, "vix_level": 12.0})
    elevated = build_results_context(weights, _metrics(), None, None, {"macro": {}, "vix_level": 30.0})
    assert "(calm)" in calm
    assert "(elevated)" in elevated


# ---------------------------------------------------------------------------
# generate_commentary — mocked chat(), never a real network call
# ---------------------------------------------------------------------------

def test_generate_commentary_passes_context_through_to_chat(monkeypatch):
    captured = {}

    def _fake_chat(messages, temperature, max_tokens):
        captured["messages"] = messages
        captured["temperature"] = temperature
        return "mocked commentary", "groq"

    monkeypatch.setattr(ai_features, "chat", _fake_chat)
    text, backend = generate_commentary("PORTFOLIO WEIGHTS...\nsome context")
    assert text == "mocked commentary"
    assert backend == "groq"
    assert "some context" in captured["messages"][-1]["content"]
    assert captured["messages"][0]["role"] == "system"