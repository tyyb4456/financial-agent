"""
conftest.py
-----------
Shared pytest fixtures used across all test modules.

Key design decisions:
  - CHECKPOINT_BACKEND=memory so tests never touch disk (fast + isolated)
  - GOOGLE_API_KEY set to a dummy value so LLM instantiation doesn't fail
    at import time — actual LLM calls are always mocked via monkeypatch
  - A fresh registry cache is cleared between tests that mutate it
  - All external I/O (Yahoo Finance, Reddit, PubMed, Serper, LLM) is patched
    at the tool / LLM layer so graphs run end-to-end with zero network calls
"""

import os
import sys
import json
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ── Path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("LANGSMITH_TRACING", "false")

# ── Force memory checkpointer for all tests ───────────────────────────────────
os.environ["CHECKPOINT_BACKEND"] = "memory"
os.environ["GOOGLE_API_KEY"]     = "test-api-key-not-real"

# ── Configure logging (suppress output during tests) ──────────────────────────
from config.logging import configure_logging
configure_logging()


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app():
    """Import and return the FastAPI app (session-scoped — built once)."""
    from main import app as _app
    return _app


@pytest.fixture()
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTPX client wired to the FastAPI app. Fresh per test."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── LLM mock ──────────────────────────────────────────────────────────────────

def make_llm_response(content: str = "mocked LLM response"):
    """Return a MagicMock that looks like a LangChain AIMessage."""
    msg = MagicMock()
    msg.content = content
    return msg


@pytest.fixture()
def mock_llm(monkeypatch):
    """
    Patch get_llm() everywhere so no real LLM is ever called.
    Returns the mock so individual tests can set .invoke.return_value.
    """
    llm = MagicMock()
    llm.invoke.return_value = make_llm_response("## Mocked Report\n\nThis is a test.")
    monkeypatch.setattr("config.llm.get_llm", lambda *a, **kw: llm)

    # Also patch already-imported module-level _llm references in each graph
    for module_path in [
        "graphs.financial", "graphs.investment", "graphs.query",
        "graphs.reddit",    "graphs.news",       "graphs.medical",
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            monkeypatch.setattr(mod, "_llm", llm)
        except (ImportError, AttributeError):
            pass

    return llm


# ── Tool mocks ────────────────────────────────────────────────────────────────

FAKE_FINANCIALS = {
    "financials": {"AAPL": {"totalRevenue": 394_000_000_000, "netIncome": 96_000_000_000}},
    "summary":    {"AAPL": {"marketCap": 3_000_000_000_000, "trailingPE": 31.5}},
    "price":      {"AAPL": {"regularMarketPrice": 195.0, "currency": "USD"}},
}

FAKE_INVESTMENT = {
    "valuation":               {"AAPL": {"enterpriseValue": 3_100_000_000_000}},
    "performance":             {"AAPL": {}},
    "analyst_recommendations": {"AAPL": {"buy": 28, "hold": 10, "sell": 2}},
}

FAKE_POSTS = [
    {"title": "AI is taking over", "score": 4200, "sentiment": 0.42, "url": "https://reddit.com/1"},
    {"title": "Markets down today", "score": 1100, "sentiment": -0.31, "url": "https://reddit.com/2"},
]

FAKE_ABSTRACTS = (
    "1. Smith J et al. (2024) Metformin reduces HbA1c by 1.2% in T2D patients. PMID: 99999.\n\n"
    "Abstract: A randomised trial of 500 patients showed significant glycaemic improvement..."
)

FAKE_NEWS = [
    {"title": "AI regulation bill passes", "snippet": "Congress passed...",
     "url": "https://news.com/1", "date": "2025-01-01", "source": "Reuters"},
    {"title": "Tech stocks rally",         "snippet": "Markets up 3%...",
     "url": "https://news.com/2", "date": "2025-01-01", "source": "Bloomberg"},
]


@pytest.fixture()
def mock_tools(monkeypatch):
    """
    Patch every external tool so graphs run with zero network calls.

    LangChain's StructuredTool is a Pydantic model — you cannot setattr
    `.invoke` directly on it. The correct approach is to patch the underlying
    `._run` method (which `.invoke` delegates to) using `unittest.mock.patch.object`.

    We also patch the graph node functions directly so the mock data flows
    through state correctly.
    """
    from unittest.mock import patch

    import tools.finance as fin
    import tools.reddit  as red
    import tools.pubmed  as pub
    import tools.serper  as ser

    patches = [
        patch.object(fin.fetch_financials,          "_run", return_value=FAKE_FINANCIALS),
        patch.object(fin.fetch_investment_analysis, "_run", return_value=FAKE_INVESTMENT),
        patch.object(red.fetch_trending_posts,      "_run", return_value=FAKE_POSTS),
        patch.object(pub.fetch_pubmed_abstracts,    "_run", return_value=FAKE_ABSTRACTS),
        patch.object(ser.search_news,               "_run", return_value=FAKE_NEWS),
    ]

    for p in patches:
        p.start()

    yield {
        "financials": FAKE_FINANCIALS,
        "investment": FAKE_INVESTMENT,
        "posts":      FAKE_POSTS,
        "abstracts":  FAKE_ABSTRACTS,
        "news":       FAKE_NEWS,
    }

    for p in patches:
        p.stop()


# ── Registry cache reset ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    """
    Clear the registry singleton cache before every test.
    Prevents state bleed between tests that patch graph builders.
    """
    import registry as reg_module
    original = reg_module._registry
    yield
    reg_module._registry = original