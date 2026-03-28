"""
Graph registry
--------------
Single source of truth that maps a graph name (string) to:
  - the compiled LangGraph instance
  - the Pydantic input model for request validation
  - a list of required field names (for clear error messages)

The graphs themselves are imported lazily (inside get_registry()) so that
import errors in one graph don't bring down the whole API.

Usage:
    from registry import get_registry

    registry = get_registry()
    entry    = registry["financial_reporter"]
    result   = await entry.graph.ainvoke(entry.schema(**inputs).model_dump())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel


# ── Per-entry input schemas (Pydantic v2) ─────────────────────────────────────

class FinancialReporterInput(BaseModel):
    symbol: str


class InvestmentAdvisorInput(BaseModel):
    symbol: str


class FinancialQueryInput(BaseModel):
    symbol: str
    query:  str


class RedditInput(BaseModel):
    subreddit_name: str


class NewsInput(BaseModel):
    topic: str


class MedicalInput(BaseModel):
    term:  str
    query: str = ""   # optional


# ── Registry entry ────────────────────────────────────────────────────────────

@dataclass
class RegistryEntry:
    graph:       Any            # compiled LangGraph (CompiledStateGraph)
    schema:      Type[BaseModel]
    description: str


# ── Builder ───────────────────────────────────────────────────────────────────

def get_registry() -> dict[str, RegistryEntry]:
    """
    Build and return the registry dict.

    Graphs are imported here (not at module level) to keep startup clean
    and allow individual graphs to fail without cascading.
    """

    # Import graphs lazily
    from graphs.financial   import financial_graph
    from graphs.investment  import investment_graph
    from graphs.query       import query_graph
    from graphs.reddit      import reddit_graph
    from graphs.news        import news_graph
    from graphs.medical     import medical_graph

    return {
        "financial_reporter": RegistryEntry(
            graph=financial_graph,
            schema=FinancialReporterInput,
            description="Retrieve and analyse financial reports for a stock symbol.",
        ),
        "investment_advisor": RegistryEntry(
            graph=investment_graph,
            schema=InvestmentAdvisorInput,
            description="Provide tailored investment recommendations for a stock symbol.",
        ),
        "financial_query": RegistryEntry(
            graph=query_graph,
            schema=FinancialQueryInput,
            description="Answer a specific financial question about a stock symbol.",
        ),
        "trending_posts": RegistryEntry(
            graph=reddit_graph,
            schema=RedditInput,
            description="Retrieve and analyse trending posts from a subreddit.",
        ),
        "news_reporter": RegistryEntry(
            graph=news_graph,
            schema=NewsInput,
            description="Generate a news report on a given topic.",
        ),
        "medical_researcher": RegistryEntry(
            graph=medical_graph,
            schema=MedicalInput,
            description="Retrieve and summarise medical research from PubMed.",
        ),
    }