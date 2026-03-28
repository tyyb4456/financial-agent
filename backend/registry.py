"""
Graph registry
--------------
Maps graph name → RegistryEntry.

Each entry stores the graph BUILDER (_build function), not just the compiled
graph, so the API layer can recompile with a checkpointer injected per-request.

The bare compiled graph (no checkpointer) is also cached for streaming
use-cases where persistence isn't needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel


# ── Input schemas ─────────────────────────────────────────────────────────────

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
    query: str = ""


# ── Registry entry ────────────────────────────────────────────────────────────

@dataclass
class RegistryEntry:
    builder:     Callable          # _build(checkpointer=None) function
    schema:      Type[BaseModel]
    description: str
    _graph:      Any = field(default=None, repr=False)   # cached bare graph

    @property
    def graph(self):
        """Bare compiled graph (no checkpointer). Cached on first access."""
        if self._graph is None:
            self._graph = self.builder()
        return self._graph

    def with_checkpointer(self, saver):
        """Return a fresh compiled graph with `saver` as checkpointer."""
        return self.builder(checkpointer=saver)


# ── Builder (singleton) ───────────────────────────────────────────────────────

_registry: dict[str, RegistryEntry] | None = None


def get_registry() -> dict[str, RegistryEntry]:
    global _registry
    if _registry is not None:
        return _registry

    from graphs.financial  import _build as fb
    from graphs.investment import _build as ib
    from graphs.query      import _build as qb
    from graphs.reddit     import _build as rb
    from graphs.news       import _build as nb
    from graphs.medical    import _build as mb

    _registry = {
        "financial_reporter": RegistryEntry(
            builder=fb, schema=FinancialReporterInput,
            description="Retrieve and analyse financial reports for a stock symbol.",
        ),
        "investment_advisor": RegistryEntry(
            builder=ib, schema=InvestmentAdvisorInput,
            description="Provide tailored investment recommendations for a stock symbol.",
        ),
        "financial_query": RegistryEntry(
            builder=qb, schema=FinancialQueryInput,
            description="Answer a specific financial question about a stock symbol.",
        ),
        "trending_posts": RegistryEntry(
            builder=rb, schema=RedditInput,
            description="Retrieve and analyse trending posts from a subreddit.",
        ),
        "news_reporter": RegistryEntry(
            builder=nb, schema=NewsInput,
            description="Generate a news report on a given topic.",
        ),
        "medical_researcher": RegistryEntry(
            builder=mb, schema=MedicalInput,
            description="Retrieve and summarise medical research from PubMed.",
        ),
    }
    return _registry