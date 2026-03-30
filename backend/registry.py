"""
Graph registry
--------------
Maps graph name → RegistryEntry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel, ConfigDict


# ── Input schemas (with Swagger examples) ─────────────────────────────────────

class FinancialReporterInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"symbol": "AAPL"}})
    symbol: str

class InvestmentAdvisorInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"symbol": "MSFT"}})
    symbol: str

class FinancialQueryInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"symbol": "GOOG", "query": "What is the P/E ratio?"}})
    symbol: str
    query:  str

class RedditInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"subreddit_name": "investing"}})
    subreddit_name: str

class NewsInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"topic": "AI regulation 2025"}})
    topic: str

class MedicalInput(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"term": "type 2 diabetes", "query": "What are the latest treatments?"}})
    term:  str
    query: str = ""


# ── Registry entry ────────────────────────────────────────────────────────────

@dataclass
class RegistryEntry:
    builder:     Callable
    schema:      Type[BaseModel]
    description: str
    _graph:      Any = field(default=None, repr=False)

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.builder()
        return self._graph

    def with_checkpointer(self, saver):
        return self.builder(checkpointer=saver)


# ── Singleton ─────────────────────────────────────────────────────────────────

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