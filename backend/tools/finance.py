"""
tools/finance.py
----------------
Yahoo Finance tools using the latest LangChain @tool pattern.

Two tools are exposed:
  - fetch_financials   → company financials, market summary, live price
  - fetch_investment   → valuation metrics, earnings history, analyst recs

Both use Pydantic BaseModel args schemas (recommended over bare type hints
for complex tools) and tenacity retries for network resilience.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from yahooquery import Ticker

log = structlog.get_logger(__name__)


# ── Retry decorator shared by all finance tools ───────────────────────────────

_finance_retry = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


# ── Arg schemas ───────────────────────────────────────────────────────────────

class FinancialsInput(BaseModel):
    """Input for fetching company financials."""
    symbol: str = Field(description="Stock ticker symbol, e.g. 'AAPL' or 'TSLA'.")


class InvestmentInput(BaseModel):
    """Input for fetching investment analysis data."""
    symbol: str = Field(description="Stock ticker symbol, e.g. 'GOOGL' or 'MSFT'.")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("fetch_financials", args_schema=FinancialsInput)
def fetch_financials(symbol: str) -> dict:
    """
    Fetch company financials, market summary, and live stock price from Yahoo Finance.

    Returns a dict with keys: financials, summary, price.
    Use this when you need revenue, net income, market cap, or current price data.
    """
    log.info("tool.fetch_financials", symbol=symbol)

    @_finance_retry
    def _fetch() -> dict:
        ticker = Ticker(symbol)
        return {
            "financials": ticker.financial_data,
            "summary":    ticker.summary_detail,
            "price":      ticker.price,
        }

    try:
        data = _fetch()
        log.info("tool.fetch_financials.done", symbol=symbol)
        return data
    except Exception as exc:
        log.error("tool.fetch_financials.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}


@tool("fetch_investment_analysis", args_schema=InvestmentInput)
def fetch_investment_analysis(symbol: str) -> dict:
    """
    Fetch investment analysis data for a stock: valuation multiples, earnings history,
    and analyst buy/sell/hold recommendations from Yahoo Finance.

    Use this when you need P/E ratio, PEG ratio, EPS history, or analyst consensus.
    """
    log.info("tool.fetch_investment_analysis", symbol=symbol)

    @_finance_retry
    def _fetch() -> dict:
        ticker = Ticker(symbol)
        return {
            "valuation":            ticker.valuation_measures,
            "performance":          ticker.earning_history,
            "analyst_recommendations": ticker.recommendation_trend,
        }

    try:
        data = _fetch()
        log.info("tool.fetch_investment_analysis.done", symbol=symbol)
        return data
    except Exception as exc:
        log.error("tool.fetch_investment_analysis.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}