"""
tools/finance.py
----------------
Yahoo Finance tools using yfinance — clean, no API key needed.

yfinance's .info dict contains all fields in one call, so we make ONE
request per tool instead of multiple yahooquery property fetches.

Two tools exposed (same interface as always):
  - fetch_financials          → financials, market summary, live price
  - fetch_investment_analysis → valuation, earnings history, analyst recs
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import yfinance as yf

log = structlog.get_logger(__name__)

_retry = retry(
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


# ── Shared data fetcher ───────────────────────────────────────────────────────

@_retry
def _get_info(symbol: str) -> dict:
    """Fetch yfinance .info dict with retry. Raises on failure."""
    ticker = yf.Ticker(symbol)
    info   = ticker.info
    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
        raise ValueError(f"No data returned for symbol '{symbol}' — check the ticker.")
    return info


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("fetch_financials", args_schema=FinancialsInput)
def fetch_financials(symbol: str) -> dict:
    """
    Fetch company financials, market summary, and live stock price from Yahoo Finance.

    Returns a dict with keys: financials, summary, price.
    Use this when you need revenue, net income, market cap, or current price data.
    """
    log.info("tool.fetch_financials", symbol=symbol)

    try:
        info = _get_info(symbol)
        data = {
            "financials": {
                symbol: {
                    "totalRevenue":    info.get("totalRevenue"),
                    "netIncome":       info.get("netIncomeToCommon"),
                    "grossProfit":     info.get("grossProfits"),
                    "operatingIncome": info.get("operatingIncome"),
                    "ebitda":          info.get("ebitda"),
                    "returnOnEquity":  info.get("returnOnEquity"),
                    "debtToEquity":    info.get("debtToEquity"),
                    "currentRatio":    info.get("currentRatio"),
                }
            },
            "summary": {
                symbol: {
                    "marketCap":        info.get("marketCap"),
                    "trailingPE":       info.get("trailingPE"),
                    "forwardPE":        info.get("forwardPE"),
                    "priceToBook":      info.get("priceToBook"),
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                    "fiftyTwoWeekLow":  info.get("fiftyTwoWeekLow"),
                    "averageVolume":    info.get("averageVolume"),
                    "dividendYield":    info.get("dividendYield"),
                }
            },
            "price": {
                symbol: {
                    "regularMarketPrice":        info.get("currentPrice") or info.get("regularMarketPrice"),
                    "regularMarketChange":        info.get("regularMarketChange"),
                    "regularMarketChangePercent": info.get("regularMarketChangePercent"),
                    "regularMarketDayHigh":       info.get("dayHigh"),
                    "regularMarketDayLow":        info.get("dayLow"),
                    "currency":                   info.get("currency", "USD"),
                }
            },
        }
        log.info("tool.fetch_financials.done", symbol=symbol)
        return data

    except Exception as exc:
        log.error("tool.fetch_financials.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}


@tool("fetch_investment_analysis", args_schema=InvestmentInput)
def fetch_investment_analysis(symbol: str) -> dict:
    """
    Fetch investment analysis data: valuation multiples, earnings history,
    and analyst buy/sell/hold recommendations from Yahoo Finance.

    Use this when you need P/E ratio, PEG ratio, EPS history, or analyst consensus.
    """
    log.info("tool.fetch_investment_analysis", symbol=symbol)

    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info

        earnings_history = []
        try:
            hist = ticker.quarterly_earnings
            if hist is not None and not hist.empty:
                for date, row in hist.tail(4).iterrows():
                    earnings_history.append({
                        "quarter":         str(date),
                        "epsActual":       row.get("Earnings"),
                        "epsEstimate":     None,
                        "surprisePercent": None,
                    })
        except Exception:
            pass  # earnings history is optional

        data = {
            "valuation": {
                symbol: {
                    "enterpriseValue":              info.get("enterpriseValue"),
                    "pegRatio":                     info.get("pegRatio"),
                    "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
                    "enterpriseToRevenue":          info.get("enterpriseToRevenue"),
                    "enterpriseToEbitda":           info.get("enterpriseToEbitda"),
                }
            },
            "performance": {symbol: earnings_history},
            "analyst_recommendations": {
                symbol: {
                    "numberOfAnalysts": info.get("numberOfAnalystOpinions"),
                    "consensusRating":  info.get("recommendationKey", "").replace("_", " ").title(),
                    "targetMeanPrice":  info.get("targetMeanPrice"),
                    "targetHighPrice":  info.get("targetHighPrice"),
                    "targetLowPrice":   info.get("targetLowPrice"),
                }
            },
        }
        log.info("tool.fetch_investment_analysis.done", symbol=symbol)
        return data

    except Exception as exc:
        log.error("tool.fetch_investment_analysis.error", symbol=symbol, error=str(exc))
        return {"error": str(exc)}