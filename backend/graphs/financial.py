"""
graphs/financial.py
--------------------
Financial Reporter workflow:

  START → fetch_data → write_report → END
"""

from __future__ import annotations
import structlog
from langgraph.graph import StateGraph, START, END
from config.llm import get_llm
from state.schemas import FinancialState
from tools.finance import fetch_financials

log = structlog.get_logger(__name__)
_llm = get_llm()

_REPORT_PROMPT = """\
You are a senior financial analyst. Using the raw Yahoo Finance data below,
write a structured financial report for {symbol}. Include:

1. **Company Financials** — revenue, net income, key ratios
2. **Market Summary** — market cap, P/E, 52-week range, volume
3. **Stock Price Insights** — current price, day change, analyst targets

Be concise, factual, and use markdown formatting.

Raw data:
{raw_data}
"""

def fetch_data(state: FinancialState) -> dict:
    log.info("financial_graph.fetch_data", symbol=state["symbol"])
    data = fetch_financials.invoke({"symbol": state["symbol"]})
    return {"raw_financials": data}

def write_report(state: FinancialState) -> dict:
    log.info("financial_graph.write_report", symbol=state["symbol"])
    prompt = _REPORT_PROMPT.format(
        symbol=state["symbol"],
        raw_data=str(state["raw_financials"])[:6000],
    )
    response = _llm.invoke(prompt)
    return {"result": response.content}

def _build(checkpointer=None):
    builder = StateGraph(FinancialState)
    builder.add_node("fetch_data",   fetch_data)
    builder.add_node("write_report", write_report)
    builder.add_edge(START,          "fetch_data")
    builder.add_edge("fetch_data",   "write_report")
    builder.add_edge("write_report",  END)
    return builder.compile(checkpointer=checkpointer)

financial_graph = _build()