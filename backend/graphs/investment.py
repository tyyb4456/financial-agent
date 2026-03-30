"""
graphs/investment.py  — Now fully async
---------------------------------------
Investment recommendation graph with async nodes and async tool calls.
"""

from __future__ import annotations
import structlog
from langgraph.graph import StateGraph, START, END
from config.llm import get_llm
from state.schemas import InvestmentState
from tools.finance import fetch_financials, fetch_investment_analysis

log = structlog.get_logger(__name__)
_llm = get_llm()

_PROMPT = """\
You are a seasoned investment advisor. Using the data below for {symbol},
provide a structured investment recommendation. Include:

1. **Investment Thesis** — bull and bear case in 2–3 sentences each
2. **Valuation Assessment** — is the stock cheap, fair, or expensive vs peers?
3. **Key Risks** — top 3 risks to the thesis
4. **Recommendation** — Buy / Hold / Sell with a clear rationale
5. **Price Target** — estimated fair value range

Use markdown formatting. Be direct and actionable.

Financials:
{financials}

Valuation & Analyst Data:
{investment}
"""

async def fetch_data(state: InvestmentState) -> dict:
    log.info("investment_graph.fetch_data", symbol=state["symbol"])
    data = await fetch_financials.ainvoke({"symbol": state["symbol"]})
    if "error" in data:
        raise RuntimeError(
            f"Failed to fetch financials for {state['symbol']}: {data['error']}"
        )
    return {"raw_financials": data}

async def fetch_analysis(state: InvestmentState) -> dict:
    log.info("investment_graph.fetch_analysis", symbol=state["symbol"])
    data = await fetch_investment_analysis.ainvoke({"symbol": state["symbol"]})
    if "error" in data:
        raise RuntimeError(
            f"Failed to fetch investment data for {state['symbol']}: {data['error']}"
        )
    return {"raw_investment": data}

async def write_recommendation(state: InvestmentState) -> dict:
    log.info("investment_graph.write_recommendation", symbol=state["symbol"])
    prompt = _PROMPT.format(
        symbol=state["symbol"],
        financials=str(state["raw_financials"])[:3000],
        investment=str(state["raw_investment"])[:3000],
    )
    response = _llm.invoke(prompt)
    return {"result": response.content}

def _build(checkpointer=None):
    builder = StateGraph(InvestmentState)
    builder.add_node("fetch_data",          fetch_data)
    builder.add_node("fetch_analysis",      fetch_analysis)
    builder.add_node("write_recommendation",write_recommendation)
    builder.add_edge(START,                 "fetch_data")
    builder.add_edge("fetch_data",          "fetch_analysis")
    builder.add_edge("fetch_analysis",      "write_recommendation")
    builder.add_edge("write_recommendation", END)
    return builder.compile(checkpointer=checkpointer)

investment_graph = _build()