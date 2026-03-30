"""
graphs/query.py  — FIXED
"""

from __future__ import annotations
import structlog
from langgraph.graph import StateGraph, START, END
from config.llm import get_llm
from state.schemas import QueryState
from tools.finance import fetch_financials, fetch_investment_analysis

log = structlog.get_logger(__name__)
_llm = get_llm()

_PROMPT = """\
You are a financial expert. Using the data below for {symbol},
answer the following question precisely and concisely:

Question: {query}

Cite specific numbers from the data where relevant.
Use markdown formatting.

Financials:
{financials}

Investment Data:
{investment}
"""

def fetch_data(state: QueryState) -> dict:
    log.info("query_graph.fetch_data", symbol=state["symbol"])
    data = fetch_financials.invoke({"symbol": state["symbol"]})
    if "error" in data:                           # ← ADDED
        raise RuntimeError(
            f"Failed to fetch financials for {state['symbol']}: {data['error']}"
        )
    return {"raw_financials": data}

def fetch_analysis(state: QueryState) -> dict:
    log.info("query_graph.fetch_analysis", symbol=state["symbol"])
    data = fetch_investment_analysis.invoke({"symbol": state["symbol"]})
    if "error" in data:                           # ← ADDED
        raise RuntimeError(
            f"Failed to fetch investment data for {state['symbol']}: {data['error']}"
        )
    return {"raw_investment": data}

def answer_query(state: QueryState) -> dict:
    log.info("query_graph.answer_query", symbol=state["symbol"], query=state["query"])
    prompt = _PROMPT.format(
        symbol=state["symbol"],
        query=state["query"],
        financials=str(state["raw_financials"])[:3000],
        investment=str(state["raw_investment"])[:3000],
    )
    response = _llm.invoke(prompt)
    return {"result": response.content}

def _build(checkpointer=None):
    builder = StateGraph(QueryState)
    builder.add_node("fetch_data",    fetch_data)
    builder.add_node("fetch_analysis",fetch_analysis)
    builder.add_node("answer_query",  answer_query)
    builder.add_edge(START,           "fetch_data")
    builder.add_edge("fetch_data",    "fetch_analysis")
    builder.add_edge("fetch_analysis","answer_query")
    builder.add_edge("answer_query",   END)
    return builder.compile(checkpointer=checkpointer)

query_graph = _build()