"""
graphs/medical.py
------------------
Medical Researcher workflow with a conditional branch:

  START → fetch_abstracts → route ─┬─ (query provided) → answer_query → END
                                   └─ (no query)        → summarise    → END

This is the first graph that uses add_conditional_edges — demonstrating
LangGraph's branching capability over the old sequential-only CrewAI setup.
"""

from __future__ import annotations
from typing import Literal
import structlog
from langgraph.graph import StateGraph, START, END
from config.llm import get_llm
from state.schemas import MedicalState
from tools.pubmed import fetch_pubmed_abstracts

log = structlog.get_logger(__name__)
_llm = get_llm()

_SUMMARY_PROMPT = """\
You are an expert medical analyst. Based on the PubMed abstracts below for the
topic "{term}", write a structured medical summary:

1. **Key Findings** — most important results from the literature
2. **Clinical Implications** — what this means for patients or practitioners
3. **Limitations** — gaps or caveats in the current evidence
4. **Recommendations** — evidence-based takeaways

Use clear, professional medical language with markdown formatting.

Abstracts:
{abstracts}
"""

_QUERY_PROMPT = """\
You are an expert medical analyst. Answer the following question using ONLY
the PubMed evidence provided. Be precise, cite findings, and note uncertainty.

Question: {query}

Abstracts:
{abstracts}
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def fetch_abstracts(state: MedicalState) -> dict:
    log.info("medical_graph.fetch_abstracts", term=state["term"])
    abstracts = fetch_pubmed_abstracts.invoke({"term": state["term"], "max_results": 2})
    return {"raw_abstract": abstracts}


def summarise(state: MedicalState) -> dict:
    log.info("medical_graph.summarise", term=state["term"])
    prompt = _SUMMARY_PROMPT.format(
        term=state["term"],
        abstracts=state["raw_abstract"][:6000],
    )
    response = _llm.invoke(prompt)
    return {"result": response.content}


def answer_query(state: MedicalState) -> dict:
    log.info("medical_graph.answer_query", term=state["term"], query=state["query"])
    prompt = _QUERY_PROMPT.format(
        query=state["query"],
        abstracts=state["raw_abstract"][:6000],
    )
    response = _llm.invoke(prompt)
    return {"result": response.content}


# ── Conditional router ────────────────────────────────────────────────────────

def _route(state: MedicalState) -> Literal["answer_query", "summarise"]:
    """Branch: if a specific query was provided → answer it; else → general summary."""
    if (state.get("query") or "").strip():
        return "answer_query"
    return "summarise"


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build(checkpointer=None):
    builder = StateGraph(MedicalState)
    builder.add_node("fetch_abstracts", fetch_abstracts)
    builder.add_node("summarise",       summarise)
    builder.add_node("answer_query",    answer_query)

    builder.add_edge(START, "fetch_abstracts")
    builder.add_conditional_edges(
        "fetch_abstracts",
        _route,
        {"answer_query": "answer_query", "summarise": "summarise"},
    )
    builder.add_edge("summarise",    END)
    builder.add_edge("answer_query", END)
    return builder.compile(checkpointer=checkpointer)

medical_graph = _build()