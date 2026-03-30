"""
graphs/news.py — Now fully async
---------------------------------
News Reporting workflow with async nodes.

  START → search_articles → write_report → END

Async nodes with async tool calls via ainvoke.
"""

from __future__ import annotations
import structlog
from langgraph.graph import StateGraph, START, END
from config.llm import get_llm
from state.schemas import NewsState
from tools.serper import search_news

log = structlog.get_logger(__name__)
_llm = get_llm()

_PROMPT = """\
You are an experienced news analyst. Below are recent news articles about: {topic}

Write a comprehensive news report that includes:

1. **Headline Summary** — 2-sentence overview of what's happening
2. **Key Developments** — the 3–5 most important stories with brief explanations
3. **Context & Background** — relevant background for each development
4. **Sources** — list article titles and URLs in a reference section

Use markdown formatting. For each story include the source URL.

Articles:
{articles}
"""

async def search_articles(state: NewsState) -> dict:
    log.info("news_graph.search_articles", topic=state["topic"])
    articles = await search_news.ainvoke({"query": state["topic"], "num_results": 5})
    return {"raw_articles": articles}

async def write_report(state: NewsState) -> dict:
    log.info("news_graph.write_report", topic=state["topic"])
    prompt = _PROMPT.format(
        topic=state["topic"],
        articles=str(state["raw_articles"])[:5000],
    )
    response = _llm.invoke(prompt)
    return {"result": response.content}

def _build(checkpointer=None):
    builder = StateGraph(NewsState)
    builder.add_node("search_articles", search_articles)
    builder.add_node("write_report",    write_report)
    builder.add_edge(START,             "search_articles")
    builder.add_edge("search_articles", "write_report")
    builder.add_edge("write_report",     END)
    return builder.compile(checkpointer=checkpointer)

news_graph = _build()