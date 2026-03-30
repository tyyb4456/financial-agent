"""
graphs/reddit.py — Now fully async
-----------------------------------
Reddit Trending Posts workflow with async nodes.

  START → fetch_posts → summarise → END

Async nodes with async tool calls via ainvoke.
"""

from __future__ import annotations
import structlog
from langgraph.graph import StateGraph, START, END
from config.llm import get_llm
from state.schemas import RedditState
from tools.reddit import fetch_trending_posts

log = structlog.get_logger(__name__)
_llm = get_llm()

_PROMPT = """\
You are a Reddit trends analyst. Below are the top trending posts from r/{subreddit}.

Analyse the data and produce a structured report with:

1. **Overview** — overall mood and dominant themes (2–3 sentences)
2. **Top Posts** — table with columns: Title | Score | Sentiment
3. **Sentiment Summary** — average sentiment, most positive post, most negative post
4. **Key Takeaways** — 3 bullet points on what the community is focused on

Use markdown formatting.

Posts:
{posts}
"""

async def fetch_posts(state: RedditState) -> dict:
    log.info("reddit_graph.fetch_posts", subreddit=state["subreddit_name"])
    posts = await fetch_trending_posts.ainvoke({"subreddit_name": state["subreddit_name"]})
    return {"raw_posts": posts}

async def summarise(state: RedditState) -> dict:
    log.info("reddit_graph.summarise", subreddit=state["subreddit_name"])
    prompt = _PROMPT.format(
        subreddit=state["subreddit_name"],
        posts=str(state["raw_posts"])[:5000],
    )
    response = _llm.invoke(prompt)
    return {"result": response.content}

def _build(checkpointer=None):
    builder = StateGraph(RedditState)
    builder.add_node("fetch_posts", fetch_posts)
    builder.add_node("summarise",   summarise)
    builder.add_edge(START,         "fetch_posts")
    builder.add_edge("fetch_posts", "summarise")
    builder.add_edge("summarise",    END)
    return builder.compile(checkpointer=checkpointer)

reddit_graph = _build()