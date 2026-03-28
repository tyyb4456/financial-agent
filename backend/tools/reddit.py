"""
tools/reddit.py
---------------
Reddit trending posts tool using PRAW + VADER sentiment analysis.

Reads credentials from config.settings (no raw os.getenv calls here).
"""

from __future__ import annotations

from functools import lru_cache

import praw
import structlog
from pydantic import BaseModel, Field
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import settings

log = structlog.get_logger(__name__)

_analyzer = SentimentIntensityAnalyzer()


# ── PRAW client — lazy singleton ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent="ai-workflows-api/2.0",
    )


# ── Arg schema ────────────────────────────────────────────────────────────────

class TrendingPostsInput(BaseModel):
    """Input for fetching trending Reddit posts."""
    subreddit_name: str = Field(
        description="Name of the subreddit to fetch posts from, e.g. 'technology' or 'investing'."
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of hot posts to retrieve (1–50).",
    )


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool("fetch_trending_posts", args_schema=TrendingPostsInput)
def fetch_trending_posts(subreddit_name: str, limit: int = 10) -> list[dict]:
    """
    Fetch trending (hot) posts from a given subreddit and analyse the sentiment
    of each post title using VADER.

    Returns a list of dicts, each with: title, score, sentiment (compound -1→1), url.
    Use this when you need to understand what's currently popular or discussed on Reddit.
    """
    log.info("tool.fetch_trending_posts", subreddit=subreddit_name, limit=limit)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def _fetch() -> list[dict]:
        reddit   = _reddit_client()
        posts    = []
        for post in reddit.subreddit(subreddit_name).hot(limit=limit):
            sentiment = _analyzer.polarity_scores(post.title)
            posts.append({
                "title":     post.title,
                "score":     post.score,
                "sentiment": round(sentiment["compound"], 4),
                "url":       post.url,
            })
        return posts

    try:
        data = _fetch()
        log.info("tool.fetch_trending_posts.done", subreddit=subreddit_name, count=len(data))
        return data
    except Exception as exc:
        log.error("tool.fetch_trending_posts.error", subreddit=subreddit_name, error=str(exc))
        return [{"error": str(exc)}]