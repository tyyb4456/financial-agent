"""
tools/
------
All LangChain @tool functions, split by domain.
Import directly from the sub-module for clarity:

    from tools.finance import fetch_financials, fetch_investment_analysis
    from tools.reddit  import fetch_trending_posts
    from tools.pubmed  import fetch_pubmed_abstracts
    from tools.serper  import search_news
"""

from .finance import fetch_financials, fetch_investment_analysis
from .reddit  import fetch_trending_posts
from .pubmed  import fetch_pubmed_abstracts
from .serper  import search_news

__all__ = [
    "fetch_financials",
    "fetch_investment_analysis",
    "fetch_trending_posts",
    "fetch_pubmed_abstracts",
    "search_news",
]