from crewai.tools import tool,BaseTool
import requests
import os
from yahooquery import Ticker
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



import json

from dotenv import load_dotenv

load_dotenv()


class CustomSerperDevTool(BaseTool):
    name: str = "Custom Serper Dev Tool"
    description: str = "Search the internet for news."

    def _run(self, query: str) -> str:
        """
        Search the internet for news.
        """

        url = "https://google.serper.dev/news"

        payload = json.dumps({
            "q": query,
            "num": 5,
            "autocorrect": False,
            "tbs": "qdr:d"
        })

        headers = {
            'X-API-KEY': os.getenv('SERPER_API_KEY'),
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        # Parse the JSON response
        response_data = response.json()

        # Extract only the 'news' property
        news_data = response_data.get('news', [])

        # Convert the news data back to a JSON string
        return json.dumps(news_data, indent=2)
    

@tool("get financial reports and market data")
def fetch_financial_reports(symbol: str):
    """
    Fetches financial reports and market data for a given stock symbol using Yahoo Finance API.
    
    Args:
        symbol (str): The stock symbol (e.g., 'AAPL' for Apple).
    
    Returns:
        dict: A dictionary containing financial reports and market data.
    """
    try:
        ticker = Ticker(symbol)
        data = {
            "financials": ticker.financial_data,
            "summary": ticker.summary_detail,
            "price": ticker.price
        }
        return data
    except Exception as e:
        return {"error": str(e)}
    
@tool("fetch investment analysis")
def fetch_investment_analysis(symbol: str):
    """
    Fetches investment analysis data for a given stock symbol using various financial APIs.
    
    Args:
        symbol (str): The stock symbol (e.g., 'GOOGL' for Alphabet Inc.).
    
    Returns:
        dict: A dictionary containing analysis data such as valuation metrics, historical performance,
              and analyst recommendations.
    """
    try:
        ticker = Ticker(symbol)
        data = {
            "valuation": ticker.valuation_measures,
            "performance": ticker.earning_history,
            "analystRecommendations": ticker.recommendation_trend
        }
        return data
    except Exception as e:
        return {"error": str(e)}

import requests

clientid = os.getenv("REDDIT_CLIENT_ID")
clientsecret = os.getenv("REDDIT_CLIENT_SECRET")
# Reddit API Configuration
reddit = praw.Reddit(
    client_id = clientid,
    client_secret = clientsecret,
    user_agent="my_APP"
)

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

@tool("fetch trending posts from Reddit")
def fetch_trending_posts(subreddit_name='all', limit=10):
    """
    Fetch trending posts from a given subreddit.

    Args:
        subreddit_name (str): Name of the subreddit to fetch posts from.
        limit (int): Number of posts to retrieve.

    Returns:
        list: A list of dictionaries containing post details (title, score, sentiment, and URL).
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.hot(limit=limit):
        sentiment = analyzer.polarity_scores(post.title)
        posts.append({
            'title': post.title,
            'score': post.score,
            'sentiment': sentiment['compound'],
            'url': post.url
        })
    
    return posts


@tool("fetch data of medical articles")
def fetch_data(term: str):
    """
    Fetches the abstract of a medical article from PubMed based on a search term.
    
    This function uses NCBI's E-utilities to search for a single article ID matching the term
    and then fetches its abstract.
    
    Args:
        term (str): The search term for querying a medical article.
    
    Returns:
        str or dict: A string containing the abstract of the fetched article, or a dict with an error message.
    """
    # Define the URL and parameters for the ESearch endpoint.
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": term,
        "retmax": "2",
        "retmode": "json"
    }
    
    try:
        # Execute the ESearch request.
        esearch_response = requests.get(esearch_url, params=esearch_params)
        esearch_response.raise_for_status()
        esearch_data = esearch_response.json()
        
        # Extract the first article ID from the JSON response.
        article_id = esearch_data.get("esearchresult", {}).get("idlist", [None])[0]
        if not article_id:
            return {"error": "No articles found for the given search term."}
        
        # Define the URL and parameters for the EFetch endpoint.
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": article_id,
            "retmode": "text",
            "rettype": "abstract"
        }
        
        # Execute the EFetch request.
        efetch_response = requests.get(efetch_url, params=efetch_params)
        efetch_response.raise_for_status()
        
        # Return the fetched abstract as plain text.
        return efetch_response.text
        
    except requests.RequestException as e:
        return {"error": f"Failed to fetch medical data: {str(e)}"}

