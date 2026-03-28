from crewai import LLM, Agent
from agents.tool import *
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# # ✅ Define LLM (Groq Llama)
# llm1 = LLM(
#     model = "groq/deepseek-r1-distill-llama-70b",
#     temperature=0.7,
#     max_completion_tokens=6144
# )
# # ✅ Define LLM (Groq Llama)
# llm2 = LLM(
#     model = "groq/llama-3.3-70b-versatile",
#     temperature=0.7,
#     max_completion_tokens=6144
# )
# # ✅ Define LLM (Groq Llama)
# llm3 = LLM(
#     model = "groq/llama-3.3-70b-specdec",
#     temperature=0.7,
#     max_completion_tokens=6144
# )
# # ✅ Define LLM (Groq Llama)
# llm4 = LLM(
#     model = "groq/deepseek-r1-distill-qwen-32b",
#     temperature=0.7,
#     max_completion_tokens=6144
# )


# --- Agent Definitions ---

financial_report_agent = Agent(
    role="Financial Report Agent",
    goal=(
        """Retrieve and analyze financial reports and market data for the given stock symbol {symbol} 
        using the Yahoo Finance API. Extract key financial metrics, including company financials, 
        market summary details, and stock price information. If a specific query is provided, tailor 
        the analysis accordingly. Otherwise, provide a structured financial assessment that includes:  
        - **Company Financials:** Key financial data, such as revenue, net income, and other indicators.  
        - **Market Summary:** A general overview of the company s market performance and standing.  
        - **Stock Price Insights:** Latest stock price details and historical performance trends."""
    ),
    backstory=(
        """You are a seasoned financial expert specializing in analyzing company financials and market data. 
        Your responsibility is to gather, interpret, and summarize key financial insights from Yahoo Finance. 
        Regardless of whether a specific query is given, you provide a comprehensive financial report.  
        If a query is provided, you adjust your insights to address it while ensuring a broad overview of 
        the company's financial health and stock performance."""
    ),
    tools=[fetch_financial_reports],
    llm=model # Assumes llm is defined and configured elsewhere.
)


investment_advisor_agent = Agent(
    role='Investment Advisor Agent',
    goal=(
        """Provide tailored investment recommendations for the given stock symbol {symbol}. 
        proceed with a comprehensive evaluation, analyzing valuation metrics, historical performance, and current market trends. 
        Deliver actionable insights in a well-structured format to support informed investment decisions."""
    ),
    backstory=(
        """You are a seasoned investment advisor with extensive expertise in market trends, financial analysis, and investment strategies. 
        Regardless of whether a specific query is given, you conduct a thorough investment analysis and provide insightful recommendations.
        Tailor your advice accordingly while ensuring a well-rounded evaluation of the stock's potential risks and opportunities."""
    ),
    tools=[fetch_financial_reports,fetch_investment_analysis],
    llm=model
)

query_answerer_agent_financial = Agent(
    role='Query Answerer Agent',
    goal=(
        """Answer specific queries related to financial data, market trends, and investment opportunities for the company {symbol}.
        Utilize your expertise to provide accurate and detailed responses to user questions."""
    ),
    backstory=(
        """You are a financial expert with in-depth knowledge of market trends, investment strategies, and financial analysis.
        Your role is to address user queries with precision and clarity, offering valuable insights and recommendations."""
    ),
    tools=[fetch_financial_reports, fetch_investment_analysis],
    llm=model
)

trending_posts_agent = Agent(
    role="Reddit Trending Posts Analyst",
    goal=(
        """Retrieve trending posts from the specified subreddit {subreddit_name} using the Reddit API.  
        Analyze the sentiment of each post title and provide a structured summary that includes:  
        - **Post Title:** The title of the trending post.  
        - **Score:** The posts upvote score.  
        - **Sentiment Analysis:** A polarity score indicating the sentiment of the title.  
        - **Post URL:** A direct link to the post.  
        """
    ),
    backstory=(
        """You are a Reddit trends analyst specializing in identifying and analyzing trending posts across subreddits.  
        Your task is to fetch the most popular posts from a given subreddit and evaluate their sentiment to gauge 
        the overall mood of the discussions.Deliver a structured report containing key insights about trending content."""
    ),
    tools=[fetch_trending_posts],
    llm=model
)


reporting_analyst = Agent(
    role="{topic} News Reporting Analyst",
    goal=("Create detailed reports based on {topic} news analysis and research findings"),
    backstory=("You're a meticulous analyst with a keen eye for detail. You're known for your ability to turn complex data into clear and concise reports, making it easy for others to understand and act on the information you provide."),
    tools=[CustomSerperDevTool()],
    llm=model
)


medical_agent = Agent(
    role="Medical Agent",
    goal="Retrieve medical research from PubMed using E-utilities for {term}. Provide a short expert-level response to specific queries or a structured summary of key findings if no query is given.",
    backstory="An expert medical analyst specializing in research analysis. Retrieves and synthesizes PubMed data to provide clear, medically insightful responses.",
    tools=[fetch_data],
    llm=model
)