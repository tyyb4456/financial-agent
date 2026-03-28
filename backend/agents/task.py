from agents.agent import *
from crewai import Task

financial_report_task = Task(
    description=(
        """Retrieve and analyze financial reports and market data for the given stock symbol {symbol} 
        using the Yahoo Finance API. Extract key financial metrics and provide a structured assessment 
        of the company's financial health. The report should include:  
        - **Company Financials:** Revenue, net income, and key financial indicators.  
        - **Market Summary:** Overview of the company’s market standing and recent trends.  
        - **Stock Price Insights:** Latest stock price data and historical performance.  
        """
    ),
    agent=financial_report_agent,
    expected_output=(
        """A structured financial summary containing:  
        - **Key Financial Metrics:** Revenue, profitability, and other financial indicators.  
        - **Market Performance Overview:** Trends, risks, and company standing in the industry.  
        - **Stock Price Data:** Latest stock price, historical trends, and relevant insights.
        """
    )
)

investment_advisor_task = Task(
    description=(
        """Provide tailored investment recommendations for the given stock symbol {symbol}. 
        perform a comprehensive evaluation by analyzing valuation metrics, historical performance, and current market trends. 
        Deliver actionable insights in a well-structured format to support informed investment decisions."""
    ),
    agent=investment_advisor_agent,
    expected_output=(
        """A structured analysis that includes tailored investment recommendations, 
        a detailed review of valuation metrics, historical performance, current market trends, 
        and actionable insights to guide investment decisions."""
    )
)

query_answerer_task_financial = Task(
    description=(
        """Answer specific queries related to financial data, market trends, and investment opportunities for the company {symbol}.
        Utilize your expertise to provide accurate and detailed responses to user questions."""
    ),
    agent=query_answerer_agent_financial,
    expected_output=(
        """Accurate and detailed responses to user queries related to financial data, market trends, and investment opportunities."""
    )
)

trending_posts_task = Task(
    description=(
        """Retrieve trending posts from the specified subreddit {subreddit_name} using the Reddit API.  
        Analyze the sentiment of each post title and return a structured summary that includes:  
        - **Post Title:** The title of the trending post.  
        - **Score:** The post upvote score.  
        - **Sentiment Analysis:** A polarity score indicating the sentiment of the title.  
        - **Post URL:** A direct link to the post.
        """
    ),
    agent=trending_posts_agent,
    expected_output=(
        """A structured summary of trending posts from the given subreddit, including:  
        - **Post Title:** Title of the trending post.  
        - **Score:** Number of upvotes received.  
        - **Sentiment Analysis:** Sentiment polarity score of the title.  
        - **Post URL:** Direct link to the post.  
        """
    )
)

reporting_task = Task(
    description=("Generate a report on the {topic}"),
    agent=reporting_analyst,
    expected_output=("""A fully fledge reporting of the news articles.  Make sure you have the title, summary, url and the image.
    Formatted as markdown without '```
    """)

)

medical_task = Task(
    description="Retrieve up to PubMed articles for {term}. Provide a short expert response if a query is given; otherwise, summarize key findings and recommendations.",
    agent=medical_agent,
    expected_output="A clear medical report: expert response for queries or a structured summary of key insights if no query is provided."
)