from crewai import Crew, Process
from agents.agent import *
from agents.task import *


from typing import Dict, Any

class CrewDefinition:
    def __init__(self, crew, required_inputs, optional_inputs=None):
        self.crew = crew
        self.required_inputs = required_inputs  # Define the expected inputs
        self.optional_inputs = optional_inputs or {}  # Define the optional inputs

# Define available crews and their required inputs
crews = {
    "financial reporter": CrewDefinition(
        Crew(
            agents=[financial_report_agent],
            tasks=[financial_report_task],
            process=Process.sequential,
        ),
        required_inputs={"symbol": str},
        
    ),

    "investment advisor": CrewDefinition(
        Crew(
            agents=[investment_advisor_agent],
            tasks=[investment_advisor_task],
            process=Process.sequential,
        ),
        required_inputs={"symbol": str},
    ),
    "financial query answerer": CrewDefinition(
        Crew(
            agents=[query_answerer_agent_financial],
            tasks=[query_answerer_task_financial],
            process=Process.sequential,
        ),
        required_inputs={"symbol": str, "query": str},
    ),

    "trending posts searcher": CrewDefinition(
        Crew(
            agents=[trending_posts_agent],
            tasks=[trending_posts_task],
            process=Process.sequential,
        ),
        required_inputs={"subreddit_name": str},
    ),

    "news reporting analyst": CrewDefinition(
        Crew(
            agents=[reporting_analyst],
            tasks=[reporting_task],
            process=Process.sequential,
        ),
        required_inputs={"topic": str},
    ),

    "medical researcher": CrewDefinition(
        Crew(
            agents=[medical_agent],
            tasks=[medical_task],
            process=Process.sequential,
        ),
        required_inputs={"term": str},
        optional_inputs={"query": str},
    )

}