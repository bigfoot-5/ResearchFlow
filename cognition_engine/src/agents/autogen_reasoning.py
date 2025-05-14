import json
from typing import List, Sequence
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os
from typing import AsyncGenerator, Sequence, List, Dict
import json
import adalflow as adal
import openai

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_core import CancellationToken
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, RequestUsage, UserMessage

from dotenv import load_dotenv

load_dotenv()


# ---- Query Agent ----
class QueryAgent(BaseChatAgent):
    def __init__(self, name="QueryAgent"):
        super().__init__(name=name, description="Converts hypothesis into SQL filter.")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        import sqlalchemy
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine, text
        from sqlalchemy.ext.declarative import declarative_base

        hypothesis = messages[-1].content.strip()

        # Use LLM to generate SQL filter condition from hypothesis
        filter_prompt = f"""
You are a SQL expert assisting in CRM data analysis. Use the following schema to help convert the hypothesis into a SQL WHERE clause filter condition.

Schema:
Table: companies
- id (PK), company_name, industry, numberofemployees, annual_revenue, country

Table: contacts
- id (PK), company_id (FK to companies.id), first_name, last_name, job_title

Table: deals
- id (PK), company_id (FK to companies.id), company_name, deal_name, amount, dealstage, closedate, createdate, days_to_close, hs_is_closed_won, hs_is_closed_lost, closed_won_reason, closed_lost_reason, hs_analytics_source

Relationships:
- companies.id = deals.company_id
- companies.id = contacts.company_id

The following are some of the entries from the sql tables:

companies aliased as c table:
      id      |            company_name            |              industry               | numberofemployees | annual_revenue |    country     
--------------+------------------------------------+-------------------------------------+-------------------+----------------+----------------
 138616529118 | HubSpot                            |                                     |               648 |              0 | United Kingdom
 145102662894 | Hernandez, Jackson and Gonzalez    | FOOD_BEVERAGES                      |               730 |              0 | United Kingdom
 145102662895 | Hammond Inc                        | PHARMACEUTICALS                     |               288 |              0 | United Kingdom
 145102662896 | Williams, Hudson and Burton        | PHARMACEUTICALS                     |               773 |              0 | United Kingdom

 contacts aliased as ct table:

       id      |  company_id  | first_name  |         last_name         |       job_title       
--------------+--------------+-------------+---------------------------+-----------------------
 283360955622 | 145707875564 | Jared       | Ponce                     | VP People
 283365953751 | 145720339698 | James       | Roy                       | VP People
 272110637250 |              | vincenzo    | migliore                  | 
 283365953752 | 145558241488 | Paige       | Mcfarland                 | People Analytics Lead
 283365953753 | 145560029376 | Jeffrey     | Ashley                    | VP People
 283365953757 | 145556515034 | Shawn       | Moreno                    | People Analytics Lead

 deals aliased as d table:

       id      |  company_id  |          company_name          |                     deal_name                     | amount |       dealstage       |        closedate        | hs_is_closed_lost | hs_is_closed_won | hs_is_closed | hs_analytics_source | closed_lost_reason  |     closed_won_reason     |       createdate        | days_to_close 
--------------+--------------+--------------------------------+---------------------------------------------------+--------+-----------------------+-------------------------+-------------------+------------------+--------------+---------------------+---------------------+---------------------------+-------------------------+---------------
 199285499119 |              |                                | Moore, Wilson and Stanley New Business 223        | 104674 | appointmentscheduled  | 2025-07-11 05:00:00     | f                 | f                | f            | OFFLINE             |                     | Compensation Benchmarking | 2025-05-31 05:00:00     |            41
 199285499121 |              |                                | Riggs-Hoffman New Business 4226                   |  85064 | presentationscheduled | 2025-05-11 05:00:00     | f                 | f                | f            | PAID_SEARCH         |                     | Workforce Planning        | 2025-04-03 05:00:00     |            38
 199285499122 | 145722137802 | Friedman-Coleman               | Cruz, Hunter and Oconnell New Business 2423       |  67093 | closedlost            | 2025-07-19 05:00:00     | f                 | f                | f            | OFFLINE             | Pricing/Budget      | Diversity Reporting       | 2025-06-10 05:00:00     |            39


In the query, companies is aliased as c, deals is aliased as d, contacts in aliased as ct
Given the following hypothesis, generate a SQL WHERE clause condition (without 'WHERE') that could be used to filter results from these joined tables:


Hypothesis:
{hypothesis}
"""
        client = openai.OpenAI()
        # This could be replaced with your preferred LLM call
        openai.api_key = os.getenv("OPENAI_API_KEY")
        llm_response = client.responses.create(
            model="gpt-3.5-turbo",
            input=[{"role": "user", "content": filter_prompt}]
        )
        import re
        raw_output = llm_response.output_text.strip()
        filter_condition = raw_output
        match = re.search(r"```sql\s*(.*?)\s*```", raw_output, re.DOTALL)
        print(match)
        if match:
            filter_condition = match.group(1).strip()
        else:
            filter_condition = raw_output.strip().splitlines()[0]

        # Create DB connection and session
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        query = text(f"""
            SELECT 
                d.hs_is_closed_won, 
                d.hs_is_closed_lost, 
                d.closed_won_reason, 
                d.closed_lost_reason, 
                d.hs_analytics_source,
                d.deal_name,
                d.days_to_close,
                d.dealstage,
                d.amount,
                d.closedate,
                d.createdate,
                c.company_name,
                c.annual_revenue,
                c.industry,
                c.country,
                c.numberofemployees,
                ct.first_name,
                ct.last_name,
                ct.job_title
            FROM deals d
            LEFT JOIN companies c ON d.company_id = c.id
            LEFT JOIN contacts ct ON d.company_id = ct.company_id
            WHERE {filter_condition};
        """)

        result = session.execute(query)
        from datetime import datetime

        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        rows = [
            {k: serialize(v) for k, v in dict(r._mapping).items()}
            for r in result
        ]

        session.close()

        return Response(
            chat_message=TextMessage(content=json.dumps({"filter": filter_condition, "results": rows}), source=self.name),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)


# ---- Insight Summarizer ----
class InsightSummarizerAgent(BaseChatAgent):
    def __init__(self, name="InsightSummarizerAgent"):
        super().__init__(name=name, description="Summarizes insights from query results.")

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken, data) -> Response:
        # Normally this would execute a query; here we mock a result
        metrics = data
        return Response(chat_message=TextMessage(content=json.dumps(metrics), source=self.name), inner_messages=[])

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)







# ---- OllamaAgent and model_info definition ----
# Example model info for Ollama
model_info = {
    "family": "gemma",
    "function_calling": False,
    "json_output": False,
    "structured_output": False,
    "vision": False,
    "multiple_system_messages": False,
}

# Replace Hypothesis, Reflection, and Narrative agents with OllamaAgent using Gemma
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, RequestUsage

class OllamaAgent(BaseChatAgent):
    def __init__(self, name: str, model: str = "gemma3:1b", system_message: str = None, custom_context: str = None):
        super().__init__(name=name, description="LLM-based agent using Ollama")
        self._model_client = OllamaChatCompletionClient(model=model, model_info=model_info)
        self._model_context = UnboundedChatCompletionContext()
        self._system_message = system_message or "You are a helpful assistant."
        self._custom_context = custom_context or ""  # Default to empty string if no custom context is provided.

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        # Add the custom context to the message history, if present, as an AssistantMessage from system
        if self._custom_context:
            await self._model_context.add_message(AssistantMessage(content=self._custom_context, source="system"))
        
        # Add the user messages to the history
        for msg in messages:
            await self._model_context.add_message(msg)
        
        # Fetch all messages from the context
        history = await self._model_context.get_messages()
        
        # Send the history to the model
        response = await self._model_client.create(history)
        
        # Usage tracking (optional)
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        
        # Add the assistant's response to the model context
        await self._model_context.add_message(AssistantMessage(content=response.content, source=self.name))
        
        # Yield the response message
        yield Response(
            chat_message=TextMessage(content=response.content, source=self.name, models_usage=usage),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("No response was generated by the model.")
# ---- Main execution logic ----
import asyncio


async def run_pipeline(question, data):
    # question= "Why do you think United States is the region with the fastest sales cycle?"
    cancellation_token = CancellationToken()

    # data = {"status": "success", "filters": {"amount": {"operator": ">", "value": "0"}}, "metadata": {"deals_analyzed": 202, "won_deals": 58, "win_rate": 28.71, "avg_deal_size": 78759.65, "filters_applied": {"amount": {"operator": ">", "value": "0"}}}, "dimensions": [{"attribute": "Industry", "highest_win_rate": {"value": "INFORMATION_TECHNOLOGY_AND_SERVICES", "metric": "100.0%", "raw_value": 100.0, "confidence": 0.85}, "fastest_sales_cycle": {"value": "INFORMATION_TECHNOLOGY_AND_SERVICES", "metric": "15 days", "raw_value": 15.0, "confidence": 0.85}}, {"attribute": "Geography", "highest_win_rate": {"value": "United Kingdom", "metric": "100.0%", "raw_value": 100.0, "confidence": 0.85}, "fastest_sales_cycle": {"value": "United Kingdom", "metric": "15 days", "raw_value": 15.0, "confidence": 0.85}}, {"attribute": "Company Size", "highest_win_rate": {"value": "201-1000", "metric": "100.0%", "raw_value": 100.0, "confidence": 0.85}, "fastest_sales_cycle": {"value": "201-1000", "metric": "15 days", "raw_value": 15.0, "confidence": 0.85}}]}
    hypothesis_agent = OllamaAgent(name="HypothesisAgent", model="gemma3:1b", system_message="Generate insightful hypotheses for GTM triangulation based on CRM data.", custom_context=json.dumps(data))
    query_agent = QueryAgent()
    summarizer_agent = InsightSummarizerAgent()
    narrative_agent = OllamaAgent(name="NarrativeGeneratorAgent", model="gemma3:1b", system_message=f"Write a concise narrative summarizing insights, emphasizing business implications. Based on that answer the {question}", custom_context=json.dumps(data))
    # Step 1: Hypothesis
    hyp_resp = await hypothesis_agent.on_messages([UserMessage(content=question, source="user")], cancellation_token)
    print(f"\nHypothesis: {hyp_resp.chat_message.content}")

    if not hyp_resp.chat_message.content.strip():
        print("Hypothesis agent returned no content. Exiting pipeline.")
        return

    # Step 2: Query Generation
    query_resp = await query_agent.on_messages([hyp_resp.chat_message], cancellation_token)
    print(f"\nGenerated Query Condition: {query_resp.chat_message}")
    msg = query_resp.chat_message  # This is a TextMessage object

    # Step 1: Extract the JSON string
    json_str = msg.content

    # Step 2: Parse the string into a Python dict
    parsed_data = json.loads(json_str)

    # Step 3: Extract the results
    results = parsed_data.get("results", [])
    # Step 3: Insight Summary (send structured results to the Insight Summarizer)
    insight_resp = await summarizer_agent.on_messages([query_resp.chat_message], cancellation_token, data=results)
    print(f"\nInsights: {insight_resp.chat_message.content}")
    print(f"\ntype of Insights: {type(insight_resp.chat_message)}")

    # Step 4: Pass the structured 'results' to the reflection agent for analysis
    reflection_agent = OllamaAgent(
        name="ReflectionAgent", 
        model="gemma3:1b", 
        system_message="Reflect on the insights provided and suggest improvements or deeper observations based on the data provided",
        custom_context=json.dumps(results)  # Pass structured results (JSON) as custom context
    )

    # Now you can pass this structured data to the next agent (Reflection Agent)
    reflection_resp = await reflection_agent.on_messages(
        [UserMessage(content=f"Based on the given question {question} and the given context from the given data, give me some reflections. Also answer the question", source="user")], 
        cancellation_token
    )

    # Print the reflection
    print(f"\nReflection: {reflection_resp.chat_message.content}")

    # # Step 5: Narrative
    # narrative_resp = await narrative_agent.on_messages([UserMessage(content=question, source="user")], cancellation_token)
    # print(f"\nNarrative Summary:\n{narrative_resp.chat_message}")
    final_response = hyp_resp.chat_message.content + "\n\n" + reflection_resp.chat_message.content
    return final_response

if __name__ == "__main__":
    asyncio.run(run_pipeline())