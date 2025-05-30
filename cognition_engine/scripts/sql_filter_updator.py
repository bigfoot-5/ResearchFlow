import psycopg2
import psycopg2.extras
import os
import json
import asyncio
from typing import Dict, List, Any, Sequence, AsyncGenerator
from datetime import datetime, timedelta

# Import Autogen components
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage, UserMessage, BaseAgentEvent
from autogen_core import CancellationToken
# Assuming you have Ollama or OpenAI set up for the analysis agent
# from autogen_ext.models.ollama import OllamaChatCompletionClient
# from autogen_ext.models.openai import OpenAIChatCompletionClient

# --- Database Configuration ---
# Ensure these environment variables are set or replace with your actual config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cognition_user:cognition_password@localhost:5432/cognition_db")
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "cognition_db"),
    "user": os.getenv("DB_USER", "cognition_user"),
    "password": os.getenv("DB_PASSWORD", "cognition_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# --- Dimension Values and Threshold ---
HS_ANALYTICS_SOURCES = [
    'OFFLINE', 'EMAIL_MARKETING', 'DIRECT_TRAFFIC',
    'PAID_SEARCH', 'ORGANIC_SEARCH'
]

JOB_TITLES = [
    'Salesperson', 'Head of Talent', 'CEO', 'CHRO', 'HR Director',
    'Executive Chairperson', 'VP People', 'People Analytics Lead',
    'Head of Business Development Europe'
]

INDUSTRIES = [
    'PHARMACEUTICALS', 'COMPUTER_SOFTWARE', 'ACCOUNTING',
    'INFORMATION_TECHNOLOGY_AND_SERVICES', 'RETAIL', 'FOOD_BEVERAGES'
]

MIN_DEAL_COUNT = 5

# --- Helper function to build dynamic WHERE clauses ---
def build_filter_conditions(filters: Dict[str, List[str]]) -> str:
    """Builds SQL WHERE clause fragments for IN conditions."""
    conditions = []
    if 'hs_analytics_source' in filters and filters['hs_analytics_source']:
        sources = ", ".join(f"'{s}'" for s in filters['hs_analytics_source'])
        conditions.append(f"d.hs_analytics_source IN ({sources})")
    if 'job_title' in filters and filters['job_title']:
        titles = ", ".join(f"'{t}'" for t in filters['job_title'])
        # Assuming contacts table is joined and job_title is on contacts (ct)
        conditions.append(f"ct.job_title IN ({titles})")
    if 'industry' in filters and filters['industry']:
        industries = ", ".join(f"'{i}'" for i in filters['industry'])
        # Assuming companies table is joined and industry is on companies (c)
        conditions.append(f"c.industry IN ({industries})")

    return " AND ".join(conditions) if conditions else "TRUE" # Return TRUE if no filters

# --- Agent 1: Database Query Agent ---
class DatabaseQueryAgent(BaseChatAgent):
    def __init__(self, name: str, db_config: dict):
        super().__init__(name=name, description="Agent that queries the database to get deal metrics by segment")
        self.db_config = db_config

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        latest_message = messages[-1]
        # The message content is expected to be a JSON string defining the filters
        try:
            filter_definition = json.loads(latest_message.content)
            hs_source_filters = filter_definition.get("hs_analytics_source", HS_ANALYTICS_SOURCES)
            job_title_filters = filter_definition.get("job_title", JOB_TITLES)
            industry_filters = filter_definition.get("industry", INDUSTRIES)
            # Combine all filter values for grouping
            all_filters = {
                'hs_analytics_source': hs_source_filters,
                'job_title': job_title_filters,
                'industry': industry_filters
            }
            where_condition = build_filter_conditions(all_filters)

            conn = None
            cursor = None
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                # Construct the query to group by all three dimensions and calculate metrics
                query = f"""
                    SELECT
                        c.industry,
                        ct.job_title,
                        d.hs_analytics_source,
                        COUNT(d.id) AS total_deals,
                        COUNT(CASE WHEN d.dealstage = 'closedwon' THEN d.id END) AS won_deals,
                        AVG(d.amount) AS avg_amount,
                        AVG(d.days_to_close) AS avg_days_to_close
                    FROM dim_deals d
                    JOIN dim_companies c ON d.company_id = c.company_id
                    LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id -- Use LEFT JOIN as contact might not exist
                    WHERE {where_condition}
                    GROUP BY
                        c.industry,
                        ct.job_title,
                        d.hs_analytics_source
                    HAVING COUNT(d.id) >= {MIN_DEAL_COUNT}; -- Filter out segments below threshold

                """
                # print("Executing Query:", query) # Debug print
                cursor.execute(query)
                rows = cursor.fetchall()

                result_data = {"status": "success", "data": rows}

            except Exception as e:
                result_data = {"status": "error", "message": str(e)}
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            yield Response(
                chat_message=TextMessage(content=json.dumps(result_data), source=self.name),
                inner_messages=[],
            )

        except json.JSONDecodeError:
            yield Response(
                 chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Invalid JSON input"}), source=self.name),
                 inner_messages=[]
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass # No state to reset

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("DatabaseQueryAgent did not produce a response.")


# --- Agent 2: Revenue Velocity Analysis Agent (LLM-backed) ---
class RevenueVelocityAnalysisAgent(BaseChatAgent):
    def __init__(self, name: str, llm_model: str = "gemma3:1b"): # Use your preferred LLM
        super().__init__(name=name, description="Agent that analyzes database query results to identify high revenue velocity segments")

        # Define model info for the Ollama client
        ollama_model_info = {
            "family": "gemma",  # Assuming 'gemma3' belongs to the 'gemma' family
            "function_calling": False, # Set based on model capability, default False
            "json_output": False,      # Set based on model capability, default False
            "structured_output": False, # Set based on model capability, default False
            "vision": False,           # Set based on model capability, default False
            "multiple_system_messages": False, # Set based on model capability, default False
        }

        # Configure your LLM client here
        # Example with Ollama:
        from autogen_ext.models.ollama import OllamaChatCompletionClient
        # Passed model_info to the constructor
        self._llm_client = OllamaChatCompletionClient(model=llm_model, base_url="http://localhost:11434", model_info=ollama_model_info)
        # Example with OpenAI:
        # from autogen_ext.models.openai import OpenAIChatCompletionClient
        # self._llm_client = OpenAIChatCompletionClient(model=llm_model, api_key=os.getenv("OPENAI_API_KEY"))
        # Placeholder if no LLM client is configured:
        # self._llm_client = None # Commented out placeholder

        if self._llm_client is None:
            print("Warning: No LLM client configured for RevenueVelocityAnalysisAgent. Will only perform calculations, not natural language analysis.")

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    def calculate_metrics_and_velocity(self, raw_data: List[Dict]) -> List[Dict]:
        """Calculates win rate, ACV, sales cycle, and revenue velocity."""
        analyzed_segments = []
        for row in raw_data:
            total_deals = row.get('total_deals', 0)
            won_deals = row.get('won_deals', 0)
            avg_amount = row.get('avg_amount', 0)
            avg_days_to_close = row.get('avg_days_to_close', 0)

            win_rate = (won_deals / total_deals * 100) if total_deals > 0 else 0
            acv = avg_amount or 0
            sales_cycle = avg_days_to_close or 0

            # Calculate Revenue Velocity: (Win Rate / Sales Cycle) * ACV
            # Avoid division by zero for sales cycle
            revenue_velocity = (win_rate / sales_cycle * acv) if sales_cycle > 0 else 0 # Handle sales_cycle = 0

            analyzed_segments.append({
                "industry": row.get('industry'),
                "job_title": row.get('job_title'),
                "hs_analytics_source": row.get('hs_analytics_source'),
                "total_deals": total_deals,
                "won_deals": won_deals,
                "win_rate": round(win_rate, 2),
                "acv": round(acv, 2),
                "sales_cycle_days": round(sales_cycle, 2),
                "revenue_velocity": round(revenue_velocity, 2)
            })

        # Sort by Revenue Velocity (highest first)
        analyzed_segments.sort(key=lambda x: x.get("revenue_velocity", 0), reverse=True)

        return analyzed_segments


    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        latest_message = messages[-1]
        # The message content is expected to be a JSON string of raw DB results
        try:
            db_result = json.loads(latest_message.content)

            if db_result.get("status") != "success":
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "error", "message": f"Error from Database Agent: {db_result.get('message')}"}), source=self.name),
                     inner_messages=[]
                 )
                 return

            raw_data = db_result.get("data", [])
            analyzed_data = self.calculate_metrics_and_velocity(raw_data)

            if not analyzed_data:
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "success", "analysis": "No segments found matching criteria.", "data": []}), source=self.name),
                     inner_messages=[]
                 )
                 return


            # Use LLM for natural language analysis if client is available
            if self._llm_client:
                # Prepare data for LLM prompt
                data_summary = json.dumps(analyzed_data, indent=2)
                prompt = f"""
                Analyze the following sales segment performance data, sorted by Revenue Velocity.
                Identify the top 3-5 segments with the highest Revenue Velocity.
                Explain what Revenue Velocity means in this context.
                Provide insights on why these top segments might be performing well based on their metrics (Win Rate, Sales Cycle, ACV).
                Suggest potential actions based on these insights.

                Segment Data (JSON):
                {data_summary}
                """
                # print("LLM Prompt:", prompt) # Debug print

                try:
                    # Use the appropriate method for your LLM client (e.g., create or invoke)
                    # This is a placeholder and needs to be adapted to your specific client
                    # llm_response_obj = await self._llm_client.create([UserMessage(content=prompt)])
                    # llm_analysis_text = llm_response_obj.content

                    # Placeholder if LLM client is not fully integrated
                    llm_analysis_text = "LLM analysis is not configured. Showing raw data and calculations.\n\n" + data_summary

                except Exception as llm_error:
                     llm_analysis_text = f"Error during LLM analysis: {llm_error}\n\nShowing raw data and calculations.\n\n" + data_summary


                yield Response(
                    chat_message=TextMessage(content=json.dumps({"status": "success", "analysis": llm_analysis_text, "data": analyzed_data}), source=self.name),
                    inner_messages=[],
                )

            else:
                # If no LLM client, just return the processed data
                 yield Response(
                     chat_message=TextMessage(content=json.dumps({"status": "success", "analysis": "LLM analysis not available.", "data": analyzed_data}), source=self.name),
                     inner_messages=[]
                 )

        except json.JSONDecodeError:
            yield Response(
                 chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Invalid JSON input from previous agent"}), source=self.name),
                 inner_messages=[]
            )


    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass # No state to reset

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("RevenueVelocityAnalysisAgent did not produce a response.")


# --- Main orchestration function ---
async def predict_high_revenue_velocity_segments():
    print("Starting Revenue Velocity Prediction workflow...")

    # Instantiate agents
    db_agent = DatabaseQueryAgent(name="DBQueryAgent", db_config=DB_CONFIG)
    # Initialize with your preferred LLM model
    analysis_agent = RevenueVelocityAnalysisAgent(name="AnalysisAgent", llm_model="gemma3:1b")

    # Initial message to trigger the workflow - empty content or filter definition
    # Sending the default filter values to the DB agent as JSON
    filter_input = {
        "hs_analytics_source": HS_ANALYTICS_SOURCES,
        "job_title": JOB_TITLES,
        "industry": INDUSTRIES
    }
    initial_message = UserMessage(content=json.dumps(filter_input), source="user") # Send filter definition

    # --- Workflow Execution ---
    # Step 1: Send filter request to Database Agent
    print(f"Sending filter request to {db_agent.name}...")
    db_response = await db_agent.on_messages([initial_message], CancellationToken())

    if db_response and db_response.chat_message:
        print(f"Received response from {db_agent.name}: {db_response.chat_message.content[:200]}...") # Print snippet

        # Step 2: Send database results to Analysis Agent
        analysis_request_message = UserMessage(content=db_response.chat_message.content, source="db_agent")
        print(f"Sending database results to {analysis_agent.name}...")
        analysis_response = await analysis_agent.on_messages([analysis_request_message], CancellationToken())

        if analysis_response and analysis_response.chat_message:
            final_result_json = analysis_response.chat_message.content
            print(f"Received final response from {analysis_agent.name}:")
            # Parse and pretty print the final result
            try:
                final_result = json.loads(final_result_json)
                print(json.dumps(final_result, indent=2))
            except json.JSONDecodeError:
                print("Error decoding final JSON response.")
                print(final_result_json)
        else:
            print(f"No valid response received from {analysis_agent.name}.")
    else:
        print(f"No valid response received from {db_agent.name}.")

    print("Workflow finished.")


# --- Run the workflow ---
if __name__ == "__main__":
    asyncio.run(predict_high_revenue_velocity_segments())
