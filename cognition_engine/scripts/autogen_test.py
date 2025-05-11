import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os
from typing import AsyncGenerator, Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_core import CancellationToken
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, RequestUsage, UserMessage

class DatabaseAgent(BaseChatAgent):
    def __init__(self, name: str, db_config: dict):
        super().__init__(name=name, description="Agent that queries PostgreSQL database to evaluate win rates.")
        self.db_config = db_config

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    def _calculate_win_rate(self, filter_condition: str) -> str:
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Adjusting query to join deals with companies and contacts
            query_total = f"""
                SELECT COUNT(*) 
                FROM deals d
                JOIN companies c ON d.company_id = c.id
                JOIN contacts co ON d.company_id = co.company_id  -- Using company_id to join deals and contacts
                WHERE {filter_condition};
            """
            
            query_won = f"""
                SELECT COUNT(*) 
                FROM deals d
                JOIN companies c ON d.company_id = c.id
                JOIN contacts co ON d.company_id = co.company_id  -- Using company_id to join deals and contacts
                WHERE {filter_condition} AND d.dealstage = 'closedwon';
            """
            
            cursor.execute(query_total)
            total = cursor.fetchone()["count"]

            cursor.execute(query_won)
            won = cursor.fetchone()["count"]

            cursor.close()
            conn.close()

            if total == 0:
                return "No deals match the filter criteria."
            win_rate = (won / total) * 100
            return f"Win rate: {win_rate:.2f}% ({won}/{total} deals won)"
        except Exception as e:
            return f"Error calculating win rate: {str(e)}"

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        latest_message = messages[-1]
        filter_condition = latest_message.content.strip()
        result = self._calculate_win_rate(filter_condition)

        yield Response(
            chat_message=TextMessage(content=result, source=self.name),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("No response generated.")
    # Example model info for Ollama
model_info = {
    "family": "gemma",  # Or another supported family
    "function_calling": False,
    "json_output": False,
    "structured_output": False,
    "vision": False,
    "multiple_system_messages": False,
}

class OllamaAgent(BaseChatAgent):
    def __init__(self, name: str, model: str = "gemma3:1b", system_message: str = None):
        super().__init__(name=name, description="Test agent for Ollama")
        self._model_client = OllamaChatCompletionClient(model=model, model_info=model_info)
        self._model_context = UnboundedChatCompletionContext()
        self._system_message = system_message or "You are a helpful assistant."

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        for msg in messages:
            await self._model_context.add_message(msg.to_model_message())

        history = await self._model_context.get_messages()

        response = await self._model_client.create(history)

        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)  # Optional token usage

        await self._model_context.add_message(
            AssistantMessage(content=response.content, source=self.name)
        )

        yield Response(
            chat_message=TextMessage(content=response.content, source=self.name, models_usage=usage),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        results = []
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message  # return the first full response
        raise RuntimeError("No response was generated by the model.")

async def main():
    async def test_ollama_agent():
        agent = OllamaAgent(name="OllamaAgent")
        message = TextMessage(content="What is the capital of France?", source="user")

        async for output in agent.on_messages_stream([message], CancellationToken()):
            if isinstance(output, Response):
                print("Response:", output.chat_message.content)

    async def test_database_agent():
        db_agent = DatabaseAgent(
            name="PostgresAgent",
            db_config={
                "dbname": "cognition_db",
                "user": "cognition_user",
                # "password": "cognition_password",
                # "host": "localhost",
                # "port": 5432,
            },
        )
        filter_msg = TextMessage(content="amount > 50000 AND country = 'United Kingdom'", source="user")
        async for output in db_agent.on_messages_stream([filter_msg], CancellationToken()):
            if isinstance(output, Response):
                print("DB Agent Response:", output.chat_message.content)

    # await test_ollama_agent()
    await test_database_agent()

if __name__ == "__main__":
    asyncio.run(main())

# import asyncio
# from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
# from autogen_core.models import UserMessage
# from autogen_ext.models.ollama import OllamaChatCompletionClient
# from autogen_agentchat.messages import TextMessage


# model_info = {
#     "family": "gemma",  # or "unknown" if not supported
#     "function_calling": False,
#     "json_output": False,
#     "structured_output": False,
#     "vision": False,
#     "multiple_system_messages": False,
# }

# async def main():
#     ollama_model_client = OllamaChatCompletionClient(model="gemma3:1b", model_info=model_info)
#     response = await ollama_model_client.create([TextMessage(content="What is the capital of France?", source="user")])
#     print(response)
#     await ollama_model_client.close()

# if __name__ == "__main__":
#     asyncio.run(main())

# # --- Agent 1: Data Analyst (executes SQL queries) ---
# analyst = AssistantAgent(
#     name="analyst",
#     system_message="You are a data analyst. Generate and execute SQL queries to evaluate win rates for filtered subsets of deals.",
#     code_execution_config={"work_dir": "./"},
# )
# run_sql_query(analyst, db_path="hubspot_etl.db")

# # --- Agent 2: Strategist (asks for insights, evaluates patterns) ---
# strategist = AssistantAgent(
#     name="strategist",
#     system_message="You are a GTM strategist. Suggest possible filters that could increase deal win rate. Request SQL summaries from analyst and evaluate them.",
# )

# # --- User Proxy (goal initiator) ---
# user = UserProxyAgent(
#     name="user",
#     human_input_mode="NEVER",
#     system_message="You are a user requesting insights about why a specific deal won or lost. You can respond with questions.",
# )

# # --- Group Chat Setup ---
# chat = GroupChat(
#     agents=[user, analyst, strategist],
#     messages=[],
#     max_round=15
# )

# manager = GroupChatManager(
#     groupchat=chat,
#     system_message="The goal is to analyze historical HubSpot deal data and find filters that correlate with a high win rate."
# )

# # --- Start the agentic workflow ---
# user.initiate_chat(
#     manager,
#     message="Ask the analyst to compute win rate for deals where amount > 50000 and country = 'United Kingdom'."
# )
