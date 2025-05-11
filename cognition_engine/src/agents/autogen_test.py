import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
import os
from typing import AsyncGenerator, Sequence
import json

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

    def _parse_filter_condition(self, filter_condition: str) -> dict:
        """Parse the filter condition into structured components."""
        filters = {}
        
        # Split by AND if multiple conditions
        conditions = filter_condition.split(" AND ")
        
        for condition in conditions:
            if ">" in condition:
                field, value = condition.split(">")
                filters[field.strip()] = {"operator": ">", "value": value.strip()}
            elif "<" in condition:
                field, value = condition.split("<")
                filters[field.strip()] = {"operator": "<", "value": value.strip()}
            elif "=" in condition:
                field, value = condition.split("=")
                filters[field.strip()] = {"operator": "=", "value": value.strip().strip("'")}
            elif "BETWEEN" in condition:
                # Handle BETWEEN conditions
                field = condition.split("BETWEEN")[0].strip()
                values = condition.split("BETWEEN")[1].strip()
                start, end = values.split("AND")
                filters[field] = {
                    "operator": "BETWEEN",
                    "value": {
                        "start": start.strip(),
                        "end": end.strip()
                    }
                }
        
        return filters

    def _calculate_win_rate(self, filter_condition: str) -> dict:
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Simplified query to only use deals table
            query_total = f"""
                SELECT COUNT(*) 
                FROM deals
                WHERE {filter_condition};
            """
            
            query_won = f"""
                SELECT COUNT(*) 
                FROM deals
                WHERE {filter_condition} AND dealstage = 'closedwon';
            """
            
            cursor.execute(query_total)
            total = cursor.fetchone()["count"]

            cursor.execute(query_won)
            won = cursor.fetchone()["count"]

            cursor.close()
            conn.close()

            if total == 0:
                return {
                    "status": "no_data",
                    "message": "No deals match the filter criteria.",
                    "filters": self._parse_filter_condition(filter_condition),
                    "metrics": {
                        "win_rate": 0,
                        "won_deals": 0,
                        "total_deals": 0
                    }
                }

            win_rate = (won / total) * 100
            
            return {
                "status": "success",
                "filters": self._parse_filter_condition(filter_condition),
                "metrics": {
                    "win_rate": round(win_rate, 2),
                    "won_deals": won,
                    "total_deals": total
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "filters": self._parse_filter_condition(filter_condition),
                "metrics": {
                    "win_rate": 0,
                    "won_deals": 0,
                    "total_deals": 0
                }
            }

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        latest_message = messages[-1]
        filter_condition = latest_message.content.strip()
        result = self._calculate_win_rate(filter_condition)

        # Convert the result dictionary to a JSON string
        result_json = json.dumps(result)

        yield Response(
            chat_message=TextMessage(content=result_json, source=self.name),
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
        filter_msg = TextMessage(content="amount > 50000", source="user")
        async for output in db_agent.on_messages_stream([filter_msg], CancellationToken()):
            if isinstance(output, Response):
                # Parse the JSON response
                result = json.loads(output.chat_message.content)
                print("DB Agent Response:", json.dumps(result, indent=2))

    # await test_ollama_agent()
    await test_database_agent()

if __name__ == "__main__":
    asyncio.run(main())