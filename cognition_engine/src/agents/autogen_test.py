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

class EnhancedFilterSuggestionAgent(OllamaAgent):
    def __init__(self, name: str = "FilterSuggester"):
        system_message = """You are an AI agent specialized in suggesting SQL filters to optimize deal win rates.
        Analyze patterns across multiple dimensions:
        - Industry segments
        - Company sizes
        - Deal values
        - Geographic regions
        - Sales cycles
        
        For each suggestion:
        1. Consider historical performance
        2. Look for correlations between dimensions
        3. Suggest filters that can be combined with AND operators
        4. Focus on actionable insights
        """
        super().__init__(name=name, model="gemma3:1b", system_message=system_message)

class OptimizedFilterSuggestionAgent(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()
        
        # Define the system prompt as a parameter that can be optimized
        system_prompt = adal.Parameter(
            data="""You are an AI agent specialized in suggesting SQL filters to optimize deal win rates.
            
            Your task is to analyze patterns and suggest SQL filters that will help identify deals with high win probability.
            
            Consider these aspects when suggesting filters:
            1. Industry segments and their historical performance
            2. Deal sizes and their correlation with success
            3. Geographic patterns and regional variations
            4. Sales cycle length and its impact
            5. Company size and its relationship to deal success
            
            Format your suggestions as SQL WHERE clauses that can be combined with AND operators.
            Example: "amount > 50000 AND industry = 'Technology' AND deal_stage = 'Proposal'"
            """,
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            role_desc="System prompt for the filter suggestion agent"
        )
        
        # Define dimension information to help the model understand available filters
        dimension_info = [
            {
                "name": "industry",
                "description": "The industry sector of the prospect company",
                "values": ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]
            },
            {
                "name": "deal_size",
                "description": "The monetary value of the deal",
                "values": ["amount > X where X is a numerical value"]
            },
            {
                "name": "company_size",
                "description": "Size of the prospect company by employee count",
                "values": ["1-50", "51-200", "201-1000", "1000+"]
            },
            {
                "name": "geography",
                "description": "Geographic region of the prospect",
                "values": ["North America", "Europe", "Asia Pacific", "Latin America"]
            }
        ]
        
        # Create the generator with our template
        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "dimension_info": dimension_info,
                # historical_filters can be updated dynamically during runtime
                "historical_filters": []  
            }
        )
    
    def update_historical_filters(self, filters: List[Dict]):
        """Update the historical filters based on recent performance"""
        self.generator.prompt_kwargs["historical_filters"] = filters
    
    def __call__(self, input_str: str) -> str:
        """Generate filter suggestions based on input"""
        return self.generator(prompt_kwargs={"input_str": input_str})

class MemoryEnabledOrchestratorAgent(BaseChatAgent):
    def __init__(self, name: str = "Orchestrator"):
        super().__init__(name=name, description="Agent that orchestrates workflow with memory")
        self.filter_memory = {}  # Store successful filter combinations
        
    def _update_memory(self, filter_condition: str, win_rate: float):
        """Update memory with successful filters"""
        if win_rate > 0.3:  # Store filters that led to >30% win rate
            self.filter_memory[filter_condition] = win_rate
            
    def _get_similar_successful_filters(self, current_filter: str) -> List[str]:
        """Get historically successful similar filters"""
        similar_filters = []
        for stored_filter, win_rate in self.filter_memory.items():
            if self._are_filters_similar(current_filter, stored_filter):
                similar_filters.append((stored_filter, win_rate))
        return sorted(similar_filters, key=lambda x: x[1], reverse=True)

class EnhancedOrchestratorAgent(BaseChatAgent):
    def __init__(self, name: str = "Orchestrator"):
        super().__init__(name=name, description="Agent that orchestrates enhanced workflow")
        self.filter_suggester = EnhancedFilterSuggestionAgent()
        self.analyzer = WinRateAnalysisAgent()
        self.memory = MemoryEnabledOrchestratorAgent()
        
    async def on_messages_stream(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        user_msg = messages[-1].content
        
        # 1. Get filter suggestions
        yield TextMessage(content="Getting filter suggestions...", source=self.name)
        suggestion_msg = TextMessage(content=f"What filters might improve win rate for deals? Context: {user_msg}", source="user")
        async for resp in self.filter_suggester.on_messages_stream([suggestion_msg], cancellation_token):
            if isinstance(resp, Response):
                suggested_filters = resp.chat_message.content
                
                # 2. Check memory for similar successful filters
                similar_filters = self.memory._get_similar_successful_filters(suggested_filters)
                if similar_filters:
                    yield TextMessage(content=f"Found similar successful filters: {similar_filters}", source=self.name)
                
                # 3. Evaluate win rates
                evaluator = DatabaseAgent(
                    name="Evaluator",
                    db_config={
                        "dbname": "cognition_db",
                        "user": "cognition_user",
                    }
                )
                
                filter_msg = TextMessage(content=suggested_filters, source="user")
                async for eval_resp in evaluator.on_messages_stream([filter_msg], cancellation_token):
                    if isinstance(eval_resp, Response):
                        results = json.loads(eval_resp.chat_message.content)
                        
                        # 4. Update memory with results
                        if results["status"] == "success":
                            self.memory._update_memory(suggested_filters, results["metrics"]["win_rate"])
                        
                        # 5. Analyze patterns
                        analysis = self.analyzer.analyze_win_rate_trends([results])
                        
                        # 6. Return comprehensive response
                        response = {
                            "suggested_filters": suggested_filters,
                            "similar_successful_filters": similar_filters,
                            "current_results": results,
                            "analysis": analysis
                        }
                        
                        yield Response(
                            chat_message=TextMessage(content=json.dumps(response), source=self.name),
                            inner_messages=[]
                        )

class WinRateAnalysisAgent(BaseChatAgent):
    def __init__(self, name: str = "Analyzer"):
        super().__init__(name=name, description="Agent that analyzes win rate patterns")
        
    def analyze_win_rate_trends(self, filter_results: List[Dict]) -> Dict:
        """Analyze trends in win rates across different filters"""
        trends = {
            'industry_performance': {},
            'deal_size_impact': {},
            'geographic_patterns': {},
            'recommendations': []
        }
        # Implement analysis logic here
        return trends

async def main():
    orchestrator = EnhancedOrchestratorAgent()
    message = TextMessage(content="amount > 50000", source="user")

    async for output in orchestrator.on_messages_stream([message], CancellationToken()):
        if isinstance(output, Response):
            print("Final Output:", output.chat_message.content)
        elif isinstance(output, TextMessage):
            print(f"{output.source}: {output.content}")

template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}

{% if output_format_str is not none %}
{{output_format_str}}
{% endif %}

{% if historical_filters %}
<HISTORICAL_PERFORMANCE>
Previous successful filters:
{% for filter in historical_filters %}
{{loop.index}}. Filter: {{filter.condition}}
   Win Rate: {{filter.win_rate}}%
   Total Deals: {{filter.total_deals}}
{% endfor %}
</HISTORICAL_PERFORMANCE>
{% endif %}

{% if dimension_info %}
<DIMENSION_INFO>
Available dimensions for filtering:
{% for dim in dimension_info %}
- {{dim.name}}: {{dim.description}}
  Possible values: {{dim.values|join(', ')}}
{% endfor %}
</DIMENSION_INFO>
{% endif %}
<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
{{input_str}}
<END_OF_USER>"""