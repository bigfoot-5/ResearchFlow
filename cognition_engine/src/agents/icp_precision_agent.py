"""
ICP Precision Agent - specialized agent for analyzing and refining Ideal Customer Profiles.
"""

import json
import logging
import os
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta

from src.config import settings
from src.database.vector_store_manager import VectorStoreManager
from src.analysis.reflective_icp_engine import ReflectiveICPEngine
from src.agents.base_agent import CognitionAgent


class ICPPrecisionAgent(CognitionAgent):
    """
    A specialized agent for analyzing CRM data to identify and refine 
    Ideal Customer Profile based on successful deals.
    """
    
    def __init__(
        self,
        name: str = "ICP Precision Agent",
        description: str = "Analyzes CRM data to identify and refine ideal customer profiles",
        system_prompt: str = None,
        vector_store: Optional[VectorStoreManager] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ICP Precision Agent.
        
        Args:
            name: Name of the agent (default provided)
            description: Description of the agent (default provided)
            system_prompt: System prompt (default will be generated if None)
            vector_store: Optional vector store for agent data
            agent_id: Optional agent ID
            metadata: Optional additional metadata
        """
        # Generate default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert at analyzing CRM data to identify ideal customer profiles. "
                "Your goal is to identify patterns and characteristics of customers that lead to "
                "successful deals. You should focus on identifying customer segments with high "
                "conversion rates, short sales cycles, and high lifetime value. "
                "When analyzing data, be precise, data-driven, and objective. "
                "Your insights should be actionable and lead to clear targeting strategies. "
                "When asked about specific industries, market segments, or metrics, provide detailed analysis "
                "with supporting evidence from the data. Explain the underlying factors and reasons "
                "behind performance patterns. Make connections between different data points to "
                "derive deeper insights, and suggest next steps for targeting or further analysis. "
                "Consider multiple factors including market dynamics, decision-making processes, "
                "value proposition fit, budget allocation patterns, and technical alignment."
            )
        
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            vector_store=vector_store,
            agent_id=agent_id,
            metadata=metadata
        )
        
        # Additional attributes specific to the ICP Precision Agent
        self.last_analysis_date = None
        self.current_icp = None
        self.reflective_engine = ReflectiveICPEngine()
    
    def analyze_closed_deals(self, timeframe: str = "last_90_days", min_deal_count: int = 10) -> Dict[str, Any]:
        """
        Analyze closed deals within the specified timeframe to identify patterns.
        
        Args:
            timeframe: Time period to analyze ("last_30_days", "last_90_days", "last_year", etc.)
            min_deal_count: Minimum number of deals required for meaningful analysis
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert timeframe to actual date range
        end_date = datetime.now()
        
        if timeframe == "last_30_days":
            start_date = end_date - timedelta(days=30)
        elif timeframe == "last_90_days":
            start_date = end_date - timedelta(days=90)
        elif timeframe == "last_year":
            start_date = end_date - timedelta(days=365)
        else:
            # Default to 90 days
            start_date = end_date - timedelta(days=90)
        
        # In a real implementation, this would query the HubSpot API or database
        # For now, we'll simulate with a function that would be replaced with actual implementation
        deals = self._get_closed_deals(start_date, end_date)
        
        # Check if we have enough data for analysis
        if len(deals) < min_deal_count:
            return {
                "status": "insufficient_data",
                "message": f"Not enough closed deals for analysis. Found {len(deals)}, need at least {min_deal_count}.",
                "deals_found": len(deals)
            }
        
        # Analyze the deals to extract patterns
        analysis_result = self._analyze_deal_patterns(deals)
        
        # Update the last analysis date
        self.last_analysis_date = datetime.now()
        
        return analysis_result
    
    def generate_icp_definition(self, include_raw_data: bool = False) -> Dict[str, Any]:
        """
        Generate an Ideal Customer Profile definition based on the latest analysis.
        
        Args:
            include_raw_data: Whether to include the raw data used for analysis
            
        Returns:
            Dictionary containing the ICP definition
        """
        # If we haven't run an analysis yet, do so
        if self.last_analysis_date is None:
            self.analyze_closed_deals()
        
        # Generate the ICP definition based on the analysis
        icp_definition = {
            "name": f"ICP Definition as of {datetime.now().strftime('%Y-%m-%d')}",
            "description": "Ideal Customer Profile based on analysis of successful deals",
            "criteria": self._generate_icp_criteria(),
            "created_at": datetime.now().isoformat(),
            "generated_by": self.name,
            "confidence_score": 0.85,  # This would be dynamically calculated in a real implementation
        }
        
        # Store the current ICP
        self.current_icp = icp_definition
        
        # Include raw data if requested
        if include_raw_data:
            icp_definition["raw_data"] = {
                "deals_analyzed": 25,  # This would be the actual count in a real implementation
                "date_range": {
                    "start": (datetime.now() - timedelta(days=90)).isoformat(),
                    "end": datetime.now().isoformat()
                }
            }
        
        return icp_definition
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent with the given query and context.
        
        Args:
            query: User query or task for the agent
            context: Additional context information
            
        Returns:
            A dictionary containing the agent's response and any additional data
        """
        context = context or {}
        
        # Basic context defaults if not provided
        if "timeframe" not in context:
            context["timeframe"] = "last_90_days"
        
        # Check if this is a triangulation data analysis query
        if context.get("data_type") in ["triangulation", "icp_triangulation", "triangulation_matrix"]:
            # This is a follow-up question about triangulation data sent from agent_endpoints.py
            triangulation_data = context.get("triangulation_data", {})
            user_question = query.replace("Analyze this triangulation data and answer: ", "")
            
            # Extract dimensions data for analysis
            dimensions = triangulation_data.get("dimensions", [])
            metadata = triangulation_data.get("metadata", {})
            
            # Use the reflective engine to generate insights based on the question and data
            analysis_params = {
                "question": user_question,
                "dimensions": dimensions,
                "metadata": metadata,
                "context": context
            }
            
            # Generate insights using the reflective engine instead of hardcoded responses
            insights = self._generate_dynamic_insights(analysis_params)
            
            # Format the response based on the generated insights
            response = self._format_insights_response(user_question, insights, triangulation_data)
            result = triangulation_data
            
        # Parse regular queries to determine what action to take    
        elif "analyze" in query.lower() and ("deals" in query.lower() or "closed deals" in query.lower()):
            # Analyze deals
            result = self.analyze_closed_deals(timeframe=context.get("timeframe"))
            response = f"I analyzed {result.get('deals_analyzed', 0)} closed deals from the {context.get('timeframe')}."
            
            if result.get("status") == "insufficient_data":
                response = result.get("message")
            else:
                response += " " + self._format_analysis_insights(result)
            
        elif "generate" in query.lower() and "icp" in query.lower():
            # Generate ICP definition
            result = self.generate_icp_definition(include_raw_data="raw" in query.lower() or "data" in query.lower())
            response = f"I've generated an updated ICP definition based on our analysis:\n\n"
            response += self._format_icp_definition(result)
            
        elif "triangulation" in query.lower() or "triangulate" in query.lower() or "matrix" in query.lower():
            # Generate ICP triangulation matrix
            result = self.generate_icp_triangulation()
            response = (
                f"I've generated an ICP Triangulation Matrix based on {result['metadata']['deals_analyzed']} deals. "
                f"This analysis reveals your most promising customer segments across different dimensions:\n\n"
                f"- Industry: Fintech shows the best performance with {result['dimensions'][0]['highest_win_rate']['metric']} and {result['dimensions'][0]['fastest_sales_cycle']['metric']}\n"
                f"- Company Size: Companies with 100-500 employees have {result['dimensions'][1]['highest_win_rate']['metric']} and {result['dimensions'][1]['fastest_sales_cycle']['metric']}\n"
                f"- Geography: EMEA region shows {result['dimensions'][2]['highest_win_rate']['metric']} with {result['dimensions'][2]['fastest_sales_cycle']['metric']}\n\n"
                f"The triangulation data is available for visualization."
            )
            
        elif "compare" in query.lower() and "icp" in query.lower():
            # Compare current ICP with previous
            response = "ICP comparison feature is not yet implemented."
            result = {"status": "not_implemented", "feature": "icp_comparison"}
            
        else:
            # Default - general information about the agent
            response = (
                f"I'm the {self.name}, specialized in analyzing CRM data to identify and refine "
                f"your Ideal Customer Profile. You can ask me to:\n"
                f"- Analyze closed deals (e.g., 'Analyze our closed deals from the last 90 days')\n"
                f"- Generate an ICP definition (e.g., 'Generate our latest ICP definition')\n"
                f"- Create an ICP triangulation matrix (e.g., 'Generate ICP triangulation matrix')\n"
                f"- Compare current ICP with previous versions (coming soon)\n\n"
                f"What would you like me to do?"
            )
            result = {"status": "info"}
        
        # Construct the full response
        full_result = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "query": query,
            "response": response,
            "context": context,
            "reflection_enabled": self.reflection_enabled,
            "result": result,
            "status": "success"
        }
        
        # Track execution
        self.execution_history.append({
            "query": query,
            "context": context,
            "result": full_result,
            "timestamp": datetime.now().isoformat()
        })
        
        return full_result
    
    def _generate_dynamic_insights(self, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dynamic insights using the reflective engine based on the question and data.
        
        Args:
            analysis_params: Dictionary containing analysis parameters
            
        Returns:
            Dictionary containing generated insights
        """
        question = analysis_params.get("question", "")
        dimensions = analysis_params.get("dimensions", [])
        metadata = analysis_params.get("metadata", {})
        
        # Prepare the data for analysis
        analysis_data = {
            "question": question,
            "dimensions_data": dimensions,
            "deals_analyzed": metadata.get("deals_analyzed", 0),
            "date_range": metadata.get("date_range", {})
        }
        
        # Categorize the question type to guide the analysis
        question_categories = self._categorize_question(question)
        
        # Use the reflective engine to generate insights
        if hasattr(self, 'reflective_engine') and self.reflective_engine:
            # If we have a reflective engine, use it for dynamic analysis
            insights = self.reflective_engine.analyze(
                data=analysis_data,
                question=question,
                categories=question_categories
            )
        else:
            # Fallback to rule-based analysis if reflective engine isn't available
            insights = self._rule_based_analysis(analysis_data, question_categories)
        
        return insights
    
    def _categorize_question(self, question: str) -> List[str]:
        """
        Categorize the question to guide the analysis.
        
        Args:
            question: The user's question
            
        Returns:
            List of identified question categories
        """
        categories = []
        
        # Identify the question type based on keywords
        if any(keyword in question.lower() for keyword in ["target", "industry", "industries", "segment", "vertical"]):
            categories.append("targeting")
        
        if any(keyword in question.lower() for keyword in ["why", "reason", "explain", "performance", "outperform"]):
            categories.append("performance_analysis")
            
        if any(keyword in question.lower() for keyword in ["compare", "versus", "vs", "difference", "better"]):
            categories.append("comparison")
            
        if any(keyword in question.lower() for keyword in ["recommend", "suggest", "advice", "strategy", "next steps"]):
            categories.append("recommendation")
            
        if any(keyword in question.lower() for keyword in ["forecast", "predict", "future", "trend", "growth"]):
            categories.append("forecasting")
            
        # If no specific category is identified, use general analysis
        if not categories:
            categories.append("general_analysis")
            
        return categories
    
    def _rule_based_analysis(self, analysis_data: Dict[str, Any], categories: List[str]) -> Dict[str, Any]:
        """
        Perform rule-based analysis when the reflective engine isn't available.
        
        Args:
            analysis_data: The data to analyze
            categories: List of question categories
            
        Returns:
            Dictionary containing analysis results
        """
        dimensions_data = analysis_data.get("dimensions_data", [])
        question = analysis_data.get("question", "")
        
        # Extract key information from dimensions
        industry_dimension = next((dim for dim in dimensions_data if dim["attribute"] == "Industry"), None)
        company_size_dimension = next((dim for dim in dimensions_data if dim["attribute"] == "Company Size"), None)
        geography_dimension = next((dim for dim in dimensions_data if dim["attribute"] == "Geography"), None)
        
        # Initialize insights structure
        insights = {
            "main_insight": "",
            "supporting_points": [],
            "recommendations": [],
            "trends": [],
            "data_summary": {}
        }
        
        # Generate insights based on the question categories
        if "targeting" in categories:
            # Questions about targeting industries
            if industry_dimension:
                top_industry = industry_dimension["highest_win_rate"]["value"]
                win_rate = industry_dimension["highest_win_rate"]["metric"]
                sales_cycle = industry_dimension["fastest_sales_cycle"]["metric"]
                
                insights["main_insight"] = f"Based on our analysis, {top_industry} represents your strongest industry segment with {win_rate} and {sales_cycle}."
                
                # Add supporting points
                insights["supporting_points"] = [
                    f"{top_industry} shows significantly higher conversion rates compared to other industries.",
                    "This industry segment appears to have both higher win rates and faster sales cycles, indicating good product-market fit.",
                    f"The data suggests that your value proposition resonates particularly well with {top_industry} customers."
                ]
                
                # Add recommendations
                insights["recommendations"] = [
                    f"Consider developing specialized messaging and case studies for the {top_industry} vertical.",
                    f"Allocate more marketing and sales resources to target the {top_industry} segment.",
                    "Develop a specific ICP for this industry to further optimize targeting."
                ]
                
                # Add potential other industries to target
                if "other" in question.lower() or "additional" in question.lower() or "more" in question.lower():
                    # This would typically come from deeper analysis, using placeholder for now
                    insights["supporting_points"].append(
                        "Other promising industries could include Healthcare and Manufacturing, though with lower performance metrics than Fintech."
                    )
                    insights["recommendations"].append(
                        "Consider exploring adjacent industries with similar characteristics to your top performer."
                    )
            else:
                insights["main_insight"] = "We don't have sufficient industry-specific data to make targeting recommendations."
        
        elif "performance_analysis" in categories:
            # Questions about why certain segments perform better
            if industry_dimension and industry_dimension["highest_win_rate"]["value"] == "Fintech":
                insights["main_insight"] = "Fintech is outperforming other industries due to several key factors in market dynamics, decision processes, and value alignment."
                
                insights["supporting_points"] = [
                    "Market Dynamics: The fintech sector is experiencing rapid digital transformation, creating urgent needs for solutions.",
                    "Decision-Making Process: Fintech companies typically have more streamlined procurement processes with fewer decision-makers involved.",
                    "Value Proposition Fit: Your product likely addresses critical pain points specific to the fintech space.",
                    "Budget Allocation: Fintech companies typically allocate larger technology budgets as a percentage of revenue.",
                    "Technical Alignment: The technical sophistication of fintech buyers means less education is needed during the sales process."
                ]
            elif company_size_dimension and "Company Size" in question:
                insights["main_insight"] = f"Companies with {company_size_dimension['highest_win_rate']['value']} employees perform better due to organizational structure and budget alignment factors."
                
                insights["supporting_points"] = [
                    "Mid-sized companies often have the right balance of budget flexibility and established processes.",
                    "These organizations typically have clear decision-making structures without excessive bureaucracy.",
                    "They often have sufficient pain points to justify the investment while still being agile enough to implement quickly."
                ]
            elif geography_dimension and "Geography" in question:
                insights["main_insight"] = f"The {geography_dimension['highest_win_rate']['value']} region shows stronger performance due to market maturity and regional business practices."
                
                insights["supporting_points"] = [
                    f"{geography_dimension['highest_win_rate']['value']} businesses may have higher digital transformation priorities.",
                    "Regional compliance requirements might make your solution particularly valuable in this market.",
                    "Cultural and business practice differences can impact sales cycles and decision processes."
                ]
        elif "general_analysis" in categories:
            # Default general analysis
            top_segments = []
            
            if industry_dimension:
                top_segments.append(f"Industry: {industry_dimension['highest_win_rate']['value']} ({industry_dimension['highest_win_rate']['metric']})")
                
            if company_size_dimension:
                top_segments.append(f"Company Size: {company_size_dimension['highest_win_rate']['value']} ({company_size_dimension['highest_win_rate']['metric']})")
                
            if geography_dimension:
                top_segments.append(f"Geography: {geography_dimension['highest_win_rate']['value']} ({geography_dimension['highest_win_rate']['metric']})")
                
            top_segments_text = ", ".join(top_segments)
            
            insights["main_insight"] = f"Based on our analysis, your top performing segments are: {top_segments_text}."
            insights["supporting_points"] = [
                "These segments show significantly higher conversion rates and shorter sales cycles.",
                "Your value proposition appears to resonate strongly with these customer types.",
                "Consider focusing your go-to-market strategy on these key segments."
            ]
        
        return insights
    
    def _format_insights_response(self, question: str, insights: Dict[str, Any], data: Dict[str, Any]) -> str:
        """
        Format the insights into a coherent response.
        
        Args:
            question: The original user question
            insights: The generated insights
            data: The original triangulation data
            
        Returns:
            Formatted response string
        """
        deals_analyzed = data.get("metadata", {}).get("deals_analyzed", "multiple")
        
        response = f"Based on the analysis of {deals_analyzed} deals, I can provide insights on your question: \"{question}\"\n\n"
        
        # Add the main insight
        response += f"{insights.get('main_insight', '')}\n\n"
        
        # Add supporting points
        supporting_points = insights.get("supporting_points", [])
        if supporting_points:
            for i, point in enumerate(supporting_points, 1):
                response += f"{i}. {point}\n"
            
            response += "\n"
            
        # Add recommendations if available
        recommendations = insights.get("recommendations", [])
        if recommendations:
            response += "Recommendations:\n"
            for i, recommendation in enumerate(recommendations, 1):
                response += f"• {recommendation}\n"
                
        return response
    
    # Helper methods (would connect to actual data sources in real implementation)
    
    def _get_closed_deals(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get closed deals within the specified date range.
        
        In a real implementation, this would connect to HubSpot or another CRM.
        For now, return simulated data.
        """
        # Simulated data - this would be replaced with actual API calls
        return [
            {
                "deal_id": "deal_1",
                "company_name": "TechCorp Inc.",
                "deal_size": 75000,
                "close_date": "2024-05-15",
                "industry": "Technology",
                "employee_count": 250,
                "deal_stage": "closed_won",
                "sales_cycle_days": 45
            },
            {
                "deal_id": "deal_2",
                "company_name": "HealthPlus",
                "deal_size": 120000,
                "close_date": "2024-06-01",
                "industry": "Healthcare",
                "employee_count": 500,
                "deal_stage": "closed_won",
                "sales_cycle_days": 60
            },
            # More simulated deals would be here
        ]
    
    def _analyze_deal_patterns(self, deals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze deals to extract patterns and insights.
        
        In a real implementation, this would perform sophisticated analysis.
        """
        # Simulated analysis - this would be replaced with actual analysis
        return {
            "deals_analyzed": len(deals),
            "average_deal_size": 98000,
            "median_sales_cycle": 52,
            "top_industries": ["Technology", "Healthcare", "Financial Services"],
            "employee_count_distribution": {
                "0-50": 0.05,
                "51-200": 0.25,
                "201-500": 0.45,
                "501+": 0.25
            },
            "insights": [
                "Technology companies with 201-500 employees show highest conversion rates",
                "Healthcare deals have 30% higher average deal sizes",
                "Companies with dedicated IT teams close 40% faster"
            ]
        }
    
    def _generate_icp_criteria(self) -> List[Dict[str, Any]]:
        """
        Generate ICP criteria based on the latest analysis.
        
        In a real implementation, this would use the actual analysis results.
        """
        # Simulated criteria - this would be based on actual analysis in real implementation
        return [
            {
                "field": "industry",
                "values": ["Technology", "Healthcare"],
                "weight": 0.30,
                "description": "Technology and Healthcare industries show the highest success rates"
            },
            {
                "field": "employee_count",
                "values": ["201-500"],
                "weight": 0.25,
                "description": "Mid-size companies with 201-500 employees convert at highest rates"
            },
            {
                "field": "annual_revenue",
                "values": ["$10M-$50M"],
                "weight": 0.20,
                "description": "Companies with annual revenue between $10M-$50M show best ROI"
            },
            {
                "field": "buying_committee",
                "values": ["IT Involved"],
                "weight": 0.15,
                "description": "Deals with IT department involvement close 40% faster"
            },
            {
                "field": "pain_point",
                "values": ["Efficiency", "Integration"],
                "weight": 0.10,
                "description": "Customers seeking efficiency improvements and integration solutions show highest LTV"
            }
        ]
    
    def _format_analysis_insights(self, analysis: Dict[str, Any]) -> str:
        """Format analysis insights into a readable string."""
        insights = analysis.get("insights", [])
        if not insights:
            return "No significant insights were found in the data."
            
        insight_text = "Key insights from the analysis:\n"
        for i, insight in enumerate(insights, 1):
            insight_text += f"{i}. {insight}\n"
            
        return insight_text
    
    def _format_icp_definition(self, icp_definition: Dict[str, Any]) -> str:
        """Format ICP definition as a readable string."""
        formatted = f"ICP Name: {icp_definition.get('name')}\n\n"
        formatted += f"Description: {icp_definition.get('description')}\n\n"
        formatted += "Key Criteria:\n"
        
        for criterion in icp_definition.get('criteria', []):
            formatted += f"- {criterion.get('field')}: {', '.join(criterion.get('values', []))}\n"
            formatted += f"  Weight: {criterion.get('weight')}\n"
            formatted += f"  Description: {criterion.get('description')}\n\n"
        
        return formatted
        
    def generate_icp_triangulation(self) -> Dict[str, Any]:
        """
        Generate ICP triangulation matrix showing key attributes vs. metrics like sales cycle and win rate.
        
        Returns:
            Dictionary containing triangulation data structured for visualization
        """
        # In a real implementation, this would analyze actual CRM data
        # For now, we'll return simulated data that matches the expected format
        
        triangulation_data = {
            "title": "ICP Triangulation Matrix",
            "description": "Mapping company attributes against GTM signals reveals your most promising customer segments",
            "dimensions": [
                {
                    "attribute": "Industry",
                    "fastest_sales_cycle": {
                        "value": "Fintech",
                        "metric": "-22% faster close",
                        "confidence": 0.83,
                        "raw_value": 22
                    },
                    "highest_win_rate": {
                        "value": "Fintech",
                        "metric": "31% win rate",
                        "confidence": 0.87,
                        "raw_value": 31
                    }
                },
                {
                    "attribute": "Company Size",
                    "fastest_sales_cycle": {
                        "value": "100-500",
                        "metric": "37 days avg",
                        "confidence": 0.76,
                        "raw_value": 37
                    },
                    "highest_win_rate": {
                        "value": "100-500",
                        "metric": "28% win rate",
                        "confidence": 0.82,
                        "raw_value": 28
                    }
                },
                {
                    "attribute": "Geography",
                    "fastest_sales_cycle": {
                        "value": "EMEA",
                        "metric": "41 days avg",
                        "confidence": 0.72,
                        "raw_value": 41
                    },
                    "highest_win_rate": {
                        "value": "EMEA",
                        "metric": "31% win rate", 
                        "confidence": 0.78,
                        "raw_value": 31
                    }
                }
            ],
            "metadata": {
                "deals_analyzed": 250,
                "date_range": {
                    "start": (datetime.now() - timedelta(days=365)).isoformat(),
                    "end": datetime.now().isoformat()
                },
                "generated_at": datetime.now().isoformat()
            }
        }
        
        return triangulation_data
        
    def _calculate_triangulation_metrics(self, deals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for triangulation based on deals data.
        This would be implemented with actual analysis logic in a production system.
        
        Args:
            deals: List of deal data for analysis
            
        Returns:
            Dictionary with calculated metrics by dimension
        """
        # This would contain the actual logic for calculating metrics like:
        # - Average sales cycle by industry, company size, geography
        # - Win rates by industry, company size, geography
        # - Confidence scores based on sample sizes
        
        # For MVP, this is a placeholder that would be replaced with actual analysis
        return {
            "industry": {
                "Fintech": {"win_rate": 0.31, "sales_cycle_days": 37, "sample_size": 45},
                "Healthcare": {"win_rate": 0.25, "sales_cycle_days": 52, "sample_size": 38},
                "Manufacturing": {"win_rate": 0.22, "sales_cycle_days": 61, "sample_size": 29}
            },
            "company_size": {
                "1-99": {"win_rate": 0.18, "sales_cycle_days": 42, "sample_size": 31},
                "100-500": {"win_rate": 0.28, "sales_cycle_days": 37, "sample_size": 62},
                "501+": {"win_rate": 0.24, "sales_cycle_days": 48, "sample_size": 44}
            },
            "geography": {
                "North America": {"win_rate": 0.26, "sales_cycle_days": 45, "sample_size": 78},
                "EMEA": {"win_rate": 0.31, "sales_cycle_days": 41, "sample_size": 53},
                "APAC": {"win_rate": 0.22, "sales_cycle_days": 49, "sample_size": 32}
            }
        } 