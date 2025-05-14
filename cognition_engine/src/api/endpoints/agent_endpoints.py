"""
API endpoints for agent operations in the Cognition Engine.
"""

import os
import json
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime
import time
from src.agents.autogen_reasoning import run_pipeline
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Body, Query, Path, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import logging

from ...agents.base_agent import CognitionAgent
from ...agents.icp_precision_agent import ICPPrecisionAgent
from src.database.session import get_db
from src.database.models import AgentExecution
from src.agents.manager import AgentManager
from src.core_logic.reflective_icp_engine import ReflectiveICPEngine

# Create router
router = APIRouter(
    prefix="/agents",
    tags=["agents"],
    responses={404: {"description": "Not found"}},
)

# Set up logging
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class AgentCreate(BaseModel):
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of what the agent does")
    type: str = Field(..., description="Type of agent to create (e.g., 'icp_precision')")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt for the agent")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the agent")

class AgentResponse(BaseModel):
    agent_id: str
    name: str
    description: str
    type: str
    created_at: str
    reflection_enabled: bool
    metadata: Dict[str, Any]

class AgentExecuteRequest(BaseModel):
    query: str = Field(..., description="The query or task for the agent to execute")
    context: Optional[Dict[str, Any]] = Field({}, description="Additional context for the agent")
    reflection_enabled: Optional[bool] = Field(None, description="Enable/disable reflection for this execution")
    reflection_steps: Optional[int] = Field(None, description="Number of reflection steps if enabled")

class AgentExecuteResponse(BaseModel):
    agent_id: str
    agent_name: str
    query: str
    response: str
    reflection_applied: Optional[bool] = None
    execution_time: str
    metadata: Dict[str, Any]

class AnalyzeDataRequest(BaseModel):
    """Request model for analyzing specific data with follow-up questions."""
    query: str
    # data_type: str  # e.g., "triangulation", "icp_definition", etc.
    data: Dict[str, Any]  # The data to be analyzed
    additional_context: Optional[Dict[str, Any]] = {}

class AnalyzeDataResponse(BaseModel):
    """Response model for data analysis."""
    response: str
    metadata: Dict[str, Any]

# Create a shared agent manager instance
agent_manager = AgentManager()

@router.post("/", response_model=AgentResponse, status_code=201)
async def create_agent(agent_data: AgentCreate):
    """Create a new agent of the specified type."""
    try:
        # Create the agent using the agent manager
        agent = agent_manager.create_agent(
            name=agent_data.name,
            description=agent_data.description,
            agent_type=agent_data.type,
            system_prompt=agent_data.system_prompt,
            metadata=agent_data.metadata
        )
        
        # Create the response
        response = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "type": agent_data.type,
            "created_at": datetime.now().isoformat(),
            "reflection_enabled": agent.reflection_enabled,
            "metadata": agent.metadata
        }
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agent: {str(e)}")

@router.post("/analyze", response_model=AnalyzeDataResponse)

async def analyze_data(
    request: AnalyzeDataRequest,
    db: Session = Depends(get_db)
):
    print("Entered pipeline function")
    try:
        result = await run_pipeline(question=request.query, data = request.data)
        return {
            "response": result,
            "metadata": {} 
        }
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing data: {str(e)}")
#     """
#     Analyze specific data (like triangulation matrix) with a follow-up question.
    
#     This endpoint handles follow-up questions about specific data structures,
#     providing more contextual responses based on the provided data.
#     """
#     start_time = time.time()
    
#     # Select the appropriate agent based on data_type
#     if request.data_type in ["triangulation", "icp_triangulation", "triangulation_matrix"]:
#         # Create a specialized system prompt for triangulation data analysis
#         triangulation_system_prompt = (
#             "You are an expert CRM data analyst specializing in Ideal Customer Profile (ICP) triangulation analysis. "
#             "When analyzing triangulation data, you should provide deep, thoughtful insights about WHY certain "
#             "segments outperform others by considering these factors:\n"
#             "1. MARKET DYNAMICS: Explain industry-specific factors that might contribute to performance differences\n"
#             "2. BUYING PROCESS: Analyze how decision-making processes differ across segments\n"
#             "3. VALUE PROPOSITION FIT: Explain why your product/service likely resonates better with top segments\n"
#             "4. COMPETITIVE LANDSCAPE: Consider competitive differences across segments\n"
#             "5. HIDDEN CORRELATIONS: Look for relationships between dimensions (e.g., industry + company size)\n\n"
#             "Your analysis should be insightful, data-driven, and provide specific actionable recommendations. "
#             "Go beyond surface-level observations to explain underlying factors and business implications. "
#             "When asked about why a segment performs better, don't just restate the metrics - explain the business "
#             "and market dynamics that likely drive these differences."
#         )
        
#         # Use an ICP agent for triangulation data with enhanced system prompt
#         agent = ICPPrecisionAgent(
#             name="Data Analysis Agent",
#             system_prompt=triangulation_system_prompt
#         )
        
#         # Build a context that includes the data
#         context = {
#             "data_type": request.data_type,
#             "triangulation_data": request.data,
#             **request.additional_context
#         }
#     else:
#         # Default agent for other data types
#         agent = CognitionAgent(name="Data Analysis Agent")
#         context = {
#             "data_type": request.data_type,
#             "analysis_data": request.data,
#             **request.additional_context
#         }
    
#     # Construct a prompt that includes information about the data type
#     query = f"Analyze this {request.data_type} data and answer: {request.query}"
    
#     # Execute the agent
#     try:
#         result = agent.execute(query=query, context=context)
        
#         # Log the execution
#         execution = AgentExecution(
#             agent_id=agent.agent_id,
#             query=query,
#             response=result.get("response", ""),
#             reflection_enabled=agent.reflection_enabled,
#             reflection_steps=agent.reflection_steps if agent.reflection_enabled else 0,
#             meta_data={"data_type": request.data_type, "analysis_request": True}
#         )
#         db.add(execution)
#         db.commit()
        
#         execution_time = time.time() - start_time
        
#         return {
#             "response": result.get("response", "No analysis generated"),
#             "metadata": {
#                 "execution_time": f"{execution_time:.2f} seconds",
#                 "agent_id": agent.agent_id,
#                 "reflection_applied": agent.reflection_enabled,
#                 "data_type": request.data_type
#             }
#         }
#     except Exception as e:
#         logger.error(f"Error analyzing data: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error analyzing data: {str(e)}")

@router.get("/", response_model=List[AgentResponse])
async def list_agents():
    """List all available agents."""
    agents = []
    for agent in agent_manager.get_agents():
        agents.append({
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "type": agent.__class__.__name__.lower().replace("agent", ""),
            "created_at": agent.metadata.get("created_at", datetime.now().isoformat()),
            "reflection_enabled": agent.reflection_enabled,
            "metadata": agent.metadata
        })
    
    return agents

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get details for a specific agent."""
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "description": agent.description,
        "type": agent.__class__.__name__.lower().replace("agent", ""),
        "created_at": agent.metadata.get("created_at", datetime.now().isoformat()),
        "reflection_enabled": agent.reflection_enabled,
        "metadata": agent.metadata
    }

@router.post("/{agent_id}/execute", response_model=AgentExecuteResponse)
async def execute_agent(
    agent_id: str,
    execution_data: AgentExecuteRequest
):
    """Execute an agent with the given query and context."""
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    # Configure reflection for this execution if specified
    original_reflection_state = agent.reflection_enabled
    original_reflection_steps = getattr(agent, "reflection_steps", 1)
    
    try:
        # Apply temporary reflection settings if provided
        if execution_data.reflection_enabled is not None:
            if execution_data.reflection_enabled:
                agent.enable_reflection(execution_data.reflection_steps or original_reflection_steps)
            else:
                agent.disable_reflection()
        
        # Execute the agent
        start_time = datetime.now()
        
        if agent.reflection_enabled:
            result = agent.execute_with_reflection(execution_data.query, execution_data.context)
        else:
            result = agent.execute(execution_data.query, execution_data.context)
            
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Create the response
        response = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "query": execution_data.query,
            "response": result.get("response", "No response generated"),
            "reflection_applied": result.get("reflection_applied", None),
            "execution_time": f"{execution_time:.2f} seconds",
            "metadata": {
                "context": execution_data.context,
                "status": result.get("status", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return response
    
    finally:
        # Restore original reflection settings
        if execution_data.reflection_enabled is not None:
            if original_reflection_state:
                agent.enable_reflection(original_reflection_steps)
            else:
                agent.disable_reflection()

@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str):
    """Delete an agent from the registry."""
    if not agent_manager.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    return None

@router.post("/{agent_id}/reflection", status_code=200)
async def configure_reflection(
    agent_id: str,
    reflection_enabled: bool = Body(..., embed=True),
    reflection_steps: int = Body(1, embed=True)
):
    """Configure reflection settings for an agent."""
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
    
    if reflection_enabled:
        agent.enable_reflection(reflection_steps)
    else:
        agent.disable_reflection()
    
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "reflection_enabled": agent.reflection_enabled,
        "reflection_steps": getattr(agent, "reflection_steps", 1) if agent.reflection_enabled else 0
    } 