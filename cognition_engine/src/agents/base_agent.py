"""
Base Agent class for the Cognition Engine agent framework.
This serves as the foundation for all agent types in the system.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union

from ..database.vector_store_manager import VectorStoreManager


class CognitionAgent:
    """
    Base class for all agents in the Cognition Engine.
    Provides core functionality for agent persistence, execution and reflection.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        vector_store: Optional[VectorStoreManager] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new agent.
        
        Args:
            name: The name of the agent
            description: A description of what the agent does
            system_prompt: The system prompt that defines the agent's behavior
            vector_store: Optional vector store for agent data
            agent_id: Optional agent ID (generated if not provided)
            metadata: Optional additional metadata for the agent
        """
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.agent_id = agent_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.reflection_enabled = False
        self.reflection_steps = 1
        
        # Connect to the vector store if provided, otherwise use default
        self.vector_store = vector_store or VectorStoreManager(collection_name="agent_data")
        
        # Track the agent's execution history
        self.execution_history = []
    
    def enable_reflection(self, reflection_steps: int = 1) -> None:
        """
        Enable the reflection capability for this agent.
        
        Args:
            reflection_steps: Number of reflection iterations to perform
        """
        self.reflection_enabled = True
        self.reflection_steps = reflection_steps
    
    def disable_reflection(self) -> None:
        """Disable the reflection capability for this agent."""
        self.reflection_enabled = False
    
    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent with the given query and context.
        This is the main method that should be overridden by subclasses.
        
        Args:
            query: The user query or task for the agent
            context: Additional context information
            
        Returns:
            A dictionary containing the agent's response and any additional data
        """
        # Base implementation just returns a simple response
        # This should be overridden by subclasses
        result = {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "query": query,
            "response": f"Agent {self.name} received query: {query}",
            "context": context or {},
            "reflection_enabled": self.reflection_enabled,
            "status": "not_implemented"
        }
        
        # Add to execution history
        self.execution_history.append({
            "query": query,
            "context": context,
            "result": result
        })
        
        return result
    
    def execute_with_reflection(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent with reflection capability.
        
        Args:
            query: The user query or task for the agent
            context: Additional context information
            
        Returns:
            A dictionary containing the agent's response enhanced through reflection
        """
        # Get initial result
        initial_result = self.execute(query, context)
        
        if self.reflection_enabled:
            try:
                from ..analysis.reflective_icp_engine import ReflectiveICPEngine
                
                # Use existing ReflectiveICPEngine to improve result
                reflective_engine = ReflectiveICPEngine()
                
                # Extract the response text to be reflected upon
                response_text = initial_result.get("response", "")
                
                # Enhance the response through reflection
                enhanced_response = reflective_engine.enhance_output(
                    response_text, 
                    steps=self.reflection_steps
                )
                
                # Update the result with the enhanced response
                initial_result["response"] = enhanced_response
                initial_result["reflection_applied"] = True
                
            except ImportError:
                # Reflection engine not available
                initial_result["reflection_applied"] = False
                initial_result["reflection_error"] = "ReflectiveICPEngine not available"
        
        return initial_result
    
    def save(self) -> str:
        """
        Save the agent to persistent storage.
        
        Returns:
            The agent ID of the saved agent
        """
        # Create a serializable representation of the agent
        agent_data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "reflection_enabled": self.reflection_enabled,
            "reflection_steps": self.reflection_steps,
            "type": self.__class__.__name__
        }
        
        # Save to a JSON file in a data directory
        # In a real implementation, this might use a database
        agent_file_path = f"data/agents/{self.agent_id}.json"
        
        try:
            import os
            # Ensure the directory exists
            os.makedirs("data/agents", exist_ok=True)
            
            with open(agent_file_path, "w") as f:
                json.dump(agent_data, f, indent=2)
        except Exception as e:
            print(f"Error saving agent: {e}")
        
        return self.agent_id
    
    @classmethod
    def load(cls, agent_id: str) -> "CognitionAgent":
        """
        Load an agent from persistent storage.
        
        Args:
            agent_id: The ID of the agent to load
            
        Returns:
            The loaded agent instance
        """
        agent_file_path = f"data/agents/{agent_id}.json"
        
        try:
            with open(agent_file_path, "r") as f:
                agent_data = json.load(f)
                
            # Create the appropriate agent type
            agent_type = agent_data.get("type", "CognitionAgent")
            
            # In a real implementation, this would dynamically load the correct class
            if agent_type == cls.__name__:
                agent = cls(
                    name=agent_data["name"],
                    description=agent_data["description"],
                    system_prompt=agent_data["system_prompt"],
                    agent_id=agent_data["agent_id"],
                    metadata=agent_data.get("metadata", {})
                )
                
                # Restore reflection settings
                if agent_data.get("reflection_enabled", False):
                    agent.enable_reflection(agent_data.get("reflection_steps", 1))
                
                return agent
            else:
                raise ValueError(f"Cannot load agent of type {agent_type} with {cls.__name__}.load()")
                
        except Exception as e:
            raise ValueError(f"Error loading agent {agent_id}: {e}")
            
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_id}): {self.description}"
    
    def __repr__(self) -> str:
        return f"CognitionAgent(name='{self.name}', id='{self.agent_id}')" 