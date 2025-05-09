"""
Agent manager for handling agent instances in the Cognition Engine.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any, Type

from src.agents.base_agent import CognitionAgent
from src.agents.icp_precision_agent import ICPPrecisionAgent
from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class AgentManager:
    """
    Manages the lifecycle and retrieval of agent instances.
    
    This class handles the creation, retrieval, and management of agents
    to provide a consistent interface for working with different agent types.
    """
    
    def __init__(self):
        """Initialize the agent manager with an empty registry."""
        self.agent_registry = {}
        self._load_default_agents()
    
    def _load_default_agents(self):
        """Load default agents into the registry."""
        # Create a default ICP agent
        default_icp_agent = ICPPrecisionAgent(
            name="Default ICP Agent",
            description="Analyzes CRM data to identify ideal customer profiles"
        )
        self.agent_registry[default_icp_agent.agent_id] = default_icp_agent
        
        # Create a default general purpose agent
        default_agent = CognitionAgent(
            name="Default Agent",
            description="General purpose agent for basic tasks",
            system_prompt="You are a helpful general-purpose assistant for the Cognition Engine platform."
        )
        self.agent_registry[default_agent.agent_id] = default_agent
        
        logger.info(f"Loaded {len(self.agent_registry)} default agents")
    
    def get_agent(self, agent_id: str) -> Optional[CognitionAgent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The agent instance if found, None otherwise
        """
        return self.agent_registry.get(agent_id)
    
    def get_agents(self) -> List[CognitionAgent]:
        """
        Get all registered agents.
        
        Returns:
            List of all agent instances
        """
        return list(self.agent_registry.values())
    
    def create_agent(self, 
                     name: str, 
                     description: str, 
                     agent_type: str = "base", 
                     system_prompt: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> CognitionAgent:
        """
        Create a new agent of the specified type.
        
        Args:
            name: Name of the agent
            description: Description of what the agent does
            agent_type: Type of agent to create (default: "base")
            system_prompt: Optional custom system prompt
            metadata: Optional additional metadata
            
        Returns:
            The created agent instance
            
        Raises:
            ValueError: If the agent type is unknown
        """
        # Map agent type to class
        agent_classes = {
            "base": CognitionAgent,
            "icp_precision": ICPPrecisionAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(agent_classes.keys())}")
        
        # Create the agent
        agent_class = agent_classes[agent_type]
        agent = agent_class(
            name=name,
            description=description,
            system_prompt=system_prompt,
            metadata=metadata or {}
        )
        
        # Register the agent
        self.agent_registry[agent.agent_id] = agent
        
        logger.info(f"Created new agent of type {agent_type}: {agent.name} ({agent.agent_id})")
        
        return agent
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent by ID.
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            True if the agent was deleted, False if it wasn't found
        """
        if agent_id in self.agent_registry:
            agent = self.agent_registry[agent_id]
            del self.agent_registry[agent_id]
            logger.info(f"Deleted agent: {agent.name} ({agent_id})")
            return True
        return False 