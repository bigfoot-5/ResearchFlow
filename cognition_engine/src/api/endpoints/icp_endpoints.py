"""
ICP analysis endpoints for the Cognition Engine.

This module provides API endpoints for ICP-related operations.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import json

from src.database.models import Company, Deal, ICPDefinition
from src.database.database_setup import get_db
from src.core_logic.icp_analysis import (
    ICPAnalysisEngine, 
    ICPScoredEntity,
    ICPInsight
)
from src.core_logic.reflective_icp_engine import ReflectiveICPEngine
from src.agents.icp_precision_agent import ICPPrecisionAgent
from src.agents.manager import AgentManager

router = APIRouter(
    prefix="",
    tags=["icp"],
    responses={404: {"description": "Not found"}},
)

# --- Model for Company List Response ---
class CompanyInfo(BaseModel):
    company_id: str
    company_name: str

    class Config:
        orm_mode = True

# --- Endpoints ---

@router.get("/definition", response_model=Dict[str, Any])
async def get_icp_definition(db: Session = Depends(get_db)):
    """
    Get the current Ideal Customer Profile definition.
    
    Args:
        db: Database session dependency
        
    Returns:
        ICP definition
    """
    # Check if we have a stored ICP definition
    icp_def = db.query(ICPDefinition).order_by(ICPDefinition.created_at.desc()).first()
    
    if icp_def:
        # Return the stored definition
        return {
            "name": icp_def.name,
            "description": icp_def.description,
            "criteria": icp_def.criteria,
            "confidence_score": icp_def.confidence_score,
            "created_at": icp_def.created_at,
            "reflection_applied": icp_def.reflection_applied
        }
    
    # Generate a default ICP definition if none exists
    return {
        "name": "Default ICP Definition",
        "description": "Automatically generated Ideal Customer Profile based on successful deals",
        "criteria": [
            {
                "field": "industry",
                "values": ["Technology", "Healthcare"],
                "weight": 0.3,
                "description": "Industries showing highest conversion rates"
            },
            {
                "field": "employee_count",
                "values": ["201-500"],
                "weight": 0.25,
                "description": "Mid-market companies convert at highest rates"
            }
        ],
        "confidence_score": 0.7,
        "created_at": datetime.now().isoformat()
    }

@router.get("/companies", response_model=List[Dict[str, Any]])
async def get_icp_companies(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get a list of companies for ICP scoring.
    
    Args:
        limit: Maximum number of companies to return
        db: Database session dependency
        
    Returns:
        List of companies
    """
    companies = db.query(Company).limit(limit).all()
    
    result = []
    for company in companies:
        company_data = {
            "company_id": company.id,
            "company_name": company.company_name,
            "industry": company.industry,
            "employee_count": company.employee_count
        }
        result.append(company_data)
    
    return result

@router.get("/company/{company_id}/score", response_model=Dict[str, Any])
async def score_company_against_icp(
    company_id: str,
    db: Session = Depends(get_db)
):
    """
    Score a company against the current ICP definition.
    
    Args:
        company_id: ID of the company to score
        db: Database session dependency
        
    Returns:
        Score results
    """
    # Get company
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
    
    # Get ICP definition
    icp_definition = await get_icp_definition(db)
    
    # Simple scoring logic (in a real implementation, this would be more sophisticated)
    score = 0.75  # Sample score
    match_details = {
        "industry": {
            "icp_values": ["Technology", "Healthcare"],
            "company_value": company.industry,
            "match_score": 0.8 if company.industry in ["Technology", "Healthcare"] else 0.2
        },
        "employee_count": {
            "icp_values": ["201-500"],
            "company_value": company.employee_count,
            "match_score": 0.9 if company.employee_count and 201 <= company.employee_count <= 500 else 0.3
        }
    }
    
    return {
        "company_id": company.id,
        "company_name": company.company_name,
        "score": score,
        "match_details": match_details,
        "icp_definition_name": icp_definition["name"]
    }

@router.get("/insights", response_model=List[Dict[str, Any]])
async def get_deal_insights(db: Session = Depends(get_db)):
    """
    Get insights from deal patterns.
    
    Args:
        db: Database session dependency
        
    Returns:
        List of insights
    """
    # In a real implementation, this would analyze actual deal data
    # For now, return sample insights
    return [
        {
            "title": "Technology Sector High Value",
            "description": "Technology sector deals have 40% higher average deal sizes compared to other industries",
            "confidence": 0.85,
            "supporting_evidence": [
                {"metric": "Avg Deal Size (Tech)", "value": "$120,000"},
                {"metric": "Avg Deal Size (Other)", "value": "$85,000"}
            ]
        },
        {
            "title": "Sales Cycle Length",
            "description": "Healthcare deals take 30% longer to close but have higher retention rates",
            "confidence": 0.78,
            "supporting_evidence": [
                {"metric": "Avg Sales Cycle (Healthcare)", "value": "75 days"},
                {"metric": "Avg Sales Cycle (Overall)", "value": "58 days"}
            ]
        }
    ]

@router.get("/reflective/definition", response_model=Dict[str, Any])
async def get_reflective_icp_definition(
    reflection_steps: int = Query(1, ge=1, le=3),
    db: Session = Depends(get_db)
):
    """
    Get an enhanced ICP definition using reflection for improvement.
    
    Args:
        reflection_steps: Number of reflection steps to perform
        db: Database session dependency
        
    Returns:
        Enhanced ICP definition
    """
    # Get base ICP definition
    base_definition = await get_icp_definition(db)
    
    # Enhance it with reflection
    reflective_engine = ReflectiveICPEngine()
    enhanced_definition = reflective_engine.generate_icp_definition(
        data=base_definition,
        reflection_steps=reflection_steps
    )
    
    return enhanced_definition

@router.get("/reflective/insights", response_model=List[Dict[str, Any]])
async def get_reflective_deal_insights(
    reflection_steps: int = Query(1, ge=1, le=3),
    db: Session = Depends(get_db)
):
    """
    Get enhanced insights using reflection for improvement.
    
    Args:
        reflection_steps: Number of reflection steps to perform
        db: Database session dependency
        
    Returns:
        Enhanced insights
    """
    # Get base insights
    base_insights = await get_deal_insights(db)
    
    # Apply reflection (in a real implementation, this would be more sophisticated)
    reflective_engine = ReflectiveICPEngine()
    
    # Convert insights to string for reflection
    insights_text = json.dumps(base_insights, indent=2)
    
    # Enhance through reflection
    enhanced_text = reflective_engine.enhance_output(insights_text, steps=reflection_steps)
    
    # For demo purposes, just add a note that reflection was applied
    enhanced_insights = base_insights.copy()
    
    # Add additional insight generated through reflection
    enhanced_insights.append({
        "title": "Decision Maker Involvement",
        "description": "Deals with C-level engagement close 45% faster and have 60% higher win rates",
        "confidence": 0.92,
        "supporting_evidence": [
            {"metric": "Win Rate (C-level)", "value": "73%"},
            {"metric": "Win Rate (No C-level)", "value": "45%"},
            {"metric": "Avg Days to Close (C-level)", "value": "42 days"}
        ],
        "reflection_generated": True
    })
    
    return enhanced_insights

@router.get("/vector-insights", response_model=List[Dict[str, Any]])
async def get_vector_insights(db: Session = Depends(get_db)):
    """
    Get insights from vector similarity searches.
    
    Args:
        db: Database session dependency
        
    Returns:
        List of vector-based insights
    """
    # In a real implementation, this would use vector search on actual data
    # For now, return sample insights
    return [
        {
            "concept": "Enterprise Sales",
            "related_patterns": [
                "Decision committees take 3x longer for approval than single decision makers",
                "Proof of concepts increase close rates by 35% for enterprise deals",
                "Multiple stakeholder demos improve win rates by 28%"
            ],
            "similarity_score": 0.92
        },
        {
            "concept": "Product Adoption",
            "related_patterns": [
                "Companies with technical champions show 50% faster implementation times",
                "Early user feedback sessions correlate with 40% higher expansion rates",
                "Training programs reduce support tickets by 65% in first 90 days"
            ],
            "similarity_score": 0.87
        }
    ]

@router.post("/search", response_model=List[Dict[str, Any]])
async def search_crm_data(
    query: str,
    n_results: int = Query(5, ge=1, le=20),
    entity_types: Optional[List[str]] = None
):
    """
    Search across CRM data using semantic similarity.
    
    Args:
        query: Natural language search query
        n_results: Number of results to return
        entity_types: Optional list of entity types to search (company, deal, contact)
        
    Returns:
        List of search results
    """
    # In a real implementation, this would use vector search on actual data
    # For now, return sample results
    results = [
        {
            "document": "TechCorp Inc. is a technology company founded in 2010. They specialize in AI solutions for healthcare. Annual revenue is $45M with 250 employees.",
            "metadata": {
                "entity_type": "company",
                "entity_id": "company_1",
                "name": "TechCorp Inc.",
                "industry": "Technology"
            },
            "distance": 0.15
        },
        {
            "document": "HealthTech Solutions signed a $120,000 deal for our enterprise plan. The sales cycle took 60 days and involved their CTO and Director of Operations.",
            "metadata": {
                "entity_type": "deal",
                "entity_id": "deal_15",
                "deal_name": "HealthTech Enterprise Deal",
                "amount": 120000
            },
            "distance": 0.22
        }
    ]
    
    # Filter by entity type if specified
    if entity_types:
        results = [r for r in results if r["metadata"]["entity_type"] in entity_types]
        
    return results[:n_results]

@router.get("/reflective/report", response_model=Dict[str, Any])
async def get_reflective_unified_icp_report(
    reflection_steps: int = Query(1, ge=1, le=3),
    db: Session = Depends(get_db)
):
    """
    Get a comprehensive ICP report with reflection for improved quality.
    
    Args:
        reflection_steps: Number of reflection steps to perform
        db: Database session dependency
        
    Returns:
        Comprehensive ICP report
    """
    # Get all the individual components with reflection
    icp_definition = await get_reflective_icp_definition(reflection_steps, db)
    insights = await get_reflective_deal_insights(reflection_steps, db)
    vector_insights = await get_vector_insights(db)
    
    # Combine them into a unified report
    report = {
        "icp_definition": icp_definition,
        "deal_insights": insights,
        "vector_insights": vector_insights,
        "generation_metadata": {
            "reflection_steps": reflection_steps,
            "generated_at": datetime.now().isoformat(),
            "based_on_deals": 2,  # In a real implementation, this would be actual count
            "confidence_score": 0.85
        }
    }
    
    return report

class ICPTriangulationResponse(BaseModel):
    """Response model for ICP triangulation data."""
    title: str
    description: str
    dimensions: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@router.get("/triangulation", response_model=ICPTriangulationResponse)
def get_icp_triangulation(
    agent_id: Optional[str] = Query(None, description="Agent ID to use for triangulation (if not provided, uses default ICP agent)"),
    db: Session = Depends(get_db)
):
    """
    Generate ICP triangulation matrix showing which customer segments perform best.
    
    The triangulation maps dimensions (industry, company size, geography, etc.) 
    against key metrics like sales velocity and win rates.
    """
    # Get the agent
    agent_manager = AgentManager()
    
    if agent_id:
        agent = agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")
        
        # Check if this is an ICP agent
        if not isinstance(agent, ICPPrecisionAgent):
            raise HTTPException(status_code=400, detail="Specified agent is not an ICP analysis agent")
    else:
        # Use default ICP agent
        agent = ICPPrecisionAgent(name="Default ICP Agent")
    
    # Generate triangulation data
    triangulation_data = agent.generate_icp_triangulation()
    
    return triangulation_data 