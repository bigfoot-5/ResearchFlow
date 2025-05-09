"""
Data access endpoints for the Cognition Engine.

This module provides API endpoints for accessing data from the database.
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

# Relative imports for modules within the same package (src)
from ...database.database_setup import get_db
from ...database.models import Company, Contact, Deal

# Create router
router = APIRouter(
    prefix="/data",
    tags=["data"],
    responses={404: {"description": "Not found"}},
)

@router.get("/companies", response_model=List[Dict[str, Any]])
async def get_companies(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get a list of companies from the database.
    
    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        db: Database session dependency
        
    Returns:
        List of company data
    """
    companies = db.query(Company).offset(skip).limit(limit).all()
    
    # Convert to dictionary format
    result = []
    for company in companies:
        company_data = {
            "company_id": company.id,
            "company_name": company.company_name,
            "industry": company.industry,
            "employee_count": company.employee_count,
            "annual_revenue": company.annual_revenue,
            "meta_data": company.meta_data
        }
        result.append(company_data)
        
    return result

@router.get("/companies/{company_id}", response_model=Dict[str, Any])
async def get_company(
    company_id: str,
    db: Session = Depends(get_db)
):
    """
    Get details for a specific company.
    
    Args:
        company_id: ID of the company to retrieve
        db: Database session dependency
        
    Returns:
        Company data
    """
    company = db.query(Company).filter(Company.id == company_id).first()
    
    if company is None:
        raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found")
    
    # Convert to dictionary format
    company_data = {
        "company_id": company.id,
        "company_name": company.company_name,
        "industry": company.industry,
        "employee_count": company.employee_count,
        "annual_revenue": company.annual_revenue,
        "country": company.country,
        "website": company.website,
        "founded_year": company.founded_year,
        "meta_data": company.meta_data,
        "created_at": company.created_at,
        "updated_at": company.updated_at
    }
    
    return company_data

@router.get("/deals", response_model=List[Dict[str, Any]])
async def get_deals(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get a list of deals from the database.
    
    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        db: Database session dependency
        
    Returns:
        List of deal data
    """
    deals = db.query(Deal).offset(skip).limit(limit).all()
    
    # Convert to dictionary format
    result = []
    for deal in deals:
        deal_data = {
            "deal_id": deal.id,
            "company_id": deal.company_id,
            "deal_name": deal.deal_name,
            "amount": deal.amount,
            "stage": deal.stage,
            "close_date": deal.close_date,
            "meta_data": deal.meta_data
        }
        result.append(deal_data)
        
    return result

@router.get("/contacts", response_model=List[Dict[str, Any]])
async def get_contacts(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get a list of contacts from the database.
    
    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        db: Database session dependency
        
    Returns:
        List of contact data
    """
    contacts = db.query(Contact).offset(skip).limit(limit).all()
    
    # Convert to dictionary format
    result = []
    for contact in contacts:
        contact_data = {
            "contact_id": contact.id,
            "company_id": contact.company_id,
            "first_name": contact.first_name,
            "last_name": contact.last_name,
            "email": contact.email,
            "job_title": contact.job_title,
            "meta_data": contact.meta_data
        }
        result.append(contact_data)
        
    return result 