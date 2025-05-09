from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class Company(BaseModel):
    company_id: str = Field(..., description="Primary identifier for the company")
    company_name: str = Field(..., description="Name of the company")
    industry: Optional[str] = Field(None, description="Industry of the company")
    geography: Optional[str] = Field(None, description="Region-based patterns")
    employee_count: Optional[int] = Field(None, description="Thermographic segmentation")
    annual_revenue: Optional[float] = Field(None, description="For ACV normalization")

class Deal(BaseModel):
    deal_id: str = Field(..., description="Primary identifier for the deal")
    deal_name: str = Field(..., description="Human-readable name for the deal")
    create_date: datetime = Field(..., description="Date the deal was created")
    close_date: Optional[datetime] = Field(None, description="Date the deal was closed")
    amount: Optional[float] = Field(None, description="ACV / deal size")
    deal_stage: Optional[str] = Field(None, description="Stage progression analysis")
    pipeline: Optional[str] = Field(None, description="New Biz vs Renewal segmentation")
    buyer_persona: Optional[str] = Field(None, description="Persona segmentation")
    use_case: Optional[str] = Field(None, description="Strategic GTM mapping")
    closed_lost_reason: Optional[str] = Field(None, description="Disqualification insight")
    # Foreign key to Company
    company_id: str = Field(..., description="Identifier for the associated company")

# Optional for persona mapping, as per instruction.txt
class Contact(BaseModel):
    contact_id: str = Field(..., description="Primary identifier for the contact")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    # Foreign key to Company
    company_id: Optional[str] = Field(None, description="Identifier for the associated company")
    # Potentially other fields relevant for persona mapping 