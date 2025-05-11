"""
SQLAlchemy models for the Cognition Engine.

This module defines the database models for the system.
"""

import uuid
from datetime import datetime
from typing import Dict, Any

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Company(Base):
    """Company model representing organizations in the CRM data."""
    __tablename__ = "companies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_name = Column(String)
    industry = Column(String)
    numberofemployees = Column(Integer)
    annual_revenue = Column(Float)
    # founded_year = Column(Integer)
    # website = Column(String)
    country = Column(String)
    # created_at = Column(DateTime, default=datetime.utcnow)
    # updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    deals = relationship("Deal", back_populates="company")
    contacts = relationship("Contact", back_populates="company")
    
    # Additional data as JSON
    meta_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<Company {self.company_name}>"


class Contact(Base):
    """Contact model representing individuals in the CRM data."""
    __tablename__ = "contacts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, ForeignKey("companies.id", ondelete="CASCADE"))
    first_name = Column(String)
    last_name = Column(String)
    # email = Column(String)
    # phone = Column(String)
    job_title = Column(String)
    # created_at = Column(DateTime, default=datetime.utcnow)
    # updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="contacts")
    
    # Additional data as JSON
    meta_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<Contact {self.first_name} {self.last_name}>"


class Deal(Base):
    """Deal model representing sales opportunities in the CRM data."""
    __tablename__ = "deals"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    company_id = Column(String, ForeignKey("companies.id", ondelete="CASCADE"))
    company_name=Column(String)
    deal_name = Column(String)
    amount = Column(Float)
    dealstage = Column(String)  # e.g., "prospecting", "qualified", "closed_won", "closed_lost"
    closedate = Column(DateTime)
    hs_is_closed_lost=Column(Boolean)
    hs_is_closed_won=Column(Boolean)
    hs_is_closed=Column(Boolean)
    hs_analytics_source=Column(String)
    closed_lost_reason=Column(String)
    closed_won_reason=Column(String)
    createdate = Column(DateTime, default=datetime.utcnow)
    days_to_close=Column(Float)
    # updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="deals")
    
    # Additional data as JSON
    meta_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<Deal {self.deal_name}>"


class ICPDefinition(Base):
    """Model to store generated Ideal Customer Profile definitions."""
    __tablename__ = "icp_definitions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    description = Column(Text)
    criteria = Column(JSON)  # Structured ICP criteria
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    reflection_applied = Column(Boolean, default=False)
    reflection_steps = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<ICPDefinition {self.name}>"


class AgentExecution(Base):
    """Model to store agent execution history."""
    __tablename__ = "agent_executions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String)
    agent_name = Column(String)
    query = Column(Text)
    response = Column(Text)
    reflection_enabled = Column(Boolean, default=False)
    reflection_steps = Column(Integer, default=0)
    reflection_applied = Column(Boolean, default=False)
    execution_time_seconds = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Store context and result as JSON
    context = Column(JSON, default=dict)
    result = Column(JSON, default=dict)
    meta_data = Column(JSON, default=dict)
    
    def __repr__(self):
        return f"<AgentExecution {self.agent_name} {self.created_at}>"

# If you want to create an engine and session for use elsewhere, you could define them here or in database_setup.py
# For example:
# engine = create_engine(settings.DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# To create tables (usually called once at app startup or via a script):
# Base.metadata.create_all(bind=engine) 