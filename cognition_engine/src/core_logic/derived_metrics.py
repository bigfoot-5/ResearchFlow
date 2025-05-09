from datetime import datetime, date
from typing import Optional, List

from src.database.models import DealOrm, CompanyOrm # Assuming CompanyOrm might be needed for ICP fit

# Constants for Dead Zone logic
DEFAULT_DEAD_ZONE_AGE_THRESHOLD_DAYS = 90
DEAD_ZONE_CLOSED_LOST_REASONS_KEYWORDS = ["no champion", "no decision", "project shelved", "timing", "stalled"]

# Constants for Price Sensitivity logic
PRICE_SENSITIVITY_KEYWORDS = ["price", "pricing", "budget", "cost", "expensive", "cheaper"]

def calculate_deal_age(deal: DealOrm) -> Optional[int]:
    """Calculates the age of a deal in days from its creation date to today."""
    if deal.create_date:
        try:
            # deal.create_date is already a datetime object from the generator/model
            create_date_obj = deal.create_date.date() if isinstance(deal.create_date, datetime) else datetime.fromisoformat(str(deal.create_date)).date()
            today = datetime.utcnow().date()
            return (today - create_date_obj).days
        except (ValueError, TypeError):
            # Handle cases where conversion might fail or if it's not a datetime object initially
            return None
    return None

def calculate_velocity(deal: DealOrm) -> Optional[int]:
    """Calculates the velocity of a deal in days if it's closed."""
    closed_stages = ["closed_won", "closed_lost"]
    # Using deal.deal_stage which we mapped in mock data generator
    if deal.deal_stage and deal.deal_stage.lower() in closed_stages and deal.create_date and deal.close_date:
        try:
            # deal.create_date and deal.close_date are already datetime objects
            create_date_obj = deal.create_date.date() if isinstance(deal.create_date, datetime) else datetime.fromisoformat(str(deal.create_date)).date()
            close_date_obj = deal.close_date.date() if isinstance(deal.close_date, datetime) else datetime.fromisoformat(str(deal.close_date)).date()
            return (close_date_obj - create_date_obj).days
        except (ValueError, TypeError):
            return None
    return None

def calculate_icp_fit_signal(deal: DealOrm, company: Optional[CompanyOrm] = None) -> str:
    """Placeholder logic for ICP Fit Signal Score based on company attributes.
    Requires the associated CompanyOrm object to be passed or fetched.
    """
    # This function might need access to the company associated with the deal.
    # For now, we'll keep it simple. If 'company' is not passed, it can't determine.
    # In a real scenario, you'd fetch company = db.query(CompanyOrm).filter(CompanyOrm.id == deal.company_id).first()
    # For this example, let's assume company attributes are directly on the deal or passed in.
    
    company_industry = getattr(company, 'industry', None) if company else None
    company_employee_count = getattr(company, 'size', None) # Assuming 'size' can be mapped to employee count later or is a proxy

    # Convert company_employee_count (e.g., "101-500 employees") to a number if possible
    # This is a placeholder, real mapping would be needed.
    employee_count_numeric = None
    if isinstance(company_employee_count, str):
        if "1000+" in company_employee_count or "Enterprise" in company_employee_count :
             employee_count_numeric = 1001
        elif "500" in company_employee_count or "501" in company_employee_count:
             employee_count_numeric = 501
        elif "101" in company_employee_count or "201" in company_employee_count or "Medium" in company_employee_count:
            employee_count_numeric = 200
        elif "Small" in company_employee_count or "50" in company_employee_count:
            employee_count_numeric = 50


    if not company_industry or employee_count_numeric is None:
        return "Indeterminate"

    if company_industry.lower() in ["technology", "saas", "finance"]:
        if employee_count_numeric > 500:
            return "High"
        elif employee_count_numeric > 100:
            return "Medium"
        else:
            return "Low (Tech/SaaS/Finance)"
    elif company_industry.lower() in ["healthcare", "manufacturing"]:
        if employee_count_numeric > 1000:
            return "High"
        elif employee_count_numeric > 200:
            return "Medium"
        else:
            return "Low (Healthcare/Manufacturing)"
    else:
        return "Low (Other Industries)"


def is_in_dead_zone(
    deal: DealOrm,
    dead_zone_age_threshold_days: int = DEFAULT_DEAD_ZONE_AGE_THRESHOLD_DAYS
) -> bool:
    """Checks if a deal is in a 'Dead Zone'."""
    deal_age = calculate_deal_age(deal)
    
    if deal_age is None: # Cannot determine age
        return False

    # If deal is very old and not yet won
    if deal.deal_stage and deal.deal_stage.lower() != "closed won" and deal_age > dead_zone_age_threshold_days:
        # If it's Closed Lost for a dead zone reason, or simply very old and open/lost for other reasons
        if deal.deal_stage.lower() == "closed lost" and deal.closed_lost_reason:
            for keyword in DEAD_ZONE_CLOSED_LOST_REASONS_KEYWORDS:
                if keyword in deal.closed_lost_reason.lower():
                    return True
        # An old open deal can also be considered stalled
        open_stages = ["prospecting", "qualified", "proposal", "negotiation"]
        if deal.deal_stage.lower() in open_stages: 
            return True 

    # Specifically check Closed Lost reasons for dead zone keywords
    if deal.deal_stage and deal.deal_stage.lower() == "closed lost" and deal.closed_lost_reason:
        for keyword in DEAD_ZONE_CLOSED_LOST_REASONS_KEYWORDS:
            if keyword in deal.closed_lost_reason.lower():
                return True
                
    return False

def is_price_sensitive(deal: DealOrm) -> bool:
    """Checks if a deal was lost due to price sensitivity."""
    if deal.deal_stage and deal.deal_stage.lower() == "closed lost" and deal.closed_lost_reason:
        for keyword in PRICE_SENSITIVITY_KEYWORDS:
            if keyword in deal.closed_lost_reason.lower():
                return True
    return False

# Example of how you might fetch the company if not passed:
# from sqlalchemy.orm import Session
# def get_company_for_deal(db: Session, deal: DealOrm) -> Optional[CompanyOrm]:
#     return db.query(CompanyOrm).filter(CompanyOrm.id == deal.company_id).first()

# Note: Conversion Rate is an aggregate metric and typically calculated per segment, 
# so it's not included here as a per-deal calculation. 
# It would be computed later during an aggregation or analysis step. 