import os
import sys
import requests
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.models import Company, Deal, Contact
from src.database.session import SessionLocal

HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUBSPOT_API_KEY}"}
DEALS_URL = "https://api.hubapi.com/crm/v3/objects/deals"
COMPANIES_URL = "https://api.hubapi.com/crm/v3/objects/companies"
CONTACTS_URL = "https://api.hubapi.com/crm/v3/objects/contacts"

def fetch_hubspot_data(url):
    all_results = []
    while url:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        all_results.extend(data.get("results", []))
        paging = data.get("paging", {}).get("next", {})
        url = paging.get("link")
    return all_results

def transform_company(hs_company):
    props = hs_company["properties"]
    return Company(
        id=hs_company["id"],
        company_name=props.get("company_name"),
        industry=props.get("industry"),
        numberofemployees=int(props.get("numberofemployees") or 0),
        annual_revenue=float(props.get("annualrevenue") or 0),
        country=props.get("country"),
        meta_data=props,
    )

def transform_deal(hs_deal, db_session):
    props = hs_deal["properties"]
    logger.info(f"Deal properties: {props}")  # Log all properties
    
    # Get company name from the deal properties
    company_name = props.get("deal_name")
    logger.info(f"Processing deal for company: {company_name}")
    
    # Fetch company from the companies table based on company name
    # Use case-insensitive comparison and trim whitespace
    company = None
    if company_name:
        company = db_session.query(Company).filter(
            func.lower(Company.company_name) == func.lower(company_name.strip())
        ).first()
        logger.info(f"Found company match: {company.id if company else 'None'}")
    
    # If company is found, use the company_id, otherwise, set company_id as None
    company_id = company.id if company else None

    return Deal(
        id=hs_deal["id"],
        company_id=company_id,
        company_name=company_name,  # Store the original company name from HubSpot
        deal_name=props.get("deal_name"),
        amount=float(props.get("amount") or 0),
        dealstage=props.get("dealstage"),
        closedate=parse_date(props.get("closedate")),
        hs_is_closed=props.get("hs_is_closed") == "true",
        hs_is_closed_won=props.get("hs_is_closed_won") == "true",
        hs_is_closed_lost=props.get("hs_is_closed_lost") == "true",
        hs_analytics_source=props.get("hs_analytics_source"),
        closed_lost_reason=props.get("closed_lost_reason"),
        closed_won_reason=props.get("closed_won_reason"),
        createdate=parse_date(props.get("createdate")),
        days_to_close=float(props.get("days_to_close") or 0),
        meta_data=props,
    )

def transform_contact(hs_contact, db_session):
    props = hs_contact["properties"]
    logger.info(f"Contact properties: {props}")  # Log all properties
    
    # Get the associated company ID from HubSpot
    associated_company_id = props.get("associatedcompanyid")
    logger.info(f"Processing contact with associated company ID: {associated_company_id}")
    
    # Fetch company from the companies table based on HubSpot's company ID
    company = None
    if associated_company_id:
        company = db_session.query(Company).filter(Company.id == associated_company_id).first()
        logger.info(f"Found company match: {company.id if company else 'None'}")
    
    # If company is found, use the company_id, otherwise, set company_id as None
    company_id = company.id if company else None

    return Contact(
        id=hs_contact["id"],
        company_id=company_id,
        first_name=props.get("firstname"),
        last_name=props.get("lastname"),
        job_title=props.get("jobtitle"),
        company=props.get("company"),
        meta_data=props,
    )

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None

def load_to_db():
    db: Session = SessionLocal()

    try:
        # Fetch and insert company data first
        logger.info("Fetching companies...")
        companies = fetch_hubspot_data(COMPANIES_URL)
        logger.info(f"Found {len(companies)} companies")
        for item in companies:
            db.merge(transform_company(item))
        db.commit()  # Commit companies first
        logger.info("Companies committed to database")

        # Fetch and insert deal data
        logger.info("Fetching deals...")
        deals = fetch_hubspot_data(DEALS_URL)
        logger.info(f"Found {len(deals)} deals")
        for item in deals:
            db.merge(transform_deal(item, db))
        db.commit()  # Commit deals
        logger.info("Deals committed to database")

        # Fetch and insert contact data
        logger.info("Fetching contacts...")
        contacts = fetch_hubspot_data(CONTACTS_URL)
        logger.info(f"Found {len(contacts)} contacts")
        for item in contacts:
            db.merge(transform_contact(item, db))
        db.commit()  # Commit contacts
        logger.info("Contacts committed to database")

    except Exception as e:
        logger.error(f"Error during database load: {str(e)}")
        db.rollback()
        raise e
    finally:
        db.close()


# FastAPI router for triggering load_to_db via HTTP GET
from fastapi import APIRouter

router = APIRouter()

@router.get("/load-hubspot")
def load_hubspot_data():
    load_to_db()
    return {"status": "success"}

if __name__ == "__main__":
    load_to_db()