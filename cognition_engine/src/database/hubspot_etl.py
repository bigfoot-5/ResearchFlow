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

def list_object_properties(object_type):
    url = f"https://api.hubapi.com/crm/v3/properties/{object_type}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return [prop["name"] for prop in response.json().get("results", [])]
    else:
        raise Exception(f"Error fetching properties for {object_type}: {response.status_code} - {response.text}")
    
def get_info(item):
    props = list_object_properties(item)
    url = f"https://api.hubapi.com/crm/v4/objects/{item}"
    all_results = []
    after = None

    while True:
        params = {
            "properties": ",".join(props),
            "limit": 100
        }
        if after:
            params["after"] = after

        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching {item}: {response.status_code} - {response.text}")

        data = response.json()
        all_results.extend(data.get('results', []))

        paging = data.get('paging')
        if paging and "next" in paging:
            after = paging["next"]["after"]
        else:
            break

    return all_results
    
def get_associated_companies(item, deal_id):
    """
    Retrieves all companies associated with a given deal.
    """
    url = f"https://api.hubapi.com/crm/v4/objects/{item}/{deal_id}/associations/companies"
    logger.info(f"Fetching associations from URL: {url}")
    
    response = requests.get(url, headers=HEADERS)
    logger.info(f"Association API response status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        logger.info(f"Association API response: {data}")
        associations = data.get('results', [])
        company_ids = [assoc['toObjectId'] for assoc in associations]
        logger.info(f"Extracted company IDs: {company_ids}")
        return company_ids
    else:
        error_msg = f"Error fetching associations: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise Exception(error_msg)

def fetch_hubspot_data(url):
    all_results = []
    if url == COMPANIES_URL:
        params = {
            "properties": "id,name,industry, numberofemployees,annual_revenue,country"
        }
    elif url == DEALS_URL:
        params = {
            "properties": "id, company_id, company_name, dealname, hs_analytics_source,closed_lost_reason,closed_won_reason,days_to_close,amount,dealstage,closedate"
        }
    elif url == CONTACTS_URL:
        params = {
            "properties": "id,company_id,firstname ,lastname,jobtitle"
        } 
    while url:
        response = requests.get(url, headers=HEADERS, params=params)
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
        company_name=props.get("name"),
        industry=props.get("industry"),
        numberofemployees=int(props.get("numberofemployees") or 0),
        annual_revenue=float(props.get("annualrevenue") or 0),
        country=props.get("country"),
        meta_data=props,
    )
def transform_deal(hs_deal, db_session):
    props = hs_deal["properties"]
    deal_id = hs_deal["id"]

    # Get associated company IDs using the v4 API
    try:
        company_ids = get_associated_companies("deals", deal_id)
    except Exception as e:
        logger.error(f"Error fetching associated companies for deal {deal_id}: {e}")
        company_ids = []

    company_id = company_ids[0] if company_ids else None

    # Optionally, fetch company_name from DB for reference
    company_name = None
    if company_id:
        company = db_session.query(Company).filter(Company.id == str(company_id)).first()
        company_name = company.company_name if company else None

    return Deal(
        id=deal_id,
        company_id=company_id,
        company_name=company_name,
        deal_name=props.get("dealname"),
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
# def transform_deal(hs_deal, db_session):
#     props = hs_deal["properties"]
#     logger.info(f"Deal properties: {props}")  # Log all properties
    
#     # Get company name from the deal properties
#     company_name = props.get("deal_name")
#     logger.info(f"Processing deal for company: {company_name}")
    
#     # Fetch company from the companies table based on company name
#     # Use case-insensitive comparison and trim whitespace
#     company = None
#     if company_name:
#         company = db_session.query(Company).filter(
#             func.lower(Company.company_name) == func.lower(company_name.strip())
#         ).first()
#         logger.info(f"Found company match: {company.id if company else 'None'}")
    
#     # If company is found, use the company_id, otherwise, set company_id as None
#     company_id = company.id if company else None

#     return Deal(
#         id=hs_deal["id"],
#         company_id=company_id,
#         company_name=company_name,  # Store the original company name from HubSpot
#         deal_name=props.get("deal_name"),
#         amount=float(props.get("amount") or 0),
#         dealstage=props.get("dealstage"),
#         closedate=parse_date(props.get("closedate")),
#         hs_is_closed=props.get("hs_is_closed") == "true",
#         hs_is_closed_won=props.get("hs_is_closed_won") == "true",
#         hs_is_closed_lost=props.get("hs_is_closed_lost") == "true",
#         hs_analytics_source=props.get("hs_analytics_source"),
#         closed_lost_reason=props.get("closed_lost_reason"),
#         closed_won_reason=props.get("closed_won_reason"),
#         createdate=parse_date(props.get("createdate")),
#         days_to_close=float(props.get("days_to_close") or 0),
#         meta_data=props,
#     )

def transform_contact(hs_contact, db_session):
    props = hs_contact["properties"]
    contact_id = hs_contact["id"]
    logger.info(f"Processing contact {contact_id} - {props.get('firstname')} {props.get('lastname')}")
    
    # Get associated company IDs using the v4 API
    try:
        company_ids = get_associated_companies("contacts", contact_id)
        logger.info(f"Found {len(company_ids)} associated companies for contact {contact_id}: {company_ids}")
    except Exception as e:
        logger.error(f"Error fetching associated companies for contact {contact_id}: {e}")
        company_ids = []

    company_id = company_ids[0] if company_ids else None
    logger.info(f"Selected company_id for contact {contact_id}: {company_id}")

    # Optionally, fetch company_name from DB for reference
    company_name = None
    if company_id:
        company = db_session.query(Company).filter(Company.id == str(company_id)).first()
        company_name = company.company_name if company else None
        logger.info(f"Found company name for company_id {company_id}: {company_name}")
    else:
        logger.warning(f"No company_id found for contact {contact_id}")

    return Contact(
        id=contact_id,
        company_id=company_id,
        first_name=props.get("firstname"),
        last_name=props.get("lastname"),
        job_title=props.get("jobtitle"),
        # company=company_name,  # Use the company name from our database if found
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
        
        # First pass: insert all companies
        for item in companies:
            company = transform_company(item)
            db.merge(company)
        db.commit()
        logger.info("Companies committed to database")

        # Verify companies were loaded
        company_count = db.query(Company).count()
        logger.info(f"Verified {company_count} companies in database")

        # Fetch and insert deal data
        logger.info("Fetching deals...")
        deals = fetch_hubspot_data(DEALS_URL)
        logger.info(f"Found {len(deals)} deals")
        for item in deals:
            db.merge(transform_deal(item, db))
        db.commit()
        logger.info("Deals committed to database")

        # Fetch and insert contact data
        logger.info("Fetching contacts...")
        contacts = fetch_hubspot_data(CONTACTS_URL)
        logger.info(f"Found {len(contacts)} contacts")
        for item in contacts:
            contact = transform_contact(item, db)
            db.merge(contact)
        db.commit()
        logger.info("Contacts committed to database")

        # Final verification
        contact_count = db.query(Contact).count()
        deal_count = db.query(Deal).count()
        # logger.info(f"Final counts - Companies: {company_count}, Deals: {deal_count}, Contacts: {contact_count}")

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