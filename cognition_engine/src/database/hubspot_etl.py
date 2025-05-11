import os
import requests
from datetime import datetime
from sqlalchemy.orm import Session

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
    
    # Retrieve company_name from HubSpot data
    company_name = props.get("company_name")
    
    # Fetch company_id from the companies table based on company_name
    company = db_session.query(Company).filter(Company.company_name == company_name).first()
    
    # If company is found, use the company_id, otherwise, set company_id as None
    company_id = company.id if company else None

    return Deal(
        id=hs_deal["id"],
        company_id=company_id,  # Now mapping the company_id
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
def transform_contact(hs_contacts):
    props = hs_contacts["properties"]
    return Contact(
        id=hs_contacts["id"],
        company_id=props.get("associatedcompanyid"),
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

    # Fetch and insert company data
    companies = fetch_hubspot_data(COMPANIES_URL)
    for item in companies:
        db.merge(transform_company(item))

    # Fetch and insert deal data
    deals = fetch_hubspot_data(DEALS_URL)
    for item in deals:
        db.merge(transform_deal(item, db))  # Pass db session here

    # Fetch and insert contact data
    contacts = fetch_hubspot_data(CONTACTS_URL)
    for item in contacts:
        db.merge(transform_contact(item))

    db.commit()
    db.close()

if __name__ == "__main__":
    load_to_db()