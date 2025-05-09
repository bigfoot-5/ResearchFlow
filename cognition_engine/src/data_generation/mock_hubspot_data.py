import random
import uuid
from datetime import datetime, timedelta
from faker import Faker
from sqlalchemy.orm import Session

from src.database.models import CompanyOrm, DealOrm, ContactOrm
from src.data_models.hubspot_entities import Company, Deal, Contact

fake = Faker()

def generate_mock_companies(num_companies: int) -> list[CompanyOrm]:
    """Generates a list of mock companies."""
    companies = []
    for i in range(num_companies):
        size_str = random.choice(["Small", "Medium", "Enterprise"])
        employee_count_val = None
        if size_str == "Small":
            employee_count_val = random.randint(1, 50)
        elif size_str == "Medium":
            employee_count_val = random.randint(51, 500)
        else:  # Enterprise
            employee_count_val = random.randint(501, 5000)

        company = CompanyOrm(
            company_id=str(uuid.uuid4()),
            company_name=fake.company(),
            industry=random.choice(["Technology", "Finance", "Healthcare", "Manufacturing", "Retail"]),
            geography=fake.city(),
            employee_count=employee_count_val,
            annual_revenue=float(random.randint(100000, 1000000000)),
        )
        companies.append(company)
    return companies

def generate_mock_deals(num_deals: int, company_ids: list[int]) -> list[DealOrm]:
    """Generates a list of mock deals."""
    deals = []
    for i in range(num_deals):
        creation_datetime = datetime.now() - timedelta(days=random.randint(30, 365))
        
        # Mocking status and stage logic
        current_status = random.choice(["open", "closed_won", "closed_lost"])
        deal_stage_val = None
        close_datetime = None

        if current_status == "open":
            deal_stage_val = random.choice(["Prospecting", "Qualified", "Proposal", "Negotiation"])
            # close_date remains None
        elif current_status == "closed_won":
            deal_stage_val = "Closed Won"
            close_datetime = creation_datetime + timedelta(days=random.randint(1, 180))
        else: # closed_lost
            deal_stage_val = "Closed Lost"
            close_datetime = creation_datetime + timedelta(days=random.randint(1, 180))

        mock_deal_type = random.choice(["New Business", "Upsell", "Renewal"])

        deal = DealOrm(
            deal_id=str(uuid.uuid4()),  # Generate deal_id
            company_id=random.choice(company_ids),
            deal_name=f"Deal for {fake.word().capitalize()} Services", # Changed from name
            amount=float(random.randint(1000, 500000)),
            deal_stage=deal_stage_val,  # Changed from stage, and uses new logic
            create_date=creation_datetime, # Pass datetime object
            close_date=close_datetime, # Pass datetime object or None
            # status field is handled by deal_stage and close_date logic above
            closed_lost_reason=fake.sentence() if deal_stage_val == "Closed Lost" else None, # Changed from lost_reason
            pipeline=mock_deal_type, # Mapped deal_type to pipeline
            # Removed fields not in DealOrm: description, source, priority
            # Buyer persona and use case are in the model but not previously generated, can add if needed.
            buyer_persona=random.choice(["IT Manager", "Sales Director", "Marketing Head", "CEO"]) if random.choice([True,False]) else None,
            use_case=random.choice(["Platform Modernization", "Cost Reduction", "Improved Efficiency"]) if random.choice([True,False]) else None
        )
        deals.append(deal)
    return deals

def generate_mock_contacts(num_contacts: int, company_ids: list[int]) -> list[ContactOrm]:
    """Generates a list of mock contacts."""
    contacts = []
    for i in range(num_contacts):
        # Removed created_date and last_activity_date as they are not in ContactOrm
        contact = ContactOrm(
            contact_id=str(uuid.uuid4()), # Generate contact_id
            company_id=random.choice(company_ids) if company_ids else None,
            first_name=fake.first_name(),
            last_name=fake.last_name(),
            email=fake.email()
            # Removed fields not in ContactOrm: 
            # phone, title, department, created_date, last_activity, 
            # lead_source, lead_status, lifecycle_stage
        )
        contacts.append(contact)
    return contacts

if __name__ == '__main__':
    # Example usage (requires database setup)
    # from src.database.database_setup import create_db_and_tables, get_db
    # create_db_and_tables()
    # db = next(get_db())

    # mock_companies = generate_mock_companies(5)
    # for comp in mock_companies:
    #     db.add(comp)
    # db.commit()
    # company_ids = [c.id for c in db.query(CompanyOrm.id).all()]

    # mock_deals = generate_mock_deals(10, company_ids)
    # for deal_obj in mock_deals: # Renamed deal to deal_obj to avoid conflict with Deal model
    #     db.add(deal_obj)
    # db.commit()

    # mock_contacts = generate_mock_contacts(20, company_ids)
    # for contact_obj in mock_contacts: # Renamed contact to contact_obj to avoid conflict with Contact model
    #     db.add(contact_obj)
    # db.commit()

    # print(f"Generated {len(mock_companies)} companies, {len(mock_deals)} deals, and {len(mock_contacts)} contacts.")
    # db.close()
    pass