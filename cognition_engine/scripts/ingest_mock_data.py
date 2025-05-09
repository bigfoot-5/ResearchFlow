import sys
import os

# Add the project root to the Python path to allow for absolute imports
# This is often needed when running scripts directly from a subdirectory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming this script is in cognition_engine/scripts/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import specific generation functions and ORM models
from src.data_generation.mock_hubspot_data import (
    generate_mock_companies, 
    generate_mock_deals, 
    generate_mock_contacts
)
# Removed Pydantic imports as we'll work with ORM directly
from src.database.database_setup import SessionLocal, engine # For session and engine
from src.database.models import CompanyOrm, DealOrm, ContactOrm, Base # SQLAlchemy ORM models
# Import refactored derived metric functions that expect ORM objects
from src.core_logic.derived_metrics import (
    calculate_deal_age,
    calculate_velocity,
    calculate_icp_fit_signal,
    is_in_dead_zone,
    is_price_sensitive
)
# Keep PerceptionService import for classification if available
from src.core_logic.perception import PerceptionService 
from datetime import date, datetime # Need datetime

# Configuration for mock data generation
NUM_COMPANIES = 20
AVG_DEALS_PER_COMPANY = 5
AVG_CONTACTS_PER_COMPANY = 3

def ingest_data():
    print(f"Generating mock data: {NUM_COMPANIES} companies, approx {AVG_DEALS_PER_COMPANY} deals/co, {AVG_CONTACTS_PER_COMPANY} contacts/co...")
    
    db = SessionLocal() # Create a new session
    perception_service = PerceptionService() # Initialize PerceptionService (may need LLM setup)

    try:
        print("Starting data ingestion...")
        
        # 1. Generate and Ingest Companies
        mock_companies_orm = generate_mock_companies(NUM_COMPANIES)
        db.add_all(mock_companies_orm)
        db.flush() # Flush to assign IDs if needed by FKs, and ensure objects are queryable
        print(f"Generated {len(mock_companies_orm)} company ORM objects.")

        # Get company IDs *after* flushing or committing
        company_ids = [c.company_id for c in mock_companies_orm]
        # Create a map from company_id to ORM object for easy lookup
        companies_orm_map = {c.company_id: c for c in mock_companies_orm}
        print(f"Successfully staged {len(mock_companies_orm)} companies.")

        # 2. Generate Deals (ORM objects directly)
        num_deals = NUM_COMPANIES * AVG_DEALS_PER_COMPANY
        mock_deals_orm = generate_mock_deals(num_deals, company_ids)
        print(f"Generated {len(mock_deals_orm)} deal ORM objects.")

        # 3. Calculate Derived Metrics for Deals & Classify Reason
        deals_to_ingest = []
        for deal_orm in mock_deals_orm:
            associated_company_orm = companies_orm_map.get(deal_orm.company_id)

            # Calculate metrics using ORM object
            deal_orm.deal_age_days = calculate_deal_age(deal_orm)
            deal_orm.velocity_days = calculate_velocity(deal_orm)
            
            if associated_company_orm:
                 deal_orm.icp_fit_signal_score = calculate_icp_fit_signal(deal_orm, associated_company_orm)
            else:
                 deal_orm.icp_fit_signal_score = "Unknown Company" # Or handle as needed

            deal_orm.is_dead_zone = is_in_dead_zone(deal_orm)
            deal_orm.is_price_sensitive = is_price_sensitive(deal_orm)

            # Classify closed lost reason if applicable
            deal_orm.classified_closed_lost_reason = None
            if deal_orm.deal_stage == "Closed Lost" and deal_orm.closed_lost_reason:
                try:
                    # Check if LLM is available within the perception service instance
                    if perception_service.llm_client and perception_service.llm_client.llm:
                        deal_orm.classified_closed_lost_reason = perception_service.classify_closed_lost_reason(deal_orm.closed_lost_reason)
                    else:
                        # Only print this once or use logging to avoid clutter
                        if not hasattr(ingest_data, '_printed_llm_warning'):
                            print("LLM client not available in PerceptionService, skipping classification.")
                            ingest_data._printed_llm_warning = True 
                except Exception as llm_err:
                     # Handle potential errors during LLM call gracefully
                     print(f"Warning: Error during LLM classification for deal {deal_orm.deal_id}: {llm_err}")
                     deal_orm.classified_closed_lost_reason = "Classification Error"

            deals_to_ingest.append(deal_orm)
            
        db.add_all(deals_to_ingest)
        print(f"Successfully staged {len(deals_to_ingest)} deals with derived metrics.")

        # 4. Generate and Ingest Contacts
        num_contacts = NUM_COMPANIES * AVG_CONTACTS_PER_COMPANY
        mock_contacts_orm = generate_mock_contacts(num_contacts, company_ids)
        db.add_all(mock_contacts_orm)
        print(f"Successfully staged {len(mock_contacts_orm)} contacts.")

        # 5. Commit everything
        db.commit()
        print("Commit successful. Mock data ingestion completed!")

    except Exception as e:
        print(f"Error during data ingestion: {e}")
        print("Rolling back changes...")
        db.rollback() # Rollback in case of error
        raise
    finally:
        db.close() # Always close the session

if __name__ == "__main__":
    print("Running mock data ingestion script...")
    # Optional: Clear existing data before ingesting new mock data
    print("Clearing existing data (dropping and recreating tables)...")
    Base.metadata.drop_all(bind=engine) # Drop all tables
    Base.metadata.create_all(bind=engine) # Recreate all tables
    print("Tables cleared and recreated.")

    ingest_data() 