#!/usr/bin/env python3
"""
Test script for ICP Analysis Engine

This script creates mock data and tests the ICP analysis features.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session

from src.database.database_setup import create_db_and_tables, get_db
from src.database.models import CompanyOrm, DealOrm, ContactOrm
from src.data_generation.mock_hubspot_data import (
    generate_mock_companies,
    generate_mock_deals,
    generate_mock_contacts
)
from src.core_logic.derived_metrics import (
    calculate_deal_age,
    calculate_velocity,
    calculate_icp_fit_signal,
    is_in_dead_zone,
    is_price_sensitive
)
from src.core_logic.icp_analysis import ICPAnalysisEngine
from src.vector_store.vector_store_manager import VectorStoreManager


def clear_and_generate_mock_data(db: Session):
    """Clear existing data and generate new mock data with derivated metrics"""
    # Clear existing data
    db.query(ContactOrm).delete()
    db.query(DealOrm).delete()
    db.query(CompanyOrm).delete()
    db.commit()
    
    # Generate new mock data
    companies = generate_mock_companies(30)  # Generate 30 companies
    for company in companies:
        db.add(company)
    db.commit()
    
    # Create a company lookup by ID
    companies_by_id = {company.company_id: company for company in companies}

    # Get company IDs
    company_ids = [company.company_id for company in companies]
    
    # Generate deals
    deals = generate_mock_deals(50, company_ids)  # Generate 50 deals
    
    # Calculate derived metrics
    for deal in deals:
        # Calculate deal age
        deal.deal_age_days = calculate_deal_age(deal)
        
        # Calculate velocity
        deal.velocity_days = calculate_velocity(deal)
        
        # Get the company for the current deal
        associated_company = companies_by_id.get(deal.company_id)
        
        # Calculate ICP fit
        deal.icp_fit_signal_score = calculate_icp_fit_signal(deal, associated_company)
        
        # Set dead zone flag
        deal.is_dead_zone = is_in_dead_zone(deal)
        
        # Set price sensitive flag
        deal.is_price_sensitive = is_price_sensitive(deal)
        
        # Add to database
        db.add(deal)
    db.commit()
    
    # Generate contacts
    contacts = generate_mock_contacts(100, company_ids)  # Generate 100 contacts
    for contact in contacts:
        db.add(contact)
    db.commit()
    
    logging.info(f"Generated {len(companies)} companies, {len(deals)} deals, and {len(contacts)} contacts")
    
    return companies, deals, contacts


def test_vector_store_manager(db: Session, companies, deals, contacts):
    """Test the VectorStoreManager"""
    logging.info("Testing Vector Store Manager...")
    
    # Initialize the manager
    manager = VectorStoreManager(collection_name="test_crm_collection")
    
    # Index some data
    manager.index_companies(companies[:5])
    manager.index_deals(deals[:5])
    manager.index_contacts(contacts[:5])
    
    # Test search
    logging.info("Testing vector search...")
    results = manager.search_crm_data("technology enterprise deals", n_results=3)
    
    logging.info(f"Found {len(results)} results")
    for i, result in enumerate(results):
        logging.info(f"Result {i+1}:")
        logging.info(f"  Similarity: {result['similarity']:.4f}")
        logging.info(f"  Entity Type: {result['metadata']['entity_type']}")
        logging.info(f"  Document preview: {result['document'][:50]}...")
    
    logging.info("Vector Store Manager test completed")


def test_icp_analysis_engine(db: Session):
    """Test the ICPAnalysisEngine"""
    logging.info("Testing ICP Analysis Engine...")
    
    # Initialize the engine
    engine = ICPAnalysisEngine()
    
    try:
        # Generate ICP definition
        logging.info("Generating ICP definition...")
        icp = engine.generate_icp_definition_from_data(db)
        logging.info(f"ICP Name: {icp.name}")
        logging.info(f"ICP Description: {icp.description}")
        logging.info(f"Number of criteria: {len(icp.criteria)}")
        
        # Score companies
        logging.info("\nScoring companies against ICP...")
        companies = db.query(CompanyOrm).limit(3).all()
        for company in companies:
            score = engine.score_company_against_icp(company, icp)
            logging.info(f"Company ID {company.company_id} ({company.company_name}): Score = {score.score:.2f}")
            
            # Log match details for the first company
            if company.company_id == companies[0].company_id:
                logging.info("Match details:")
                for field, details in score.match_details.items():
                    logging.info(f"  {field}: Value={details['value']}, Match={details['is_match']}, Weight={details['weight']}")
        
        # Generate deal insights
        logging.info("\nGenerating deal insights...")
        insights = engine.analyze_deal_patterns(db)
        for i, insight in enumerate(insights):
            logging.info(f"Insight {i+1}: {insight.title}")
            logging.info(f"Confidence: {insight.confidence:.2f}")
            logging.info(f"Description: {insight.description}")
        
        # Generate vector insights
        logging.info("\nGenerating vector-enhanced insights...")
        vector_insights = engine.generate_vector_enhanced_icp_insights(db)
        for i, insight in enumerate(vector_insights):
            logging.info(f"Vector Insight {i+1} for query: {insight['query']}")
            logging.info(f"Dominant patterns: {insight['dominant_patterns']}")
            logging.info(f"Average similarity: {insight['average_similarity']:.4f}")
        
        # Generate unified report
        logging.info("\nGenerating unified ICP report...")
        report = engine.generate_unified_icp_report(db)
        if "error" in report:
            logging.error(f"Error in report generation: {report['error']}")
        else:
            logging.info(f"Report generated at: {report['report_generated_at']}")
            logging.info(f"Top companies count: {len(report['top_companies'])}")
            logging.info(f"Deal insights count: {len(report['deal_insights'])}")
            logging.info(f"Vector insights count: {len(report['vector_insights'])}")
        
        logging.info("ICP Analysis Engine test completed successfully!")
    except Exception as e:
        logging.error(f"Error in ICP Analysis Engine test: {e}", exc_info=True)


def main():
    """Main test function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create database tables
    create_db_and_tables()
    
    # Get database session
    db = next(get_db())
    
    try:
        # Generate mock data
        logging.info("Generating mock data...")
        companies, deals, contacts = clear_and_generate_mock_data(db)
        
        # Test Vector Store Manager
        test_vector_store_manager(db, companies, deals, contacts)
        
        # Test ICP Analysis Engine
        test_icp_analysis_engine(db)
        
        logging.info("All tests completed successfully!")
    finally:
        db.close()


if __name__ == "__main__":
    main() 