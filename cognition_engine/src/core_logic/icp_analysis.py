import sys
import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
from collections import Counter

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.config import settings
from src.llm.llm_client import OllamaLLMClient
from src.vector_store.vector_store_manager import VectorStoreManager
from src.database.models import Company, Deal, Contact

logger = logging.getLogger(__name__)


class ICPCriterion(BaseModel):
    """A single criterion for ICP definition"""
    field: str
    values: List[str]
    weight: float
    description: str


class ICPDefinition(BaseModel):
    """Definition of an Ideal Customer Profile"""
    name: str
    description: str
    criteria: List[ICPCriterion]
    created_at: str = datetime.now().isoformat()
    updated_at: str = datetime.now().isoformat()


class ICPScoredEntity(BaseModel):
    """An entity with an ICP fit score"""
    entity_type: str
    entity_id: int
    score: float
    match_details: Dict[str, Any]
    

class ICPInsight(BaseModel):
    """An insight about ICP related patterns"""
    title: str
    description: str
    supporting_evidence: List[Dict[str, Any]]
    confidence: float
    generated_at: str = datetime.now().isoformat()


class ICPAnalysisEngine:
    """Engine for analyzing CRM data to generate ICP insights"""
    
    def __init__(self, vector_collection_name: str = None):
        """Initialize the ICP Analysis Engine"""
        self.vector_store_manager = VectorStoreManager(collection_name=vector_collection_name)
        self.llm_client = OllamaLLMClient()
        
    def _get_successful_deals(self, db: Session) -> List[Deal]:
        """Get all successful deals from the database"""
        return db.query(Deal).filter(Deal.deal_stage == "Closed Won").all()
    
    def _get_unsuccessful_deals(self, db: Session) -> List[Deal]:
        """Get all unsuccessful deals from the database"""
        return db.query(Deal).filter(Deal.deal_stage == "Closed Lost").all()
    
    def _get_companies_with_successful_deals(self, db: Session) -> List[Company]:
        """Get all companies with successful deals"""
        # Get all successful deals
        successful_deals = self._get_successful_deals(db)
        
        # Extract unique company IDs
        company_ids = set(deal.company_id for deal in successful_deals)
        
        # Get companies using company_id
        return db.query(Company).filter(Company.company_id.in_(company_ids)).all()
    
    def _extract_common_attributes(self, companies: List[Company]) -> Dict[str, Counter]:
        """Extract common attributes from a list of companies"""
        # Use actual Company fields and desired analysis keys
        attributes_to_analyze = {
            "industry": "industry",
            "employee_count_bucket": "employee_count", # Key for analysis vs ORM field name
            "geography": "geography",
            "annual_revenue_bucket": "annual_revenue"
            # Removed: founded_year, is_public, size, location
        }
        
        analysis_counters = {key: Counter() for key in attributes_to_analyze.keys()}

        # Count occurrences of each value
        for company in companies:
            for analysis_key, orm_field in attributes_to_analyze.items():
                value = getattr(company, orm_field, None) # Use getattr with default None
                
                if value is not None:
                    # Bucket numeric fields
                    if analysis_key == "annual_revenue_bucket":
                        if value < 1000000: bucket = "< $1M"
                        elif value < 10000000: bucket = "$1M - $10M"
                        elif value < 100000000: bucket = "$10M - $100M"
                        elif value < 1000000000: bucket = "$100M - $1B"
                        else: bucket = "> $1B"
                        analysis_counters[analysis_key][bucket] += 1
                    elif analysis_key == "employee_count_bucket":
                        # Bucket employee count (adjust ranges as needed)
                        if value <= 50: bucket = "1-50"
                        elif value <= 200: bucket = "51-200"
                        elif value <= 500: bucket = "201-500"
                        elif value <= 1000: bucket = "501-1000"
                        elif value <= 5000: bucket = "1001-5000"
                        else: bucket = "5000+"
                        analysis_counters[analysis_key][bucket] += 1
                    else: # Handle categorical fields like industry, geography
                        analysis_counters[analysis_key][str(value)] += 1 # Ensure value is string for Counter key
        
        return analysis_counters
    
    def generate_icp_definition_from_data(self, db: Session) -> ICPDefinition:
        """Generate an ICP definition from successful deals and companies"""
        # Get companies with successful deals
        successful_companies = self._get_companies_with_successful_deals(db)
        
        if not successful_companies:
            raise ValueError("No successful companies found in the database.")
        
        # Extract common attributes using the updated method
        attribute_counters = self._extract_common_attributes(successful_companies)
        
        # Generate criteria from attributes
        criteria = []
        for field, counter in attribute_counters.items():
            if not counter:
                continue
                
            # Get the top values (up to 3) for each field
            top_values = [value for value, _ in counter.most_common(3)]
            
            # Calculate the weight based on how dominant the top values are
            total = sum(counter.values())
            top_values_count = sum(counter[value] for value in top_values)
            weight = min(0.9, top_values_count / total) if total > 0 else 0 # Cap at 0.9, handle total=0
            
            # Add criterion - Use the analysis key (e.g., 'employee_count_bucket') as the field name
            criterion = ICPCriterion(
                field=field, # Use the key from attribute_counters (e.g., 'employee_count_bucket')
                values=top_values,
                weight=weight,
                description=f"Top {field} values in successful customers"
            )
            criteria.append(criterion)
        
        # Generate ICP description using LLM
        prompt = f"""Based on the following criteria extracted from successful customer data, write a concise description of the Ideal Customer Profile (ICP) in 2-3 sentences:

{json.dumps([c.dict() for c in criteria], indent=2)}

Description:"""
        
        # Use the correct method name and remove non-standard invoke params
        description_raw = self.llm_client.generate_text(
            prompt=prompt
            # max_tokens=150 - Not standard invoke param
        )
        description = description_raw.strip() if description_raw else "Could not generate description."
        
        return ICPDefinition(
            name="Data-Driven ICP",
            description=description,
            criteria=criteria
        )
    
    def score_company_against_icp(self, company: Company, icp: ICPDefinition) -> ICPScoredEntity:
        """Score a company against an ICP definition"""
        score = 0.0
        total_weight = 0.0
        match_details = {}
        
        # Map analysis keys back to ORM fields for fetching values
        analysis_key_to_orm_field = {
            "industry": "industry",
            "employee_count_bucket": "employee_count", 
            "geography": "geography",
            "annual_revenue_bucket": "annual_revenue"
        }

        for criterion in icp.criteria:
            analysis_key = criterion.field
            orm_field = analysis_key_to_orm_field.get(analysis_key)
            
            # If the analysis key doesn't map to a known ORM field, skip (shouldn't happen with current setup)
            if not orm_field:
                continue
                
            company_raw_value = getattr(company, orm_field, None)
            
            # Skip if the company doesn't have this attribute
            if company_raw_value is None:
                continue
                
            company_processed_value = None
            field_match = False
            
            # Handle bucketing for specific analysis keys
            if analysis_key == "annual_revenue_bucket":
                if company_raw_value < 1000000: bucket = "< $1M"
                elif company_raw_value < 10000000: bucket = "$1M - $10M"
                elif company_raw_value < 100000000: bucket = "$10M - $100M"
                elif company_raw_value < 1000000000: bucket = "$100M - $1B"
                else: bucket = "> $1B"
                company_processed_value = bucket
                field_match = company_processed_value in criterion.values
            elif analysis_key == "employee_count_bucket":
                if company_raw_value <= 50: bucket = "1-50"
                elif company_raw_value <= 200: bucket = "51-200"
                elif company_raw_value <= 500: bucket = "201-500"
                elif company_raw_value <= 1000: bucket = "501-1000"
                elif company_raw_value <= 5000: bucket = "1001-5000"
                else: bucket = "5000+"
                company_processed_value = bucket
                field_match = company_processed_value in criterion.values
            else: # Handle direct comparison for categorical fields
                company_processed_value = str(company_raw_value)
                field_match = company_processed_value in criterion.values
            
            # Calculate weighted score contribution
            criterion_score = 1.0 if field_match else 0.0
            weighted_score = criterion_score * criterion.weight
            
            score += weighted_score
            total_weight += criterion.weight
            
            # Add match details
            match_details[analysis_key] = { # Use analysis_key for details dict
                "value": str(company_raw_value), # Show original value
                "processed_value": company_processed_value, # Show bucket/string value used for matching
                "icp_values": criterion.values,
                "is_match": field_match,
                "weight": criterion.weight
            }
        
        # Normalize score
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0.0
            
        # Use company.company_id and handle int conversion for entity_id
        entity_id_int = int(company.company_id) if company.company_id.isdigit() else hash(company.company_id)

        return ICPScoredEntity(
            entity_type="company",
            entity_id=entity_id_int,
            score=final_score,
            match_details=match_details
        )
    
    def batch_score_companies(self, companies: List[Company], icp: ICPDefinition) -> List[ICPScoredEntity]:
        """Score multiple companies against an ICP definition"""
        return [self.score_company_against_icp(company, icp) for company in companies]
    
    def analyze_deal_patterns(self, db: Session) -> List[ICPInsight]:
        """Analyze deal patterns to generate insights"""
        # Get successful and unsuccessful deals
        successful_deals = self._get_successful_deals(db)
        unsuccessful_deals = self._get_unsuccessful_deals(db)
        
        if not successful_deals:
            # Return an empty list or a specific insight indicating no data
            # raise ValueError("No successful deals found in the database.")
             return [
                ICPInsight(
                    title="No Successful Deals",
                    description="Cannot analyze deal patterns as no successful deals were found.",
                    supporting_evidence=[],
                    confidence=0.0
                )
            ]           
        
        # Prepare deal data for LLM analysis (Limit size for prompt)
        MAX_DEALS_FOR_LLM = 20
        successful_deal_data = [
            {
                "deal_id": deal.deal_id,
                "amount": deal.amount,
                "deal_stage": deal.deal_stage,
                "velocity_days": getattr(deal, "velocity_days", None), # Use getattr for safety
                # "source": deal.source, # Not in DealOrm
                "pipeline": deal.pipeline, # Was deal_type
                # "priority": deal.priority, # Not in DealOrm
                "company_id": deal.company_id,
                "buyer_persona": deal.buyer_persona,
                "use_case": deal.use_case
            }
            for deal in successful_deals[:MAX_DEALS_FOR_LLM]
        ]
        
        unsuccessful_deal_data = [
            {
                "deal_id": deal.deal_id,
                "amount": deal.amount,
                "deal_stage": deal.deal_stage,
                "velocity_days": getattr(deal, "velocity_days", None), # Use getattr for safety
                # "source": deal.source, # Not in DealOrm
                "pipeline": deal.pipeline, # Was deal_type
                # "priority": deal.priority, # Not in DealOrm
                "company_id": deal.company_id,
                "closed_lost_reason": deal.closed_lost_reason, # Was lost_reason
                "classified_lost_reason": getattr(deal, "classified_closed_lost_reason", None), # Use getattr
                "buyer_persona": deal.buyer_persona,
                "use_case": deal.use_case
            }
            for deal in unsuccessful_deals[:MAX_DEALS_FOR_LLM]
        ]
        
        # Get company data for context
        company_ids = set()
        for deal in successful_deals[:MAX_DEALS_FOR_LLM] + unsuccessful_deals[:MAX_DEALS_FOR_LLM]:
            company_ids.add(deal.company_id)
            
        companies = db.query(Company).filter(Company.company_id.in_(company_ids)).all()
        company_data = {
            company.company_id: { # Use company_id as key
                "company_id": company.company_id,
                "company_name": company.company_name, # Was name
                "industry": company.industry,
                "employee_count": company.employee_count, # Was size
                "geography": company.geography # Was location
            }
            for company in companies
        }
        
        # Generate insights using LLM
        prompt = f"""As a sales analytics expert, analyze the following deal data to identify 3-5 key insights about what differentiates successful deals from unsuccessful ones. Focus on patterns related to ideal customer profiles.

Successful Deals:
{json.dumps(successful_deal_data, indent=2)}

Unsuccessful Deals:
{json.dumps(unsuccessful_deal_data, indent=2)}

Company Information:
{json.dumps(company_data, indent=2)}

Please structure each insight with:
1. A title (one sentence)
2. A detailed description (2-3 sentences)
3. Supporting evidence from the data (as a LIST of JSON objects/dictionaries, where each object might contain keys like 'deal_id', 'company_id', 'attribute', 'value', or 'reasoning')
4. Confidence level (0.0 to 1.0)

Format your response as a JSON array of insights.
"""
        
        # Use the correct method name and remove non-standard invoke params
        # Temperature override might be possible via kwargs if needed, but client already has temp set.
        insights_json_raw = self.llm_client.generate_text(
            prompt=prompt
            # max_tokens=1500 - Not standard invoke param
            # temperature=0.2 - Temperature usually set at client init
        )
        insights_json = insights_json_raw.strip() if insights_json_raw else '[]' # Default to empty JSON array
        
        # Parse insights JSON
        try:
            raw_insights = json.loads(insights_json)
            insights = []
            for raw_insight in raw_insights:
                insight = ICPInsight(
                    title=raw_insight.get("title", "Untitled Insight"),
                    description=raw_insight.get("description", "No description provided"),
                    supporting_evidence=raw_insight.get("supporting_evidence", []),
                    confidence=raw_insight.get("confidence", 0.5)
                )
                insights.append(insight)
            return insights
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse insights JSON: {e}")
            logger.debug(f"Raw LLM output: {insights_json}")
            # Return a fallback insight
            return [
                ICPInsight(
                    title="Analysis Error",
                    description="The system encountered an error while analyzing the deal patterns. Please try again later.",
                    supporting_evidence=[],
                    confidence=0.0
                )
            ]
    
    def generate_vector_enhanced_icp_insights(self, db: Session) -> List[Dict[str, Any]]:
        """Generate ICP insights enhanced with vector similarity analysis"""
        # Ensure CRM data is indexed in the vector store
        self.vector_store_manager.index_all_crm_data(db)
        
        # Get successful and unsuccessful deals
        successful_deals = self._get_successful_deals(db)
        unsuccessful_deals = self._get_unsuccessful_deals(db)
        
        # Create semantic queries to find patterns
        queries = [
            "high-value enterprise deals with fast closing velocity",
            "deals with technology companies in the enterprise segment",
            "deals that were lost due to pricing concerns",
            "successful deals with financial services companies",
            "deals with manufacturing companies"
        ]
        
        # Run vector searches for each query
        insights = []
        for query in queries:
            # Search across all data
            results = self.vector_store_manager.search_crm_data(query, n_results=10)
            
            # Count entity types in results
            entity_type_counts = Counter()
            for result in results:
                entity_type = result["metadata"].get("entity_type")
                if entity_type:
                    entity_type_counts[entity_type] += 1
            
            # Extract key metadata patterns
            metadata_patterns = {}
            for result in results:
                metadata = result["metadata"]
                for key, value in metadata.items():
                    if key == "entity_type":
                        continue
                    if key not in metadata_patterns:
                        metadata_patterns[key] = Counter()
                    metadata_patterns[key][value] += 1
            
            # Find dominant patterns (top value for each metadata field)
            dominant_patterns = {}
            for key, counter in metadata_patterns.items():
                if counter:
                    top_value, count = counter.most_common(1)[0]
                    # Only include if it appears in at least 30% of results
                    if count >= 0.3 * len(results):
                        dominant_patterns[key] = top_value
            
            # Calculate average similarity score
            avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0
            
            # Create insight
            insight = {
                "query": query,
                "dominant_patterns": dominant_patterns,
                "entity_distribution": dict(entity_type_counts),
                "average_similarity": avg_similarity,
                "top_examples": results[:3]  # Include top 3 examples
            }
            insights.append(insight)
        
        return insights
    
    def generate_unified_icp_report(self, db: Session) -> Dict[str, Any]:
        """Generate a comprehensive ICP report"""
        try:
            # Generate ICP definition
            icp_definition = self.generate_icp_definition_from_data(db)
            
            # Score all companies against the ICP
            all_companies = db.query(Company).all()
            scored_companies = self.batch_score_companies(all_companies, icp_definition)
            
            # Get top and bottom companies by score
            scored_companies.sort(key=lambda x: x.score, reverse=True)
            top_companies = scored_companies[:5]
            bottom_companies = scored_companies[-5:]
            
            # Analyze deal patterns
            deal_insights = self.analyze_deal_patterns(db)
            
            # Get vector-enhanced insights
            vector_insights = self.generate_vector_enhanced_icp_insights(db)
            
            # Compile the report
            report = {
                "icp_definition": icp_definition.dict(),
                "top_companies": [
                    {
                        "company_id": sc.entity_id,
                        "score": sc.score,
                        "match_details": sc.match_details
                    }
                    for sc in top_companies
                ],
                "bottom_companies": [
                    {
                        "company_id": sc.entity_id,
                        "score": sc.score,
                        "match_details": sc.match_details
                    }
                    for sc in bottom_companies
                ],
                "deal_insights": [insight.dict() for insight in deal_insights],
                "vector_insights": vector_insights,
                "report_generated_at": datetime.now().isoformat()
            }
            
            return report
        except Exception as e:
            logger.error(f"Error generating unified ICP report: {e}")
            return {
                "error": str(e),
                "report_generated_at": datetime.now().isoformat()
            }


# Test function to validate the module
def test_icp_analysis():
    """Test the ICPAnalysisEngine with sample data"""
    from src.database.database_setup import create_db_and_tables, get_db
    
    # Make sure tables are created
    create_db_and_tables()
    
    # Get session
    db = next(get_db())
    
    try:
        # Create engine
        engine = ICPAnalysisEngine()
        
        # Generate ICP definition
        print("Generating ICP definition...")
        icp = engine.generate_icp_definition_from_data(db)
        print(f"ICP Name: {icp.name}")
        print(f"ICP Description: {icp.description}")
        print(f"Number of criteria: {len(icp.criteria)}")
        
        # Score a company
        print("\nScoring companies against ICP...")
        companies = db.query(Company).limit(2).all()
        for company in companies:
            score = engine.score_company_against_icp(company, icp)
            print(f"Company ID {company.id} ({company.name}): Score = {score.score:.2f}")
        
        # Generate insights
        print("\nGenerating deal insights...")
        insights = engine.analyze_deal_patterns(db)
        for i, insight in enumerate(insights):
            print(f"Insight {i+1}: {insight.title}")
            print(f"Confidence: {insight.confidence:.2f}")
            print(f"Description: {insight.description}")
            print()
        
        print("Test completed successfully!")
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_icp_analysis() 