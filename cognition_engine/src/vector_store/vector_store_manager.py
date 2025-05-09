import sys
import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

from sqlalchemy.orm import Session
from pydantic import BaseModel

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import settings
from src.vector_store.vector_client import ChromaVectorClient
from src.database.models import Company, Deal, Contact
from src.data_models.hubspot_entities import Company as CompanyModel, Deal as DealModel, Contact as ContactModel

logger = logging.getLogger(__name__)

class CRMDocument(BaseModel):
    """Representation of a CRM entity as a document for vector storage"""
    id: str
    text: str
    entity_type: str  # 'company', 'deal', or 'contact'
    entity_id: int
    metadata: Dict[str, Any] = {}

class VectorStoreManager:
    """Manager for integrating CRM data with vector storage"""
    
    def __init__(self, collection_name: str = None):
        """Initialize the vector store manager with a ChromaDB client"""
        self.collection_name = collection_name or settings.DEFAULT_CHROMA_COLLECTION
        self.vector_client = ChromaVectorClient(collection_name=self.collection_name)
        
    def company_to_document(self, company: Union[CompanyModel, Company]) -> CRMDocument:
        """Convert a company to a document for vector storage"""
        if isinstance(company, Company):
            # Convert ORM object to Pydantic model for consistent handling
            company_data = {
                "company_id": company.company_id,
                "company_name": company.company_name,
                "industry": company.industry,
                "geography": company.geography,
                "employee_count": company.employee_count,
                "annual_revenue": company.annual_revenue
                # Fields like domain, founded_year, description, is_public, website are not in CompanyOrm or Company Pydantic model
            }
            # Filter out None values before passing to Pydantic model
            company_data_filtered = {k: v for k, v in company_data.items() if v is not None}
            company_model = CompanyModel(**company_data_filtered)
        else:
            company_model = company # Assuming it's already a Pydantic Company model
        
        # Create a text representation of the company
        text = f"""Company: {company_model.company_name}
Industry: {company_model.industry}
Geography: {company_model.geography}
Employee Count: {company_model.employee_count}
Annual Revenue: {company_model.annual_revenue}
"""
        # Removed: Domain, Founded, Description from text as they are not in the model
        
        # Create metadata for filtering and retrieval
        metadata = {
            "entity_type": "company",
            "company_id": company_model.company_id, # Use company_id from Pydantic model
            "industry": company_model.industry,
            "employee_count": str(company_model.employee_count) if company_model.employee_count is not None else None, # Was 'size'
            "geography": company_model.geography, # Was 'location'
            "revenue": str(company_model.annual_revenue) if company_model.annual_revenue is not None else None,
            # "is_public": company_model.is_public # Not in Pydantic model
        }
        # Filter out None metadata values
        metadata_filtered = {k: v for k, v in metadata.items() if v is not None}
        
        return CRMDocument(
            id=f"company_{company_model.company_id}", # Use company_id
            text=text,
            entity_type="company",
            entity_id=int(company_model.company_id) if company_model.company_id.isdigit() else hash(company_model.company_id), # entity_id is int. company_id is str.
            metadata=metadata_filtered
        )
    
    def deal_to_document(self, deal: Union[DealModel, Deal]) -> CRMDocument:
        """Convert a deal to a document for vector storage"""
        if isinstance(deal, Deal):
            # Convert ORM object to Pydantic model for consistent handling
            deal_data = {
                "deal_id": deal.deal_id,
                "company_id": deal.company_id,
                "deal_name": deal.deal_name, # Was 'name'
                "amount": deal.amount,
                "deal_stage": deal.deal_stage, # Was 'stage'
                "close_date": deal.close_date,
                "create_date": deal.create_date, # Was 'creation_date'
                "pipeline": deal.pipeline, # Mapped from 'deal_type'
                "buyer_persona": deal.buyer_persona,
                "use_case": deal.use_case,
                "closed_lost_reason": deal.closed_lost_reason, # Was 'lost_reason'
                # Derived metrics from DealOrm - ensure they are handled if needed by Pydantic model, or used directly if not
                # For now, assume Pydantic Deal model does not include these derived metrics explicitly
                # "deal_age_days": getattr(deal, "deal_age_days", None),
                # "velocity_days": getattr(deal, "velocity_days", None),
                # "icp_fit_signal_score": getattr(deal, "icp_fit_signal_score", None),
                # "is_dead_zone": getattr(deal, "is_dead_zone", None),
                # "is_price_sensitive": getattr(deal, "is_price_sensitive", None),
                # "classified_closed_lost_reason": getattr(deal, "classified_closed_lost_reason", None)
            }
            deal_data_filtered = {k: v for k, v in deal_data.items() if v is not None}
            deal_model = DealModel(**deal_data_filtered)
        else:
            deal_model = deal # Assuming it's already a Pydantic Deal model
        
        # Create a text representation of the deal
        text = f"""Deal: {deal_model.deal_name} 
Company ID: {deal_model.company_id}
Amount: {deal_model.amount}
Stage: {deal_model.deal_stage}
Create Date: {deal_model.create_date}
Close Date: {deal_model.close_date}
Pipeline: {deal_model.pipeline}
Buyer Persona: {deal_model.buyer_persona}
Use Case: {deal_model.use_case}
"""
        # Removed: Status, Description, Source, Priority from text
        if deal_model.deal_stage and "lost" in deal_model.deal_stage.lower() and deal_model.closed_lost_reason:
            text += f"Lost Reason: {deal_model.closed_lost_reason}\n"
            
        # Add derived metrics to text if they were part of ORM and are needed for vectorization
        # These are not on the Pydantic Deal model based on hubspot_entities.py
        # if hasattr(deal, "deal_age_days") and deal.deal_age_days is not None:
        #     text += f"Deal Age (days): {deal.deal_age_days}\n"
        # if hasattr(deal, "velocity_days") and deal.velocity_days is not None:
        #     text += f"Velocity (days): {deal.velocity_days}\n"
        # if hasattr(deal, "icp_fit_signal_score") and deal.icp_fit_signal_score is not None:
        #     text += f"ICP Fit Score: {deal.icp_fit_signal_score}\n"
        # if hasattr(deal, "classified_closed_lost_reason") and deal.classified_closed_lost_reason:
        #     text += f"Classified Lost Reason: {deal.classified_closed_lost_reason}\n"
            
        # Create metadata for filtering and retrieval
        metadata = {
            "entity_type": "deal",
            "deal_id": deal_model.deal_id,
            "company_id": deal_model.company_id,
            "deal_stage": deal_model.deal_stage, # Was 'stage'
            # "status": deal_model.status, # Not in Pydantic Deal model
            "amount": str(deal_model.amount) if deal_model.amount is not None else None,
            "pipeline": deal_model.pipeline, # Was 'deal_type'
            "buyer_persona": deal_model.buyer_persona,
            "use_case": deal_model.use_case,
            "closed_lost_reason": deal_model.closed_lost_reason,
        }
        
        # Add derived ORM metrics to metadata if available and needed
        # if hasattr(deal, "icp_fit_signal_score") and deal.icp_fit_signal_score is not None:
        #     metadata["icp_fit_score"] = str(deal.icp_fit_signal_score)
        # if hasattr(deal, "is_dead_zone") and deal.is_dead_zone is not None:
        #     metadata["is_dead_zone"] = str(deal.is_dead_zone)
        # if hasattr(deal, "is_price_sensitive") and deal.is_price_sensitive is not None:
        #     metadata["is_price_sensitive"] = str(deal.is_price_sensitive)
        # if hasattr(deal, "classified_closed_lost_reason") and deal.classified_closed_lost_reason:
        #     metadata["classified_lost_reason"] = deal.classified_closed_lost_reason
        
        metadata_filtered = {k: v for k, v in metadata.items() if v is not None}

        return CRMDocument(
            id=f"deal_{deal_model.deal_id}",
            text=text,
            entity_type="deal",
            entity_id=int(deal_model.deal_id) if deal_model.deal_id.isdigit() else hash(deal_model.deal_id), # deal_id is str
            metadata=metadata_filtered
        )
    
    def contact_to_document(self, contact: Union[ContactModel, Contact]) -> CRMDocument:
        """Convert a contact to a document for vector storage"""
        if isinstance(contact, Contact):
            # Convert ORM object to Pydantic model for consistent handling
            contact_data = {
                "contact_id": contact.contact_id,
                "company_id": contact.company_id,
                "first_name": contact.first_name,
                "last_name": contact.last_name,
                "email": contact.email,
                # Fields like phone, title, department, created_date, etc. are not in ContactOrm or Contact Pydantic model
            }
            contact_data_filtered = {k: v for k, v in contact_data.items() if v is not None}
            contact_model = ContactModel(**contact_data_filtered)
        else:
            contact_model = contact # Assuming it's already a Pydantic Contact model
        
        # Create a text representation of the contact
        text = f"""Contact: {contact_model.first_name or ""} {contact_model.last_name or ""}
Email: {contact_model.email}
Company ID: {contact_model.company_id}
"""
        # Removed: Phone, Title, Department, Lead Source, Lead Status, Lifecycle Stage, Created Date, Last Activity
        
        # Create metadata for filtering and retrieval
        metadata = {
            "entity_type": "contact",
            "contact_id": contact_model.contact_id,
            "company_id": contact_model.company_id,
            "email": contact_model.email
            # Removed: department, title, lead_status, lifecycle_stage
        }
        metadata_filtered = {k: v for k, v in metadata.items() if v is not None}
        
        return CRMDocument(
            id=f"contact_{contact_model.contact_id}",
            text=text,
            entity_type="contact",
            entity_id=int(contact_model.contact_id) if contact_model.contact_id.isdigit() else hash(contact_model.contact_id), # contact_id is str
            metadata=metadata_filtered
        )
    
    def index_companies(self, companies: List[Union[CompanyModel, Company]]) -> None:
        """Index a list of companies in the vector store"""
        documents = [self.company_to_document(company) for company in companies]
        self._index_documents(documents)
        
    def index_deals(self, deals: List[Union[DealModel, Deal]]) -> None:
        """Index a list of deals in the vector store"""
        documents = [self.deal_to_document(deal) for deal in deals]
        self._index_documents(documents)
        
    def index_contacts(self, contacts: List[Union[ContactModel, Contact]]) -> None:
        """Index a list of contacts in the vector store"""
        documents = [self.contact_to_document(contact) for contact in contacts]
        self._index_documents(documents)
    
    def _index_documents(self, documents: List[CRMDocument]) -> None:
        """Index a list of CRM documents in the vector store"""
        if not documents:
            return
        
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add documents to the vector store
        self.vector_client.add_documents(
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Indexed {len(documents)} documents in collection '{self.collection_name}'")
    
    def similar_companies(self, query: str, n_results: int = 5, 
                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find companies similar to the query text"""
        # Add entity type filter
        if filter_metadata is None:
            filter_metadata = {}
        filter_metadata["entity_type"] = "company"
        
        return self._query_similar(query, n_results, filter_metadata)
    
    def similar_deals(self, query: str, n_results: int = 5,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find deals similar to the query text"""
        # Add entity type filter
        if filter_metadata is None:
            filter_metadata = {}
        filter_metadata["entity_type"] = "deal"
        
        return self._query_similar(query, n_results, filter_metadata)
    
    def similar_contacts(self, query: str, n_results: int = 5,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find contacts similar to the query text"""
        # Add entity type filter
        if filter_metadata is None:
            filter_metadata = {}
        filter_metadata["entity_type"] = "contact"
        
        return self._query_similar(query, n_results, filter_metadata)
    
    def _query_similar(self, query: str, n_results: int = 5,
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query the vector store for documents similar to the query text"""
        results = self.vector_client.query_collection(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format the results
        formatted_results = []
        for idx, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            formatted_results.append({
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": 1.0 - dist  # Convert distance to similarity score
            })
            
        return formatted_results
    
    def get_entity_by_id(self, entity_type: str, entity_id: int, db: Session) -> Optional[Union[Company, Deal, Contact]]:
        """Get a database entity by its type and ID"""
        if entity_type == "company":
            return db.query(Company).filter(Company.id == entity_id).first()
        elif entity_type == "deal":
            return db.query(Deal).filter(Deal.id == entity_id).first()
        elif entity_type == "contact":
            return db.query(Contact).filter(Contact.id == entity_id).first()
        else:
            raise ValueError(f"Invalid entity type: {entity_type}")
    
    def search_crm_data(self, query: str, entity_types: List[str] = None, 
                         n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search across all CRM data types (companies, deals, contacts)
        
        Args:
            query: The search query text
            entity_types: List of entity types to include ('company', 'deal', 'contact')
            n_results: Maximum number of results to return
            
        Returns:
            List of matching documents with metadata and similarity scores
        """
        # If no entity types specified, search all
        if not entity_types:
            entity_types = ["company", "deal", "contact"]
            
        filter_metadata = {"entity_type": {"$in": entity_types}}
        return self._query_similar(query, n_results, filter_metadata)
    
    def index_all_crm_data(self, db: Session) -> None:
        """Index all CRM data from the database"""
        # Index companies
        companies = db.query(Company).all()
        self.index_companies(companies)
        logger.info(f"Indexed {len(companies)} companies")
        
        # Index deals
        deals = db.query(Deal).all()
        self.index_deals(deals)
        logger.info(f"Indexed {len(deals)} deals")
        
        # Index contacts
        contacts = db.query(Contact).all()
        self.index_contacts(contacts)
        logger.info(f"Indexed {len(contacts)} contacts")
        
        logger.info(f"Completed indexing of all CRM data")


# Simple test function to validate the module
def test_vector_store_manager():
    """Test the VectorStoreManager with some sample data"""
    manager = VectorStoreManager(collection_name="test_crm_collection")
    
    # Create sample company
    company = CompanyModel(
        id=1,
        name="Acme Corporation",
        domain="acme.com",
        industry="Technology",
        size="Enterprise",
        location="San Francisco, CA",
        founded_year=1985,
        description="Acme Corporation is a fictional company that manufactures everything from portable holes to earthquake pills.",
        annual_revenue=10000000,
        is_public=True,
        website="https://acme.com"
    )
    
    # Create sample deal
    deal = DealModel(
        id=1,
        company_id=1,
        name="Enterprise License",
        amount=50000,
        stage="Qualified",
        close_date="2023-06-30",
        creation_date="2023-01-15",
        status="open",
        description="Enterprise license agreement for Acme Corporation",
        source="Referral",
        lost_reason=None,
        deal_type="New Business",
        priority="High"
    )
    
    # Create sample contact
    contact = ContactModel(
        id=1,
        company_id=1,
        first_name="John",
        last_name="Doe",
        email="john.doe@acme.com",
        phone="555-123-4567",
        title="CTO",
        department="Engineering",
        created_date="2023-01-10",
        last_activity="2023-05-20",
        lead_source="Website",
        lead_status="Qualified",
        lifecycle_stage="Customer"
    )
    
    # Test document conversions
    company_doc = manager.company_to_document(company)
    deal_doc = manager.deal_to_document(deal)
    contact_doc = manager.contact_to_document(contact)
    
    print(f"Company Document ID: {company_doc.id}")
    print(f"Deal Document ID: {deal_doc.id}")
    print(f"Contact Document ID: {contact_doc.id}")
    
    # Test indexing
    manager.index_companies([company])
    manager.index_deals([deal])
    manager.index_contacts([contact])
    
    # Test querying
    results = manager.search_crm_data("Technology company enterprise", n_results=2)
    
    print("\nSearch Results:")
    for idx, result in enumerate(results):
        print(f"Result {idx+1}:")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Entity Type: {result['metadata']['entity_type']}")
        print(f"  Document: {result['document'][:100]}...")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_vector_store_manager() 