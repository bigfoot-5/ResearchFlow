import psycopg2
import psycopg2.extras
import os
import json
import asyncio
from typing import Dict, List, Any, Sequence, AsyncGenerator, Optional
from datetime import datetime, timedelta
import hashlib

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import Autogen components
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage, UserMessage, BaseAgentEvent
from autogen_core import CancellationToken

# Import for Vector DB and Embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
# Import OllamaEmbeddings and ChatOllama from langchain_community or langchain_ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Import OpenAI components
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# --- Database Configuration ---
# Ensure these environment variables are set or replace with your actual config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://cognition_user:cognition_password@localhost:5432/cognition_db")
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "cognition_db"),
    "user": os.getenv("DB_USER", "cognition_user"),
    "password": os.getenv("DB_PASSWORD", "cognition_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "deals-index")
PINECONE_NAMESPACE = "all_deals_context"

# --- Dimension Values and Threshold ---
# These are the possible values the agents can use for filtering
HS_ANALYTICS_SOURCES = [
    "PAID_SOCIAL",
    "EMAIL_MARKETING",
    "OTHER_CAMPAIGNS",
    "PAID_SEARCH",
    "ORGANIC_SEARCH",
    "REFERRALS",    
    "OFFLINE",
    "DIRECT_TRAFFIC",
    "SOCIAL_MEDIA"
]

JOB_TITLES = [
    "Partner",
    "Head of Business Development Europe",
    "Head of Talent",
    "Salesperson",
    "Global Talent Lead",
    "CEO",
    "CHRO",
    "HR Director",
    "VP People",
    "Executive Chairperson",
    "People Analytics Lead"
]

INDUSTRIES = [
            "Database & File Management Software", "Education", "Agriculture", "TELECOMMUNICATIONS", "Physicians Clinics", "Repair Services", "Motor Vehicles", "Security Software", "CONSUMER_SERVICES", "INSURANCE", "Grocery Retail", "Sporting Goods", "Record, Video & Book Stores", "Facilities Management & Commercial Cleaning", "Convenience Stores, Gas Stations & Liquor Stores", "PUBLISHING", "DEFENSE_SPACE", "MEDICAL_PRACTICE", "INFORMATION_SERVICES", "THINK_TANKS", "Industrial Machinery & Equipment", "DESIGN", "Computer Equipment & Peripherals", "Organizations", "INDUSTRIAL_AUTOMATION", "OUTSOURCING_OFFSHORING", "Government", "Finance", "Venture Capital & Private Equity", "Animals & Livestock", "K-12 Schools", "Professional Services", "DAIRY", "SUPERMARKETS", "Gambling & Gaming", "Electronics", "Home Improvement & Hardware Retail", "BIOTECHNOLOGY", "HIGHER_EDUCATION", "Automotive Parts", "Glass & Clay", "Sporting & Recreational Equipment Retail", "PRINTING", "GRAPHIC_DESIGN", "WHOLESALE", "Architecture, Engineering & Design", "PERFORMING_ARTS", "INFORMATION_TECHNOLOGY_AND_SERVICES", "GOVERNMENT_ADMINISTRATION", "PACKAGING_AND_CONTAINERS", "Security Products & Services", "ENVIRONMENTAL_SERVICES", "Chemicals & Related Products", "Motor Vehicle Manufacturing", "Civil Engineering Construction", "JUDICIARY", "RESTAURANTS", "Management Consulting", "Medical Laboratories & Imaging Centers", "COMPUTER_HARDWARE", "CHEMICALS", "Ticket Sales", "Training", "MACHINERY", "MEDICAL_DEVICES", "AUTOMOTIVE", "Cleaning Services", "Software", "PROFESSIONAL_TRAINING_COACHING", "LAW_ENFORCEMENT", "Energy and Utilities", "CONSTRUCTION", "ARTS_AND_CRAFTS", "Water Treatment", "Call Centers & Business Centers", "Vitamins, Supplements & Health Stores", "Food Service", "Hospitals & Physicians Clinics", "Financial Software", "ELECTRICAL_ELECTRONIC_MANUFACTURING", "Watches & Jewelry", "Rail, Bus & Taxi", "Medical & Surgical Hospitals", "Tribal Nations", "Photographic & Optical Equipment", "Pet Products", "Pulp & Paper", "Storage & System Management Software", "Apparel & Accessories Retail", "Elderly Care Services", "Customer Relationship Management (CRM) Software", "Public Sector", "OIL_ENERGY", "Debt Collection", "Software Development", "Test & Measurement Equipment", "RETAIL", "FOOD_BEVERAGES", "EXECUTIVE_OFFICE", "PACKAGE_FREIGHT_DELIVERY", "WIRELESS", "RELIGIOUS_INSTITUTIONS", "MARKETING_AND_ADVERTISING", "COMPUTER_SOFTWARE", "Transportation", "Software and IT", "Museums & Art Galleries", "Information & Document Management", "BUILDING_MATERIALS", "Energy, Utilities & Waste", "Automobile Parts Stores", "MECHANICAL_OR_INDUSTRIAL_ENGINEERING", "Textiles & Apparel", "Electricity, Oil & Gas", "Manufacturing", "NON_PROFIT_ORGANIZATION_MANAGEMENT", "PHOTOGRAPHY", "LEGAL_SERVICES", "REAL_ESTATE", "Office Products Retail & Distribution", "Data Collection & Internet Portals", "LUXURY_GOODS_JEWELRY", "Aerospace & Defense", "Minerals & Mining", "Business Intelligence (BI) Software", "Freight & Logistics Services", "Blood & Organ Banks", "SPORTS", "COMPUTER_GAMES", "INTERNATIONAL_AFFAIRS", "Crops", "Automobile Dealers", "Wire & Cable", "Healthcare Services", "FINANCIAL_SERVICES", "Computer and Network Security", "Barber Shops & Beauty Salons", "AVIATION_AEROSPACE", "Fitness & Dance Facilities", "Oil & Gas Exploration & Services", "Local", "Medical Specialists", "Airlines, Airports & Air Services", "MUSIC", "Plastic, Packaging & Containers", "NEWSPAPERS", "Lodging & Resorts", "Non-Profit & Charitable Organizations", "Custom Software & IT Services", "Legal Software", "TRANSPORTATION_TRUCKING_RAILROAD", "State", "PRIMARY_SECONDARY_EDUCATION", "Tires & Rubber", "Transportation and Logistics", "Amusement Parks, Arcades & Attractions", "Hand, Power & Lawn-care Tools", "INTERNET", "Human Resources Services", "Dental Offices", "Healthcare Software", "GOVERNMENT_RELATIONS", "INVESTMENT_MANAGEMENT", "Appliances", "Travel Agencies & Services", "Networking Software", "Broadcasting", "HUMAN_RESOURCES", "Ambulance Services", "VENTURE_CAPITAL_PRIVATE_EQUITY", "BANKING", "Pharmaceutical Manufacturing", "GLASS_CERAMICS_CONCRETE", "TRANSLATION_AND_LOCALIZATION", "PAPER_FOREST_PRODUCTS", "Lending & Brokerage", "Department Stores, Shopping Centers & Superstores", "Other Rental Stores (Furniture, A/V, Construction & Industrial Equipment)", "Business Services", "HOSPITALITY", "Food & Beverage", "Consumer Electronics & Computers Retail", "Advertising & Marketing", "RENEWABLES_ENVIRONMENT", "Newspapers & News Services", "Accounting Services", "ENTERTAINMENT", "ALTERNATIVE_MEDICINE", "Telephony & Wireless", "Telecommunication Equipment", "FUND_RAISING", "Colleges & Universities", "Drug Stores & Pharmacies", "LEGISLATIVE_OFFICE", "COMPUTER_NETWORK_SECURITY", "ACCOUNTING", "MINING_METALS", "Multimedia, Games & Graphics Software", "Commercial Printing", "Religious Organizations", "LEISURE_TRAVEL_TOURISM", "WRITING_AND_EDITING", "HR & Staffing", "Real Estate", "Healthcare", "Boats & Submarines", "UTILITIES", "TOBACCO", "Medical Devices & Equipment", "Internet Service Providers, Website Hosting & Internet-related Services", "Jewelry & Watch Retail", "Holding Companies & Conglomerates", "PUBLIC_RELATIONS_AND_COMMUNICATIONS", "Law Firms & Legal Services", "Research & Development", "Childcare", "GAMBLING_CASINOS", "STAFFING_AND_RECRUITING", "Human Resources Software", "CONSUMER_GOODS", "Photography Studio", "Membership Organizations", "IT Services and IT Consulting", "APPAREL_FASHION", "SHIPBUILDING", "Consumer Services", "Technology, Information and Internet", "Chambers of Commerce", "MARKET_RESEARCH", "Engineering Software", "Cleaning Products", "COSMETICS", "Automotive Service & Collision Repair", "Investment Banking", "Cosmetics, Beauty Supply & Personal Care Products", "Building Materials", "EDUCATION_MANAGEMENT", "Enterprise Resource Planning (ERP) Software", "Credit Cards & Transaction Processing", "Media & Internet", "Flowers, Gifts & Specialty Stores", "AIRLINES_AVIATION", "Retail and Consumer", "FACILITIES_SERVICES", "Weight & Health Management", "MANAGEMENT_CONSULTING", "CIVIC_SOCIAL_ORGANIZATION", "Cultural & Informational Centers", "SEMICONDUCTORS", "SECURITY_AND_INVESTIGATIONS"
        ]

# New dimensions
COUNTRIES = [
    "Andorra",
    "Billericay, United Kingdom",
    "Indonesia",
    "Bangladesh",
    "Atlanta, Georgia",
    "UAE",
    "Luxembourg",
    "Czech Republic",
    "Sweden",
    "Unknown",
    "San Francisco, California",
    "Montenegro",
    "Jordan",
    "Ireland",
    "Singapore",
    "San Marino",
    "London, United Kingdom",
    "Morristown, New Jersey",
    "United KIngdom",
    "Portugal",
    "Finland",
    "Levallois-Perret, France",
    "Colombia",
    "Malta",
    "Cayman Islands",
    "Ukraine",
    "Saudi Arabia",
    "11, Allée de l'arche, Tour Egée, Courbevoie, Ile-de-France 92400, FR",
    "Latvia",
    "Iran, Islamic Republic of",
    "England,United Kingdom",
    "France",
    "Israel",
    "Madrid, Spain",
    "Cardiff",
    "Netherland",
    "København, Denmark",
    "United states",
    "Malaysia",
    "Kenya",
    "Iceland",
    "Bologna, Italy",
    "Kuwait",
    "Taiwan, Province of China",
    "Hong Kong",
    "Germany",
    "Kuala Lumpur",
    "Philippines",
    "United States",
    "Cyprus",
    "Turkey",
    "Nigeria",
    "Zimbabwe",
    "Boston, Massachusetts",
    "London, England",
    "China",
    "Toronto, Ontario",
    "Belarus",
    "Qatar",
    "Netherlands",
    "England, United Kingdom",
    "Congo, the Democratic Republic of the",
    "UK",
    "Australia",
    "Serbia",
    "Shinagawa, Tokyo",
    "Burton upon Trent, England",
    "Bahrain",
    "Spain",
    "United Arab Emirates",
    "Chicago, Illinois",
    "Belgium",
    "Marousi, Attica",
    "Guernsey",
    "Leeds, United Kingdom",
    "San Jose, California",
    "Miami, Florida",
    "Taiwan",
    "Republic of Korea",
    "Northern Europe",
    "Holland",
    "Thailand",
    "United Kingdom",
    "Perth",
    "El Salvador",
    "Italy",
    "Uruguay",
    "Oman",
    "Istanbul, Turkey",
    "USA",
    "United Kingdom",
    "Macedonia, the former Yugoslav Republic of",
    "Minneapolis, MN",
    "Germany",
    "Ohio",
    "VietNam",
    "Canada",
    "Brussels, Brussels Region",
    "Burgess Hill",
    "Newport, Shropshire",
    "Loughborough, Leicestershire",
    "ireland",
    "Bermuda",
    "Argentina",
    "Nottingham, United Kingdom",
    "Unites States",
    "NETHERLANDS",
    "Liechtenstein",
    "Sevenoaks",
    "England",
    "Roma, Italy",
    "Azerbaijan",
    "Slovenia",
    "Greece",
    "Egypt",
    "United Kindgom",
    "Puerto Rico",
    "India",
    "Iran",
    "Chile",
    "Buckden",
    "Estonia",
    "Korea, Republic of",
    "South Africa",
    "Boston",
    "Troy, Michigan",
    "Isle of Man",
    "Japan",
    "Taipei City, Taiwan",
    "Denmark",
    "United kingdom",
    "Crowborough, United States",
    "Jamaica",
    "Trinidad and Tobago",
    "Ashburn",
    "Bath, United Kingdom",
    "United States of America",
    "Delaware",
    "Hongkong",
    "Switzerland",
    "Ecuador",
    "London",
    "New Zealand",
    "Austalia",
    "Hungary",
    "Russia",
    "Mannheim, Germany",
    "Newcastle, New South Wales",
    "Norway",
    "Russian Federation",
    "Pakistan",
    "Romania",
    "Crawley, England",
    "Brazil",
    "Austria",
    "Dubai, United Arab Emirates",
    "Guatemala",
    "Guinea-Bissau",
    "Business Park Abu Dhabi Abu Dhabi 6316 AE",
    "Panama",
    "Lithuania",
    "Bulgaria",
    "Holywell Green, Halifax, England",
    "Croatia",
    "Tunisia",
    "Morocco",
    "GB",
    "Europe",
    "Wokingham, Berkshire",
    "Via Marco d'Aviano Milan Lombardy 20131 IT",
    "Lausanne",
    "Nicaragua",
    "Mexico",
    "Poland",
    "EU",
    "Lebanon",
    "Poway, CA",
    "Jersey",
    "Klarabergsviadukten 63 Stockholm SE",
    "Gibraltar",
    "Tel Aviv-Yafo, Israel",
    "united Kingdom",
]

EMPLOYEE_BUCKETS = [
    "1-1000",
    "5000+",
    "4001-5000",
    "2001-3000",
    "1001-2000",
    "3001-4000",
]

MIN_DEAL_COUNT = 5 # Minimum deals for a segment to be considered statistically significant

# Global variables to hold initialized Pinecone components
pinecone_index = None
embeddings_model = None

print("Attempting to initialize Pinecone and embeddings...")
try:
    print(f"[INIT] Initializing Pinecone client...")
    # Initialize Pinecone client with new method
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("[INIT] Pinecone client initialized successfully.")

    # Initialize embeddings model
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Attempt to initialize OpenAI embeddings first
    if openai_api_key:
        openai_embedding_model_name = "text-embedding-ada-002"
        print(f"[INIT] Attempting to initialize OpenAI embeddings with model '{openai_embedding_model_name}'...")
        try:
            embeddings_model = OpenAIEmbeddings(api_key=openai_api_key, model=openai_embedding_model_name)
            # Perform a small test invocation to check if it works
            embeddings_model.embed_documents(["test"])
            print("[INIT] Successfully initialized OpenAI embeddings.")
        except Exception as e:
            print(f"[INIT] Failed to initialize OpenAI embeddings: {e}")
            embeddings_model = None
    else:
        print("[INIT] OPENAI_API_KEY not found in environment variables. Skipping OpenAI embeddings initialization.")

    # If OpenAI initialization failed, attempt to initialize Ollama embeddings
    if embeddings_model is None:
        ollama_embedding_model_name = "qwen3:1.7b"
        print(f"[INIT] Attempting to initialize Ollama embeddings with model '{ollama_embedding_model_name}'...")
        try:
            embeddings_model = OllamaEmbeddings(model=ollama_embedding_model_name, base_url="http://localhost:11434")
            # Perform a small test invocation to check if it works
            embeddings_model.embed_documents(["test"])
            print("[INIT] Successfully initialized Ollama embeddings.")
        except Exception as e:
            print(f"[INIT] Failed to initialize Ollama embeddings: {e}")
            embeddings_model = None

    # If neither LLM initialized, raise an error
    if embeddings_model is None:
        raise RuntimeError("Failed to initialize both OpenAI and Ollama embedding models.")

    # Get or create Pinecone index
    print(f"[INIT] Getting Pinecone index: {PINECONE_INDEX_NAME}...")
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"[INIT] Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # OpenAI ada-002 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print(f"[INIT] Successfully accessed Pinecone index '{PINECONE_INDEX_NAME}'.")

except Exception as e:
    print(f"[INIT ERROR] An unexpected error occurred during Pinecone/Embeddings initialization: {e}")
    pinecone_index = None
    embeddings_model = None

def normalize_industry_name(industry: str) -> str:
    """Normalize industry names to match database values."""
    industry_mappings = {
        "CUSTOM_SOFTWARE_IT_SERVICES": "COMPUTER_SOFTWARE",
        "ENTERPRISE_RESOURCE_PLANNING": "COMPUTER_SOFTWARE",
        "GAMBLING_CASINOS": "GAMBLING_CASINOS",
        "SPORTS": "SPORTS",
        "ENERGY": "OIL_ENERGY",
        "OIL_ENERGY": "OIL_ENERGY",
        "HEALTHCARE": "HEALTHCARE",
        "MEDICAL_DEVICES": "MEDICAL_DEVICES",
        "RETAIL": "RETAIL",
        "ELECTRONICS": "ELECTRONICS",
        "ACCOUNTING": "ACCOUNTING",
        "LEGAL_SERVICES": "LEGAL_SERVICES",
        "COMPUTER_SOFTWARE": "COMPUTER_SOFTWARE",
        "INFORMATION_TECHNOLOGY_AND_SERVICES": "INFORMATION_TECHNOLOGY_AND_SERVICES"
    }
    return industry_mappings.get(industry.upper(), industry.upper())

# --- Helper function to build dynamic WHERE clause for a single filter combo ---
def build_single_filter_condition(filter_combo: Dict[str, List[str]]) -> str:
    """Builds SQL WHERE clause for a single filter combination."""
    print("\nDebug - Building filter condition for:")
    print(json.dumps(filter_combo, indent=2))
    
    conditions = []
    
    # Define the field mappings for SQL conditions
    field_mappings = {
        'hs_analytics_source': ('d.hs_analytics_source', 'source'),
        'job_title': ('ct.job_title', 'title'),
        'industry': ('c.industry', 'industry'),
        'country': ('c.country', 'country'),
        'employee_bucket': ('c.employee_bucket', 'bucket')
    }
    
    for field, (sql_field, debug_name) in field_mappings.items():
        if field in filter_combo:
            values = filter_combo[field]
            if values:  # Only add condition if there are values
                # Normalize industry names if this is the industry field
                if field == 'industry':
                    values = [normalize_industry_name(v) for v in values]
                quoted_values = ", ".join(f"'{v}'" for v in values)
                conditions.append(f"{sql_field} IN ({quoted_values})")
                print(f"Debug - Added {debug_name} condition: {sql_field} IN ({quoted_values})")

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    print("\nDebug - Generated WHERE clause:")
    print(where_clause)
    return where_clause

# --- Relevance Scoring Function ---
def calculate_relevance_score(deal_data: Dict[str, Any], filter_combo: Dict[str, List[str]]) -> float:
    """Calculates a relevance score for a deal against a filter combination based on predefined weights."""
    weights = {
        'industry': 0.30,
        'job_title': 0.25,
        'country': 0.15,
        'hs_analytics_source': 0.10,
        'employee_bucket': 0.20,
    }

    score = 0.0
    for dimension, weight in weights.items():
        if dimension in filter_combo:
            filter_values = filter_combo[dimension]
            deal_value = deal_data.get(dimension)
            if deal_value is not None and any(str(deal_value).lower() == str(fv).lower() for fv in filter_values):
                score += weight
                print(f"Debug - Match found for {dimension}: {deal_value} in {filter_values}")

    return score

# --- Function to Fetch Open Deals ---
async def fetch_open_deals(db_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetches details for all open deals (not closedwon or closedlost)."""
    print("Fetching open deals...")
    conn = None
    cursor = None
    open_deals = []
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query = f"""
            SELECT
                d.id AS deal_id,
                d.deal_name,
                d.amount,
                d.dealstage,
                d.closedate,
                c.company_id,
                d.company_name,
                c.industry,
                c.country,
                c.employee_bucket,
                ct.job_title
            FROM dim_deals d
            JOIN dim_companies c ON d.company_id = c.company_id
            LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
            WHERE d.dealstage NOT IN ('closedwon', 'closedlost');
        """
        cursor.execute(query)
        open_deals = cursor.fetchall()
        print(f"Fetched {len(open_deals)} open deals.")

    except Exception as e:
        print(f"Error fetching open deals: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return open_deals

# --- Filter Prediction Agent ---
class FilterPredictionAgent(BaseChatAgent):
    def __init__(self, name: str, openai_model: str = "gpt-4o", ollama_model: str = "gemma3:1b"):
        super().__init__(name=name, description="Agent that predicts high-performing filter combinations using LLM reasoning and vector DB context")
        self._llm = None
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key:
            print(f"Attempting to initialize OpenAI model '{openai_model}'...")
            try:
                self._llm = ChatOpenAI(api_key=openai_api_key, model=openai_model)
                print("Successfully initialized OpenAI model.")
            except Exception as e:
                print(f"Failed to initialize OpenAI model: {e}")
                self._llm = None
        else:
            print("OPENAI_API_KEY not found in environment variables. Skipping OpenAI initialization.")

        if self._llm is None:
            print(f"Attempting to initialize Ollama model '{ollama_model}'...")
            try:
                ollama_test_llm = ChatOllama(model=ollama_model, base_url="http://localhost:11434")
                ollama_test_llm.invoke("hello")
                self._llm = ollama_test_llm
                print("Successfully initialized Ollama model.")
            except Exception as e:
                print(f"Failed to initialize Ollama model: {e}")
                self._llm = None

        if self._llm is None:
            raise RuntimeError("Failed to initialize both OpenAI and Ollama models.")

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        try:
            global pinecone_index, embeddings_model
            if pinecone_index is None or embeddings_model is None:
                yield Response(
                    chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Vector database or embeddings not initialized."}), source=self.name),
                    inner_messages=[]
                )
                return

            rag_query = "Examples of deals with high win rate and good performance metrics."
            print(f"Querying vector DB for context: '{rag_query}'...")

            # Get embeddings for the query
            query_embedding = embeddings_model.embed_query(rag_query)

            # Query Pinecone
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=20,
                namespace=PINECONE_NAMESPACE,
                include_metadata=True
            )

            matches = results.matches
            if not matches:
                context = "No relevant deal examples found in the vector database."
                print(context)
            else:
                context = "Context from similar high-performing deals:\n\n"
                for i, match in enumerate(matches):
                    metadata = match.metadata
                    context += f"Deal {i+1}: {match.id}\n"
                    dealstage = str(metadata.get('dealstage', 'N/A'))
                    amount = float(metadata.get('amount', 0) or 0)
                    days_to_close = float(metadata.get('days_to_close', 0) or 0)
                    industry = str(metadata.get('industry', 'N/A'))
                    employees = str(metadata.get('numberofemployees', 'N/A'))
                    job_title = str(metadata.get('job_title', 'N/A'))
                    context += f"  - Dealstage: {dealstage}, Amount: ${amount:.2f}, Sales Cycle: {days_to_close:.2f} days\n"
                    context += f"  - Industry: {industry}, Employees: {employees}, Job Title: {job_title}\n"
                    context += "\n"
                print(f"Retrieved {len(matches)} deals for context.")

            # Initialize variables for retry logic
            max_retries = 7
            valid_segments = []
            retry_count = 0
            failed_filters = []  # Track failed filter combinations

            while len(valid_segments) < 5 and retry_count < max_retries:
                if retry_count > 0:
                    print(f"\nRetry {retry_count}: Attempting to find more valid segments...")
                    # Add feedback about previous attempts
                    feedback = f"\nPrevious attempt found {len(valid_segments)} valid segments. Need at least 5 segments with deals.\n"
                    if failed_filters:
                        feedback += "\nThe following filter combinations did not yield enough deals:\n"
                        for i, failed in enumerate(failed_filters[-3:], 1):  # Show last 3 failures
                            feedback += f"{i}. {json.dumps(failed, indent=2)}\n"
                    feedback += "\nTry using simpler combinations with fewer filters, focusing on the most common values in each dimension."
                    context += feedback

                # Step 2: Formulate prompt for the LLM with context and available dimensions
                prompt = f"""
Analyze the following examples of high-performing deals provided as context.
Based on these examples and your general knowledge about sales dynamics, predict 5 to 10 specific filter combinations (segments) that are likely to have high revenue velocity. Revenue velocity is calculated as (Win Rate ÷ Sales Cycle in Days) × Average Contract Value (ACV).

IMPORTANT: 
1. Each segment MUST have at least 5 deals in the database.
2. You don't need to use all dimensions in each filter - focus on the most important ones.
3. Use simpler combinations with fewer filters if needed.
4. Prioritize common values that are likely to exist in the data.

Consider combinations of the following dimensions and their possible values:

- **Industries**: {', '.join(INDUSTRIES)}
- **Job Titles**: {', '.join(JOB_TITLES)}
- **Analytics Sources**: {', '.join(HS_ANALYTICS_SOURCES)}
- **Countries**: {', '.join(COUNTRIES)}
- **Employee Buckets**: {', '.join(EMPLOYEE_BUCKETS)}

For each predicted combination, provide a brief 'reasoning' field explaining *why* you predict this combination will have high revenue velocity, referencing patterns observed in the context examples or general sales principles. Also, provide the 'filter' field with the specific values chosen for each dimension in that combination as lists. Ensure each filter combination is a dictionary with 'filter' (a dictionary of lists) and 'reasoning' (a string) keys.

Example JSON format for the list of predictions:
[
  {{
    "filter": {{
      "industry": ["TECHNOLOGY"],
      "job_title": ["CEO"]
    }},
    "reasoning": "Based on the context, deals in the technology sector with C-level contacts seem to close faster with higher amounts."
  }},
  {{
    "filter": {{
      "industry": ["PHARMACEUTICALS"],
      "employee_bucket": ["1001-5000"]
    }},
    "reasoning": "Pharmaceutical companies in this size range often represent specialized software needs."
  }},
  ...
]

Predicted Segments (JSON array):
"""
                # Add the context to the prompt for the LLM
                prompt_with_context = f"{context}\n\n{prompt}"

                # Print the full prompt being sent to the LLM for inspection
                print("--- START LLM PROMPT ---")
                print(prompt_with_context)
                print("--- END LLM PROMPT ---")

                # Use the LLM to generate the predicted filters based on context
                try:
                    # Langchain ChatOllama uses invoke
                    llm_response = self._llm.invoke(prompt_with_context)
                    print("Successfully invoked LLM.")
                    
                    # Parse the response
                    response_text = llm_response.content
                    print("\nDebug - Raw LLM Response:")
                    print(response_text)
                    
                    # Extract JSON array from response
                    import re
                    json_matches = list(re.finditer(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL))
                    if not json_matches:
                        print("Debug - No JSON array found in response")
                        continue
                    
                    # Use the last match (most recent complete response)
                    json_str = json_matches[-1].group(0)
                    print("\nDebug - Extracted JSON string:")
                    print(json_str)
                    
                    try:
                        predicted_segments = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"\nDebug - JSON decode error: {e}")
                        continue

                    # Validate segments and check for deals
                    for segment in predicted_segments:
                        if not isinstance(segment, dict) or "filter" not in segment:
                            continue
                            
                        # Build SQL query to count deals
                        where_condition = build_single_filter_condition(segment['filter'])
                        conn = None
                        cursor = None
                        try:
                            conn = psycopg2.connect(**DB_CONFIG)
                            cursor = conn.cursor()
                            
                            query = f"""
                                SELECT COUNT(*) as deal_count
                                FROM dim_deals d
                                JOIN dim_companies c ON d.company_id = c.company_id
                                LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
                                WHERE {where_condition}
                            """
                            cursor.execute(query)
                            result = cursor.fetchone()
                            deal_count = result[0] if result else 0
                            
                            if deal_count >= 5:
                                valid_segments.append(segment)
                                print(f"\nFound valid segment with {deal_count} deals:")
                                print(json.dumps(segment, indent=2))
                            else:
                                failed_filters.append({
                                    "filter": segment['filter'],
                                    "deal_count": deal_count,
                                    "reasoning": segment.get('reasoning', '')
                                })
                                print(f"\nFilter did not yield enough deals ({deal_count}):")
                                print(json.dumps(segment['filter'], indent=2))
                            
                        except Exception as e:
                            print(f"Error checking deals for segment: {e}")
                        finally:
                            if cursor:
                                cursor.close()
                            if conn:
                                conn.close()

                except Exception as e:
                    print(f"Error in LLM processing: {e}")
                    continue

                retry_count += 1

            if not valid_segments:
                yield Response(
                    chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Could not find any segments with sufficient deals after multiple attempts."}), source=self.name),
                    inner_messages=[]
                )
                return

            # Return the valid segments
            yield Response(
                chat_message=TextMessage(content=json.dumps({"predicted_filters": valid_segments}), source=self.name),
                inner_messages=[]
            )

        except Exception as e:
            print(f"Error in filter prediction agent: {e}")
            yield Response(
                chat_message=TextMessage(content=json.dumps({"status": "error", "message": str(e)}), source=self.name),
                inner_messages=[]
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("FilterPredictionAgent did not produce a response.")

# --- Function to vectorize deals for a filter ---
def vectorize_deals_for_filter(filter_combo: dict, filter_id: str) -> int:
    """Extracts deals matching the filter_combo and stores them in Pinecone with filter_id in metadata."""
    global pinecone_index, embeddings_model
    if not pinecone_index or not embeddings_model:
        print("Pinecone index or embedding model not initialized.")
        return 0

    where_condition = build_single_filter_condition(filter_combo)
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = f"""
            SELECT
                d.id AS deal_id,
                d.deal_name,
                d.amount,
                d.dealstage,
                d.closedate,
                d.days_to_close,
                d.hs_analytics_source,
                c.company_id,
                c.industry,
                c.country,
                c.employee_bucket,
                c.numberofemployees,
                d.company_name,
                ct.id AS contact_id,
                ct.first_name,
                ct.last_name,
                ct.job_title
            FROM dim_deals d
            JOIN dim_companies c ON d.company_id = c.company_id
            LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
            WHERE {where_condition}
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        vectors_to_upsert = []
        for row in rows:
            doc_id = f"filter_{filter_id}_deal_{row['deal_id']}"
            if row['contact_id']:
                doc_id += f"_contact_{row['contact_id']}"
            
            doc_content = (
                f"Deal: '{row['deal_name']}' (ID: {row['deal_id']}). "
                f"Status: {row['dealstage']}, Amount: ${row['amount'] or 0:.2f}, "
                f"Sales Cycle: {row['days_to_close'] or 0:.2f} days. "
                f"Source: {row['hs_analytics_source'] or 'N/A'}. "
                f"Company: '{row['company_name']}' (ID: {row['company_id']}, Industry: {row['industry'] or 'N/A'}, Country: {row.get('country') or 'N/A'}, Employee Bucket: {row.get('employee_bucket') or 'N/A'}, Employees: {row['numberofemployees'] or 'N/A'}). "
                f"Contact: {row['first_name'] or ''} {row['last_name'] or ''} (Job Title: {row['job_title'] or 'N/A'}, ID: {row['contact_id'] or 'N/A'})."
            )
            
            # Get embeddings for the document
            embedding = embeddings_model.embed_query(doc_content)
            
            metadata = {
                "filter_id": filter_id,
                "deal_id": str(row['deal_id']),
                "company_id": str(row['company_id']),
                "contact_id": str(row['contact_id']) if row['contact_id'] is not None else "N/A",
                "dealstage": row['dealstage'] if row['dealstage'] is not None else "N/A",
                "amount": row['amount'] if row['amount'] is not None else 0.0,
                "days_to_close": row['days_to_close'] if row['days_to_close'] is not None else 0.0,
                "hs_analytics_source": row['hs_analytics_source'] if row['hs_analytics_source'] is not None else "N/A",
                "industry": row['industry'] if row['industry'] is not None else "N/A",
                "country": row['country'] if row['country'] is not None else "N/A",
                "employee_bucket": row['employee_bucket'] if row['employee_bucket'] is not None else "N/A",
                "numberofemployees": row['numberofemployees'] if row['numberofemployees'] is not None else 0,
                "job_title": row['job_title'] if row['job_title'] is not None else "N/A"
            }
            
            vectors_to_upsert.append((doc_id, embedding, metadata))

        if vectors_to_upsert:
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                pinecone_index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
            print(f"Added {len(vectors_to_upsert)} documents for filter_id {filter_id} to Pinecone.")
        else:
            print(f"No deals found for filter_id {filter_id}.")
        return len(vectors_to_upsert)
    except Exception as e:
        print(f"Error vectorizing deals for filter: {e}")
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# --- Function to answer questions for a filter ---
def answer_question_for_filter(filter_id: str, question: str, n_results: int = 10) -> str:
    """Retrieves docs for filter_id from Pinecone, uses LLM to answer the question using those docs as context."""
    global pinecone_index, embeddings_model
    if not pinecone_index or not embeddings_model:
        print("Pinecone index or embedding model not initialized.")
        return ""

    try:
        # Get embeddings for the question
        query_embedding = embeddings_model.embed_query(question)

        # Query Pinecone with filter_id in metadata
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=n_results,
            namespace=PINECONE_NAMESPACE,
            filter={"filter_id": filter_id},
            include_metadata=True
        )

        matches = results.matches
        if not matches:
            print(f"No documents found for filter_id {filter_id}.")
            return "No relevant documents found for this filter."

        # Build context for LLM
        context = "Context from relevant deals:\n\n"
        for match in matches:
            metadata = match.metadata
            context += f"Deal: {metadata.get('deal_name', 'N/A')}\n"
            context += f"  - Amount: ${float(metadata.get('amount', 0) or 0):,.2f}\n"
            context += f"  - Stage: {metadata.get('dealstage', 'N/A')}\n"
            context += f"  - Industry: {metadata.get('industry', 'N/A')}\n"
            context += f"  - Country: {metadata.get('country', 'N/A')}\n"
            context += f"  - Employee Size: {metadata.get('employee_bucket', 'N/A')}\n"
            context += f"  - Job Title: {metadata.get('job_title', 'N/A')}\n"
            context += f"  - Sales Cycle: {float(metadata.get('days_to_close', 0) or 0):.1f} days\n"
            context += "\n"

        prompt = f"""
You are an expert sales analyst. Use ONLY the following context (deals) to answer the user's question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Provide a detailed analysis based on the deal data above. Include specific numbers and patterns you observe.
Answer as a helpful analyst:
"""
        # Use the same LLM as FilterPredictionAgent
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm = None
        if openai_api_key:
            try:
                llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")
            except Exception as e:
                print(f"Failed to initialize OpenAI model: {e}")
        if llm is None:
            try:
                llm = ChatOllama(model="gemma3:1b", base_url="http://localhost:11434")
            except Exception as e:
                print(f"Failed to initialize Ollama model: {e}")
                return "No LLM available."
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            return "Error invoking LLM."
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return f"Error querying vector database: {e}"

# --- Analysis Agent ---
class AnalysisAgent(BaseChatAgent):
    def __init__(self, name: str, openai_model: str = "gpt-4o", ollama_model: str = "gemma3:1b"):
        super().__init__(name=name, description="Agent that analyzes segment performance and calculates revenue velocity")
        self._llm = None
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key:
            print(f"Attempting to initialize OpenAI model '{openai_model}'...")
            try:
                self._llm = ChatOpenAI(api_key=openai_api_key, model=openai_model)
                print("Successfully initialized OpenAI model.")
            except Exception as e:
                print(f"Failed to initialize OpenAI model: {e}")
                self._llm = None
        else:
            print("OPENAI_API_KEY not found in environment variables. Skipping OpenAI initialization.")

        if self._llm is None:
            print(f"Attempting to initialize Ollama model '{ollama_model}'...")
            try:
                ollama_test_llm = ChatOllama(model=ollama_model, base_url="http://localhost:11434")
                ollama_test_llm.invoke("hello")
                self._llm = ollama_test_llm
                print("Successfully initialized Ollama model.")
            except Exception as e:
                print(f"Failed to initialize Ollama model: {e}")
                self._llm = None

        if self._llm is None:
            raise RuntimeError("Failed to initialize both OpenAI and Ollama models.")

    def calculate_revenue_velocity(self, total_deals: int, won_deals: int, avg_amount: float, avg_days_to_close: float) -> float:
        """Calculate revenue velocity for a segment."""
        if total_deals == 0 or avg_days_to_close == 0:
            return 0.0
        
        win_rate = (won_deals / total_deals) * 100
        revenue_velocity = (win_rate / avg_days_to_close) * avg_amount
        return round(revenue_velocity, 2)

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        try:
            latest_message = messages[-1]
            segments_data = json.loads(latest_message.content)

            # Process each segment
            analyzed_results = []
            for segment in segments_data:
                # If this is an initial segment (has filter and reasoning but no metrics)
                if "filter" in segment and "reasoning" in segment:
                    metrics = segment.get("metrics", {})
                    total = metrics.get("total_deals", 0)
                    won = metrics.get("won_deals", 0)
                    amount = metrics.get("avg_amount", 0)
                    days = metrics.get("avg_days_to_close", 0)
                    
                    revenue_velocity = self.calculate_revenue_velocity(total, won, amount, days)

                    analyzed_results.append({
                        "filter": segment["filter"],
                        "reasoning": segment["reasoning"],
                        "measured_metrics": {
                            "total_deals": total,
                            "won_deals": won,
                            "avg_amount": amount,
                            "avg_days_to_close": days,
                            "revenue_velocity": revenue_velocity
                        }
                    })

            # Sort results by revenue velocity (highest first)
            analyzed_results.sort(key=lambda x: x["measured_metrics"]["revenue_velocity"], reverse=True)

            # Prepare analysis prompt for LLM
            analysis_prompt = f"""
Analyze the following segment performance data, sorted by Revenue Velocity.
Focus on the top performing segments and explain why they might be successful.
Consider factors like:
- Win rates and their impact on revenue velocity
- Sales cycle length and its effect on velocity
- Average deal size (ACV) and its contribution
- Any patterns in the filter combinations that correlate with high performance

Segment Data:
{json.dumps(analyzed_results, indent=2)}

Provide a concise analysis focusing on actionable insights.
"""

            # Get LLM analysis
            try:
                llm_response = self._llm.invoke(analysis_prompt)
                analysis_text = llm_response.content
            except Exception as e:
                print(f"Error getting LLM analysis: {e}")
                analysis_text = "LLM analysis unavailable. Showing raw metrics only."

            # Prepare final response
            final_response = {
                "status": "success",
                "analysis": analysis_text,
                "results": analyzed_results
            }

            yield Response(
                chat_message=TextMessage(content=json.dumps(final_response), source=self.name),
                inner_messages=[]
            )

        except Exception as e:
            print(f"Error in analysis agent: {e}")
            yield Response(
                chat_message=TextMessage(content=json.dumps({"status": "error", "message": str(e)}), source=self.name),
                inner_messages=[]
            )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise RuntimeError("AnalysisAgent did not produce a response.")

# --- Main Workflow Orchestration ---
async def predict_measure_analyze_segments():
    """Main workflow function that predicts, measures, and analyzes segments."""
    try:
        # Initialize agents
        filter_agent = FilterPredictionAgent(name="FilterPredictionAgent")
        analysis_agent = AnalysisAgent(name="AnalysisAgent", openai_model="gpt-4o", ollama_model="gemma3:1b")

        # Step 1: Get predicted segments from FilterPredictionAgent
        print("\n--- Step 1: Predicting Segments ---")
        filter_response = await filter_agent.on_messages(
            [UserMessage(content="Predict segments with high revenue velocity.", source="user")],
            CancellationToken()
        )

        if not filter_response or not filter_response.chat_message:
            return {"status": "error", "message": "No response from filter prediction agent"}

        try:
            # Parse the response and extract predicted filters
            response_content = filter_response.chat_message.content
            print("\nDebug - Raw Filter Response:")
            print(response_content)
            
            # First try to parse the entire response as JSON
            try:
                response_json = json.loads(response_content)
                if isinstance(response_json, dict) and "predicted_filters" in response_json:
                    predicted_segments = response_json["predicted_filters"]
                else:
                    predicted_segments = response_json
            except json.JSONDecodeError:
                # If that fails, try to extract JSON array from the response
                import re
                json_matches = list(re.finditer(r'\[\s*\{.*\}\s*\]', response_content, re.DOTALL))
                if not json_matches:
                    print("Debug - No JSON array found in response")
                    return {"status": "error", "message": "No valid JSON array found in response"}
                
                # Use the last match (most recent complete response)
                json_str = json_matches[-1].group(0)
                print("\nDebug - Extracted JSON string:")
                print(json_str)
                
                try:
                    predicted_segments = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"\nDebug - JSON decode error: {e}")
                    return {"status": "error", "message": f"Error decoding JSON: {str(e)}"}

            if not isinstance(predicted_segments, list):
                print("\nDebug - Parsed data is not a list")
                return {"status": "error", "message": "Expected a list of segments"}

            print(f"\nDebug - Number of segments before validation: {len(predicted_segments)}")

            # Validate and clean up segments
            valid_segments = []
            for i, segment in enumerate(predicted_segments):
                print(f"\nDebug - Validating segment {i+1}:")
                print(json.dumps(segment, indent=2))
                
                if not isinstance(segment, dict):
                    print(f"Debug - Segment {i+1} is not a dictionary")
                    continue
                    
                if "filter" not in segment:
                    print(f"Debug - Segment {i+1} missing 'filter' field")
                    continue
                    
                filter_data = segment["filter"]
                if not isinstance(filter_data, dict):
                    print(f"Debug - Filter data in segment {i+1} is not a dictionary")
                    continue

                # Create a new normalized filter dictionary
                normalized_filter = {}
                
                # Normalize field names and ensure all values are lists
                field_mappings = {
                    "countries": "country",
                    "employee_buckets": "employee_bucket",
                    "industry": "industry",
                    "job_title": "job_title",
                    "hs_analytics_source": "hs_analytics_source"
                }
                
                for old_key, new_key in field_mappings.items():
                    if old_key in filter_data:
                        value = filter_data[old_key]
                        if isinstance(value, list):
                            normalized_filter[new_key] = value
                        else:
                            normalized_filter[new_key] = [value] if value else []
                
                # Create a new segment with normalized filter
                valid_segment = {
                    "filter": normalized_filter,
                    "reasoning": segment.get("reasoning", "")
                }
                
                valid_segments.append(valid_segment)
                print(f"Debug - Segment {i+1} is valid after normalization:")
                print(json.dumps(valid_segment, indent=2))

            predicted_segments = valid_segments
            print(f"\nSuccessfully extracted {len(predicted_segments)} valid predicted filters from LLM response.")
            print("\nDebug - Final Valid Segments:")
            print(json.dumps(predicted_segments, indent=2))

        except Exception as e:
            print(f"Error parsing filter prediction response: {e}")
            return {"status": "error", "message": f"Error parsing filter prediction response: {str(e)}"}

        if not predicted_segments:
            return {"status": "error", "message": "No valid segments found in response"}

        # Step 2: Measure performance for each predicted segment
        print("\n--- Step 2: Measuring Segment Performance ---")
        measured_segments = []
        for i, segment in enumerate(predicted_segments):
            try:
                print(f"\nTesting predicted segment {i+1}:")
                print(json.dumps(segment, indent=2))
                
                # Build SQL query with LIMIT for faster testing
                where_condition = build_single_filter_condition(segment['filter'])
                conn = None
                cursor = None
                try:
                    conn = psycopg2.connect(**DB_CONFIG)
                    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    
                    # Add LIMIT to the query for faster testing
                    query = f"""
                        SELECT
                            COUNT(*) as total_deals,
                            COUNT(CASE WHEN d.dealstage = 'closedwon' THEN 1 END) as won_deals,
                            AVG(d.amount) as avg_amount,
                            AVG(d.days_to_close) as avg_days_to_close
                        FROM dim_deals d
                        JOIN dim_companies c ON d.company_id = c.company_id
                        LEFT JOIN dim_contacts ct ON d.company_id = ct.company_id
                        WHERE {where_condition}
                        AND d.dealstage IN ('closedwon', 'closedlost')
                    """
                    print("\nDebug - Executing query:")
                    print(query)
                    
                    cursor.execute(query)
                    metrics = cursor.fetchone()
                    
                    if metrics:
                        segment['metrics'] = {
                            'total_deals': metrics['total_deals'] or 0,
                            'won_deals': metrics['won_deals'] or 0,
                            'avg_amount': float(metrics['avg_amount'] or 0),
                            'avg_days_to_close': float(metrics['avg_days_to_close'] or 0)
                        }
                        measured_segments.append(segment)
                        print(f"\nDebug - Added metrics for segment {i+1}:")
                        print(json.dumps(segment['metrics'], indent=2))
                    else:
                        print(f"No metrics found for segment {i+1}")
                        
                except Exception as e:
                    print(f"Error measuring segment {i+1}: {e}")
                finally:
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()
            except Exception as e:
                print(f"Error processing segment {i+1}: {e}")
                continue

        if not measured_segments:
            print("No valid segments found. Will prompt LLM again with feedback.")
            return {"status": "error", "message": "No valid segments found"}

        # Step 3: Analyze measured performance
        print("\n--- Analyzing Measured Performance ---")
        print(f"Sending results of {len(measured_segments)} tested segments to AnalysisAgent for final analysis...")
        
        analysis_response = await analysis_agent.on_messages(
            [UserMessage(content=json.dumps(measured_segments), source="user")],
            CancellationToken()
        )

        if not analysis_response or not analysis_response.chat_message:
            return {"status": "error", "message": "No response from analysis agent"}

        try:
            final_analysis = json.loads(analysis_response.chat_message.content)
            print("\nReceived final analysis from AnalysisAgent:")
            print("\n--- Final Analysis Result ---")
            print(json.dumps(final_analysis, indent=2))
        except json.JSONDecodeError as e:
            print(f"Error decoding analysis response: {e}")
            return {"status": "error", "message": "Invalid response from analysis agent"}

        # Step 4: Identify relevant open deals
        print("\n--- Identifying Relevant Open Deals ---")
        print("Fetching open deals...")
        open_deals = await fetch_open_deals(DB_CONFIG)
        print(f"Fetched {len(open_deals)} open deals.")

        # Score and match open deals to segments
        print(f"Scoring {len(open_deals)} open deals against {len(measured_segments)} predicted segments...")
        relevant_open_deals_by_segment = {}
        for segment in measured_segments:
            segment_key = json.dumps(segment['filter'], sort_keys=True)
            scored_deals = []
            
            # Limit the number of deals to process for faster testing
            for deal in open_deals[:100]:  # Process only first 100 deals
                score = calculate_relevance_score(deal, segment['filter'])
                if score > 0:
                    scored_deals.append((deal, score))
            
            # Sort by score and take top 3
            scored_deals.sort(key=lambda x: x[1], reverse=True)
            relevant_open_deals_by_segment[segment_key] = [deal for deal, _ in scored_deals[:3]]

        # Add open deals to final analysis
        final_analysis['relevant_open_deals'] = relevant_open_deals_by_segment
        return final_analysis

    except Exception as e:
        print(f"Error in predict_measure_analyze_segments: {e}")
        return {"status": "error", "message": str(e)}

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Example: Test the vector database with a sample filter and question
    print("\n--- TEST: Vectorize deals for a sample filter and ask a question ---")
    
    # Example filter (you can modify these values based on your data)
    sample_filter = {
        "industry": ["INTERNET"]
    }
    
    # Create a unique filter_id (hash of filter dict)
    filter_id = hashlib.md5(json.dumps(sample_filter, sort_keys=True).encode()).hexdigest()
    print(f"Generated filter_id: {filter_id}")
    
    # Step 1: Vectorize deals for this filter
    print("\nStep 1: Vectorizing deals for the sample filter...")
    num_added = vectorize_deals_for_filter(sample_filter, filter_id)
    print(f"Number of deals vectorized for filter: {num_added}")
    
    if num_added > 0:
        # Step 2: Ask some sample questions about this segment
        print("\nStep 2: Testing question answering for this segment...")
        
        sample_questions = [
            "What is the average deal amount for this segment?",
            "What is the typical sales cycle length for deals in this segment?",
            "What is the win rate for this segment?",
            "What are the most common deal stages in this segment?"
        ]
        
        for question in sample_questions:
            print(f"\nQ: {question}")
            answer = answer_question_for_filter(filter_id, question)
            print(f"A: {answer}")
            
        # Step 3: Test the filter prediction agent
        print("\nStep 3: Testing the filter prediction agent...")
        async def test_filter_prediction():
            filter_agent = FilterPredictionAgent(name="FilterPredictionAgent")
            response = await filter_agent.on_messages(
                [UserMessage(content="Predict segments with high revenue velocity.", source="user")],
                CancellationToken()
            )
            if response and response.chat_message:
                try:
                    result = json.loads(response.chat_message.content)
                    print("\nPredicted Segments:")
                    print(json.dumps(result, indent=2))
                except json.JSONDecodeError:
                    print("Error decoding filter prediction response")
            else:
                print("No response received from filter prediction agent")
        
        # Run the async test
        asyncio.run(test_filter_prediction())
    else:
        print("No deals found for this filter. Cannot test question answering.")
