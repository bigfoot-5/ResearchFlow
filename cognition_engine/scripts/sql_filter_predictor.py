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

# --- Helper function to build dynamic WHERE clause for a single filter combo ---
def build_single_filter_condition(filter_combo: Dict[str, List[str]]) -> str:
    """Builds SQL WHERE clause for a single filter combination."""
    conditions = []
    if filter_combo.get('hs_analytics_source'):
        sources = ", ".join(f"'{s}'" for s in filter_combo['hs_analytics_source'])
        conditions.append(f"d.hs_analytics_source IN ({sources})")
    if filter_combo.get('job_title'):
        titles = ", ".join(f"'{t}'" for t in filter_combo['job_title'])
        conditions.append(f"ct.job_title IN ({titles})")
    if filter_combo.get('industry'):
        industries = ", ".join(f"'{i}'" for i in filter_combo['industry'])
        conditions.append(f"c.industry IN ({industries})")
    if filter_combo.get('country'):
        countries = ", ".join(f"'{c}'" for c in filter_combo['country'])
        conditions.append(f"c.country IN ({countries})")
    if filter_combo.get('employee_bucket'):
        buckets = ", ".join(f"'{b}'" for b in filter_combo['employee_bucket'])
        conditions.append(f"c.employee_bucket IN ({buckets})")

    return " AND ".join(conditions) if conditions else "TRUE"

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
        filter_values = filter_combo.get(dimension, [])
        deal_value = deal_data.get(dimension)
        if deal_value is not None and any(str(deal_value).lower() == str(fv).lower() for fv in filter_values):
            score += weight

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
            WHERE d.dealstage NOT IN ('closedwon', 'closedlost')
            LIMIT 1000;
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

            # Step 2: Formulate prompt for the LLM with context and available dimensions
            prompt = f"""
Analyze the following examples of high-performing deals provided as context.
Based on these examples and your general knowledge about sales dynamics, predict 5 to 10 specific filter combinations (segments) that are likely to have high revenue velocity. Revenue velocity is calculated as (Win Rate ÷ Sales Cycle in Days) × Average Contract Value (ACV).

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
      "job_title": ["CEO", "VP People"],
      "hs_analytics_source": ["ORGANIC_SEARCH", "DIRECT_TRAFFIC"]
    }},
    "reasoning": "Based on the context, deals in the technology sector with C-level contacts from organic sources seem to close faster with higher amounts."
  }},
  {{
    "filter": {{
      "industry": ["PHARMACEUTICALS"],
      "job_title": ["Head of Talent"],
      "hs_analytics_source": ["EMAIL_MARKETING"]
    }},
    "reasoning": "Pharmaceutical deals with talent heads often represent specialized software needs, and email campaigns can effectively reach these niche roles."
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

                # Extract the string content from the LLM's response message
                llm_response_content = llm_response.content
                print("--- START RAW LLM RESPONSE CONTENT ---")
                print(llm_response_content)
                print("--- END RAW LLM RESPONSE CONTENT ---")

                # Attempt to extract JSON array from the response
                import re
                # Updated regex to be more robust, looking for the first occurrence of a list containing dicts
                match = re.search(r'(\[\s*\{.*?\}\s*(,\s*\{.*?\}\s*)*\])', llm_response_content, re.DOTALL)
                if match:
                    try:
                        # Take the first captured group, which should be the JSON array
                        json_string = match.group(1)
                        predicted_filters_with_reasoning = json.loads(json_string)
                        # Validate basic structure of the expected output
                        if isinstance(predicted_filters_with_reasoning, list) and \
                           all(isinstance(item, dict) and 'filter' in item and isinstance(item.get('filter'), dict) and 'reasoning' in item for item in predicted_filters_with_reasoning):

                             print(f"Successfully extracted {len(predicted_filters_with_reasoning)} predicted filters from LLM response.")
                             yield Response(
                                chat_message=TextMessage(content=json.dumps({"status": "success", "predicted_filters": predicted_filters_with_reasoning}), source=self.name),
                                inner_messages=[],
                             )
                        else:
                             print("LLM response did not match expected JSON structure.")
                             yield Response(
                                chat_message=TextMessage(content=json.dumps({"status": "error", "message": "LLM response format incorrect, expected list of dictionaries with 'filter' and 'reasoning'."}), source=self.name),
                                inner_messages=[],
                             )
                    except json.JSONDecodeError:
                        print("Could not parse JSON from LLM response.")
                        print(f"LLM response content was:\n{llm_response_content}") # Log response for debugging
                        yield Response(
                            chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Could not parse JSON from LLM response."}), source=self.name),
                            inner_messages=[],
                        )
                else:
                    print("Could not find JSON array in LLM response.")
                    print(f"LLM response content was:\n{llm_response_content}") # Log response for debugging
                    yield Response(
                        chat_message=TextMessage(content=json.dumps({"status": "error", "message": "Could not extract filter combinations (JSON array) from LLM response."}), source=self.name),
                        inner_messages=[],
                    )
            except Exception as e:
                 # Catch any other exceptions during the LLM call or response processing
                 print(f"[LLM INVOKE ERROR] An unexpected error occurred during LLM invocation or initial processing: {e}")
                 yield Response(
                      chat_message=TextMessage(content=json.dumps({"status": "error", "message": f"LLM invocation or processing error: {e}"}), source=self.name),
                      inner_messages=[]
                 )
        except Exception as e:
            # Outer catch block for errors during Pinecone query or context building
            print(f"[RAG ERROR] Error during vector database query or context building: {e}")
            yield Response(
                chat_message=TextMessage(content=json.dumps({"status": "error", "message": f"Vector database query or context building error: {e}"}), source=self.name),
                inner_messages=[],
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
        context = "\n\n".join([match.id for match in matches])
        prompt = f"""
You are an expert sales analyst. Use ONLY the following context (deals) to answer the user's question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}
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

# --- Main Workflow Orchestration ---
async def predict_measure_analyze_segments():
    print("Starting Predict, Measure, and Analyze Revenue Velocity workflow...")

    # Instantiate agents
    filter_agent = FilterPredictionAgent(name="FilterPredictionAgent", openai_model="gpt-4o", ollama_model="gemma3:1b")

    max_attempts = 10
    attempt = 0
    found_valid_segment = False
    feedback_message = None
    final_result = None
    predicted_filters_with_reasoning = []
    tested_segments_results = []

    while attempt < max_attempts and not found_valid_segment:
        attempt += 1
        print(f"\n--- Attempt {attempt} to find valid segments ---")
        # Step 1: Use LLM to predict promising filter combinations
        if feedback_message:
            user_prompt = f"Predict segments with high revenue velocity. Previous attempt feedback: {feedback_message}"
        else:
            user_prompt = "Predict segments with high revenue velocity."
        filter_response = await filter_agent.on_messages(
            [UserMessage(content=user_prompt, source="user")],
            CancellationToken()
        )

        if not filter_response or not filter_response.chat_message:
            print(f"No valid response received from {filter_agent.name}.")
            break # Exit the loop

        try:
            filter_data = json.loads(filter_response.chat_message.content)
            if filter_data.get("status") != "success":
                print(f"Error from {filter_agent.name}: {filter_data.get('message')}")
                break # Exit the loop
            predicted_filters_with_reasoning = filter_data.get("predicted_filters", [])
            print(f"Received {len(predicted_filters_with_reasoning)} predicted filter combinations.")
            if not predicted_filters_with_reasoning:
                print("No filter combinations were predicted. Aborting workflow.")
                break # Exit the loop
        except json.JSONDecodeError:
            print("Error decoding predicted filters from LLM response.")
            break # Exit the loop

        # Step 2: For each predicted filter combination, query the database to get measured metrics
        tested_segments_results = []
        print("\n--- Measuring Performance of Predicted Segments ---")
        for idx, segment_prediction in enumerate(predicted_filters_with_reasoning, start=1):
            filter_combo = segment_prediction.get("filter", {})
            prediction_reasoning = segment_prediction.get("reasoning", "No reasoning provided.")

            if not filter_combo or not any(filter_combo.values()):
                print(f"\nSkipping predicted segment {idx}: Empty filter combination.")
                tested_segments_results.append({
                    "filter": filter_combo,
                    "reasoning": prediction_reasoning,
                    "db_status": "skipped",
                    "db_message": "Empty filter combination predicted.",
                    "metrics": None
                })
                continue

            print(f"\nTesting predicted segment {idx}: Filter={filter_combo} (Reasoning: {prediction_reasoning[:100]}...)")

            db_request_message = UserMessage(content=json.dumps(filter_combo), source=filter_agent.name)
            db_response = await filter_agent.on_messages([db_request_message], CancellationToken())

            if db_response and db_response.chat_message:
                try:
                    db_result = json.loads(db_response.chat_message.content)
                    tested_segments_results.append({
                        "filter": filter_combo,
                        "reasoning": prediction_reasoning,
                        "db_status": db_result.get("status"),
                        "db_message": db_result.get("message"),
                        "metrics": db_result.get("metrics")
                    })
                    print(f"  DB Query Status: {db_result.get('status')}")
                    if db_result.get('status') == 'success' and db_result.get('metrics'):
                        metrics = db_result.get('metrics')
                        total = metrics.get('total_deals', 0)
                        won = metrics.get('won_deals', 0)
                        amount = metrics.get('avg_amount', 0)
                        days = metrics.get('avg_days_to_close', 0)
                        measured_rv = analysis_agent.calculate_revenue_velocity(total, won, amount, days)
                        print(f"  Measured Metrics: Total Deals={total}, Won Deals={won}, ACV=${amount:.2f}, Sales Cycle={days:.2f} days, Measured RV={measured_rv:.2f}")
                except json.JSONDecodeError:
                    print(f"  Error decoding DB response for segment {idx}.")
                    tested_segments_results.append({
                        "filter": filter_combo,
                        "reasoning": prediction_reasoning,
                        "db_status": "error",
                        "db_message": "Invalid JSON response from FilterPredictionAgent",
                        "metrics": None
                    })
            else:
                print(f"No valid response received from {filter_agent.name} for segment {idx}.")
                tested_segments_results.append({
                    "filter": filter_combo,
                    "reasoning": prediction_reasoning,
                    "db_status": "error",
                    "db_message": "No valid response from FilterPredictionAgent",
                    "metrics": None
                })

        # Check if any segment has measured_metrics (i.e., metrics is not None)
        found_valid_segment = any(seg.get("metrics") for seg in tested_segments_results)
        if not found_valid_segment:
            # Prepare feedback for the LLM
            failed_filters = [seg["filter"] for seg in tested_segments_results]
            feedback_message = f"None of the predicted segments had at least 5 deals. Previous filters tried: {json.dumps(failed_filters)}. Please suggest broader or alternative segments."
            print("No valid segments found. Will prompt LLM again with feedback.")
        else:
            print("At least one valid segment found with measured metrics.")

    # Step 3: Send tested segments results to Analysis Agent for final analysis and comparison
    if tested_segments_results:
        analyzable_results = [res for res in tested_segments_results if res.get("db_status") not in ["skipped", "error", "no_data"]]
        if analyzable_results:
            print(f"\n--- Analyzing Measured Performance ---")
            print(f"Sending results of {len(analyzable_results)} tested segments to AnalysisAgent for final analysis...")
            analysis_request_message = UserMessage(content=json.dumps(analyzable_results), source=filter_agent.name)
            analysis_response = await analysis_agent.on_messages([analysis_request_message], CancellationToken())

            if analysis_response and analysis_response.chat_message:
                final_result_json = analysis_response.chat_message.content
                print(f"Received final analysis from {analysis_agent.name}:")
                try:
                    final_result = json.loads(final_result_json)
                    # Print the full analysis result
                    print("--- Final Analysis Result ---")
                    print(json.dumps(final_result, indent=2))

                    # Extract the top predicted segments (which are sorted by RV in analyzed_results)
                    # The 'results' key in the final_result already contains the sorted analyzed_results
                    top_segments = final_result.get("results", [])

                    # Define output filename
                    output_filename = "top_predicted_segments.json"

                    # Write the top segments to a JSON file
                    if top_segments:
                        with open(output_filename, 'w') as f:
                            json.dump(top_segments, f, indent=2)
                        print(f"Successfully saved top predicted segments to {output_filename}")
                    else:
                        print("No analyzable segments with measured data to save.")

                except json.JSONDecodeError:
                    print("Error decoding final JSON response from Analysis agent.")
                    print(final_result_json)
            else:
                print(f"No valid response received from {analysis_agent.name}.")
        else:
            print("\nNo segments with measured data met criteria for analysis, or no segments were predicted.")

    # --- Step 4: Identify and Score Relevant Open Deals ---
    print("\n--- Identifying Relevant Open Deals ---")
    # Fetch all open deals
    open_deals = await fetch_open_deals(DB_CONFIG)
    relevant_open_deals_by_segment = {}

    if open_deals and predicted_filters_with_reasoning:
        print(f"Scoring {len(open_deals)} open deals against {len(predicted_filters_with_reasoning)} predicted segments...")
        for segment_prediction in predicted_filters_with_reasoning:
            filter_combo = segment_prediction.get("filter", {})
            segment_key = json.dumps(filter_combo, sort_keys=True) # Use sorted JSON as a unique key for the segment
            relevant_deals_for_segment = []

            print(f"\n-- Scoring open deals for segment: {filter_combo} --")
            for deal in open_deals:
                score = calculate_relevance_score(deal, filter_combo)
                # Log the score for debugging
                # print(f"  Deal ID: {deal.get('deal_id')}, Score: {score:.2f}")
                if score > 0: # Only include deals with a non-zero score (at least one match)
                    relevant_deals_for_segment.append({
                        "deal": deal, # Include full deal data
                        "relevance_score": score
                    })

            # Sort relevant deals by score (highest first)
            relevant_deals_for_segment.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Store top N relevant deals for this segment (e.g., top 3)
            top_n = 3 # Limit to top 3 relevant deals as requested
            relevant_open_deals_by_segment[segment_key] = relevant_deals_for_segment[:top_n]
            print(f"Found {len(relevant_deals_for_segment)} open deals matching this segment filter (score > 0). Top {min(top_n, len(relevant_deals_for_segment))} relevant deals saved.")
    else:
        print("No open deals fetched or no segments predicted to score against.")

    # --- Combine Results and Output ---
    # Start building the final output structure, based on the analysis result or a default structure
    final_output_structure = final_result if final_result is not None else {"status": "success", "analysis": "Workflow completed, but analysis step may have been skipped.", "results": []}

    # Iterate through the segments in the results and add relevant open deals
    processed_segments_for_output = []
    for segment_data in final_output_structure.get("results", []):
        # Create a key for lookup in relevant_open_deals_by_segment
        filter_combo = segment_data.get("filter", {})
        segment_key = json.dumps(filter_combo, sort_keys=True)

        # Find the relevant open deals for this segment
        relevant_deals = relevant_open_deals_by_segment.get(segment_key, [])

        # Add the top relevant open deals to the segment data
        # We will simplify the deal objects to just include key info for JSON serialization
        top_relevant_open_deals_simplified = [{
            "deal_id": d['deal'].get('deal_id'),
            "deal_name": d['deal'].get('deal_name'),
            "dealstage": d['deal'].get('dealstage'),
            "amount": d['deal'].get('amount'),
            "relevance_score": d['relevance_score']
        } for d in relevant_deals]

        segment_data["top_relevant_open_deals"] = top_relevant_open_deals_simplified

        # Add top-level revenue_velocity field
        measured_metrics = segment_data.get("measured_metrics")
        if measured_metrics and isinstance(measured_metrics, dict):
            segment_data["revenue_velocity"] = measured_metrics.get("revenue_velocity")
        else:
            segment_data["revenue_velocity"] = None

        processed_segments_for_output.append(segment_data)

    # Replace the original results with the processed ones
    final_output_structure["results"] = processed_segments_for_output

    # --- Final Output ---
    print("\n--- Final Combined Output ---")
    # The structure should now be JSON serializable since we simplified the deal objects
    try:
        print(json.dumps(final_output_structure, indent=2))
        return final_output_structure
    except TypeError as e:
        # This catch is a fallback, should not be hit if simplification worked
        print(f"Error serializing final output structure: {e}")
        print("Final output structure:", final_output_structure)

    print("\nWorkflow finished.")

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
