import streamlit as st
import requests
import pandas as pd
import json
import time
from datetime import datetime

from components.agent_workbench import render_agent_workbench
from components.icp_triangulation import render_icp_triangulation_matrix

# --- Configuration ---
BACKEND_URL = "http://localhost:8005"  # Updated to match backend port
PAGE_TITLE = "Cognition Engine AI"
PAGE_ICON = "🧠"
LAYOUT = "wide"

# --- Page Setup ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

# --- Styling (Optional - Dark Theme Example) ---
# You can customize this further in Streamlit's settings or using CSS
# st.markdown("""
# <style>
#     /* Basic Dark Theme Adjustments */
#     .stApp {
#         background-color: #1E1E1E;
#         color: #EAEAEA;
#     }
#     /* ... more specific styles ... */
# </style>
# """, unsafe_allow_html=True)

# --- Title and Introduction ---
st.title(f"{PAGE_ICON} Cognition Engine Demo")
st.markdown("""
*Leveraging AI to unlock deep insights from your CRM data and define your Ideal Customer Profile.*
""")
st.divider()

# --- API Helper Function ---
def fetch_from_backend(endpoint: str, params: dict = None):
    """Helper function to fetch data from the FastAPI backend."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        # Use a longer timeout for reflective endpoints
        timeout_seconds = 120 if 'reflective' in endpoint else 30
        
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend at {BACKEND_URL}. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        st.error(f"Timeout Error: The request to {endpoint} timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error for {endpoint}: {e}")
        if e.response is not None:
            try:
                st.error(f"Backend Response: {e.response.json()}")
            except json.JSONDecodeError:
                st.error(f"Backend Response (non-JSON): {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error(f"JSON Decode Error: Failed to parse response from {endpoint}.")
        return None

def post_to_backend(endpoint: str, data: dict = None):
    """Helper function to send data to the FastAPI backend."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        # Use a longer timeout for reflective endpoints
        timeout_seconds = 120 if 'reflective' in endpoint else 30
        
        response = requests.post(url, json=data, timeout=timeout_seconds)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend at {BACKEND_URL}. Is the backend running?")
        return None
    except requests.exceptions.Timeout:
        st.error(f"Timeout Error: The request to {endpoint} timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error for {endpoint}: {e}")
        if e.response is not None:
            try:
                st.error(f"Backend Response: {e.response.json()}")
            except json.JSONDecodeError:
                st.error(f"Backend Response (non-JSON): {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error(f"JSON Decode Error: Failed to parse response from {endpoint}.")
        return None

def delete_from_backend(endpoint: str):
    """Helper function to delete data from the FastAPI backend."""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        response = requests.delete(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return True
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend at {BACKEND_URL}. Is the backend running?")
        return False
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error for {endpoint}: {e}")
        return False

# --- Sidebar Navigation (Example) ---
st.sidebar.header("Cognition Features")
selected_feature = st.sidebar.radio(
    "Explore:",
    [
        "Ideal Customer Profile (ICP)",
        "Company Scoring",
        "Deal Pattern Insights",
        "Vector Search Insights",
        "Live CRM Search",
        "Agent Workbench",  # New option for Agent functionality
        "ICP Triangulation Matrix"  # New option for ICP Triangulation Matrix
    ],
    captions=[
        "AI-Generated ICP Definition",
        "Score Companies Against ICP",
        "LLM Analysis of Deals",
        "Semantic Search Patterns",
        "Search CRM Data",
        "Create & Run AI Agents",  # Caption for Agent Workbench
        "ICP Triangulation Matrix"  # Caption for ICP Triangulation Matrix
    ]
)

# --- Main Content Area ---
if selected_feature == "Ideal Customer Profile (ICP)":
    st.header("🧬 Ideal Customer Profile (ICP) Definition")
    st.write("The Cognition Engine analyzes successful deals to automatically generate your ICP.")
    
    # Add option to use reflection
    use_reflection = st.toggle("Use AI Reflection (Enhanced Analysis)", value=False, 
                               help="Enable AI reflection to improve analysis quality through self-critique")
    
    if use_reflection:
        reflection_steps = st.slider("Reflection Depth", min_value=1, max_value=3, value=2,
                                    help="Higher values enable more thorough self-critique cycles but take longer")
        st.info("💡 Reflection uses a second AI to critique and improve the initial analysis, resulting in more nuanced and actionable insights. Each reflection step adds approximately 10-15 seconds.")
        st.warning("Note: With reflection enabled, this process may take up to 2 minutes to complete. Please be patient.")
    
    if st.button("Define ICP Now", type="primary"):
        with st.spinner("🧠 Analyzing successful customers..." + (" and performing reflection cycles..." if use_reflection else "")):
            # Determine which endpoint to use
            if use_reflection:
                endpoint = "/icp/reflective/definition"
                params = {"reflection_steps": reflection_steps}
                icp_data = fetch_from_backend(endpoint, params)
            else:
                icp_data = fetch_from_backend("/icp/definition")
            
            if icp_data:
                # Add badge to show if reflection was used
                if use_reflection:
                    st.success("✨ Enhanced with AI Reflection")
                
                st.subheader(f"ICP Name: {icp_data.get('name', 'N/A')}")
                st.markdown(f"**Description:** {icp_data.get('description', 'N/A')}")
                st.markdown(f"_(Generated: {icp_data.get('created_at', 'N/A')})_" )
                
                st.subheader("Key Criteria:")
                criteria = icp_data.get('criteria', [])
                if criteria:
                    # Convert criteria to DataFrame for better display
                    df_criteria = pd.DataFrame(criteria)
                    # Format weight as percentage
                    if 'weight' in df_criteria.columns:
                         df_criteria['weight'] = df_criteria['weight'].map('{:.1%}'.format)
                    # Display selected columns
                    st.dataframe(
                        df_criteria[['field', 'values', 'weight', 'description']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                             "field": st.column_config.TextColumn("Attribute"),
                             "values": st.column_config.ListColumn("Top Values", help="Most common values for this attribute among successful customers"),
                             "weight": st.column_config.TextColumn("Importance (Weight)"),
                             "description": st.column_config.TextColumn("Details")
                        }
                    )
                else:
                    st.warning("No criteria data received from backend.")
            else:
                st.error("Failed to fetch ICP definition from the backend.")

elif selected_feature == "Company Scoring":
    st.header("📊 Company Scoring")
    st.write("Score individual companies against the AI-generated ICP definition.")

    # Fetch companies for selection
    companies_data = fetch_from_backend("/icp/companies")
    company_options = {}
    if companies_data:
        # Create a dictionary mapping display name (Name + ID) to company_id
        company_options = {
            f"{comp['company_name']} ({comp['company_id']})": comp['company_id'] 
            for comp in companies_data
        }
    
    if not company_options:
        st.warning("Could not fetch companies from the backend. Please ensure the backend is running and data is ingested.")
    else:
        selected_company_display = st.selectbox(
            "Select a company to score:", 
            options=list(company_options.keys()) # Use display names as options
        )
        
        if selected_company_display:
            selected_company_id = company_options[selected_company_display] # Get the ID from the selection
        
            if st.button("Score Company", key="score_company_button", type="primary"):
                with st.spinner(f"🧠 Scoring company {selected_company_display}..."):
                    # Construct the correct endpoint URL
                    score_endpoint = f"/icp/company/{selected_company_id}/score"
                    score_data = fetch_from_backend(score_endpoint)
                    
                    if score_data:
                        st.subheader(f"Score for {selected_company_display}")
                        # Display score prominently
                        score_value = score_data.get("score", 0)
                        st.metric(label="ICP Fit Score", value=f"{score_value:.2%}")
                        
                        st.subheader("Match Details:")
                        match_details = score_data.get("match_details", {})
                        if match_details:
                            # Convert match details to DataFrame for better display
                            df_match = pd.DataFrame.from_dict(match_details, orient='index')
                            df_match.index.name = "Attribute"
                            st.dataframe(df_match, use_container_width=True)
                        else:
                            st.info("No match details available.")
                    else:
                        st.error(f"Failed to fetch score for company {selected_company_id} from the backend.")

elif selected_feature == "Deal Pattern Insights":
    st.header("💡 Deal Pattern Insights (LLM Analysis)")
    st.write("The LLM analyzes patterns distinguishing successful from unsuccessful deals.")
    
    # Add option to use reflection
    use_reflection = st.toggle("Use AI Reflection (Enhanced Analysis)", value=False, 
                               help="Enable AI reflection to improve insight quality through self-critique")
    
    if use_reflection:
        reflection_steps = st.slider("Reflection Depth", min_value=1, max_value=3, value=2,
                                    help="Higher values enable more thorough self-critique cycles but take longer")
        st.info("💡 Reflection uses a second AI to critique and improve the initial insights, resulting in more specific and actionable recommendations. Each reflection step adds approximately 15-20 seconds.")
        st.warning("Note: With reflection enabled, this process may take up to 2 minutes to complete. Please be patient.")
    
    if st.button("Analyze Deal Patterns", key="deal_insights_button", type="primary"):
        with st.spinner("🧠 Analyzing deal data with LLM..." + (" and performing reflection cycles..." if use_reflection else "")):
            # Determine which endpoint to use
            if use_reflection:
                endpoint = "/icp/reflective/insights"
                params = {"reflection_steps": reflection_steps}
                insights_data = fetch_from_backend(endpoint, params)
            else:
                insights_data = fetch_from_backend("/icp/insights")
            
            if insights_data:
                # Add badge to show if reflection was used
                if use_reflection:
                    st.success("✨ Enhanced with AI Reflection")
                    
                st.subheader("Generated Insights:")
                if not insights_data:
                    st.info("The analysis did not yield specific insights based on the current data.")
                else:
                    for i, insight in enumerate(insights_data):
                        with st.expander(f"Insight {i+1}: {insight.get('title', 'N/A')}", expanded=i==0): # Expand first insight by default
                            st.markdown(f"**Description:** {insight.get('description', 'N/A')}")
                            st.markdown(f"**Confidence:** {insight.get('confidence', 0.0):.1f}")
                            
                            st.markdown("**Supporting Evidence:**")
                            evidence = insight.get('supporting_evidence', [])
                            if evidence:
                                # Display evidence as a DataFrame
                                df_evidence = pd.DataFrame(evidence)
                                st.dataframe(df_evidence, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("_(No specific evidence provided)_")
            else:
                st.error("Failed to fetch deal pattern insights from the backend.")

elif selected_feature == "Vector Search Insights":
    st.header("🔍 Vector Search Insights")
    st.write("Identifies CRM data patterns semantically related to key concepts (using vector search + LLM).")
    
    # Add option to use reflection
    use_reflection = st.toggle("Use AI Reflection (Enhanced Analysis)", value=False, 
                               help="Enable AI reflection to improve insight quality through self-critique")
    
    if use_reflection:
        reflection_steps = st.slider("Reflection Depth", min_value=1, max_value=3, value=2,
                                    help="Higher values enable more thorough self-critique cycles but take longer")
        st.info("💡 Reflection uses a second AI to critique and improve the initial vector insights, resulting in more nuanced connections and patterns. Each reflection step adds approximately 15-20 seconds.")
        st.warning("Note: With reflection enabled, this process may take up to 2 minutes to complete. Please be patient.")
    
    if st.button("Generate Vector Insights", key="vector_insights_button", type="primary"):
        with st.spinner("🧠 Performing vector analysis..." + (" and reflection cycles..." if use_reflection else "")):
            # Determine which endpoint to use
            if use_reflection:
                endpoint = "/icp/reflective/report"  # Use the report endpoint which includes vector insights
                params = {"reflection_steps": reflection_steps}
                report_data = fetch_from_backend(endpoint, params)
                # Extract just the vector insights part from the report
                vector_insights_data = report_data.get("vector_insights", []) if report_data else []
            else:
                vector_insights_data = fetch_from_backend("/icp/vector-insights")
            
            if vector_insights_data is not None:  # Check if the data exists (empty list is valid)
                # Add badge to show if reflection was used
                if use_reflection:
                    st.success("✨ Enhanced with AI Reflection")
                    
                st.subheader("Generated Vector Insights:")
                if not vector_insights_data:
                    st.info("The vector analysis did not yield specific insights based on the current data.")
                else:
                    # Display the raw insights data (list of dictionaries)
                    st.json(vector_insights_data)  # Use st.json for nice formatting of dict/list
            else:
                st.error("Failed to fetch vector search insights from the backend.")

elif selected_feature == "Live CRM Search":
    st.header("💬 Live CRM Search")
    st.write("Use natural language to search across your indexed CRM data (Companies, Deals, Contacts).")
    
    # Search Query Input
    search_query = st.text_input("Enter your search query:", placeholder="e.g., technology companies in finance with large deals")
    
    # Entity Type Selection
    entity_options = ["company", "deal", "contact"]
    selected_entities = st.multiselect(
        "Search within entities (optional):",
        options=entity_options,
        # default=entity_options # Optional: default to searching all
    )
    
    # Number of results
    num_results = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    
    if st.button("Search Now", key="live_search_button") and search_query:
        with st.spinner(f"🧠 Searching for '{search_query}'..."):
            # Prepare parameters for the POST request
            search_params = {
                "query": search_query,
                "n_results": num_results
            }
            # Only include entity_types if some are selected
            if selected_entities:
                search_params["entity_types"] = selected_entities
                
            # Need to make a POST request - update fetch_from_backend or make specific call
            # Modifying helper function to handle POST with JSON body
            search_results = None
            try:
                url = f"{BACKEND_URL}/icp/search"
                # Use requests.post and send data as json
                # Use a longer timeout for complex searches
                response = requests.post(url, json=search_params, timeout=60)
                response.raise_for_status()
                search_results = response.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Connection Error: Could not connect to the backend at {BACKEND_URL}. Is the backend running?")
            except requests.exceptions.Timeout:
                st.error(f"Timeout Error: The request to /icp/search timed out.")
            except requests.exceptions.RequestException as e:
                st.error(f"API Request Error for /icp/search: {e}")
                if e.response is not None:
                    try:
                        st.error(f"Backend Response: {e.response.json()}")
                    except json.JSONDecodeError:
                        st.error(f"Backend Response (non-JSON): {e.response.text}")
            except json.JSONDecodeError:
                st.error(f"JSON Decode Error: Failed to parse response from /icp/search.")

            if search_results is not None: # Check if fetch was successful
                st.subheader(f"Search Results for '{search_query}':")
                if not search_results:
                    st.info("No matching results found.")
                else:
                    # Display results (list of dictionaries)
                    # st.json(search_results)
                    # Better display:
                    for i, result in enumerate(search_results):
                        entity_type = result.get('metadata', {}).get('entity_type', 'N/A')
                        entity_id = result.get('metadata', {}).get('entity_id', 'N/A')
                        score = result.get('distance') # Chroma uses distance, smaller is better
                        
                        # Extract key info from metadata for display
                        name = result.get('metadata', {}).get('name', 'N/A') 
                        if entity_type == 'deal': name = result.get('metadata', {}).get('deal_name', name)
                        if entity_type == 'contact': name = f"{result.get('metadata', {}).get('first_name', '')} {result.get('metadata', {}).get('last_name', '')}".strip() or name
                        
                        st.markdown(f"**{i+1}. {entity_type.capitalize()}: {name}** (ID: {entity_id})")
                        if score is not None:
                            st.markdown(f"   *Relevance Score (Distance): {score:.4f}*")
                        with st.expander("Details & Document"):
                            st.json(result.get('metadata', {})) # Show metadata
                            st.text_area("Indexed Document", value=result.get('document', 'N/A'), height=100, disabled=True)
                        st.divider()

    elif not search_query and st.session_state.get("live_search_button", False):
         st.warning("Please enter a search query.")

elif selected_feature == "Agent Workbench":
    # Render the Agent Workbench
    render_agent_workbench(BACKEND_URL)

elif selected_feature == "ICP Triangulation Matrix":
    # Render the ICP Triangulation Matrix
    render_icp_triangulation_matrix(BACKEND_URL)

# You can add more sections or refine the layout further. 