import streamlit as st
import requests
import pandas as pd
import json
import time
from datetime import datetime
import sys
import os
import asyncio
import hashlib

from components.agent_workbench import render_agent_workbench
from components.icp_triangulation import render_icp_triangulation_matrix, display_triangulation_matrix
from components.icp_segmentation_agent import (
    run_postgres_query,
    ask_agents
)

# Add the path to import the function if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../cognition_engine/scripts")))

from sql_filter_predictor import predict_measure_analyze_segments, answer_question_for_filter, vectorize_deals_for_filter

# Add custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# --- Configuration ---
BACKEND_URL = "http://localhost:8005"  # Updated to match backend port
PAGE_TITLE = "Cognition Engine AI"
PAGE_ICON = "🧠"
LAYOUT = "wide"
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "cognition_db"),
    "user": os.getenv("DB_USER", "cognition_user"),
    "password": os.getenv("DB_PASSWORD", "cognition_password"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

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
        "Agent Workbench",
        "ICP Triangulation Matrix",
        "ICP Segmentation"  # ✅ New Feature
    ],
    captions=[
        "AI-Generated ICP Definition",
        "Score Companies Against ICP",
        "LLM Analysis of Deals",
        "Semantic Search Patterns",
        "Search CRM Data",
        "Create & Run AI Agents",
        "ICP Triangulation Matrix",
        "Segment Customers by ICP Fit"  # ✅ Caption for new feature
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
    # display_triangulation_matrix(BACKEND_URL)

elif selected_feature == "ICP Segmentation":
    st.header("📊 ICP Segmentation")
    st.write("Run the segmentation agent and explore AI-predicted segments and their top relevant open deals.")

    # --- Caching for results ---
    if 'segments_result' not in st.session_state:
        st.session_state['segments_result'] = None
        st.session_state['segments_error'] = None

    def run_segmentation_agent():
        try:
            # extract_and_vectorize_all_deals(DB_CONFIG)
            result = asyncio.run(predict_measure_analyze_segments())
            st.session_state['segments_result'] = result
            st.session_state['segments_error'] = None
        except Exception as e:
            st.session_state['segments_result'] = None
            st.session_state['segments_error'] = str(e)

    # Button to trigger segment generation
    if st.button("Generate Segments", type="primary"):
        with st.spinner("Running segmentation agent..."):
            run_segmentation_agent()

    # Option to refresh results
    if st.session_state['segments_result']:
        if st.button("Refresh Segments", type="secondary"):
            with st.spinner("Re-running segmentation agent..."):
                run_segmentation_agent()

    # Error handling
    if st.session_state['segments_error']:
        st.error(f"Error running segmentation agent: {st.session_state['segments_error']}")

    segments_result = st.session_state['segments_result']

    if not segments_result or not isinstance(segments_result, dict) or "results" not in segments_result:
        st.info("Click 'Generate Segments' to run the segmentation agent and view results.")
    else:
        def get_segment_options(segments_result):
            """Extract segment options from backend response."""
            if not segments_result:
                return []
            
            try:
                # Handle both direct segment list and wrapped response
                if isinstance(segments_result, dict):
                    if "results" in segments_result:
                        segments = segments_result["results"]
                    else:
                        segments = [segments_result]
                else:
                    segments = segments_result

                options = []
                for i, segment in enumerate(segments):
                    try:
                        # Build label from filter data
                        filter_data = segment.get("filter", {})
                        label_parts = []
                        
                        if "industry" in filter_data:
                            industries = filter_data["industry"]
                            if isinstance(industries, list):
                                label_parts.append(f"Industry: {', '.join(industries)}")
                            else:
                                label_parts.append(f"Industry: {industries}")
                        
                        if "job_title" in filter_data:
                            titles = filter_data["job_title"]
                            if isinstance(titles, list):
                                label_parts.append(f"Job Title: {', '.join(titles)}")
                            else:
                                label_parts.append(f"Job Title: {titles}")
                        
                        if "country" in filter_data:
                            countries = filter_data["country"]
                            if isinstance(countries, list):
                                label_parts.append(f"Country: {', '.join(countries)}")
                            else:
                                label_parts.append(f"Country: {countries}")
                        
                        if "employee_bucket" in filter_data:
                            buckets = filter_data["employee_bucket"]
                            if isinstance(buckets, list):
                                label_parts.append(f"Employee Size: {', '.join(buckets)}")
                            else:
                                label_parts.append(f"Employee Size: {buckets}")
                        
                        # Add reasoning if available
                        reasoning = segment.get("reasoning", "")
                        if reasoning:
                            label_parts.append(f"Reasoning: {reasoning}")
                        
                        # Add metrics if available
                        metrics = segment.get("measured_metrics", {})
                        if metrics:
                            metric_parts = []
                            if metrics.get("total_deals"):
                                metric_parts.append(f"Total Deals: {metrics['total_deals']}")
                            if metrics.get("won_deals"):
                                metric_parts.append(f"Won Deals: {metrics['won_deals']}")
                            if metrics.get("avg_amount"):
                                metric_parts.append(f"Avg Amount: ${metrics['avg_amount']:.2f}")
                            if metrics.get("revenue_velocity"):
                                metric_parts.append(f"Revenue Velocity: {metrics['revenue_velocity']:.2f}")
                            if metric_parts:
                                label_parts.append(f"Metrics: {' | '.join(metric_parts)}")
                        
                        # Create the full label
                        label = " | ".join(label_parts)
                        
                        # Use the segment's filter as the value
                        value = json.dumps(segment)
                        
                        options.append({"label": label, "value": value})
                        
                    except Exception as e:
                        print(f"Error processing segment {i}: {e}")
                        continue
                
                print(f"Generated {len(options)} segment options")
                return options
                
            except Exception as e:
                print(f"Error in get_segment_options: {e}")
                return []

        # Get segment options
        segment_options = get_segment_options(segments_result)
        
        if not segment_options:
            st.info("No segments available to select. Please generate segments first.")
        else:
            # Multi-select to compare multiple segments
            selected_indices = st.multiselect(
                "Select segments to compare",
                options=[opt["label"] for opt in segment_options],
                format_func=lambda x: x
            )
            
            # Display selected segments
            if selected_indices:
                for selected_label in selected_indices:
                    # Find the corresponding segment data
                    selected_segment = next(
                        (json.loads(opt["value"]) for opt in segment_options if opt["label"] == selected_label),
                        None
                    )
                    
                    if selected_segment:
                        with st.expander(f"Segment Details: {selected_label}", expanded=True):
                            st.markdown(f"**Filter:**\n```json\n{json.dumps(selected_segment.get('filter', {}), indent=2)}\n```")
                            if selected_segment.get("reasoning"):
                                st.markdown(f"**Reasoning:**\n{selected_segment['reasoning']}")
                            if selected_segment.get("measured_metrics"):
                                metrics = selected_segment["measured_metrics"]
                                st.markdown("**Metrics:**")
                                if "revenue_velocity" in metrics:
                                    rv = metrics["revenue_velocity"]
                                    if rv >= 1:
                                        st.success(f"Revenue Velocity: ${rv:.2f}")
                                    elif rv > 0:
                                        st.info(f"Revenue Velocity: ${rv:.2f}")
                                    else:
                                        st.warning(f"Revenue Velocity: ${rv:.2f}")
                                st.markdown(f"```json\n{json.dumps(metrics, indent=2)}\n```")
                            
                            # Display matching deals
                            if "relevant_open_deals" in segments_result:
                                segment_key = json.dumps(selected_segment['filter'], sort_keys=True)
                                matching_deals = segments_result["relevant_open_deals"].get(segment_key, [])
                                
                                if matching_deals:
                                    st.markdown("### Top Matching Open Deals")
                                    for i, deal in enumerate(matching_deals, 1):
                                        st.markdown(f"#### Deal {i}: {deal.get('deal_name', 'N/A')}")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown(f"**Company:** {deal.get('company_name', 'N/A')}")
                                            st.markdown(f"**Amount:** ${deal.get('amount', 0):,.2f}")
                                            st.markdown(f"**Stage:** {deal.get('dealstage', 'N/A')}")
                                        with col2:
                                            st.markdown(f"**Industry:** {deal.get('industry', 'N/A')}")
                                            st.markdown(f"**Country:** {deal.get('country', 'N/A')}")
                                            st.markdown(f"**Employee Size:** {deal.get('employee_bucket', 'N/A')}")
                                        if deal.get('job_title'):
                                            st.markdown(f"**Contact Role:** {deal['job_title']}")
                                        st.divider()
                                else:
                                    st.info("No matching open deals found for this segment.")
                            
                            # Add insights section
                            st.markdown("### Segment Insights")
                            question = st.text_input("Ask a question about this segment:", key=f"question_{selected_label}")
                            
                            if question:
                                if st.button("Get Insights", key=f"insights_{selected_label}"):
                                    with st.spinner("Analyzing segment..."):
                                        try:
                                            # Create a unique filter_id for this segment
                                            filter_id = hashlib.md5(json.dumps(selected_segment['filter'], sort_keys=True).encode()).hexdigest()
                                            
                                            # First, vectorize the deals for this filter
                                            st.info("Vectorizing deals for analysis...")
                                            num_vectorized = vectorize_deals_for_filter(selected_segment['filter'], filter_id)
                                            
                                            if num_vectorized > 0:
                                                st.success(f"Successfully vectorized {num_vectorized} deals for analysis")
                                                # Get answer from the backend
                                                answer = answer_question_for_filter(filter_id, question)
                                                
                                                st.markdown("**Insights:**")
                                                st.markdown(answer)
                                            else:
                                                st.error("No deals found to analyze for this segment. Please try a different segment or question.")
                                        except Exception as e:
                                            st.error(f"Error getting insights: {str(e)}")

        # Download button for segments as JSON
        st.download_button(
            label="Download Segments as JSON",
            data=json.dumps(segments_result, indent=2, cls=DateTimeEncoder),
            file_name="icp_segments.json",
            mime="application/json"
        )