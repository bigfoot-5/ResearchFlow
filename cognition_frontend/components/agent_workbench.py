import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

def render_agent_workbench(backend_url: str):
    """
    Render the Agent Workbench component.
    
    Args:
        backend_url: URL of the backend API
    """
    st.header("🤖 Agent Workbench")
    st.write("Create, save, and run AI agents to automate specific tasks and workflows.")
    
    # Helper functions for API calls
    def fetch_from_backend(endpoint: str, params: dict = None):
        """Helper function to fetch data from the FastAPI backend."""
        try:
            url = f"{backend_url}{endpoint}"
            
            # Use a longer timeout for reflective endpoints
            timeout_seconds = 120 if 'reflective' in endpoint else 30
            
            response = requests.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not connect to the backend at {backend_url}. Is the backend running?")
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
            url = f"{backend_url}{endpoint}"
            
            # Use a longer timeout for reflective endpoints
            timeout_seconds = 120 if 'reflective' in endpoint else 30
            
            response = requests.post(url, json=data, timeout=timeout_seconds)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not connect to the backend at {backend_url}. Is the backend running?")
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
            url = f"{backend_url}{endpoint}"
            response = requests.delete(url, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return True
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not connect to the backend at {backend_url}. Is the backend running?")
            return False
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Error for {endpoint}: {e}")
            return False
    
    # Tabs for different agent workbench functions
    agent_tab1, agent_tab2, agent_tab3 = st.tabs(["Create New Agent", "My Agents", "Run Agent"])
    
    with agent_tab1:
        st.subheader("Create a New Agent")
        
        agent_name = st.text_input("Agent Name", placeholder="Example: Sales Opportunity Analyzer")
        agent_description = st.text_area("Agent Description", 
                                        placeholder="Example: Analyzes sales opportunities to identify key factors for success and provides actionable recommendations.")
        
        agent_type = st.selectbox(
            "Agent Type",
            options=["icp_precision", "base"],
            format_func=lambda x: "ICP Precision Agent" if x == "icp_precision" else "Base Agent",
            help="ICP Precision Agent specializes in analyzing CRM data for customer profiles. Base Agent is a multipurpose agent."
        )
        
        custom_prompt = st.text_area("Custom System Prompt (Optional)", 
                                    placeholder="Leave blank to use the default prompt for the selected agent type.",
                                    help="System prompt defines how the agent behaves and what it focuses on.")
        
        if st.button("Create Agent", type="primary"):
            if not agent_name or not agent_description:
                st.error("Please provide a name and description for your agent.")
            else:
                with st.spinner("Creating agent..."):
                    agent_data = {
                        "name": agent_name,
                        "description": agent_description,
                        "type": agent_type,
                        "system_prompt": custom_prompt if custom_prompt else None,
                        "metadata": {
                            "created_at": pd.Timestamp.now().isoformat(),
                            "created_by": "User via Streamlit"
                        }
                    }
                    
                    response = post_to_backend("/agents", agent_data)
                    
                    if response:
                        st.success(f"Agent '{agent_name}' created successfully!")
                        st.json(response)  # Show the full response for debugging/information
                    else:
                        st.error("Failed to create agent. Please check the backend logs.")
    
    with agent_tab2:
        st.subheader("My Agents")
        
        if st.button("Refresh Agent List"):
            st.session_state.agents_list_refreshed = True
        
        # Fetch the list of agents
        with st.spinner("Loading agents..."):
            agents_data = fetch_from_backend("/agents")
            
            if agents_data:
                if not agents_data:
                    st.info("No agents found. Create your first agent in the 'Create New Agent' tab.")
                else:
                    # Display each agent as a card
                    for agent in agents_data:
                        with st.expander(f"{agent['name']} ({agent['type']})"):
                            cols = st.columns([3, 1])
                            with cols[0]:
                                st.write(f"**Description:** {agent['description']}")
                                st.write(f"**ID:** {agent['agent_id']}")
                                st.write(f"**Created:** {agent['created_at']}")
                                st.write(f"**Reflection Enabled:** {'Yes' if agent['reflection_enabled'] else 'No'}")
                            
                            with cols[1]:
                                reflection_enabled = st.toggle("Enable Reflection", 
                                                            value=agent['reflection_enabled'],
                                                            key=f"reflection_toggle_{agent['agent_id']}")
                                
                                if reflection_enabled:
                                    reflection_steps = st.number_input("Reflection Steps",
                                                                    min_value=1, max_value=5, value=1,
                                                                    key=f"reflection_steps_{agent['agent_id']}")
                                
                                if st.button("Update Settings", key=f"update_{agent['agent_id']}"):
                                    with st.spinner("Updating agent settings..."):
                                        update_data = {
                                            "reflection_enabled": reflection_enabled,
                                            "reflection_steps": reflection_steps if reflection_enabled else 1
                                        }
                                        response = post_to_backend(f"/agents/{agent['agent_id']}/reflection", update_data)
                                        if response:
                                            st.success("Agent settings updated!")
                                
                                if st.button("Delete Agent", key=f"delete_{agent['agent_id']}"):
                                    if st.session_state.get(f"confirm_delete_{agent['agent_id']}", False):
                                        # Actually delete if confirmed
                                        with st.spinner("Deleting agent..."):
                                            success = delete_from_backend(f"/agents/{agent['agent_id']}")
                                            if success:
                                                st.success("Agent deleted successfully!")
                                                st.rerun()  # Refresh the page
                                            else:
                                                st.error("Failed to delete agent.")
                                        
                                        # Reset confirmation state
                                        st.session_state[f"confirm_delete_{agent['agent_id']}"] = False
                                    else:
                                        # Ask for confirmation
                                        st.warning("Are you sure? This cannot be undone.")
                                        st.session_state[f"confirm_delete_{agent['agent_id']}"] = True
                                        
                                        cols2 = st.columns([1, 1])
                                        with cols2[0]:
                                            if st.button("Cancel", key=f"cancel_delete_{agent['agent_id']}"):
                                                st.session_state[f"confirm_delete_{agent['agent_id']}"] = False
                                                st.rerun()
                                        with cols2[1]:
                                            if st.button("Yes, Delete", key=f"confirm_delete_{agent['agent_id']}"):
                                                with st.spinner("Deleting agent..."):
                                                    success = delete_from_backend(f"/agents/{agent['agent_id']}")
                                                    if success:
                                                        st.success("Agent deleted successfully!")
                                                        st.rerun()  # Refresh the page
                                                    else:
                                                        st.error("Failed to delete agent.")
            else:
                st.error("Failed to fetch agents from the backend.")
    
    with agent_tab3:
        st.subheader("Run Agent")
        
        # Fetch available agents for selection
        agents_data = fetch_from_backend("/agents")
        
        if agents_data:
            if not agents_data:
                st.info("No agents found. Create your first agent in the 'Create New Agent' tab.")
            else:
                # Create a dictionary of agent names to IDs for the selectbox
                agent_options = {f"{agent['name']} ({agent['type']})": agent['agent_id'] for agent in agents_data}
                
                selected_agent_display = st.selectbox(
                    "Select an agent to run:",
                    options=list(agent_options.keys()),
                    key="agent_run_selectbox"
                )
                
                if selected_agent_display:
                    selected_agent_id = agent_options[selected_agent_display]
                    
                    # Get the selected agent's details
                    selected_agent = next((a for a in agents_data if a['agent_id'] == selected_agent_id), None)
                    
                    if selected_agent:
                        st.write(f"**Description:** {selected_agent['description']}")
                        
                        # Option to override reflection settings for this run
                        override_reflection = st.toggle("Override Reflection Setting", 
                                                       value=False,
                                                       help="Temporarily change the reflection setting for this run only")
                        
                        reflection_for_run = selected_agent['reflection_enabled']
                        reflection_steps_for_run = 1
                        
                        if override_reflection:
                            reflection_for_run = st.toggle("Enable Reflection", 
                                                         value=selected_agent['reflection_enabled'])
                            
                            if reflection_for_run:
                                reflection_steps_for_run = st.number_input("Reflection Steps",
                                                                         min_value=1, max_value=5, value=1)
                        
                        # Query input
                        query = st.text_area("Enter your query or task for the agent:",
                                           placeholder="Example: Analyze our closed deals from the last 90 days")
                        
                        # Context data (optional)
                        with st.expander("Advanced: Add Context Data (Optional)"):
                            context_json = st.text_area("Context Data (JSON format)",
                                                      placeholder='{"timeframe": "last_90_days", "min_deal_value": 10000}',
                                                      help="Additional context data in JSON format")
                            
                            context_data = {}
                            if context_json:
                                try:
                                    context_data = json.loads(context_json)
                                except json.JSONDecodeError:
                                    st.error("Invalid JSON format in context data")
                        
                        if st.button("Run Agent", type="primary"):
                            if not query:
                                st.error("Please enter a query for the agent.")
                            else:
                                with st.spinner(f"Running agent{' with reflection' if reflection_for_run else ''}..."):
                                    execution_data = {
                                        "query": query,
                                        "context": context_data,
                                        "reflection_enabled": reflection_for_run,
                                        "reflection_steps": reflection_steps_for_run if reflection_for_run else None
                                    }
                                    
                                    response = post_to_backend(f"/agents/{selected_agent_id}/execute", execution_data)
                                    
                                    if response:
                                        st.success("Agent execution complete!")
                                        
                                        # Display reflection badge if applied
                                        if response.get("reflection_applied"):
                                            st.success("✨ Enhanced with AI Reflection")
                                        
                                        # Display execution time
                                        st.info(f"Execution time: {response.get('execution_time', 'N/A')}")
                                        
                                        # Display the agent's response in a custom format
                                        st.markdown("### Agent Response:")
                                        st.markdown(response.get("response", "No response generated"))
                                        
                                        # Show raw response for debugging (expandable)
                                        with st.expander("Show complete response data"):
                                            st.json(response)
                                    else:
                                        st.error("Failed to execute agent. Please check the backend logs.")
        else:
            st.error("Failed to fetch agents from the backend.") 