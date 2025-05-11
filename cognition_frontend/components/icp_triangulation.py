import streamlit as st
import requests
import pandas as pd
import warnings
from typing import Dict, List, Any

# Try to import plotly, but provide fallback if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Chart visualizations will be disabled.")

def render_icp_triangulation_matrix(backend_url: str):
    """
    Render the ICP Triangulation Matrix component.
    
    Args:
        backend_url: URL of the backend API
    """
    st.header("🔍 ICP Triangulation Matrix")
    st.subheader("Discover your best customer segments")
    
    # Add filter inputs
    st.subheader("Filter Deals")
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Minimum Deal Amount", min_value=0, value=50000)
    
    with col2:
        industry = st.selectbox("Industry", ["All", "Technology", "Healthcare", "Financial Services"])
    
    # Create filter condition
    filter_condition = f"amount > {amount}"
    if industry != "All":
        filter_condition += f" AND industry = '{industry}'"
    
    # Add a button to generate/refresh the triangulation
    if st.button("Generate ICP Triangulation"):
        with st.spinner("Analyzing deal data to generate triangulation matrix..."):
            try:
                st.write("Entered agent")
                # Make request to backend with filter condition
                response = requests.post(
                    f"{backend_url}/api/v1/icp/triangulation",
                    json={"filter_condition": filter_condition}
                )
                st.write(response)
                if response.status_code == 200:
                    triangulation_data = response.json()
                    display_triangulation_matrix(triangulation_data)
                else:
                    st.error(f"Error generating triangulation data: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")

def display_triangulation_matrix(data: Dict[str, Any]):
    """
    Display the triangulation matrix with charts.
    
    Args:
        data: Triangulation data from the backend
    """
    # Display metadata
    st.markdown(f"**Analysis based on {data['metadata']['deals_analyzed']} deals**")
    
    # Display filters applied
    st.markdown("**Filters Applied:**")
    filters = data['metadata']['filters_applied']
    for filter_name, filter_data in filters.items():
        st.markdown(f"- {filter_name}: {filter_data['operator']} {filter_data['value']}")
    
    # Create metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Deals", data['metadata']['deals_analyzed'])
    with col2:
        st.metric("Won Deals", data['metadata']['won_deals'])
    with col3:
        st.metric("Win Rate", f"{data['metadata']['win_rate']:.1f}%")
    
    # Display dimensions
    st.subheader("Dimension Analysis")
    
    for dimension in data['dimensions']:
        st.markdown(f"### {dimension['attribute']}")
        
        # Create columns for win rate and sales cycle
        col1, col2 = st.columns(2)
        
        with col1:
            win_rate = dimension['highest_win_rate']
            st.markdown("**Win Rate Analysis**")
            st.markdown(f"Value: {win_rate['value']}")
            st.markdown(f"Win Rate: {win_rate['metric']}")
            st.progress(win_rate['confidence'])
            st.caption(f"Confidence: {win_rate['confidence']*100:.0f}%")
        
        if 'fastest_sales_cycle' in dimension:
            with col2:
                sales_cycle = dimension['fastest_sales_cycle']
                st.markdown("**Sales Cycle Analysis**")
                st.markdown(f"Value: {sales_cycle['value']}")
                st.markdown(f"Metric: {sales_cycle['metric']}")
                st.progress(sales_cycle['confidence'])
                st.caption(f"Confidence: {sales_cycle['confidence']*100:.0f}%")
    
    # Add visualization if plotly is available
    if PLOTLY_AVAILABLE:
        st.subheader("Visualization")
        
        # Prepare data for visualization
        dimensions = [dim['attribute'] for dim in data['dimensions']]
        win_rates = [dim['highest_win_rate']['raw_value'] for dim in data['dimensions']]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=dimensions,
                y=win_rates,
                text=[f"{rate:.1f}%" for rate in win_rates],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Win Rates by Dimension",
            xaxis_title="Dimension",
            yaxis_title="Win Rate (%)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add insights section
    st.subheader("Insights")
    st.markdown("""
    Based on the analysis:
    - The highest performing segment is shown in each dimension
    - Confidence scores indicate the reliability of the metrics
    - Use these insights to focus your sales efforts on the most promising segments
    """)
    
    # Add interactive question section
    st.subheader("Ask Questions")
    question = st.text_input(
        "Ask a question about the triangulation data",
        placeholder="Example: Why is this segment performing better?"
    )
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{st.session_state.backend_url}/api/v1/icp/analyze",
                        json={"question": question, "data": data}
                    )
                    if response.status_code == 200:
                        st.markdown(response.json()["response"])
                    else:
                        st.error("Error getting answer")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question") 