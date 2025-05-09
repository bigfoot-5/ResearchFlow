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
    # Store backend URL in session state for use in callbacks
    st.session_state.backend_url = backend_url
    
    st.header("🔍 ICP Triangulation Matrix")
    st.subheader("Discover your best customer segments")
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly package is not installed. Some visualizations will not be available. Please run 'pip install plotly' to enable all features.")
    
    st.markdown(
        """
        This matrix shows your best-performing customer segments across key dimensions.
        For each dimension (like industry, company size, etc.), it highlights which 
        values yield the fastest sales cycles and highest win rates.
        """
    )
    
    # Add a button to generate/refresh the triangulation
    if st.button("Generate ICP Triangulation"):
        with st.spinner("Analyzing deal data to generate triangulation matrix..."):
            try:
                # Update endpoint to include /api/v1 prefix
                response = requests.get(f"{backend_url}/api/v1/icp/triangulation")
                if response.status_code == 200:
                    triangulation_data = response.json()
                    
                    # Store in session state
                    st.session_state.triangulation_data = triangulation_data
                    
                    st.success("Triangulation matrix generated successfully!")
                else:
                    st.error(f"Error generating triangulation data: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")
    
    # Display triangulation data if available
    if hasattr(st.session_state, "triangulation_data"):
        display_triangulation_matrix(st.session_state.triangulation_data)

def display_triangulation_matrix(data: Dict[str, Any]):
    """
    Display the triangulation matrix with charts.
    
    Args:
        data: Triangulation data from the backend
    """
    # Display metadata
    st.markdown(f"**Analysis based on {data['metadata']['deals_analyzed']} deals**")
    
    # Create three columns for our metrics
    cols = st.columns(3)
    
    # Extract the dimensions
    dimensions = data['dimensions']
    
    # Prepare data for charts
    win_rate_data = []
    sales_cycle_data = []
    
    for dim in dimensions:
        attribute = dim['attribute']
        win_rate = dim['highest_win_rate']
        sales_cycle = dim['fastest_sales_cycle']
        
        win_rate_data.append({
            'Dimension': attribute,
            'Value': win_rate['value'],
            'Win Rate': win_rate['raw_value'],
            'Confidence': win_rate['confidence']
        })
        
        sales_cycle_data.append({
            'Dimension': attribute,
            'Value': sales_cycle['value'],
            'Sales Cycle': sales_cycle['raw_value'],
            'Confidence': sales_cycle['confidence']
        })
    
    # Create DataFrames
    win_rate_df = pd.DataFrame(win_rate_data)
    sales_cycle_df = pd.DataFrame(sales_cycle_data)
    
    # Display the triangulation matrix as cards
    st.subheader("Best Performing Segments")
    
    for dim in dimensions:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {dim['attribute']}")
            
            # Highest Win Rate
            st.markdown("**Highest Win Rate:**")
            win_rate = dim['highest_win_rate']
            st.markdown(f"**{win_rate['value']}** - {win_rate['metric']}")
            st.progress(win_rate['confidence'])
            st.caption(f"Confidence: {win_rate['confidence']*100:.0f}%")
            
        with col2:
            st.markdown("&nbsp;")  # Space for alignment
            
            # Fastest Sales Cycle
            st.markdown("**Fastest Sales Cycle:**")
            sales_cycle = dim['fastest_sales_cycle']
            st.markdown(f"**{sales_cycle['value']}** - {sales_cycle['metric']}")
            st.progress(sales_cycle['confidence'])
            st.caption(f"Confidence: {sales_cycle['confidence']*100:.0f}%")
    
    # Create visualization tabs only if plotly is available
    if PLOTLY_AVAILABLE:
        st.subheader("Visualizations")
        tab1, tab2, tab3 = st.tabs(["Win Rate", "Sales Cycle", "Combined View"])
        
        with tab1:
            # Create win rate chart
            fig_win = px.bar(
                win_rate_df,
                x='Dimension',
                y='Win Rate',
                color='Value',
                text='Value',
                title="Win Rate by Dimension",
                labels={'Win Rate': 'Win Rate (%)', 'Dimension': ''},
                height=400
            )
            st.plotly_chart(fig_win, use_container_width=True)
        
        with tab2:
            # Create sales cycle chart
            fig_cycle = px.bar(
                sales_cycle_df,
                x='Dimension',
                y='Sales Cycle',
                color='Value',
                text='Value',
                title="Sales Cycle by Dimension (Days)",
                labels={'Sales Cycle': 'Avg. Days to Close', 'Dimension': ''},
                height=400
            )
            st.plotly_chart(fig_cycle, use_container_width=True)
        
        with tab3:
            # Create a combined radar chart
            categories = [dim['attribute'] for dim in dimensions]
            win_rate_values = [dim['highest_win_rate']['confidence'] * 100 for dim in dimensions]
            sales_cycle_values = [dim['fastest_sales_cycle']['confidence'] * 100 for dim in dimensions]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=win_rate_values,
                theta=categories,
                fill='toself',
                name='Win Rate Confidence'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=sales_cycle_values,
                theta=categories,
                fill='toself',
                name='Sales Cycle Confidence'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="Confidence Scores by Dimension"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly to view interactive visualizations: `pip install plotly`")
        
        # Display basic table instead
        st.subheader("Win Rate Data")
        st.dataframe(win_rate_df)
        
        st.subheader("Sales Cycle Data")
        st.dataframe(sales_cycle_df)
    
    # Interactive explanation
    st.subheader("Insights")
    
    # Prepare text area for follow-up questions
    st.text_area(
        "Ask follow-up questions about the triangulation data",
        placeholder="Example: Why is Fintech performing better than other industries?",
        key="triangulation_question"
    )
    
    if st.button("Get Insights"):
        if "triangulation_question" in st.session_state and st.session_state.triangulation_question:
            with st.spinner("Analyzing..."):
                # Construct a query for the agent
                query = st.session_state.triangulation_question
                
                try:
                    # Get backend URL from session state
                    backend_url = st.session_state.backend_url
                    
                    # Call the agent API with the triangulation data for analysis
                    analysis_request = {
                        "query": query,
                        "data_type": "triangulation",
                        "data": data,
                        "additional_context": {
                            "visualization_available": True
                        }
                    }
                    
                    # Make the API call - Update endpoint to include /api/v1 prefix
                    response = requests.post(f"{backend_url}/api/v1/agents/analyze", json=analysis_request)
                    response.raise_for_status()
                    
                    # Process the response
                    analysis_result = response.json()
                    st.markdown(analysis_result["response"])
                    
                    # Show execution info
                    st.caption(f"Analysis completed in {analysis_result['metadata']['execution_time']}")
                except Exception as e:
                    st.error(f"Error getting insights: {str(e)}")
        else:
            st.warning("Please enter a question first.") 