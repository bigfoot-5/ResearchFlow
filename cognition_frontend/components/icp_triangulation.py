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
    
    # # Add filter inputs
    # st.subheader("Filter Deals")
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     amount = st.number_input("Minimum Deal Amount", min_value=0, value=50000)
    
    # with col2:
    #     industry = st.selectbox("Industry", ["All", "Technology", "Healthcare", "Financial Services"])
        
    # with col3:
    #     timeframe = st.selectbox("Timeframe", ["Last 30 Days", "Last 90 Days", "Last Year", "All Time"])
    
    # # Create filter condition
    # filter_condition = f"amount > {amount}"
    # if industry != "All":
    #     filter_condition += f" AND industry = '{industry}'"
    
    # Add a button to generate/refresh the triangulation
    if st.button("Generate ICP Triangulation"):
        with st.spinner("Analyzing deal data to generate triangulation matrix..."):
            try:
                # Make request to backend with filter condition
                response = requests.post(
                    f"{backend_url}/api/v1/icp/triangulation",
                    json={
                        "filter_condition": "amount > 0",
                        "timeframe": "All Time"
                    }
                )
                
                if response.status_code == 200:
                    triangulation_data = response.json()
                    display_triangulation_matrix(triangulation_data)
                else:
                    st.error(f"Error generating triangulation data: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")

def display_triangulation_matrix(data: Dict[str, Any]):
    """
    Display the triangulation matrix in a grid/table style matching the provided screenshot.
    """
    st.markdown(f"**Analysis based on {data['metadata']['deals_analyzed']} deals**")
    # st.markdown("**Filters Applied:**")
    # filters = data['metadata']['filters_applied']
    # for filter_name, filter_data in filters.items():
    #     st.markdown(f"- {filter_name}: {filter_data['operator']} {filter_data['value']}")

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Deals", data['metadata']['deals_analyzed'])
    with col2:
        st.metric("Won Deals", data['metadata']['won_deals'])
    with col3:
        st.metric("Win Rate", f"{data['metadata']['win_rate']:.1f}%")
    with col4:
        st.metric("Avg Deal Size", f"${data['metadata']['avg_deal_size']:,.0f}")

    # --- Triangulation Matrix Table ---
    st.subheader("Dimension Analysis Matrix")
    # Display as DataFrame
    df_rows = []
    for dim in data['dimensions']:
        df_rows.append({
            "Attribute": dim["attribute"],
            "Fastest Sales Cycle": f"{dim['fastest_sales_cycle']['value']} ({dim['fastest_sales_cycle']['metric']})",
            "Highest Win Rate": f"{dim['highest_win_rate']['value']} ({dim['highest_win_rate']['metric']})"
        })
    df = pd.DataFrame(df_rows)
    st.dataframe(df)

    # Add visualization if plotly is available
    if PLOTLY_AVAILABLE:
        st.subheader("Performance Visualization")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Win Rates", "Sales Cycle", "Combined Analysis"])
        
        with tab1:
            display_win_rates_chart(data['dimensions'])
            
        with tab2:
            display_sales_cycle_chart(data['dimensions'])
            
        with tab3:
            display_combined_analysis(data['dimensions'])
    
    # Add insights section
    st.subheader("Key Insights")
    display_insights(data)
    
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

def display_triangulation_grid(dimensions: List[Dict[str, Any]]):
    """
    Display the triangulation matrix as a table/grid with progress bars and subtext.
    """
    st.markdown(
        """
        <style>
        .triangulation-table th, .triangulation-table td {padding: 8px 16px;}
        .triangulation-table th {text-align: center;}
        .triangulation-table td {vertical-align: top;}
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown(
        "<table class='triangulation-table' style='width:100%'><tr>"
        "<th>Attribute</th>"
        "<th>Fastest Sales Cycle</th>"
        "<th>Highest Win Rate</th>"
        "</tr>", unsafe_allow_html=True
    )
    for dim in dimensions:
        attr = dim["attribute"]
        # Fastest Sales Cycle
        fsc_value = dim["fastest_sales_cycle"]["value"]
        fsc_metric = dim["fastest_sales_cycle"]["metric"]
        fsc_conf = dim["fastest_sales_cycle"].get("confidence", 0.85)
        # Highest Win Rate
        hwr_value = dim["highest_win_rate"]["value"]
        hwr_metric = dim["highest_win_rate"]["metric"]
        hwr_conf = dim["highest_win_rate"].get("confidence", 0.85)
        st.markdown(
            f"<tr>"
            f"<td style='text-align:center; font-weight:bold;'>{attr}</td>"
            f"<td style='text-align:center;'>"
            f"<div style='font-weight:bold;'>{fsc_value}</div>"
            f"<progress value='{fsc_conf}' max='1' style='width:80%; height:10px;'></progress><br>"
            f"<span style='color:green; font-size:0.95em'>{fsc_metric}</span>"
            f"</td>"
            f"<td style='text-align:center;'>"
            f"<div style='font-weight:bold;'>{hwr_value}</div>"
            f"<progress value='{hwr_conf}' max='1' style='width:80%; height:10px;'></progress><br>"
            f"<span style='color:green; font-size:0.95em'>{hwr_metric}</span>"
            f"</td>"
            f"</tr>",
            unsafe_allow_html=True
        )
    st.markdown("</table>", unsafe_allow_html=True)

def display_win_rates_chart(dimensions: List[Dict[str, Any]]):
    """Display win rates chart for all dimensions."""
    fig = go.Figure()
    
    for dimension in dimensions:
        win_rate = dimension['highest_win_rate']
        fig.add_trace(go.Bar(
            name=dimension['attribute'],
            x=[dimension['attribute']],
            y=[win_rate['raw_value']],
            text=[f"{win_rate['raw_value']:.1f}%"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Win Rates by Dimension",
        xaxis_title="Dimension",
        yaxis_title="Win Rate (%)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_sales_cycle_chart(dimensions: List[Dict[str, Any]]):
    """Display sales cycle chart for all dimensions."""
    fig = go.Figure()
    
    for dimension in dimensions:
        if 'fastest_sales_cycle' in dimension:
            sales_cycle = dimension['fastest_sales_cycle']
            fig.add_trace(go.Bar(
                name=dimension['attribute'],
                x=[dimension['attribute']],
                y=[sales_cycle['raw_value']],
                text=[f"{sales_cycle['raw_value']:.0f} days"],
                textposition='auto',
            ))
    
    fig.update_layout(
        title="Sales Cycle by Dimension",
        xaxis_title="Dimension",
        yaxis_title="Days to Close",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_combined_analysis(dimensions: List[Dict[str, Any]]):
    """Display combined analysis of win rates and sales cycles."""
    fig = go.Figure()
    
    for dimension in dimensions:
        win_rate = dimension['highest_win_rate']
        if 'fastest_sales_cycle' in dimension:
            sales_cycle = dimension['fastest_sales_cycle']
            fig.add_trace(go.Scatter(
                name=dimension['attribute'],
                x=[sales_cycle['raw_value']],
                y=[win_rate['raw_value']],
                mode='markers+text',
                text=[dimension['attribute']],
                textposition="top center",
                marker=dict(size=15)
            ))
    
    fig.update_layout(
        title="Win Rate vs Sales Cycle Analysis",
        xaxis_title="Days to Close",
        yaxis_title="Win Rate (%)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_insights(data: Dict[str, Any]):
    """Display key insights from the triangulation data."""
    # Extract top performing segments
    top_segments = []
    for dimension in data['dimensions']:
        win_rate = dimension['highest_win_rate']
        if 'fastest_sales_cycle' in dimension:
            sales_cycle = dimension['fastest_sales_cycle']
            top_segments.append({
                'dimension': dimension['attribute'],
                'segment': win_rate['value'],
                'win_rate': win_rate['raw_value'],
                'sales_cycle': sales_cycle['raw_value']
            })
    
    # Display insights in a structured format
    st.markdown("### Top Performing Segments")
    for segment in top_segments:
        st.markdown(f"""
        **{segment['dimension']}: {segment['segment']}**
        - Win Rate: {segment['win_rate']:.1f}%
        - Sales Cycle: {segment['sales_cycle']:.0f} days
        """)
    
    # Add recommendations
    st.markdown("### Recommendations")
    st.markdown("""
    1. Focus sales efforts on the highest performing segments identified above
    2. Develop targeted messaging for each high-performing segment
    3. Consider adjusting pricing or packaging for segments with longer sales cycles
    4. Invest in case studies and testimonials from top-performing segments
    """) 