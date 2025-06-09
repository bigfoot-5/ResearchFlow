# import psycopg2
from autogen import ConversableAgent
from autogen import GroupChat, GroupChatManager
import os
from dotenv import load_dotenv
load_dotenv()
your_llm_config = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": os.environ["OPENAI_API_KEY"]
        }
    ]
}

def run_postgres_query(sql_query: str):
    """Execute SQL query on metrics_table and return results as list of dictionaries"""
    try:
        conn = psycopg2.connect(
            dbname="cognition_db",
            user="cognition_user",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        cur.execute(sql_query)
        columns = [desc[0] for desc in cur.description]  # Get column names
        rows = cur.fetchall()
        conn.close()

        # Convert to list of dicts
        results = [dict(zip(columns, row)) for row in rows]
        return results
    except Exception as e:
        return {"error": f"Database error: {str(e)}"}

# Master agent - the main data analyst
master_agent = ConversableAgent(
    name="DataAnalyst",
    llm_config=your_llm_config,
    system_message="""You are an expert data analyst specializing in ICP (Ideal Customer Profile) segmentation.
    
    You have access to the 'metrics_table' database containing comprehensive customer metrics.
    
    When users ask questions:
    1. Analyze their request and determine what data they need
    2. Convert their question into appropriate SQL queries for the metrics_table
    3. Execute the queries using the run_postgres_query function
    4. Interpret the results and provide clear, business-friendly insights
    5. Always end with actionable recommendations
    
    Key skills:
    - Converting business questions into SQL queries
    - Analyzing segmentation data (geography, industry, company size)
    - Providing strategic insights from data
    - Making recommendations based on patterns in the data
    
    IMPORTANT: Always provide a final summary in business language with specific insights and recommendations.
    Don't just return raw data - interpret what the data means for the business.""",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Register the database function
master_agent.register_for_execution(name="run_postgres_query")(run_postgres_query)
master_agent.register_for_llm(
    name="run_postgres_query", 
    description="Execute SQL queries on the metrics_table database to retrieve customer segmentation data"
)(run_postgres_query)

# User proxy agent - represents the user/manager
user_proxy = ConversableAgent(
    name="BusinessManager",
    llm_config=your_llm_config,
    system_message="""You are a business manager seeking insights from customer data.
    
    Your role:
    1. Present user questions to the DataAnalyst
    2. Ask follow-up questions if the analysis needs clarification
    3. Ensure the DataAnalyst provides actionable business insights
    4. Request specific examples or deeper analysis when needed
    
    You should be engaging and ask relevant follow-up questions to get the most value from the data analysis.""",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",  # No human input needed
)

# Create simplified group chat with just two agents
group = GroupChat(
    agents=[user_proxy, master_agent],
    messages=[],
    max_round=6,  # Reduced rounds since we have fewer agents
    admin_name="DataAnalyst",  # Data analyst leads the conversation
    speaker_selection_method="auto",
)

manager = GroupChatManager(
    groupchat=group,
    llm_config=your_llm_config,
    system_message="""You manage a conversation between a business manager and a data analyst.
    
    Ensure that:
    1. The data analyst properly queries the database and analyzes results
    2. The business manager gets actionable insights
    3. The conversation stays focused on providing business value
    4. The final output includes clear recommendations
    
    Keep the conversation concise but thorough."""
)

# Function to ask questions to the simplified group
def ask_agents(question: str):
    """
    Ask a question to the simplified two-agent group chat
    """
    # Reset the group chat for a fresh conversation
    group.reset()
    
    # User proxy initiates the chat with the business question
    result = user_proxy.initiate_chat(
        manager,
        message=f"I need analysis on this business question: {question}",
        max_rounds=6,
        clear_history=True,
        summary_method="reflection_with_llm"
    )
    
    return result


"""You are an expert ICP (Ideal Customer Profile) analyst with access to customer segmentation data.

Available tables:
- metrics_table: Main customer metrics
company_name   |    country     | numberofemployees |              industry               | deal_count |        win_rate        | avg_deal_velocity | avg_acv |    win_velocity     |  revenue_velocity
---------------+----------------+-------------------+-------------------------------------+------------+------------------------+-------------------+---------+---------------------+--------------------
 Jones LLC     | United States  |               498 | INFORMATION_TECHNOLOGY_AND_SERVICES |          1 | 1.00000000000000000000 |                47 |   87399 | 0.02127659574468085 | 1859.5531914893616
 Edwards-Vance | United Kingdom |               669 | ACCOUNTING                          |          1 | 0.00000000000000000000 |                72 |       |                   0   |
 Smith-Wright  | Netherlands    |               856 | FOOD_BEVERAGES                      |          1 | 0.00000000000000000000 |                67 |       |                   0   |

IMPORTANT: You MUST use the run_postgres_query function to get actual data before providing insights.

For every user question:
1. Identify which table(s) contain the relevant data
2. Write appropriate SQL queries
3. Call run_postgres_query with your SQL
4. Analyze the actual results
5. Provide business insights with specific numbers from the data


Always use actual data from queries in your responses."""