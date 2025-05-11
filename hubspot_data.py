import sqlite3
import json
import pandas as pd

def insert_data_to_db(contacts, companies, deals, db_path="hubspot_etl.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    for c in contacts:
        props = c["properties"]
        cur.execute("""
            INSERT OR REPLACE INTO contacts (id, firstname, lastname, email, createdate, lastmodifieddate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            c["id"],
            props.get("firstname", ""),
            props.get("lastname", ""),
            props.get("email", ""),
            props.get("createdate", ""),
            props.get("lastmodifieddate", "")
        ))

    for comp in companies:
        props = comp["properties"]
        cur.execute("""
            INSERT OR REPLACE INTO companies (id, name, domain, createdate)
            VALUES (?, ?, ?, ?)
        """, (
            comp["id"],
            props.get("name", ""),
            props.get("domain", ""),
            props.get("createdate", "")
        ))

    for d in deals:
        props = d["properties"]
        cur.execute("""
            INSERT OR REPLACE INTO deals (id, dealname, amount, dealstage, pipeline, closedate, createdate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            d["id"],
            props.get("dealname", ""),
            float(props.get("amount", 0.0)),
            props.get("dealstage", ""),
            props.get("pipeline", ""),
            props.get("closedate", ""),
            props.get("createdate", "")
        ))

    conn.commit()
    conn.close()
from dotenv import load_dotenv

load_dotenv() 
import os
import requests

# Load API key from .env
HUBSPOT_API_KEY = os.getenv('HUBSPOT_API_KEY')

# Headers for API authentication
HEADERS = {
    'Authorization': f'Bearer {HUBSPOT_API_KEY}',
    'Content-Type': 'application/json'
}

# Example: Fetch Deals
def get_deals(limit=5):
    url = "https://api.hubapi.com/crm/v3/objects/deals"
    params = {
        "limit": limit,
        "properties": "dealname,amount,dealstage,createdate"
    }

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        raise Exception(f"Error fetching deals: {response.status_code} - {response.text}")
def get_deal_by_id(deal_id):
    url = f"https://api.hubapi.com/crm/v3/objects/deals/{deal_id}"
    params = {
        "properties": "dealname,amount,dealstage,createdate,closedate,hubspot_owner_id"
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching deal {deal_id}: {response.status_code} - {response.text}")
def list_object_properties(object_type):
    url = f"https://api.hubapi.com/crm/v3/properties/{object_type}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return [prop["name"] for prop in response.json().get("results", [])]
    else:
        raise Exception(f"Error fetching properties for {object_type}: {response.status_code} - {response.text}")

def get_info(item):
    props = list_object_properties(item)
    url = f"https://api.hubapi.com/crm/v3/objects/{item}"
    params = {
        "properties": ",".join(props)
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        raise Exception(f"Error fetching {item}: {response.status_code} - {response.text}")
def create_sample_contact():
    url = "https://api.hubapi.com/crm/v3/objects/contacts"
    data = {
        "properties": {
            "firstname": "Jane",
            "lastname": "Doe",
            "email": "janedoe@example.com"
        }
    }
    response = requests.post(url, headers=HEADERS, json=data)
    if response.status_code == 201:
        print("Sample contact created:", response.json()['id'])
    else:
        print("Failed:", response.status_code, response.text)

from openai import OpenAI

client = OpenAI()

def predict_deal_success(deal_data):
    # Construct prompt with relevant deal info
    prompt = "Given the following deal data, predict whether the deal was successful or not. Provide reasoning. Finally give a score from 1 to 100 as to how likely is it that the deal is successful.\n"
    for deal in deal_data:
        deal_info = ", ".join(f"{k}: {v}" for k, v in deal['properties'].items())
        prompt += f"\nDeal ID: {deal['id']}, {deal_info}"

    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[
            {"role": "system", "content": "You are a helpful assistant that predicts deal outcomes based on CRM data."},
            {"role": "user", "content": prompt}
        ]
    )
    print("LLM Prediction:\n", response.output_text)
def create_synthetic_dataframe(num_entries):
    winning_companies = [
        "Acme Technologies Ltd.",
        "Globex Corporation",
        "InnovaSoft Inc.",
        "Quantum Dynamics Group",
        "Nimbus Cloud Services",
        "Vertex Financial Partners",
        "Bluewave Solutions",
        "OmniRetail Group",
        "SecureNet Systems",
        "FutureWorks AI"
    ]

    # Companies likely to lose deals
    losing_companies = [
        "John’s Startup Idea",
        "FastMoney.biz",
        "Click4Leads",
        "Test Company",
        "CoolProject2023",
        "NoCompanyName",
        "FreeEmailClientUser",
        "Test123 Corporation",
        "SideHustle Network",
        "TempClient Placeholder"
    ]
    deal_name = winning_companies + losing_companies
    deal_label = ["deal_won"] * num_entries + ["deal_lost"] * num_entries
    deal_dict = {
        "company_name" : deal_name,
        "class_name": deal_label
    }
    synthetic_deal_df = pd.DataFrame(deal_dict)
    return synthetic_deal_df

if __name__ == "__main__":
    # deal_df = create_synthetic_dataframe(10)
    # deal_df.to_csv("synthetic_hubspot_df_basic.csv")
    # print(deal_df)
    objects_list = ["contacts", "companies", "deals"]
    data_dict = {}
    for item in objects_list:
        data = get_info(item=item)
        data_dict[item] = data
        print(f"{item}: {len(data)} records fetched")
    #138616529118
    print(json.dumps(data_dict["contacts"], indent=4))
    # print(data_dict["deals"][0].keys())

    insert_data_to_db(
        contacts=data_dict["contacts"],
        companies=data_dict["companies"],
        deals=data_dict["deals"]
    )

    # print("✅ All data saved to SQLite database.")
    # predict_deal_success(data_dict["deals"])

# def list_object_schemas():
#     url = "https://api.hubapi.com/crm/v3/schemas"
#     response = requests.get(url, headers=HEADERS)

#     if response.status_code == 200:
#         return response.json().get('results', [])
#     else:
#         raise Exception(f"Error fetching schemas: {response.status_code} - {response.text}")


# if __name__ == "__main__":
#     schemas = list_object_schemas()
#     for obj in schemas:
#         print(f"{obj['name']} (label: {obj['labels']['singular']})")
