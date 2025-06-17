import sqlite3
import json
import pandas as pd
from dotenv import load_dotenv
import os
import requests
import time

# Load environment variables
load_dotenv()

# Load API key from .env
HUBSPOT_API_KEY = os.getenv('HUBSPOT_API_KEY')
if not HUBSPOT_API_KEY:
    raise ValueError("HUBSPOT_API_KEY not found in .env file")

# Headers for API authentication
HEADERS = {
    'Authorization': f'Bearer {HUBSPOT_API_KEY}',
    'Content-Type': 'application/json'
}

def get_deal_stages():
    """Fetch all deal stages and their mappings from HubSpot"""
    url = "https://api.hubapi.com/crm/v3/pipelines/deals"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching deal stages: {response.status_code} - {response.text}")
    
    pipelines = response.json().get('results', [])
    stage_mapping = {}
    
    for pipeline in pipelines:
        pipeline_id = pipeline['id']
        for stage in pipeline['stages']:
            stage_id = stage['id']
            stage_label = stage['label'].lower().replace(' ', '')
            stage_mapping[stage_id] = stage_label
    
    return stage_mapping

def list_object_properties(object_type):
    """Fetch all available properties for a given object type"""
    url = f"https://api.hubapi.com/crm/v3/properties/{object_type}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching properties for {object_type}: {response.status_code} - {response.text}")
    
    return [prop["name"] for prop in response.json().get("results", [])]

def get_info(item):
    """Fetch all records for a given object type"""
    props = list_object_properties(item)
    url = f"https://api.hubapi.com/crm/v4/objects/{item}"
    all_results = []
    after = None

    while True:
        params = {
            "properties": ",".join(props),
            "limit": 100
        }
        if after:
            params["after"] = after

        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching {item}: {response.status_code} - {response.text}")

        data = response.json()
        all_results.extend(data.get('results', []))

        paging = data.get('paging')
        if paging and "next" in paging:
            after = paging["next"]["after"]
        else:
            break

        # Add a small delay to avoid rate limiting
        time.sleep(0.1)

    return all_results

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

def get_associated_companies(item,deal_id):
    """
    Retrieves all companies associated with a given deal.
    """
    # url = f"https://api.hubapi.com/crm/v4/objects/deals/{deal_id}/associations/companies"
    url = f"https://api.hubapi.com/crm/v4/objects/{item}/{deal_id}/associations/companies"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        associations = response.json().get('results', [])
        company_ids = [assoc['toObjectId'] for assoc in associations]
        return company_ids
    else:
        raise Exception(f"Error fetching associations: {response.status_code} - {response.text}")

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
        "John's Startup Idea",
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
    try:
        # First, get the deal stage mappings
        stage_mapping = get_deal_stages()
        print("\nDeal Stage Mappings:")
        print(json.dumps(stage_mapping, indent=2))
        
        # Then fetch the data
        objects_list = ["deals"]
        data_dict = {}
        for item in objects_list:
            print(f"\nFetching {item}...")
            data = get_info(item=item)
            data_dict[item] = data
            print(f"✅ {len(data)} records fetched")
        
        # Print a sample of the data
        print("\nSample Contact:")
        # print(json.dumps(data_dict["contacts"][0], indent=2))
        print("\nSample Deal:")
        print(json.dumps(data_dict["deals"][0], indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
