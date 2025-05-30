import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
# Replace with your actual HubSpot Private App access token
ACCESS_TOKEN = os.getenv('HUBSPOT_API_KEY')

# Define the headers for authentication
headers = {
    'Authorization': f'Bearer {ACCESS_TOKEN}',
    'Content-Type': 'application/json'
}

# List of HubSpot objects to export
objects = ['contacts', 'companies', 'deals']

# Base URL for HubSpot API
base_url = 'https://api.hubapi.com/crm/v3/objects/'

# Number of records to fetch per request (maximum is 100)
limit = 100

for obj in objects:
    print(f"Exporting {obj}...")

    all_records = []
    after = None

    while True:
        params = {
            'limit': limit,
            'archived': 'false'
        }
        if after:
            params['after'] = after

        url = f"{base_url}{obj}"
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Failed to fetch {obj}: {response.status_code} - {response.text}")
            break

        data = response.json()
        results = data.get('results', [])
        all_records.extend(results)

        paging = data.get('paging')
        if paging and 'next' in paging:
            after = paging['next']['after']
        else:
            break

    # Normalize the JSON data into a flat table
    df = pd.json_normalize(all_records)

    # Save to CSV
    csv_filename = f"{obj}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved {len(df)} records to {csv_filename}")