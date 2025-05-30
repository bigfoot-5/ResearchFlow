import os
import base64
import requests
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Retrieve Gong API credentials from environment variables
access_key = os.getenv('GONG_ACCESS_KEY')
access_secret = os.getenv('GONG_ACCESS_SECRET')
print(access_key)
print(access_secret)

# Combine access key and secret, then encode in Base64
credentials = f"{access_key}:{access_secret}"
encoded_credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')

# Set up headers with Basic Authentication
headers = {
    'Authorization': f'Basic {encoded_credentials}'
}

# Make a GET request to the Gong API to retrieve users
response = requests.get('https://us-60747.api.gong.io/v2/users', headers=headers)

# Check if the request was successful
if response.status_code == 200:
    users = response.json()
    print(f"The number of users are {len(users)}")
    print(json.dumps(users, indent=2, sort_keys=False))
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")