import requests
import os
from dotenv import load_dotenv
import base64
import json

# Load environment variables from .env file
load_dotenv()

# Get Gong credentials from environment variables
GONG_ACCESS_KEY = os.getenv('GONG_ACCESS_KEY')
GONG_ACCESS_KEY_SECRET = os.getenv('GONG_ACCESS_SECRET')
BASE_URL = 'https://us-60747.api.gong.io/v2/'

if not GONG_ACCESS_KEY or not GONG_ACCESS_KEY_SECRET:
    raise ValueError("GONG_ACCESS_KEY and GONG_ACCESS_SECRET environment variables must be set in your .env file.")

def get_headers():
    """Generate headers with proper authentication"""
    auth_string = f"{GONG_ACCESS_KEY}:{GONG_ACCESS_KEY_SECRET}"
    auth_bytes = auth_string.encode('ascii')
    base64_auth = base64.b64encode(auth_bytes).decode('ascii')
    
    return {
        'Authorization': f'Basic {base64_auth}',
        'Content-Type': 'application/json'
    }

def get_calls():
    url = BASE_URL + 'calls'
    params = {
        'limit': 10  # Adjust as needed
    }

    try:
        response = requests.get(url, headers=get_headers(), params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return None

if __name__ == "__main__":
    call_data = get_calls()
    if call_data:
        print(json.dumps(call_data, indent=2))