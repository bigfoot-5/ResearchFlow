import requests

# Replace this with your Gong API token
GONG_API_TOKEN = 'your_api_token_here'
BASE_URL = 'https://api.gong.io/v2/'

HEADERS = {
    'Authorization': f'Bearer {GONG_API_TOKEN}',
    'Content-Type': 'application/json'
}

def get_calls():
    url = BASE_URL + 'calls'
    params = {
        'limit': 10  # Adjust as needed
    }

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    call_data = get_calls()
    if call_data:
        print(call_data)