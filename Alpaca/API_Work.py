import requests
import config  # ✅ Import the config file

from datetime import datetime, timedelta

# Alpaca Market Data API Base URL
BASE_URL = "https://data.alpaca.markets/v2/stocks"

# Use API credentials from config.py
API_KEY = config.API_KEY
API_SECRET = config.API_SECRET

# Headers for authentication
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# Define the stock symbol
symbol = "AAPL"  # Example: Apple stock

# Define start and end dates for historical data
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=10)  # Get last 10 days of data

# Parameters for the API request
params = {
    "symbols": symbol,  # ✅ Fix: Ensure this is a string, NOT a list
    "timeframe": "1Day",
    "start": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "end": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "limit": 10
}

# Send request to Alpaca API
response = requests.get(f"{BASE_URL}/bars", headers=HEADERS, params=params)

# Print Response
print(f"Status Code: {response.status_code}")
print(f"Raw Response: {response.text}")

# Handle response
if response.status_code == 200:
    stock_data = response.json()
    print("Stock Data:", stock_data)
else:
    print(f"Error: {response.json()}")
