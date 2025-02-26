import requests



BASE_URL = "https://data.alpaca.markets"


headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}
response = requests.get(f"{BASE_URL}/v2/stocks/bars", headers=headers)

print("Raw Response:", response.text)  # Debugging step

stock_data = response.json()  # This is where your code is failing


#
# import requests
#
# API_KEY = "PKWLZQFMLRKZWJBQOJNB"
# API_SECRET = "8KBUACQjvvno7XzRwtgLkbjMGUoI1RZMcmDVLICW"
# BASE_URL = "https://paper-api.alpaca.markets"
#
# headers = {
#     "APCA-API-KEY-ID": API_KEY,
#     "APCA-API-SECRET-KEY": API_SECRET
# }
#
# # Get account info
# response = requests.get(f"{BASE_URL}/v2/account", headers=headers)
# print("Account Info:", response.json())
#
# # Get stock data for AAPL
# market_data_url = "https://data.alpaca.markets/v2/stocks/AAPL/bars"
# params = {"timeframe": "1Day", "limit": 5}
#
# response = requests.get(market_data_url, headers=headers, params=params)
# print("Stock Data:", response.json())
#
#
#
# # import requests
# #
# # API_KEY = "PKWLZQFMLRKZWJBQOJNB"
# # API_SECRET = "8KBUACQjvvno7XzRwtgLkbjMGUoI1RZMcmDVLICW"
# # BASE_URL = "https://paper-api.alpaca.markets"
# #
# # headers = {
# #     "APCA-API-KEY-ID": API_KEY,
# #     "APCA-API-SECRET-KEY": API_SECRET
# # }
# #
# # # Check Account Info
# # response = requests.get(f"{BASE_URL}/v2/account", headers=headers)
# # print(response.status_code, response.json())  # Should return 200 and account details
#
#
#
