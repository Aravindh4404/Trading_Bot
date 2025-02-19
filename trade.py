import yfinance as yf
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time


# Function to get real-time stock price data from Yahoo Finance
def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")  # 1-minute intraday data
    return data


# Function to scrape real-time news headlines from Yahoo Finance
def get_news_headlines(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}&.tsrc=fin-srch"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for item in soup.find_all('h3', class_="Mb(5px)"):
        headlines.append(item.text.strip())

    return headlines


# Function to perform sentiment analysis on news headlines
def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for headline in headlines:
        sentiment = analyzer.polarity_scores(headline)
        sentiment_scores.append(sentiment["compound"])  # Compound score (overall sentiment)

    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0  # Average sentiment score


# Trading decision based on sentiment score
def trading_strategy(symbol):
    stock_data = get_stock_price(symbol)
    news_headlines = get_news_headlines(symbol)

    sentiment_score = analyze_sentiment(news_headlines)

    last_price = stock_data['Close'].iloc[-1]  # Get last closing price
    print(f"Stock: {symbol}, Last Price: {last_price}, Sentiment Score: {sentiment_score}")

    if sentiment_score > 0.2:
        print("Positive sentiment detected! Buying stock...")
        # Execute buy order (pseudo-code)
        # execute_order(symbol, "BUY")
    elif sentiment_score < -0.2:
        print("Negative sentiment detected! Selling stock...")
        # Execute sell order (pseudo-code)
        # execute_order(symbol, "SELL")
    else:
        print("Neutral sentiment. Holding position.")


# Running the script for a stock symbol (Example: Apple 'AAPL')
if __name__ == "__main__":
    while True:
        trading_strategy("AAPL")
        time.sleep(60)  # Run every 1 minute (adjust as needed)
