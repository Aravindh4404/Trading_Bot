import yfinance as yf
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time

# Function to get real-time stock price data from Yahoo Finance
def get_stock_price(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")  # 1-minute intraday data
    return data

# Function to fetch news headlines using Google News RSS (No API Key required)
def get_news_headlines(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock"
    feed = feedparser.parse(url)
    headlines = [entry["title"] for entry in feed.entries]
    return headlines[:10]  # Return top 10 headlines

# Function to perform sentiment analysis on headlines
def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {}

    print("\n=== NEWS HEADLINES & SENTIMENT ANALYSIS ===")
    for headline in headlines:
        sentiment = analyzer.polarity_scores(headline)
        sentiment_scores[headline] = sentiment["compound"]
        sentiment_label = "Positive" if sentiment["compound"] > 0.05 else "Negative" if sentiment["compound"] < -0.05 else "Neutral"
        print(f"Headline: {headline}\nSentiment Score: {sentiment['compound']} → {sentiment_label}\n")

    # Calculate the average sentiment score
    avg_sentiment = sum(sentiment_scores.values()) / len(sentiment_scores) if sentiment_scores else 0
    return avg_sentiment

# Trading decision based on sentiment score
def trading_strategy(symbol):
    stock_data = get_stock_price(symbol)
    news_headlines = get_news_headlines(symbol)

    if not news_headlines:
        print("No news headlines found. Skipping sentiment analysis.")
        return

    sentiment_score = analyze_sentiment(news_headlines)

    last_price = stock_data['Close'].iloc[-1]  # Get last closing price
    print(f"\nStock: {symbol}, Last Price: {last_price}, Overall Sentiment Score: {sentiment_score}")

    if sentiment_score > 0.05:
        print("🔼 Positive sentiment detected! Buying stock...")
    elif sentiment_score < -0.05:
        print("🔽 Negative sentiment detected! Selling stock...")
    else:
        print("➡️ Neutral sentiment. Holding position.")

# Running the script for a stock symbol
if __name__ == "__main__":
    while True:
        # trading_strategy("TSLA")
        trading_strategy("BTC-USD")
        time.sleep(60)  # Run every 1 minute
