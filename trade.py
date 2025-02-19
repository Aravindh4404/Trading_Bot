import yfinance as yf
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import time
from datetime import datetime

##########################
#   Global Parameters    #
##########################
SYMBOL = "AAPL"
CHECK_INTERVAL = 60  # seconds
NEWS_URL_TEMPLATE = "https://finance.yahoo.com/quote/{symbol}?p={symbol}&.tsrc=fin-srch"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
POSITIVE_THRESHOLD = 0.2
NEGATIVE_THRESHOLD = -0.2
MAX_POSITION = 100  # Max shares to hold in this example
MIN_POSITION = 0  # Min shares


##########################
#   Helper Functions     #
##########################

def get_stock_price(symbol):
    """
    Fetches the latest intraday stock data (1-minute interval) for a given symbol.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume, etc.
    """
    stock = yf.Ticker(symbol)
    # For real-time strategies, you might use 'period="1d", interval="1m"',
    # but here we can also fetch a slightly bigger window for demonstration.
    data = stock.history(period="1d", interval="1m")
    return data


def get_news_headlines(symbol):
    """
    Scrapes the Yahoo Finance page for the given symbol to collect news headlines.
    Returns a list of headlines (strings).
    """
    url = NEWS_URL_TEMPLATE.format(symbol=symbol)
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    for item in soup.find_all('h3'):
        headline_text = item.get_text(strip=True)
        if headline_text:  # Avoid empty strings
            headlines.append(headline_text)
    return headlines


def analyze_headlines(headlines, relevant_keyword):
    """
    Performs sentiment analysis on a list of headlines using VADER.
    - relevant_keyword: A string to check for “relevance” of the headline,
      e.g., "Apple" or "AAPL"
    Returns:
      - A list of tuples: (headline, sentiment_score, sentiment_label, is_relevant)
      - The average sentiment overall
      - The average sentiment of relevant headlines only (if any)
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []
    relevant_scores = []

    for headline in headlines:
        sentiment = analyzer.polarity_scores(headline)
        score = sentiment["compound"]
        label = ("Positive" if score > POSITIVE_THRESHOLD
                 else "Negative" if score < NEGATIVE_THRESHOLD
        else "Neutral")
        # Check if the headline is "relevant" by searching for the keyword
        # (e.g., "Apple" or "AAPL" to see if it specifically mentions the company)
        is_relevant = relevant_keyword.lower() in headline.lower()
        if is_relevant:
            relevant_scores.append(score)
        results.append((headline, score, label, is_relevant))

    # Compute average sentiment across all headlines
    avg_score_all = sum([r[1] for r in results]) / len(results) if results else 0

    # Compute average sentiment across relevant headlines
    avg_score_relevant = (sum(relevant_scores) / len(relevant_scores)
                          if relevant_scores else 0)

    return results, avg_score_all, avg_score_relevant


def make_trading_decision(avg_score_relevant, stock_data, current_position):
    """
    Decides whether to buy, sell, or hold based on:
      - The average sentiment from relevant headlines.
      - A simple moving average of the last few minutes’ prices.
      - Current position (number of shares held).
    Returns a string indicating the action ("BUY", "SELL", or "HOLD")
    and the updated position.
    """
    last_price = stock_data['Close'].iloc[-1]
    short_moving_avg = stock_data['Close'].tail(5).mean()  # example of short window
    sentiment_factor = avg_score_relevant

    print(f"Last Price: {last_price:.2f}")
    print(f"Short Moving Average (5min): {short_moving_avg:.2f}")
    print(f"Relevant Headlines Sentiment: {sentiment_factor:.2f}")

    # Basic logic combining sentiment & moving average
    # This is intentionally simplistic and can be replaced with advanced logic or ML.
    if sentiment_factor > POSITIVE_THRESHOLD and last_price > short_moving_avg:
        # Potential uptrend sign
        if current_position < MAX_POSITION:
            print("🔼 Positive sentiment & price above short MA => BUY signal")
            return "BUY", current_position + 10  # buy 10 shares for example
        else:
            return "HOLD", current_position
    elif sentiment_factor < NEGATIVE_THRESHOLD and last_price < short_moving_avg:
        # Potential downtrend sign
        if current_position > MIN_POSITION:
            print("🔽 Negative sentiment & price below short MA => SELL signal")
            return "SELL", current_position - 10  # sell 10 shares
        else:
            return "HOLD", current_position
    else:
        # Otherwise, hold
        print("➡️ Neutral conditions => HOLD position")
        return "HOLD", current_position


def log_decision(log_file, symbol, action, position, sentiment_all, sentiment_relevant, last_price):
    """
    Logs each trading decision to a CSV file for performance evaluation.
    Columns: Timestamp, Symbol, Action, Position, OverallSentiment, RelevantSentiment, LastPrice
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{
        "Timestamp": now,
        "Symbol": symbol,
        "Action": action,
        "Position": position,
        "OverallSentiment": sentiment_all,
        "RelevantSentiment": sentiment_relevant,
        "LastPrice": last_price
    }])
    # Append mode to keep track of all decisions over time
    df.to_csv(log_file, mode='a', header=not pd.io.common.file_exists(log_file), index=False)


##########################
#     Main Agent Loop    #
##########################

def trading_agent(symbol=SYMBOL, log_file="trading_log.csv"):
    """
    Main loop that the agent will run periodically:
      1. Get current stock data.
      2. Scrape latest news headlines.
      3. Analyze sentiment.
      4. Decide whether to buy, sell, or hold.
      5. Log decision.
    """
    # Track the agent’s current position (start with 0 shares, can be replaced with a real system)
    current_position = 0

    while True:
        print("\n" + "=" * 50)
        print(f"[{datetime.now()}] Checking market data for {symbol}")

        # 1. Fetch stock data
        stock_data = get_stock_price(symbol)
        if stock_data.empty:
            print("Could not retrieve stock data. Retrying...")
            time.sleep(CHECK_INTERVAL)
            continue

        # 2. Fetch news headlines
        headlines = get_news_headlines(symbol)
        if not headlines:
            print("No news headlines found. Holding position.")
            time.sleep(CHECK_INTERVAL)
            continue

        # 3. Analyze sentiment
        # Here we treat the “relevant keyword” as the actual company name for better contextual checks
        # (e.g., "Apple"). Could also do "AAPL" or both.
        results, avg_score_all, avg_score_relevant = analyze_headlines(headlines, "Apple")

        # Print headlines & sentiment details
        print("\n=== NEWS HEADLINES & SENTIMENT ANALYSIS ===")
        for (headline, score, label, is_relevant) in results:
            tag = "[Relevant]" if is_relevant else "[Not Relevant]"
            print(f"{tag} Headline: {headline}")
            print(f"  → Sentiment Score: {score:.2f} ({label})\n")

        # 4. Make a trading decision
        action, updated_position = make_trading_decision(avg_score_relevant, stock_data, current_position)
        last_price = stock_data['Close'].iloc[-1]

        # 5. Log the decision for performance evaluation
        log_decision(log_file, symbol, action, updated_position, avg_score_all, avg_score_relevant, last_price)

        # Update position
        current_position = updated_position

        # Sleep until the next iteration
        print(f"Position after this action: {current_position} shares\n"
              f"Waiting {CHECK_INTERVAL} seconds before next check...")
        time.sleep(CHECK_INTERVAL)


##########################
#   Execution Starting   #
##########################
if __name__ == "__main__":
    try:
        trading_agent()
    except KeyboardInterrupt:
        print("\nStopping the trading agent. Goodbye!")
