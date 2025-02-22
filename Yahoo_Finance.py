import yfinance as yf
import feedparser
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class NewsItem:
    content: str
    timestamp: datetime
    source: str
    score: float


@dataclass
class MarketEnvironment:
    current_price: float
    volume: float
    timestamp: datetime
    news_items: List[NewsItem]
    market_hours: bool
    volatility: float


class TeslaTradingAgent:
    def __init__(self, symbol: str = "TSLA"):
        self.symbol = symbol
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Technical analysis parameters
        self.sma_short = 20
        self.sma_long = 50
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # Strategy weights
        self.weights = {
            'sentiment': 0.4,
            'technical': 0.6
        }

        # Risk parameters
        self.max_position_size = 0.2

        # Initialize data
        self.update_market_data()

    def update_market_data(self) -> bool:
        """Fetch latest market data"""
        try:
            self.stock = yf.Ticker(self.symbol)
            self.market_data = self.stock.history(period="100d")
            self.intraday_data = self.stock.history(period="1d", interval="1m")
            return True
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return False

    def get_news_headlines(self) -> List[NewsItem]:
        """Fetch and analyze news headlines"""
        url = f"https://news.google.com/rss/search?q={self.symbol}+stock"
        feed = feedparser.parse(url)
        news_items = []

        for entry in feed.entries[:10]:  # Top 10 headlines
            sentiment = self.sentiment_analyzer.polarity_scores(entry.title)
            news_items.append(NewsItem(
                content=entry.title,
                timestamp=datetime.now(),
                source="Google News",
                score=sentiment['compound']
            ))

        return news_items

    def calculate_technical_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators"""
        prices = self.market_data['Close']

        # Calculate SMAs
        sma_short = prices.rolling(window=self.sma_short).mean().iloc[-1]
        sma_long = prices.rolling(window=self.sma_long).mean().iloc[-1]

        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # Calculate MACD
        exp1 = prices.ewm(span=self.macd_fast).mean()
        exp2 = prices.ewm(span=self.macd_slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal).mean()
        histogram = macd - signal

        return {
            'sma_short': sma_short,
            'sma_long': sma_long,
            'rsi': rsi,
            'macd': macd.iloc[-1],
            'macd_signal': signal.iloc[-1],
            'macd_hist': histogram.iloc[-1]
        }

    def analyze_news_sentiment(self, news_items: List[NewsItem]) -> float:
        """Calculate weighted sentiment score"""
        if not news_items:
            return 0.0

        # Weight more recent news higher
        total_weight = 0
        weighted_sentiment = 0

        for i, news in enumerate(news_items):
            # Exponential decay weight based on position
            weight = np.exp(-0.1 * i)
            weighted_sentiment += news.score * weight
            total_weight += weight

        return weighted_sentiment / total_weight if total_weight > 0 else 0

    def get_technical_signal(self, indicators: Dict[str, float]) -> float:
        """Generate technical trading signal"""
        trend_signal = (indicators['sma_short'] - indicators['sma_long']) / indicators['sma_long']

        rsi_signal = 0.0
        if indicators['rsi'] < 30:
            rsi_signal = 1.0
        elif indicators['rsi'] > 70:
            rsi_signal = -1.0

        macd_signal = np.clip(indicators['macd_hist'] / indicators['macd'], -1, 1)

        technical_signal = (
                0.4 * trend_signal +
                0.3 * rsi_signal +
                0.3 * macd_signal
        )

        return np.clip(technical_signal, -1, 1)

    def decide_action(self) -> Tuple[str, float, Dict]:
        """Make trading decision based on technical and sentiment analysis"""
        # Update market data
        self.update_market_data()

        # Get latest price and news
        current_price = self.intraday_data['Close'].iloc[-1]
        news_items = self.get_news_headlines()

        # Calculate signals
        technical_indicators = self.calculate_technical_indicators()
        technical_signal = self.get_technical_signal(technical_indicators)
        sentiment_signal = self.analyze_news_sentiment(news_items)

        # Combine signals using weights
        combined_signal = (
                technical_signal * self.weights['technical'] +
                sentiment_signal * self.weights['sentiment']
        )

        # Determine action and position size
        position_size = min(abs(combined_signal) * self.max_position_size, self.max_position_size)

        if combined_signal > 0.2:
            action = 'BUY'
        elif combined_signal < -0.2:
            action = 'SELL'
        else:
            action = 'HOLD'
            position_size = 0

        analysis = {
            'technical_signal': technical_signal,
            'sentiment_signal': sentiment_signal,
            'combined_signal': combined_signal,
            'technical_indicators': technical_indicators
        }

        return action, position_size, analysis


def run_trading_system():
    agent = TeslaTradingAgent()

    while True:
        try:
            action, size, analysis = agent.decide_action()

            # Get current price
            current_price = agent.intraday_data['Close'].iloc[-1]

            print("\n=== TESLA TRADING ANALYSIS ===")
            print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"\nRECOMMENDED ACTION: {action}")
            print(f"Position Size: {size * 100:.1f}%")

            print("\nTECHNICAL INDICATORS:")
            print(f"RSI: {analysis['technical_indicators']['rsi']:.2f}")
            print(f"MACD: {analysis['technical_indicators']['macd']:.2f}")
            print(f"Short SMA: {analysis['technical_indicators']['sma_short']:.2f}")
            print(f"Long SMA: {analysis['technical_indicators']['sma_long']:.2f}")

            print("\nSIGNALS:")
            print(f"Technical Signal: {analysis['technical_signal']:.3f}")
            print(f"Sentiment Signal: {analysis['sentiment_signal']:.3f}")
            print(f"Combined Signal: {analysis['combined_signal']:.3f}")

            # Sleep for 1 minute before next update
            time.sleep(60)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait before retrying


if __name__ == "__main__":
    run_trading_system()