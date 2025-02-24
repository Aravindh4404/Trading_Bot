import requests
from bs4 import BeautifulSoup
import json
import time
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class NewsItem:
    headline: str
    content: str
    timestamp: datetime
    source: str
    score: float
    url: str


class StockNewsTrader:
    def __init__(self, symbol: str):
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

        # Initialize stock data
        self.stock = yf.Ticker(symbol)
        self.update_market_data()

    def fetch_detailed_news(self) -> List[NewsItem]:
        """Fetch detailed news articles using Google Search"""
        base_url = "https://www.google.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        search_query = f"{self.symbol} stock news analysis financial report"
        news_items = []

        params = {
            "q": search_query,
            "tbm": "nws",
            "tbs": "qdr:d",
            "hl": "en",
            "gl": "us"
        }

        try:
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('div', class_='SoaBEf') or soup.find_all('div', class_='g')

                for article in articles:
                    # Extract headline and content
                    headline_tag = article.find('div', role='heading') or article.find('h3')
                    headline = headline_tag.text.strip() if headline_tag else ""

                    # Extract snippet/content
                    content_div = article.find('div', class_='VwiC3b')
                    content = content_div.text.strip() if content_div else ""

                    # Extract link
                    link_tag = article.find('a', href=True)
                    link = link_tag['href'] if link_tag and link_tag['href'].startswith('http') else ""

                    # Calculate sentiment
                    combined_text = f"{headline} {content}"
                    sentiment = self.sentiment_analyzer.polarity_scores(combined_text)

                    if headline and content and link:
                        news_items.append(NewsItem(
                            headline=headline,
                            content=content,
                            timestamp=datetime.now(),  # Could be refined to extract actual publication time
                            source="Google News",
                            score=sentiment['compound'],
                            url=link
                        ))

                time.sleep(1)  # Respect rate limits

        except Exception as e:
            print(f"Error fetching news: {e}")

        return news_items

    def update_market_data(self) -> bool:
        """Fetch latest market data"""
        try:
            self.market_data = self.stock.history(period="100d")
            self.intraday_data = self.stock.history(period="1d", interval="1m")
            return True
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return False

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

    def analyze_news_sentiment(self, news_items: List[NewsItem]) -> Tuple[float, Dict]:
        """Calculate weighted sentiment score and provide detailed analysis"""
        if not news_items:
            return 0.0, {}

        # Weight more recent news higher
        total_weight = 0
        weighted_sentiment = 0

        # Detailed sentiment analysis
        sentiment_details = {
            'very_positive': [],
            'positive': [],
            'neutral': [],
            'negative': [],
            'very_negative': []
        }

        for i, news in enumerate(news_items):
            # Exponential decay weight based on position
            weight = np.exp(-0.1 * i)
            weighted_sentiment += news.score * weight
            total_weight += weight

            # Categorize news items
            if news.score >= 0.5:
                sentiment_details['very_positive'].append(news)
            elif 0.5 > news.score >= 0.1:
                sentiment_details['positive'].append(news)
            elif -0.1 < news.score < 0.1:
                sentiment_details['neutral'].append(news)
            elif -0.5 < news.score <= -0.1:
                sentiment_details['negative'].append(news)
            else:
                sentiment_details['very_negative'].append(news)

        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        return avg_sentiment, sentiment_details

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

        # Get latest price and detailed news
        current_price = self.intraday_data['Close'].iloc[-1]
        news_items = self.fetch_detailed_news()

        # Calculate signals
        technical_indicators = self.calculate_technical_indicators()
        technical_signal = self.get_technical_signal(technical_indicators)
        sentiment_signal, sentiment_details = self.analyze_news_sentiment(news_items)

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
            'technical_indicators': technical_indicators,
            'sentiment_details': sentiment_details,
            'news_items': news_items
        }

        return action, position_size, analysis


def save_analysis_report(symbol: str, action: str, size: float, analysis: Dict, filename: str = None):
    """Save detailed analysis report to JSON"""
    if filename is None:
        filename = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert news items to dictionaries
    if 'news_items' in analysis:
        analysis['news_items'] = [
            {
                'headline': item.headline,
                'content': item.content,
                'timestamp': item.timestamp.isoformat(),
                'source': item.source,
                'score': item.score,
                'url': item.url
            }
            for item in analysis['news_items']
        ]

    report = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'position_size': size,
        'analysis': analysis
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return filename


def run_trading_system(symbol: str = "TSLA", interval: int = 60):
    """Run the trading system with specified update interval"""
    trader = StockNewsTrader(symbol)

    while True:
        try:
            action, size, analysis = trader.decide_action()
            current_price = trader.intraday_data['Close'].iloc[-1]

            print("\n=== STOCK TRADING ANALYSIS ===")
            print(f"Symbol: {symbol}")
            print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"\nRECOMMENDED ACTION: {action}")
            print(f"Position Size: {size * 100:.1f}%")

            print("\nTECHNICAL INDICATORS:")
            print(f"RSI: {analysis['technical_indicators']['rsi']:.2f}")
            print(f"MACD: {analysis['technical_indicators']['macd']:.2f}")
            print(f"Short SMA: {analysis['technical_indicators']['sma_short']:.2f}")
            print(f"Long SMA: {analysis['technical_indicators']['sma_long']:.2f}")

            print("\nNEWS SENTIMENT ANALYSIS:")
            print(f"Very Positive News: {len(analysis['sentiment_details']['very_positive'])}")
            print(f"Positive News: {len(analysis['sentiment_details']['positive'])}")
            print(f"Neutral News: {len(analysis['sentiment_details']['neutral'])}")
            print(f"Negative News: {len(analysis['sentiment_details']['negative'])}")
            print(f"Very Negative News: {len(analysis['sentiment_details']['very_negative'])}")

            # Save detailed analysis report
            report_file = save_analysis_report(symbol, action, size, analysis)
            print(f"\nDetailed analysis saved to: {report_file}")

            time.sleep(interval)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    symbol = input("Enter stock symbol to analyze (default: TSLA): ").strip() or "TSLA"
    run_trading_system(symbol)