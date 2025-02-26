import json
import re
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# For sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# For Alpaca API integration
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AlpacaClient:
    """Client for interacting with Alpaca API to retrieve stock market data."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }

    def get_account_info(self) -> Dict:
        """Retrieve account information."""
        endpoint = f"{self.base_url}/v2/account"
        response = requests.get(endpoint, headers=self.headers)
        return self._handle_response(response)

    def get_stock_data(self, symbol: str, timeframe: str = "1D", limit: int = 30) -> Dict:
        """
        Retrieve historical stock data.

        Args:
            symbol: Stock symbol (e.g., 'TSLA')
            timeframe: Time interval ('1D' for daily, '1H' for hourly, etc.)
            limit: Number of data points to retrieve

        Returns:
            Dictionary containing stock data
        """
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=limit)

        # Format dates to ISO format
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build URL for bars endpoint
        endpoint = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        params = {
            "start": start_str,
            "end": end_str,
            "timeframe": timeframe,
            "limit": limit
        }

        response = requests.get(endpoint, headers=self.headers, params=params)
        return self._handle_response(response)

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get the latest quote for a symbol."""
        endpoint = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
        response = requests.get(endpoint, headers=self.headers)
        return self._handle_response(response)

    def get_company_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent news articles about the company."""
        # Note: This is a placeholder - Alpaca's news endpoint may differ
        # Check the most recent Alpaca API documentation for specifics
        endpoint = f"{self.base_url}/v2/news/{symbol}"
        params = {"limit": limit}
        response = requests.get(endpoint, headers=self.headers, params=params)
        return self._handle_response(response)

    def _handle_response(self, response):
        """Handle API response and errors."""
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")


class StockAnalyzer:
    """Performs technical analysis on stock data."""

    @staticmethod
    def calculate_metrics(stock_data: Dict) -> Dict:
        """Calculate technical indicators and metrics from stock data."""
        # Convert Alpaca data to DataFrame
        if 'bars' in stock_data:
            df = pd.DataFrame([{
                'date': bar['t'],
                'open': bar['o'],
                'high': bar['h'],
                'low': bar['l'],
                'close': bar['c'],
                'volume': bar['v']
            } for bar in stock_data['bars']])

            if df.empty:
                return {'error': 'No data available for the specified timeframe'}

            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Calculate basic metrics
            metrics = {}

            # Current price and recent changes
            metrics['current_price'] = df['close'].iloc[-1]
            metrics['daily_change'] = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
            metrics['weekly_change'] = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100 if len(df) >= 5 else 0
            metrics['monthly_change'] = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if len(df) > 0 else 0

            # Volume analysis
            metrics['current_volume'] = df['volume'].iloc[-1]
            metrics['avg_volume'] = df['volume'].mean()
            metrics['volume_change'] = ((df['volume'].iloc[-1] / df['volume'].mean()) - 1) * 100

            # Volatility (standard deviation of returns)
            df['returns'] = df['close'].pct_change()
            metrics['volatility'] = df['returns'].std() * 100

            # Moving averages
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()

            metrics['MA5'] = df['MA5'].iloc[-1] if not pd.isna(df['MA5'].iloc[-1]) else None
            metrics['MA20'] = df['MA20'].iloc[-1] if not pd.isna(df['MA20'].iloc[-1]) else None

            # MA Crossover indicator
            if metrics['MA5'] and metrics['MA20']:
                metrics['ma_trend'] = 'bullish' if metrics['MA5'] > metrics['MA20'] else 'bearish'
                # Recent crossover detection
                if len(df) > 5:
                    prev_ma5 = df['MA5'].iloc[-2]
                    prev_ma20 = df['MA20'].iloc[-2]
                    current_cross = metrics['MA5'] > metrics['MA20']
                    prev_cross = prev_ma5 > prev_ma20
                    if current_cross and not prev_cross:
                        metrics['ma_crossover'] = 'golden_cross'
                    elif not current_cross and prev_cross:
                        metrics['ma_crossover'] = 'death_cross'
                    else:
                        metrics['ma_crossover'] = 'none'

            # Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)

            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()

            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            metrics['RSI'] = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else None

            if metrics['RSI']:
                if metrics['RSI'] < 30:
                    metrics['rsi_signal'] = 'oversold'
                elif metrics['RSI'] > 70:
                    metrics['rsi_signal'] = 'overbought'
                else:
                    metrics['rsi_signal'] = 'neutral'

            # Support and resistance levels (simplified)
            metrics['support'] = df['low'].tail(10).min()
            metrics['resistance'] = df['high'].tail(10).max()

            # Create chart data for visualization
            metrics['chart_data'] = {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'close': df['close'].tolist(),
                'ma5': df['MA5'].tolist(),
                'ma20': df['MA20'].tolist(),
                'volume': df['volume'].tolist()
            }

            return metrics
        else:
            return {'error': 'Invalid data format received from Alpaca API'}


class FinancialSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

        # Financial-specific sentiment dictionary
        # Positive financial terms
        self.financial_positive = {
            'growth': 2.0,
            'profit': 1.8,
            'revenue': 1.5,
            'expansion': 1.7,
            'investment': 1.4,
            'sales': 1.3,
            'increase': 1.5,
            'earnings': 1.6,
            'dividend': 1.5,
            'bullish': 2.0,
            'outperform': 1.8,
            'upgrade': 1.7,
            'beat': 1.5,
            'innovation': 1.6,
            'partnership': 1.4
        }

        # Negative financial terms
        self.financial_negative = {
            'decrease': 1.5,
            'loss': 1.8,
            'debt': 1.5,
            'lawsuit': 1.7,
            'investigation': 1.6,
            'fine': 1.6,
            'penalty': 1.6,
            'bearish': 2.0,
            'underperform': 1.8,
            'downgrade': 1.7,
            'bankruptcy': 2.0,
            'miss': 1.6,
            'decline': 1.5,
            'recall': 1.7,
            'risk': 1.3
        }

        # Stock-specific sentiment modifiers can be added for specific companies
        self.stock_specific_modifiers = {
            'TSLA': {
                'positive': ['musk', 'gigafactory', 'autopilot', 'model y', 'cybertruck', 'deliveries'],
                'negative': ['competition', 'delays', 'twitter', 'regulatory', 'battery fire']
            },
            'AAPL': {
                'positive': ['iphone', 'services', 'ecosystem', 'app store', 'warren buffett'],
                'negative': ['china tariffs', 'app store lawsuit', 'supply chain', 'competition']
            }
            # Can add more stocks as needed
        }

    def extract_content_from_json(self, json_data: str) -> List[str]:
        """Extract and clean content from JSON data."""
        try:
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            messages = []

            # Handle different possible JSON structures
            if "results" in data and isinstance(data["results"], list):
                messages = [entry.get("content", "") for entry in data["results"] if "content" in entry]
            elif "articles" in data and isinstance(data["articles"], list):
                messages = [article.get("summary", "") + " " + article.get("headline", "")
                            for article in data["articles"]]
            elif "news" in data and isinstance(data["news"], list):
                messages = [item.get("summary", "") + " " + item.get("headline", "")
                            for item in data["news"]]

            return messages
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing JSON: {e}")
            # If JSON parsing fails, try to analyze the text directly
            if isinstance(json_data, str):
                return [json_data]
            return []

    def clean_text(self, text: str) -> str:
        """Clean and prepare text for analysis."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[\[\]\(\)\{\}]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_financial_metrics(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract key financial metrics and their context."""
        metrics = {
            'revenue': [],
            'profit': [],
            'growth': [],
            'market_position': [],
            'price_targets': []
        }

        # Revenue pattern with improved regex
        revenue_matches = re.finditer(r'revenue.*?\$?\s*(\d+\.?\d*)\s*(million|billion|trillion)', text, re.IGNORECASE)
        for match in revenue_matches:
            amount = float(match.group(1))
            unit = match.group(2).lower()
            metrics['revenue'].append((unit, amount))

        # Profit/Loss pattern with improved regex
        profit_matches = re.finditer(r'(profit|loss).*?\$?\s*(\d+\.?\d*)\s*(million|billion|trillion)', text,
                                     re.IGNORECASE)
        for match in profit_matches:
            sentiment = match.group(1).lower()
            amount = float(match.group(2))
            metrics['profit'].append((sentiment, amount))

        # Growth indicators with fixed regex
        growth_patterns = ['increase', 'decrease', 'grew', 'declined', 'up', 'down']
        for pattern in growth_patterns:
            matches = re.finditer(f'{pattern}.*?(\\d+(?:\\.\\d+)?)%', text, re.IGNORECASE)
            for match in matches:
                context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)]
                metrics['growth'].append((pattern, float(match.group(1)), context))

        # Market position indicators
        market_terms = ['market share', 'market leader', 'market position', 'competitor']
        for term in market_terms:
            if term in text.lower():
                position_match = re.search(f'{term}.*?(\\b\\w+\\b)', text, re.IGNORECASE)
                if position_match:
                    metrics['market_position'].append((term, position_match.group(1)))

        # Price targets
        price_target_patterns = [
            r'price target.*?\$(\d+\.?\d*)',
            r'target price.*?\$(\d+\.?\d*)',
            r'upgraded.*?to \$(\d+\.?\d*)',
            r'downgraded.*?to \$(\d+\.?\d*)'
        ]

        for pattern in price_target_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                target_price = float(match.group(1))
                context = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                metrics['price_targets'].append((target_price, context))

        return metrics

    def analyze_market_sentiment(self, text: str, stock_symbol: Optional[str] = None) -> Dict[str, any]:
        """Perform detailed market sentiment analysis, optionally with stock-specific context."""
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Get base sentiment scores
        base_sentiment = self.analyzer.polarity_scores(cleaned_text)

        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(cleaned_text)

        # Adjust sentiment based on financial context and stock-specific modifiers
        adjusted_sentiment = self.adjust_sentiment_for_financial_context(
            base_sentiment,
            cleaned_text,
            financial_metrics,
            stock_symbol
        )

        # Generate market signals
        market_signals = self.generate_market_signals(adjusted_sentiment, financial_metrics)

        return {
            'base_sentiment': base_sentiment,
            'adjusted_sentiment': adjusted_sentiment,
            'financial_metrics': financial_metrics,
            'market_signals': market_signals,
            'key_sentences': self.extract_key_sentences(cleaned_text, stock_symbol)
        }

    def extract_key_sentences(self, text: str, stock_symbol: Optional[str] = None) -> List[Dict[str, any]]:
        """Extract and analyze key sentences for more detailed insights."""
        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        key_sentences = []

        for sentence in sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Only analyze sentences relevant to the stock if symbol is provided
            if stock_symbol and stock_symbol.lower() not in sentence.lower():
                # Skip sentences that don't mention the stock
                # But include general market sentiment sentences
                if not any(term in sentence.lower() for term in ['market', 'economy', 'industry', 'sector']):
                    continue

            # Analyze sentiment for each sentence
            sentiment = self.analyzer.polarity_scores(sentence)

            # Apply stock-specific adjustments if applicable
            if stock_symbol and stock_symbol in self.stock_specific_modifiers:
                modifiers = self.stock_specific_modifiers[stock_symbol]

                # Check for positive modifiers
                for term in modifiers['positive']:
                    if term in sentence.lower():
                        sentiment['compound'] += 0.1
                        break

                # Check for negative modifiers
                for term in modifiers['negative']:
                    if term in sentence.lower():
                        sentiment['compound'] -= 0.1
                        break

                # Cap compound score within [-1, 1]
                sentiment['compound'] = max(min(sentiment['compound'], 1.0), -1.0)

            # Only include significant sentences (very positive or very negative)
            if abs(sentiment['compound']) > 0.3:
                key_sentences.append({
                    'text': sentence,
                    'sentiment': sentiment['compound'],
                    'type': 'positive' if sentiment['compound'] > 0 else 'negative'
                })

        # Sort by absolute sentiment strength
        key_sentences.sort(key=lambda x: abs(x['sentiment']), reverse=True)

        # Return top 5 most significant sentences
        return key_sentences[:5]

    def adjust_sentiment_for_financial_context(
            self,
            base_sentiment: Dict[str, float],
            text: str,
            metrics: Dict[str, List],
            stock_symbol: Optional[str] = None
    ) -> Dict[str, float]:
        """Adjust sentiment scores based on financial context and stock-specific factors."""
        adjusted = base_sentiment.copy()
        text_lower = text.lower()

        # Apply positive financial term boosters
        for word, boost in self.financial_positive.items():
            if word in text_lower:
                # Count occurrences to weight impact
                count = text_lower.count(word)
                # Boost positive sentiment
                adjusted['pos'] += 0.05 * count * boost
                # Reduce negative sentiment slightly
                adjusted['neg'] = max(0, adjusted['neg'] - 0.02 * count)

        # Apply negative financial term boosters
        for word, boost in self.financial_negative.items():
            if word in text_lower:
                # Count occurrences to weight impact
                count = text_lower.count(word)
                # Boost negative sentiment
                adjusted['neg'] += 0.05 * count * boost
                # Reduce positive sentiment slightly
                adjusted['pos'] = max(0, adjusted['pos'] - 0.02 * count)

        # Apply stock-specific modifiers if available
        if stock_symbol and stock_symbol in self.stock_specific_modifiers:
            modifiers = self.stock_specific_modifiers[stock_symbol]

            # Apply positive stock-specific modifiers
            for term in modifiers['positive']:
                if term in text_lower:
                    count = text_lower.count(term)
                    adjusted['pos'] += 0.07 * count

            # Apply negative stock-specific modifiers
            for term in modifiers['negative']:
                if term in text_lower:
                    count = text_lower.count(term)
                    adjusted['neg'] += 0.07 * count

        # Adjust for revenue trends
        if metrics['revenue']:
            # High revenue is generally positive
            total_revenue = sum(amount for _, amount in metrics['revenue'])
            if total_revenue > 10:  # Billions
                adjusted['pos'] *= 1.2
            elif total_revenue > 100:  # Millions
                adjusted['pos'] *= 1.1

        # Adjust for profit/loss
        if metrics['profit']:
            for type_val, amount in metrics['profit']:
                if type_val == 'profit':
                    adjusted['pos'] *= (1 + min(amount * 0.05, 0.5))
                elif type_val == 'loss':
                    adjusted['neg'] *= (1 + min(amount * 0.05, 0.5))

        # Adjust for growth trends
        if metrics['growth']:
            for pattern, percentage, _ in metrics['growth']:
                if pattern in ['increase', 'grew', 'up']:
                    adjusted['pos'] *= (1 + min(percentage * 0.01, 0.5))
                elif pattern in ['decrease', 'declined', 'down']:
                    adjusted['neg'] *= (1 + min(percentage * 0.01, 0.5))

        # Adjust for price targets
        if metrics['price_targets']:
            # This would require comparing to current price
            # For now, just count positive vs. negative mentions
            price_target_count = len(metrics['price_targets'])
            if price_target_count > 0:
                adjusted['pos'] *= (1 + min(price_target_count * 0.05, 0.3))

        # Recalculate compound score
        # Simple approximation based on positive and negative scores
        adjusted['compound'] = (adjusted['pos'] - adjusted['neg']) / (
                    adjusted['pos'] + adjusted['neg'] + adjusted['neu'])

        # Normalize compound score to [-1, 1] range
        adjusted['compound'] = max(min(adjusted['compound'], 1.0), -1.0)

        return adjusted

    def generate_market_signals(
            self,
            sentiment: Dict[str, float],
            metrics: Dict[str, List]
    ) -> Dict[str, any]:
        """Generate detailed market signals based on sentiment and metrics."""
        compound_score = sentiment['compound']

        # Initialize signal strength indicators
        signal_strength = 'strong' if abs(compound_score) > 0.5 else 'moderate' if abs(compound_score) > 0.2 else 'weak'

        # Establish confidence level based on metrics presence and consistency
        has_revenue = len(metrics['revenue']) > 0
        has_profit = len(metrics['profit']) > 0
        has_growth = len(metrics['growth']) > 0
        has_price_targets = len(metrics['price_targets']) > 0

        # Calculate metric consistency
        consistency_score = 0
        if has_growth:
            # Check if growth metrics are consistent (all positive or all negative)
            growth_sentiments = [pattern in ['increase', 'grew', 'up'] for pattern, _, _ in metrics['growth']]
            if all(growth_sentiments) or not any(growth_sentiments):
                consistency_score += 1

        if has_profit:
            # Check if profit metrics are consistent
            profit_sentiments = [type_val == 'profit' for type_val, _ in metrics['profit']]
            if all(profit_sentiments) or not any(profit_sentiments):
                consistency_score += 1

        # Determine confidence based on data richness and consistency
        data_richness = sum([has_revenue, has_profit, has_growth, has_price_targets])
        if data_richness >= 2 and consistency_score >= 1:
            confidence = 'high'
        elif data_richness >= 1:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Determine action based on sentiment score and confidence
        if compound_score > 0.3 and confidence != 'low':
            action = 'BUY'
            rationale = "Strongly positive sentiment indicates favorable market conditions"
        elif compound_score < -0.3 and confidence != 'low':
            action = 'SELL'
            rationale = "Strongly negative sentiment suggests unfavorable outlook"
        elif 0.1 <= compound_score <= 0.3:
            action = 'HOLD/BUY'
            rationale = "Moderately positive sentiment with some upside potential"
        elif -0.3 <= compound_score <= -0.1:
            action = 'HOLD/SELL'
            rationale = "Moderately negative sentiment with some downside risk"
        else:
            action = 'HOLD'
            rationale = "Neutral sentiment suggests waiting for clearer signals"

        return {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'compound_score': compound_score,
            'rationale': rationale,
            'supporting_metrics': {
                'has_revenue_data': has_revenue,
                'has_profit_data': has_profit,
                'has_growth_data': has_growth,
                'has_price_targets': has_price_targets,
                'metric_consistency': consistency_score
            }
        }


class StockTradeAnalyzer:
    """Combines technical analysis and sentiment analysis to generate trading recommendations."""

    def __init__(self, api_key: str, api_secret: str):
        self.alpaca_client = AlpacaClient(api_key, api_secret)
        self.stock_analyzer = StockAnalyzer()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()

    def analyze_stock(self, symbol: str, news_text: Optional[str] = None) -> Dict:
        """
        Perform comprehensive analysis on a stock using both technical and sentiment data.

        Args:
            symbol: Stock symbol (e.g., 'TSLA')
            news_text: Optional news text to analyze for sentiment

        Returns:
            Dictionary containing analysis results and trading recommendation
        """
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'technical_analysis': {},
            'sentiment_analysis': {},
            'combined_recommendation': {}
        }

        # Get technical data
        try:
            stock_data = self.alpaca_client.get_stock_data(symbol)
            technical_metrics = self.stock_analyzer.calculate_metrics(stock_data)
            results['technical_analysis'] = technical_metrics
        except Exception as e:
            results['technical_analysis'] = {'error': str(e)}

        # Get sentiment data
        try:
            # Try to get news from Alpaca if no text provided
            if not news_text:
                try:
                    news_data = self.alpaca_client.get_company_news(symbol)
                    news_text = ' '.join([item.get('summary', '') + ' ' + item.get('headline', '')
                                          for item in news_data])
                except Exception as news_error:
                    # Use a sample text if API fails
                    news_text = f"Recent news about {symbol} shows the company has been performing well."
                    results['sentiment_analysis']['news_warning'] = f"Could not fetch real news: {str(news_error)}"

            sentiment_results = self.sentiment_analyzer.analyze_market_sentiment(news_text, symbol)
            results['sentiment_analysis'] = sentiment_results
        except Exception as e:
            results['sentiment_analysis'] = {'error': str(e)}

        # Generate combined recommendation
        if 'error' not in results['technical_analysis'] and 'error' not in results['sentiment_analysis']:
            results['combined_recommendation'] = self._generate_trading_recommendation(
                results['technical_analysis'],
                results['sentiment_analysis']
            )

        return results

    def _generate_trading_recommendation(self, technical: Dict, sentiment: Dict) -> Dict:
        """
        Generate a combined trading recommendation based on technical and sentiment analysis.

        Args:
            technical: Technical analysis metrics
            sentiment: Sentiment analysis results

        Returns:
            Dictionary with trading recommendation
        """
        recommendation = {
            'action': 'HOLD',  # Default action
            'confidence': 'low',
            'factors': {
                'technical': [],
                'sentiment': []
            },
            'risk_level': 'medium',
            'summary': ""
        }

        # Technical factors (positive)
        if 'ma_trend' in technical and technical['ma_trend'] == 'bullish':
            recommendation['factors']['technical'].append("Bullish moving average trend")

        if 'ma_crossover' in technical and technical['ma_crossover'] == 'golden_cross':
            recommendation['factors']['technical'].append("Recent golden cross (bullish)")

        if 'rsi_signal' in technical and technical['rsi_signal'] == 'oversold':
            recommendation['factors']['technical'].append(
                "RSI indicates oversold conditions (potential buying opportunity)")

        if 'daily_change' in technical and technical['daily_change'] > 2:
            recommendation['factors']['technical'].append(
                f"Strong daily price increase ({technical['daily_change']:.2f}%)")

        # Technical factors (negative)
        if 'ma_trend' in technical and technical['ma_trend'] == 'bearish':
            recommendation['factors']['technical'].append("Bearish moving average trend")

        if 'ma_crossover' in technical and technical['ma_crossover'] == 'death_cross':
            recommendation['factors']['technical'].append("Recent death cross (bearish)")

        if 'rsi_signal' in technical and technical['rsi_signal'] == 'overbought':
            recommendation['factors']['technical'].append(
                "RSI indicates overbought conditions (potential selling opportunity)")

        if 'daily_change' in technical and technical['daily_change'] < -2:
            recommendation['factors']['technical'].append(
                f"Strong daily price decrease ({technical['daily_change']:.2f}%)")

        # Sentiment factors
        market_signals = sentiment['market_signals']
        recommendation['factors']['sentiment'].append(
            f"News sentiment: {market_signals['signal_strength']} {market_signals['action']} signal with {market_signals['confidence']} confidence"
        )

        # Add key insights from sentiment analysis
        for idx, sentence in enumerate(sentiment['key_sentences'][:2]):
            sentiment_type = "positive" if sentence['sentiment'] > 0 else "negative"
            recommendation['factors']['sentiment'].append(
                f"Key {sentiment_type} news: {sentence['text']}"
            )

        # Determine confidence level
        tech_confidence = 'medium'
        sent_confidence = market_signals['confidence']

        if len(recommendation['factors']['technical']) >= 3:
            tech_confidence = 'high'
        elif len(recommendation['factors']['technical']) <= 1:
            tech_confidence = 'low'

        # Combine confidence levels
        if tech_confidence == sent_confidence:
            recommendation['confidence'] = tech_confidence
        elif tech_confidence == 'high' and sent_confidence == 'medium' or \
                tech_confidence == 'medium' and sent_confidence == 'high':
            recommendation['confidence'] = 'high'

# Combine confidence levels
        if tech_confidence == sent_confidence:
            recommendation['confidence'] = tech_confidence
        elif tech_confidence == 'high' and sent_confidence == 'medium' or \
                tech_confidence == 'medium' and sent_confidence == 'high':
            recommendation['confidence'] = 'high'
        else:
            # If one is low and the other is high/medium, favor the higher confidence signal but with reduced confidence
            higher_confidence = max(tech_confidence, sent_confidence, key=lambda x: {'low': 0, 'medium': 1, 'high': 2}[x])
            recommendation['confidence'] = 'medium' if higher_confidence == 'high' else 'low'

        # Determine final trading action based on technical and sentiment signals
        tech_bullish_count = sum(1 for factor in recommendation['factors']['technical'] if
                                 'bullish' in factor.lower() or 'oversold' in factor.lower() or 'increase' in factor.lower())
        tech_bearish_count = sum(1 for factor in recommendation['factors']['technical'] if
                                 'bearish' in factor.lower() or 'overbought' in factor.lower() or 'decrease' in factor.lower())

        tech_signal = 'bullish' if tech_bullish_count > tech_bearish_count else 'bearish' if tech_bearish_count > tech_bullish_count else 'neutral'
        sent_signal = 'bullish' if market_signals['action'] in ['BUY', 'HOLD/BUY'] else 'bearish' if market_signals[
                                                                                                         'action'] in ['SELL',
                                                                                                                       'HOLD/SELL'] else 'neutral'

        # Weighted decision making
        if tech_signal == sent_signal:
            # Strong agreement between technical and sentiment
            if tech_signal == 'bullish':
                recommendation['action'] = 'BUY'
            elif tech_signal == 'bearish':
                recommendation['action'] = 'SELL'
            else:
                recommendation['action'] = 'HOLD'
        elif tech_signal == 'neutral' or sent_signal == 'neutral':
            # One signal is neutral, follow the other signal
            primary_signal = sent_signal if tech_signal == 'neutral' else tech_signal
            if primary_signal == 'bullish':
                recommendation['action'] = 'HOLD/BUY'
            elif primary_signal == 'bearish':
                recommendation['action'] = 'HOLD/SELL'
        else:
            # Conflicting signals, consider confidence levels
            if recommendation['confidence'] == 'high':
                # With high confidence, follow the trend based on recent momentum
                if 'daily_change' in technical:
                    if technical['daily_change'] > 0:
                        recommendation['action'] = 'HOLD/BUY'
                    else:
                        recommendation['action'] = 'HOLD/SELL'
                else:
                    recommendation['action'] = 'HOLD'
            else:
                # With lower confidence and conflicting signals, hold
                recommendation['action'] = 'HOLD'

        # Determine risk level
        risk_factors = 0

        # Technical risk indicators
        if 'volatility' in technical and technical['volatility'] > 3:
            risk_factors += 1
        if 'volume_change' in technical and abs(technical['volume_change']) > 50:
            risk_factors += 1
        if 'rsi_signal' in technical and technical['rsi_signal'] in ['overbought', 'oversold']:
            risk_factors += 1

        # Sentiment risk indicators
        if abs(market_signals['compound_score']) > 0.7:
            risk_factors += 1

        # Set risk level
        if risk_factors >= 3:
            recommendation['risk_level'] = 'high'
        elif risk_factors >= 1:
            recommendation['risk_level'] = 'medium'
        else:
            recommendation['risk_level'] = 'low'

        # Generate summary
        recommendation['summary'] = self._generate_summary(recommendation, technical, sentiment)

        return recommendation


    def _generate_summary(self, recommendation: Dict, technical: Dict, sentiment: Dict) -> str:
        """Generate a human-readable summary of the trading recommendation."""
        action = recommendation['action']
        confidence = recommendation['confidence']
        risk_level = recommendation['risk_level']

        # Start with the recommendation
        summary = f"{action} recommendation with {confidence} confidence and {risk_level} risk. "

        # Add key technical points
        if technical:
            if 'current_price' in technical:
                summary += f"Current price: ${technical['current_price']:.2f}. "

            if 'daily_change' in technical:
                change_text = "up" if technical['daily_change'] > 0 else "down"
                summary += f"Price {change_text} {abs(technical['daily_change']):.2f}% today. "

            if 'ma_trend' in technical:
                summary += f"{technical['ma_trend'].capitalize()} MA trend. "

            if 'RSI' in technical and technical['RSI'] is not None:
                summary += f"RSI: {technical['RSI']:.1f}. "

        # Add sentiment summary
        if 'market_signals' in sentiment:
            market_signals = sentiment['market_signals']
            summary += f"News sentiment: {market_signals['signal_strength']} {market_signals['confidence']}. "

        # Add rationale
        summary += "Rationale: "
        # Include top factors (up to 2 each from technical and sentiment)
        factor_count = 0
        for category in ['technical', 'sentiment']:
            factors = recommendation['factors'][category][:2]
            if factors:
                summary += "; ".join(factors) + ". "
                factor_count += len(factors)

        if factor_count == 0:
            summary += "Insufficient signals for a strong recommendation."

        return summary


    def visualize_analysis(self, analysis_results: Dict, output_path: str = None) -> None:
        """
        Create visualization of the analysis results.

        Args:
            analysis_results: The output from analyze_stock
            output_path: Optional path to save the visualization
        """
        technical = analysis_results.get('technical_analysis', {})

        if 'error' in technical or 'chart_data' not in technical:
            print("Error: Cannot visualize analysis - technical data missing or invalid")
            return

        # Extract chart data
        chart_data = technical['chart_data']
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in chart_data['dates']]
        close_prices = chart_data['close']
        ma5 = chart_data['ma5']
        ma20 = chart_data['ma20']
        volumes = chart_data['volume']

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot price and moving averages
        ax1.plot(dates, close_prices, label='Close Price', color='black')
        ax1.plot(dates, ma5, label='5-Day MA', color='blue', linestyle='--')
        ax1.plot(dates, ma20, label='20-Day MA', color='red', linestyle='--')

        # Mark supports and resistances if available
        if 'support' in technical and 'resistance' in technical:
            ax1.axhline(y=technical['support'], color='green', linestyle='-', alpha=0.3, label='Support')
            ax1.axhline(y=technical['resistance'], color='red', linestyle='-', alpha=0.3, label='Resistance')

        # Add RSI level indicators if available
        if 'RSI' in technical and technical['RSI'] is not None:
            y_min, y_max = ax1.get_ylim()
            y_text = y_min + (y_max - y_min) * 0.95
            ax1.text(dates[-1], y_text, f"RSI: {technical['RSI']:.1f}",
                     bbox=dict(facecolor='white', alpha=0.7))

        # Highlight recommendation
        recommendation = analysis_results.get('combined_recommendation', {})
        if recommendation and 'action' in recommendation:
            color_map = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange',
                         'HOLD/BUY': 'lightgreen', 'HOLD/SELL': 'salmon'}
            action = recommendation['action']
            color = color_map.get(action, 'gray')

            y_min, y_max = ax1.get_ylim()
            y_text = y_min + (y_max - y_min) * 0.05

            ax1.text(dates[0], y_text,
                     f"{action} ({recommendation.get('confidence', 'medium')} confidence)",
                     fontsize=12, weight='bold', color=color,
                     bbox=dict(facecolor='white', alpha=0.7))

        # Volume chart
        ax2.bar(dates, volumes, color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')

        # Formatting
        ax1.set_title(f"{analysis_results['symbol']} - Stock Analysis", fontsize=16)
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        fig.tight_layout()

        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()


    def batch_analyze(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Analyze multiple stocks and compare their metrics.

        Args:
            symbols: List of stock symbols to analyze

        Returns:
            Dictionary with analysis results for each stock and comparative rankings
        """
        results = {}
        compare_metrics = {
            'technical_score': {},
            'sentiment_score': {},
            'combined_score': {},
            'risk_level': {},
            'price_change': {}
        }

        # Analyze each stock
        for symbol in symbols:
            try:
                stock_result = self.analyze_stock(symbol)
                results[symbol] = stock_result

                # Extract key metrics for comparison
                technical = stock_result.get('technical_analysis', {})
                sentiment = stock_result.get('sentiment_analysis', {})
                recommendation = stock_result.get('combined_recommendation', {})

                # Calculate technical score (-10 to 10 scale)
                tech_score = 0
                if 'ma_trend' in technical:
                    tech_score += 3 if technical['ma_trend'] == 'bullish' else -3
                if 'rsi_signal' in technical:
                    if technical['rsi_signal'] == 'oversold':
                        tech_score += 2
                    elif technical['rsi_signal'] == 'overbought':
                        tech_score -= 2
                if 'daily_change' in technical:
                    tech_score += min(max(technical['daily_change'] / 2, -5), 5)  # Scale daily change

                # Calculate sentiment score (-10 to 10 scale)
                sent_score = 0
                if 'market_signals' in sentiment:
                    sent_score = sentiment['market_signals'].get('compound_score', 0) * 10

                # Combined score - weighted average
                combined_score = (tech_score * 0.6) + (sent_score * 0.4)

                # Store metrics for comparison
                compare_metrics['technical_score'][symbol] = tech_score
                compare_metrics['sentiment_score'][symbol] = sent_score
                compare_metrics['combined_score'][symbol] = combined_score
                compare_metrics['risk_level'][symbol] = {'low': 1, 'medium': 2, 'high': 3}.get(
                    recommendation.get('risk_level', 'medium'), 2)
                compare_metrics['price_change'][symbol] = technical.get('daily_change', 0)

            except Exception as e:
                results[symbol] = {'error': str(e)}

        # Generate rankings
        rankings = {
            'overall_ranking': sorted(
                compare_metrics['combined_score'].keys(),
                key=lambda x: compare_metrics['combined_score'][x],
                reverse=True
            ),
            'technical_ranking': sorted(
                compare_metrics['technical_score'].keys(),
                key=lambda x: compare_metrics['technical_score'][x],
                reverse=True
            ),
            'sentiment_ranking': sorted(
                compare_metrics['sentiment_score'].keys(),
                key=lambda x: compare_metrics['sentiment_score'][x],
                reverse=True
            ),
            'lowest_risk': sorted(
                compare_metrics['risk_level'].keys(),
                key=lambda x: compare_metrics['risk_level'][x]
            ),
            'highest_momentum': sorted(
                compare_metrics['price_change'].keys(),
                key=lambda x: compare_metrics['price_change'][x],
                reverse=True
            )
        }

        return {
            'individual_results': results,
            'rankings': rankings,
            'metrics': compare_metrics
        }


# Example usage of the StockTradeAnalyzer class
if __name__ == "__main__":
    # API credentials would be loaded from environment variables in production
    API_KEY = os.getenv("ALPACA_API_KEY", "PKWLZQFMLRKZWJBQOJNB")
    API_SECRET = os.getenv("ALPACA_API_SECRET", "8KBUACQjvvno7XzRwtgLkbjMGUoI1RZMcmDVLICW")

    # Create analyzer
    analyzer = StockTradeAnalyzer(API_KEY, API_SECRET)

    # Example usage for single stock analysis
    symbol = "AAPL"
    print(f"Analyzing {symbol}...")

    # Sample news text (would typically come from API)
    sample_news = """
    Apple Inc. reported quarterly earnings that exceeded analyst expectations. 
    Revenue increased by 8% year-over-year to $90.1 billion, while profit grew by 9% to $22.6 billion.
    The company's services division continues to show strong growth at 15%, reaching $19.2 billion in revenue.
    However, iPhone sales were slightly below expectations amid supply chain constraints.
    Analysts have raised their price targets to $180 following the earnings report.
    """

    # Analyze the stock
    analysis_results = analyzer.analyze_stock(symbol, sample_news)

    # Print the recommendation
    recommendation = analysis_results.get('combined_recommendation', {})
    print(f"Recommendation: {recommendation.get('action', 'N/A')}")
    print(f"Confidence: {recommendation.get('confidence', 'N/A')}")
    print(f"Risk Level: {recommendation.get('risk_level', 'N/A')}")
    print(f"Summary: {recommendation.get('summary', 'N/A')}")

    # Create visualization
    analyzer.visualize_analysis(analysis_results, "apple_analysis.png")

    # Example of batch analysis
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    batch_results = analyzer.batch_analyze(symbols)

    # Print rankings
    print("\nOverall Rankings:")
    for i, symbol in enumerate(batch_results['rankings']['overall_ranking']):
        score = batch_results['metrics']['combined_score'][symbol]
        print(f"{i + 1}. {symbol}: Score {score:.2f}")