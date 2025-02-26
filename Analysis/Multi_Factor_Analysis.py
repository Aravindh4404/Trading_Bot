import json
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class NewsItem:
    content: str
    timestamp: datetime
    source: str
    score: float


@dataclass
class MarketEnvironment:
    """Represents the dynamic trading environment"""
    current_price: float
    volume: float
    timestamp: datetime
    news_items: List[NewsItem]
    market_hours: bool
    volatility: float

    def get_state_features(self) -> Dict[str, float]:
        """Extract relevant features from the environment"""
        return {
            'price': self.current_price,
            'volume': self.volume,
            'volatility': self.volatility,
            'time_of_day': self.timestamp.hour + self.timestamp.minute / 60,
            'is_market_hours': float(self.market_hours)
        }


class MarketAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Custom financial terms and their impact weights
        self.financial_terms = {
            'revenue': 2.0,
            'profit': 1.8,
            'growth': 1.5,
            'decline': 0.6,
            'lawsuit': 0.4,
            'investigation': 0.5,
            'partnership': 1.4,
            'expansion': 1.3,
            'delay': 0.7,
            'production': 1.2
        }

    def analyze_news(self, news_items: List[NewsItem]) -> Dict[str, float]:
        """Perform comprehensive news analysis"""
        if not news_items:
            return {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}

        # Weight recent news more heavily
        weighted_scores = []
        current_time = max(news.timestamp for news in news_items)

        for news in news_items:
            # Calculate time decay weight (more recent = higher weight)
            time_diff = (current_time - news.timestamp).total_seconds() / 3600  # hours
            time_weight = np.exp(-0.1 * time_diff)  # exponential decay

            # Get base sentiment
            sentiment = self.sentiment_analyzer.polarity_scores(news.content)

            # Adjust for financial terms
            adjusted_score = self._adjust_for_financial_terms(
                news.content,
                sentiment['compound']
            )

            # Apply source credibility weight
            source_weight = news.score  # Using the provided source score

            weighted_scores.append({
                'score': adjusted_score,
                'weight': time_weight * source_weight
            })

        # Calculate weighted average
        total_weight = sum(ws['weight'] for ws in weighted_scores)
        if total_weight == 0:
            return {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}

        weighted_compound = sum(
            ws['score'] * ws['weight'] for ws in weighted_scores
        ) / total_weight

        return {
            'compound': weighted_compound,
            'neg': min(0, weighted_compound),
            'neu': 1 - abs(weighted_compound),
            'pos': max(0, weighted_compound)
        }

    def _adjust_for_financial_terms(self, text: str, base_score: float) -> float:
        """Adjust sentiment score based on financial terms"""
        adjustment = 1.0

        for term, weight in self.financial_terms.items():
            if term.lower() in text.lower():
                # Count occurrences and adjust impact
                count = text.lower().count(term.lower())
                # Safe logarithmic scaling to prevent numerical issues
                if count * (weight - 1) > 0:
                    adjustment *= (1 + np.log1p(count * (weight - 1)))

        return np.clip(base_score * adjustment, -1, 1)

    def extract_metrics(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract numerical metrics from text"""
        metrics = {
            'revenue': [],
            'production': [],
            'deliveries': [],
            'growth': []
        }

        # Extract revenue figures
        revenue_matches = re.finditer(
            r'revenue.*?\$?\s*(\d+\.?\d*)\s*(billion|million|B|M)',
            text,
            re.IGNORECASE
        )
        for match in revenue_matches:
            try:
                amount = float(match.group(1))
                unit = match.group(2).lower()
                if 'b' in unit:
                    amount *= 1e9
                elif 'm' in unit:
                    amount *= 1e6
                metrics['revenue'].append(('USD', amount))
            except (ValueError, IndexError):
                continue

        # Extract production/delivery numbers
        production_matches = re.finditer(
            r'(produced|delivered)\s*(\d+,?\d*)',
            text,
            re.IGNORECASE
        )
        for match in production_matches:
            try:
                action = match.group(1).lower()
                amount = float(match.group(2).replace(',', ''))
                metrics['production' if 'produce' in action else 'deliveries'].append(
                    ('units', amount)
                )
            except (ValueError, IndexError):
                continue

        # Extract growth percentages
        growth_matches = re.finditer(
            r'(increased|decreased|grew|declined)\s*by\s*(\d+\.?\d*)%',
            text,
            re.IGNORECASE
        )
        for match in growth_matches:
            try:
                direction = match.group(1).lower()
                amount = float(match.group(2))
                if 'decreased' in direction or 'declined' in direction:
                    amount = -amount
                metrics['growth'].append(('percent', amount))
            except (ValueError, IndexError):
                continue

        return metrics


class AdaptiveTradingAgent:
    def __init__(self, learning_rate: float = 0.01):
        self.market_analyzer = MarketAnalyzer()
        self.learning_rate = learning_rate

        # Initialize strategy weights
        self.strategy_weights = {
            'sentiment': 0.3,
            'technical': 0.4,
            'fundamental': 0.3
        }

        # Performance tracking
        self.performance_history = []
        self.strategy_performance = {
            'sentiment': [],
            'technical': [],
            'fundamental': []
        }

        # Risk management parameters
        self.max_position_size = 0.2  # Maximum 20% of portfolio in single position
        self.stop_loss = 0.05  # 5% stop loss
        self.take_profit = 0.15  # 15% take profit

        # Initialize volume history
        self.volume_history = []
        self.max_volume_history = 100  # Keep last 100 volume data points

    def update_volume_history(self, volume: float):
        """Update volume history with new data point"""
        self.volume_history.append(volume)
        if len(self.volume_history) > self.max_volume_history:
            self.volume_history.pop(0)

    def update_strategy_weights(self, performance_metrics: Dict[str, float]):
        """Adapt strategy weights based on performance"""
        total_performance = sum(performance_metrics.values())
        if total_performance == 0:
            return

        # Calculate new weights based on relative performance
        new_weights = {
            strategy: performance / total_performance
            for strategy, performance in performance_metrics.items()
        }

        # Smooth weight changes using learning rate
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] = (
                    (1 - self.learning_rate) * self.strategy_weights[strategy] +
                    self.learning_rate * new_weights[strategy]
            )

    def analyze_environment(self, env: MarketEnvironment) -> Dict[str, float]:
        """Analyze current market environment"""
        # Update volume history
        self.update_volume_history(env.volume)

        # Sentiment analysis
        sentiment_scores = self.market_analyzer.analyze_news(env.news_items)

        # Technical analysis
        technical_signals = self._calculate_technical_signals(env)

        # Fundamental analysis
        fundamental_scores = self._analyze_fundamentals(env)

        # Calculate risk score
        risk_score = self._calculate_risk_score(env)

        # Combine analyses using current strategy weights
        return {
            'sentiment_score': sentiment_scores['compound'] * self.strategy_weights['sentiment'],
            'technical_score': technical_signals['signal'] * self.strategy_weights['technical'],
            'fundamental_score': fundamental_scores['score'] * self.strategy_weights['fundamental'],
            'risk_score': risk_score
        }

    def decide_action(self, env: MarketEnvironment) -> Tuple[str, float, Dict]:
        """Make trading decision based on current environment"""
        analysis = self.analyze_environment(env)

        # Combine signals
        total_signal = (
                analysis['sentiment_score'] +
                analysis['technical_score'] +
                analysis['fundamental_score']
        )

        # Adjust for risk
        risk_adjusted_signal = total_signal * (1 - analysis['risk_score'])

        # Calculate position size based on confidence
        confidence = abs(risk_adjusted_signal)
        position_size = min(
            confidence * self.max_position_size,
            self.max_position_size
        )

        # Determine action
        if risk_adjusted_signal > 0.2:
            action = 'BUY'
        elif risk_adjusted_signal < -0.2:
            action = 'SELL'
        else:
            action = 'HOLD'
            position_size = 0

        return action, position_size, analysis

    def _calculate_technical_signals(self, env: MarketEnvironment) -> Dict[str, float]:
        """Calculate technical indicators and signals"""
        # This is a placeholder returning random values
        # In a real implementation, you would calculate actual technical indicators
        return {'signal': np.random.uniform(-1, 1)}

    def _analyze_fundamentals(self, env: MarketEnvironment) -> Dict[str, float]:
        """Analyze fundamental factors"""
        # Extract metrics from news
        metrics = self.market_analyzer.extract_metrics(
            ' '.join(news.content for news in env.news_items)
        )

        # Calculate score using only available metrics
        score = 0.0
        if metrics['growth']:
            growth_values = [amount for _, amount in metrics['growth']]
            if growth_values:  # Check if list is not empty
                score = np.mean(growth_values)

        return {'score': np.clip(score / 100, -1, 1)}

    def _calculate_risk_score(self, env: MarketEnvironment) -> float:
        """Calculate risk score based on market conditions"""
        # Handle case when volume history is empty
        avg_volume = np.mean(self.volume_history) if self.volume_history else env.volume

        return min(1.0, max(0.0,
                            0.3 * env.volatility +
                            0.3 * (1 - env.market_hours) +
                            0.4 * (1 if env.volume < avg_volume else 0)
                            ))


def process_json_data(json_data: str) -> List[NewsItem]:
    """Process JSON data into NewsItem objects"""
    try:
        data = json.loads(json_data)
        news_items = []

        for result in data['results']:
            news_items.append(NewsItem(
                content=result['content'],
                timestamp=datetime.now(),  # You would normally parse this from the data
                source=result['source'],
                score=result['score']
            ))

        return news_items
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing JSON data: {e}")
        return []


# Example usage
if __name__ == "__main__":
    try:
        agent = AdaptiveTradingAgent()
        analyzer = MarketAnalyzer()

        # Process sample data
        with open('../message.txt', 'r', encoding='utf-8') as file:
            json_data = file.read()

        news_items = process_json_data(json_data)

        # Create sample environment
        env = MarketEnvironment(
            current_price=250.0,
            volume=1000000,
            timestamp=datetime.now(),
            news_items=news_items,
            market_hours=True,
            volatility=0.02
        )

        # Get trading decision
        action, size, analysis = agent.decide_action(env)

        print("\n=== Trading Analysis Report ===")
        print(f"Recommended Action: {action}")
        print(f"Position Size: {size * 100:.1f}%")
        print("\nAnalysis Breakdown:")
        for key, value in analysis.items():
            print(f"{key}: {value:.3f}")

    except Exception as e:
        print(f"An error occurred: {e}")