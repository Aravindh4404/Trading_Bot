import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class MarketState:
    """Represents the current state of the market."""
    price: float
    volume: float
    sentiment_score: float
    technical_indicators: Dict[str, float]
    volatility: float


class FinancialSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

        # Financial-specific sentiment multipliers
        self.financial_boost_words = {
            'growth': 1.5,
            'profit': 1.3,
            'revenue': 1.2,
            'expansion': 1.4,
            'investment': 1.2,
            'sales': 1.3,
            'increase': 1.2,
            'decrease': 0.8,
            'loss': 0.7,
            'debt': 0.8,
            'lawsuit': 0.6,
            'investigation': 0.7,
            'fine': 0.7,
            'penalty': 0.7
        }

    def extract_content_from_json(self, json_data: str) -> List[str]:
        """Extract and clean content from JSON data."""
        try:
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            messages = [entry["content"] for entry in data["results"]]
            return messages
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """Clean and prepare text for analysis."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[\[\]\(\)\{\}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_financial_metrics(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract key financial metrics and their context."""
        metrics = {
            'revenue': [],
            'profit': [],
            'growth': [],
            'market_position': []
        }

        # Example pattern for capturing "revenue ... X billion"
        revenue_matches = re.finditer(r'revenue.*?\$?\s*(\d+\.?\d*)\s*billion', text, re.IGNORECASE)
        for match in revenue_matches:
            metrics['revenue'].append(('billion', float(match.group(1))))

        # Profit/Loss pattern: captures "profit ... X billion" or "loss ... X billion"
        profit_matches = re.finditer(r'(profit|loss).*?\$?\s*(\d+\.?\d*)\s*billion', text, re.IGNORECASE)
        for match in profit_matches:
            metrics['profit'].append((match.group(1), float(match.group(2))))

        # Growth indicators: look for "increase <X>%", "decrease <X>%", etc.
        growth_patterns = ['increase', 'decrease', 'grew', 'declined']
        for pattern in growth_patterns:
            # Use a raw-string style pattern with an f-string: (\\d+) to avoid invalid escape sequence warning
            growth_matches = re.finditer(rf'{pattern}.*?(\d+)%', text, re.IGNORECASE)
            for gmatch in growth_matches:
                metrics['growth'].append((pattern, float(gmatch.group(1))))

        return metrics

    def analyze_market_sentiment(self, text: str) -> Dict[str, any]:
        """Perform detailed market sentiment analysis."""
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Get base sentiment scores
        base_sentiment = self.analyzer.polarity_scores(cleaned_text)

        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(cleaned_text)

        # Adjust sentiment based on financial context
        adjusted_sentiment = self.adjust_sentiment_for_financial_context(
            base_sentiment,
            cleaned_text,
            financial_metrics
        )

        return {
            'base_sentiment': base_sentiment,
            'adjusted_sentiment': adjusted_sentiment,
            'financial_metrics': financial_metrics,
            'market_signals': self.generate_market_signals(adjusted_sentiment, financial_metrics)
        }

    def adjust_sentiment_for_financial_context(
        self,
        base_sentiment: Dict[str, float],
        text: str,
        metrics: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, float]:
        """Adjust sentiment scores based on financial context."""
        adjusted = base_sentiment.copy()

        # Adjust for financial terms
        for word, multiplier in self.financial_boost_words.items():
            if word in text.lower():
                adjusted['compound'] *= multiplier

        # Adjust for revenue trends
        if metrics['revenue']:
            # Example heuristic: if any revenue item is above $10 billion, slightly boost compound
            if any(amount > 10 for _, amount in metrics['revenue']):
                adjusted['compound'] *= 1.2

        # Adjust for profit/loss
        if metrics['profit']:
            profit_impact = sum(
                amount if ptype.lower() == 'profit' else -amount
                for ptype, amount in metrics['profit']
            )
            # Example: multiply compound by (1 + total_profit_impact * 0.1)
            adjusted['compound'] *= (1 + (profit_impact * 0.1))

        # Normalize compound score to [-1, 1] range
        adjusted['compound'] = max(min(adjusted['compound'], 1.0), -1.0)

        return adjusted

    def generate_market_signals(
        self,
        sentiment: Dict[str, float],
        metrics: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, str]:
        """Generate detailed market signals based on sentiment and metrics."""
        compound_score = sentiment['compound']

        # Determine signal strength
        abs_compound = abs(compound_score)
        if abs_compound > 0.5:
            signal_strength = 'strong'
        elif abs_compound > 0.2:
            signal_strength = 'moderate'
        else:
            signal_strength = 'weak'

        # Base trading decision
        if compound_score > 0.2:
            action = 'BUY'
        elif compound_score < -0.2:
            action = 'SELL'
        else:
            action = 'HOLD'

        # Confidence level: example heuristic
        has_revenue = len(metrics['revenue']) > 0
        has_profit = len(metrics['profit']) > 0
        if has_revenue and has_profit:
            confidence = 'high'
        else:
            confidence = 'medium' if has_revenue or has_profit else 'low'

        return {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'compound_score': compound_score
        }


class AdaptiveTradingAgent:
    def __init__(self, learning_rate: float = 0.01, memory_size: int = 1000):
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.learning_rate = learning_rate
        self.memory = []
        self.memory_size = memory_size
        self.success_threshold = 0.6  # Configurable threshold for strategy adaptation

        # Initialize weights for different factors
        self.weights = {
            'sentiment': 0.3,
            'technical': 0.4,
            'volume': 0.15,
            'volatility': 0.15
        }

        # Track performance of different strategies
        self.strategy_performance = {
            'trend_following': [],
            'mean_reversion': [],
            'sentiment_based': []
        }

    def update_weights(self, reward: float, factors_used: Dict[str, float]):
        """Adapt weights based on strategy performance."""
        for factor, contribution in factors_used.items():
            # Adjust weights using reward feedback
            self.weights[factor] += self.learning_rate * reward * contribution

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate_technical_indicators(self, price_history: List[float]) -> Dict[str, float]:
        """Calculate technical indicators for decision making."""
        prices = np.array(price_history)
        return {
            'price': prices[-1],                 # Current price
            'sma': np.mean(prices[-20:]),        # 20-period simple moving average
            'momentum': prices[-1] - prices[-5], # 5-period momentum
            'volatility': np.std(prices[-20:]),  # 20-period volatility
            'rsi': self._calculate_rsi(prices)   # Relative Strength Index
        }

    def _calculate_rsi(self, prices: np.array, periods: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < periods + 1:
            return 50.0  # Not enough data, return neutral RSI

        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gain[-periods:])
        avg_loss = np.mean(loss[-periods:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def decide_action(self, current_state: MarketState) -> Tuple[str, float]:
        """
        Decide trading action based on current market state and learned weights.
        Returns: (action, confidence)
        """
        # Calculate weighted decision factors
        technical_score = self._evaluate_technical_factors(current_state)
        sentiment_score = current_state.sentiment_score
        volume_score = self._normalize_volume(current_state.volume)
        volatility_score = self._normalize_volatility(current_state.volatility)

        # Combine factors using learned weights
        total_score = (
            self.weights['technical'] * technical_score +
            self.weights['sentiment'] * sentiment_score +
            self.weights['volume'] * volume_score +
            self.weights['volatility'] * volatility_score
        )

        # Determine action and confidence
        if total_score > self.success_threshold:
            return 'BUY', total_score
        elif total_score < -self.success_threshold:
            return 'SELL', abs(total_score)
        else:
            return 'HOLD', abs(total_score)

    def _evaluate_technical_factors(self, state: MarketState) -> float:
        """Evaluate technical indicators to produce a score."""
        indicators = state.technical_indicators

        # Trend signal: compare current price to SMA
        trend_signal = 1.0 if indicators['price'] > indicators['sma'] else -1.0

        # Momentum signal: sign of the 5-period momentum
        momentum_signal = np.sign(indicators['momentum'])

        # RSI signal: overbought (RSI > 70) => -1, oversold (RSI < 30) => +1, else 0
        rsi_signal = -1.0 if indicators['rsi'] > 70 else (1.0 if indicators['rsi'] < 30 else 0.0)

        return np.mean([trend_signal, momentum_signal, rsi_signal])

    def _normalize_volume(self, volume: float) -> float:
        """Normalize volume to [-1, 1] range based on memory average."""
        if not self.memory:
            return 0.0
        avg_volume = np.mean([s.volume for s, _, _ in self.memory])
        if avg_volume == 0:
            return 0.0
        return float(np.clip((volume - avg_volume) / avg_volume, -1, 1))

    def _normalize_volatility(self, volatility: float) -> float:
        """Normalize volatility to [-1, 1] range based on memory average."""
        if not self.memory:
            return 0.0
        avg_volatility = np.mean([s.volatility for s, _, _ in self.memory])
        if avg_volatility == 0:
            return 0.0
        return float(np.clip((volatility - avg_volatility) / avg_volatility, -1, 1))

    def update_memory(self, state: MarketState, action: str, reward: float):
        """Update agent's memory with new experience."""
        self.memory.append((state, action, reward))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def adapt_strategy(self):
        """Adapt trading strategy based on recent performance."""
        if len(self.memory) < 50:  # Need minimum history to adapt
            return

        recent_performance = self._calculate_strategy_performance()

        # Adjust weights based on best-performing strategy
        best_strategy = max(recent_performance.items(), key=lambda x: x[1])[0]
        if best_strategy == 'trend_following':
            self.weights['technical'] *= 1.1
        elif best_strategy == 'mean_reversion':
            self.weights['volatility'] *= 1.1
        elif best_strategy == 'sentiment_based':
            self.weights['sentiment'] *= 1.1

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def _calculate_strategy_performance(self) -> Dict[str, float]:
        """Calculate recent performance of different strategies."""
        recent_memory = self.memory[-50:]  # Look at last 50 trades

        performance = {
            'trend_following': 0.0,
            'mean_reversion': 0.0,
            'sentiment_based': 0.0
        }

        for state, action, reward in recent_memory:
            if self._would_trend_follow(state) == action:
                performance['trend_following'] += reward
            if self._would_mean_revert(state) == action:
                performance['mean_reversion'] += reward
            if self._would_follow_sentiment(state) == action:
                performance['sentiment_based'] += reward

        return performance

    def _would_trend_follow(self, state: MarketState) -> str:
        """Check if a simple trend-following strategy would suggest this action."""
        # If momentum is positive -> BUY, if negative -> SELL, else HOLD
        if state.technical_indicators['momentum'] > 0:
            return 'BUY'
        elif state.technical_indicators['momentum'] < 0:
            return 'SELL'
        return 'HOLD'

    def _would_mean_revert(self, state: MarketState) -> str:
        """Check if a mean-reversion strategy would suggest this action."""
        rsi = state.technical_indicators['rsi']
        if rsi > 70:
            return 'SELL'
        elif rsi < 30:
            return 'BUY'
        return 'HOLD'

    def _would_follow_sentiment(self, state: MarketState) -> str:
        """Check if a sentiment-based strategy would suggest this action."""
        if state.sentiment_score > 0.6:
            return 'BUY'
        elif state.sentiment_score < -0.6:
            return 'SELL'
        return 'HOLD'


def main():
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()

    # Read and parse JSON file (modify path or filename as needed)
    try:
        with open('message.txt', 'r', encoding='utf-8') as file:
            json_data = file.read()

        # Extract content
        messages = analyzer.extract_content_from_json(json_data)
        combined_text = ' '.join(messages)

        # Analyze sentiment
        analysis_result = analyzer.analyze_market_sentiment(combined_text)

        # Print detailed results
        print("\n=== FINANCIAL SENTIMENT ANALYSIS REPORT ===")

        print(f"\nBase Sentiment Scores:")
        for metric, score in analysis_result['base_sentiment'].items():
            print(f"{metric}: {score:.3f}")

        print(f"\nAdjusted Sentiment Scores:")
        for metric, score in analysis_result['adjusted_sentiment'].items():
            print(f"{metric}: {score:.3f}")

        print(f"\nMarket Signals:")
        signals = analysis_result['market_signals']
        print(f"Recommended Action: {signals['action']}")
        print(f"Signal Strength: {signals['signal_strength']}")
        print(f"Confidence Level: {signals['confidence']}")

        print("\nKey Financial Metrics Found:")
        for metric_type, values in analysis_result['financial_metrics'].items():
            if values:
                print(f"{metric_type}: {values}")

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
