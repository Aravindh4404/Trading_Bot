import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from typing import Dict, List, Tuple


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

        # Revenue pattern
        revenue_matches = re.finditer(r'revenue.*?\$?\s*(\d+\.?\d*)\s*billion', text, re.IGNORECASE)
        for match in revenue_matches:
            metrics['revenue'].append(('billion', float(match.group(1))))

        # Profit/Loss pattern
        profit_matches = re.finditer(r'(profit|loss).*?\$?\s*(\d+\.?\d*)\s*billion', text, re.IGNORECASE)
        for match in profit_matches:
            metrics['profit'].append((match.group(1), float(match.group(2))))

        # Growth indicators
        growth_patterns = ['increase', 'decrease', 'grew', 'declined']
        for pattern in growth_patterns:
            if re.search(f'{pattern}.*?(\d+)%', text, re.IGNORECASE):
                match = re.search(f'{pattern}.*?(\d+)%', text, re.IGNORECASE)
                metrics['growth'].append((pattern, float(match.group(1))))

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
            if any(amount > 10 for _, amount in metrics['revenue']):
                adjusted['compound'] *= 1.2

        # Adjust for profit/loss
        if metrics['profit']:
            profit_impact = sum(amount if type == 'profit' else -amount
                                for type, amount in metrics['profit'])
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

        # Initialize signal strength indicators
        signal_strength = 'strong' if abs(compound_score) > 0.5 else 'moderate' if abs(compound_score) > 0.2 else 'weak'

        # Base trading decision
        if compound_score > 0.2:
            action = 'BUY'
        elif compound_score < -0.2:
            action = 'SELL'
        else:
            action = 'HOLD'

        # Confidence level based on metrics presence
        confidence = 'high' if len(metrics['revenue']) > 0 and len(metrics['profit']) > 0 else 'medium'

        return {
            'action': action,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'compound_score': compound_score
        }


def main():
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()

    try:
        # Read and parse JSON file
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